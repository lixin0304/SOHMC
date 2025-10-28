"""
直接核岭回归对冲模型 (Direct Kernel Ridge Regression)
基于论文: Learning minimum variance discrete hedging directly from the market
"""
import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DirectKRRHedging:
    """
    直接核岭回归对冲模型（优化版）
    按照论文：Learning minimum variance discrete hedging directly from the market
    """
    
    def __init__(self, max_samples=10000, regularization_type='kernel'):
        """
        初始化
        
        参数:
        - max_samples: 最大训练样本数（防止内存溢出）
        - regularization_type: 'kernel' 使用λα^TKα，'identity' 使用λα^Tα（更快）
        """
        self.max_samples = max_samples
        self.regularization_type = regularization_type
        self.alpha = None
        self.X_support = None
        self.gamma = None
        self.lambda_ = None
        
    def prepare_data(self, df, shift=1):
        """准备训练数据（优化版）"""
        X_list = []
        DV_list = []
        DS_list = []
        
        # 按合约分组处理
        for contract_id in df['Contract_ID'].unique():
            contract_df = df[df['Contract_ID'] == contract_id].sort_values('Time_Step')
            
            if len(contract_df) <= shift:
                continue
            
            # 向量化计算差分
            values = contract_df[['Market_Price', 'Stock_Price']].values
            diffs = values[shift:] - values[:-shift]
            
            # 提取有效数据（ΔS筛选）
            valid_idx = np.abs(diffs[:, 1]) > 1e-6  # 避免除零
            
            if valid_idx.sum() == 0:
                continue
            
            # 特征
            features = contract_df[['Moneyness', 'Time_to_Expiry', 'BS_Delta']].values[:-shift][valid_idx]
            dv = diffs[valid_idx, 0]  # ΔV
            ds = diffs[valid_idx, 1]  # ΔS
            
            X_list.append(features)
            DV_list.append(dv)
            DS_list.append(ds)
        
        if not X_list:
            return np.array([]), np.array([]), np.array([])
        
        X = np.vstack(X_list)
        DV = np.concatenate(DV_list)
        DS = np.concatenate(DS_list)
        
        # 限制样本数量（随机采样）
        if len(X) > self.max_samples:
            idx = np.random.choice(len(X), self.max_samples, replace=False)
            X, DV, DS = X[idx], DV[idx], DS[idx]
        
        logging.info(f"准备数据完成: {len(X)} 个样本")
        
        return X, DV, DS
    
    def compute_gamma_from_bandwidth(self, X, percentile=50):
        """
        按照论文方法计算gamma
        使用成对欧氏距离的中位数作为带宽ρ
        gamma = 1 / (2 * ρ^2)
        """
        # 计算成对距离
        distances = pairwise_distances(X, metric='euclidean')
        
        # 只取上三角（避免重复和对角线0）
        triu_indices = np.triu_indices_from(distances, k=1)
        dist_values = distances[triu_indices]
        
        # 使用中位数作为带宽
        if percentile is not None:
            rho = np.percentile(dist_values, percentile)
        else:
            rho = np.std(dist_values)
        
        gamma = 1.0 / (2 * rho ** 2)
        
        logging.info(f"自动计算的gamma: {gamma:.6f} (bandwidth ρ={rho:.4f})")
        
        return gamma
    
    def solve_krr(self, K, DV, DS, lambda_):
        """
        求解KRR（按论文公式）
        
        论文公式(24): min_α (DKα - ΔṼ)^T(DKα - ΔṼ) + λα^TKα
        论文公式(25): min_α (DKα - ΔṼ)^T(DKα - ΔṼ) + λα^Tα（快速版）
        """
        n = K.shape[0]
        
        # 构造 M = diag(ΔS) @ K
        M = DS[:, np.newaxis] * K
        
        # 根据正则化类型选择公式
        if self.regularization_type == 'kernel':
            # 公式(24): A = M^T M + λK
            A = M.T @ M + lambda_ * K
        else:
            # 公式(25): A = M^T M + λI（用于交叉验证加速）
            A = M.T @ M + lambda_ * np.eye(n)
        
        b = M.T @ DV
        
        try:
            # 使用Cholesky分解（更快、更稳定）
            A_reg = A + 1e-10 * np.eye(n)
            L = np.linalg.cholesky(A_reg)
            y = np.linalg.solve(L, b)
            alpha = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # 退化到标准求解
            logging.warning("Cholesky分解失败，使用lstsq")
            alpha = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        
        return alpha
    
    def objective_function(self, X_train, DV_train, DS_train, X_val, DV_val, DS_val):
        """创建贝叶斯优化的目标函数"""
        def objective(params):
            gamma, lambda_ = params
            
            try:
                # 计算核矩阵
                K_train = rbf_kernel(X_train, X_train, gamma=gamma)
                K_val = rbf_kernel(X_val, X_train, gamma=gamma)
                
                # 训练
                alpha = self.solve_krr(K_train, DV_train, DS_train, lambda_)
                
                # 验证
                delta_pred = K_val @ alpha
                errors = DV_val - delta_pred * DS_val
                mse = np.mean(errors ** 2)
                
                return mse
                
            except Exception as e:
                logging.warning(f"优化目标函数失败: {e}")
                return 1e10
        
        return objective
    
    def optimize_hyperparameters(self, X_train, DV_train, DS_train, X_val, DV_val, DS_val, 
                                  n_calls=20, gamma_init=None):
        """
        使用贝叶斯优化寻找最优超参数
        
        参数:
        - gamma_init: 如果提供，将作为gamma搜索范围的中心
        """
        logging.info(f"开始贝叶斯优化 (n_calls={n_calls})...")
        
        # 如果没有提供初始gamma，自动计算
        if gamma_init is None:
            gamma_init = self.compute_gamma_from_bandwidth(X_train)
        
        # 定义搜索空间（以gamma_init为中心）
        gamma_range = [gamma_init / 100, gamma_init * 100]
        
        search_space = [
            Real(gamma_range[0], gamma_range[1], name='gamma', prior='log-uniform'),
            Real(1e-6, 1e3, name='lambda', prior='log-uniform')
        ]
        
        logging.info(f"Gamma搜索范围: [{gamma_range[0]:.6f}, {gamma_range[1]:.6f}]")
        
        # 目标函数
        objective = self.objective_function(X_train, DV_train, DS_train, X_val, DV_val, DS_val)
        
        # 贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            n_initial_points=min(5, n_calls // 2),
            acq_func='EI',
            random_state=42,
            verbose=False
        )
        
        best_gamma, best_lambda = result.x
        best_score = result.fun
        
        logging.info(f"最优参数: gamma={best_gamma:.6f}, lambda={best_lambda:.6f}, MSE={best_score:.6f}")
        
        return best_gamma, best_lambda
    
    def fit(self, train_df, shift=1, val_ratio=0.2, n_calls=20):
        """训练模型"""
        # 准备数据
        X, DV, DS = self.prepare_data(train_df, shift)
        
        if len(X) == 0:
            raise ValueError("没有有效的训练数据")
        
        # 划分训练/验证集
        X_train, X_val, DV_train, DV_val, DS_train, DS_val = train_test_split(
            X, DV, DS, test_size=val_ratio, random_state=42
        )
        
        # 优化超参数
        self.gamma, self.lambda_ = self.optimize_hyperparameters(
            X_train, DV_train, DS_train,
            X_val, DV_val, DS_val,
            n_calls=n_calls
        )
        
        # 在完整数据上重新训练
        K = rbf_kernel(X, X, gamma=self.gamma)
        self.alpha = self.solve_krr(K, DV, DS, self.lambda_)
        self.X_support = X
        
        logging.info(f"模型训练完成，支持向量数: {len(self.X_support)}")
        logging.info(f"最终参数: gamma={self.gamma:.6f}, lambda={self.lambda_:.6f}")
    
    def predict(self, X):
        """预测Delta"""
        if self.alpha is None:
            raise ValueError("模型未训练")
        
        # 计算核
        K = rbf_kernel(X, self.X_support, gamma=self.gamma)
        
        # 预测
        delta = K @ self.alpha
        
        # 约束到合理范围（看涨期权delta在[0,1]）
        return np.clip(delta, 0, 1)

