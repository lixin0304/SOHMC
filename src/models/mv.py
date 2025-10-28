"""
最小方差Delta对冲模型 (Minimum Variance Delta Hedging)
基于论文: Hull & White (2017) "Optimal Delta Hedging for Options"
"""
import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperMVDeltaHedging:
    """
    严格按照Hull & White (2017)论文实现的MV Delta对冲
    
    核心公式:
    - 方程(5): δ_MV = δ_BS + (ν_BS/(S√T)) * (a + b*δ_BS + c*δ_BS²)
    - 方程(6): Δf - δ_BS*ΔS = (ν_BS/√T)*(ΔS/S)*(a + b*δ_BS + c*δ_BS²) + ε
    """
    
    def __init__(self):
        self.a = None  # 二次函数常数项
        self.b = None  # 一次项系数
        self.c = None  # 二次项系数
        
    def calculate_bs_greeks(self, S, K, T, r, sigma):
        """计算Black-Scholes Greeks"""
        if T <= 1e-10 or sigma <= 1e-10:
            return {'delta': 1.0 if S > K else 0.0, 'vega': 0.0}
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        
        delta = norm.cdf(d1)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return {'delta': delta, 'vega': vega}
    
    def fit(self, train_df, r=0.02, max_samples=None):
        """
        按照论文方程(6)估计参数a, b, c
        
        回归方程:
        Δf - δ_BS*ΔS = (ν_BS/√T)*(ΔS/S) * (a + b*δ_BS + c*δ_BS²) + ε
        
        整理为标准线性回归:
        Y = X1*a + X2*b + X3*c + ε
        
        其中:
        - Y = Δf - δ_BS*ΔS
        - X1 = (ν_BS/√T)*(ΔS/S)
        - X2 = (ν_BS/√T)*(ΔS/S)*δ_BS
        - X3 = (ν_BS/√T)*(ΔS/S)*δ_BS²
        """
        logging.info("="*60)
        logging.info("开始训练MV模型（论文方法）")
        logging.info("="*60)
        
        X_list = []  # 特征矩阵
        Y_list = []  # 目标变量
        
        contract_ids = train_df['Contract_ID'].unique()
        if max_samples:
            contract_ids = contract_ids[:max_samples]
        
        valid_samples = 0
        
        for contract_id in contract_ids:
            contract_df = train_df[train_df['Contract_ID'] == contract_id].sort_values('Time_Step')
            
            if len(contract_df) < 2:
                continue
            
            # 遍历相邻时间步
            for i in range(len(contract_df) - 1):
                row0 = contract_df.iloc[i]
                row1 = contract_df.iloc[i + 1]
                
                # 提取变量
                S0 = row0['Stock_Price']
                S1 = row1['Stock_Price']
                V0 = row0['Market_Price']
                V1 = row1['Market_Price']
                K = row0['Strike_Price']
                T = row0['Time_to_Expiry']
                iv = row0['Implied_Vol']
                
                dS = S1 - S0
                dV = V1 - V0
                
                # 数据过滤
                if abs(dS) < 0.001 * S0:  # 价格变化太小
                    continue
                if T < 1/252:  # 到期时间太短
                    continue
                if iv <= 0 or iv > 2:  # 隐含波动率异常
                    continue
                
                # 计算Greeks
                greeks = self.calculate_bs_greeks(S0, K, T, r, iv)
                delta_bs = greeks['delta']
                vega = greeks['vega']
                
                if vega < 1e-6:  # vega太小
                    continue
                
                # 因变量: Y = Δf - δ_BS*ΔS（论文方程(6)左侧）
                Y = dV - delta_bs * dS
                
                # 自变量系数: (ν_BS/√T) * (ΔS/S)（论文方程(6)右侧系数）
                coef = (vega / np.sqrt(T)) * (dS / S0)
                
                # 特征向量: [coef, coef*δ_BS, coef*δ_BS²]
                X_list.append([
                    coef,                  # a的系数
                    coef * delta_bs,       # b的系数  
                    coef * delta_bs**2     # c的系数
                ])
                Y_list.append(Y)
                valid_samples += 1
        
        if len(X_list) < 10:
            logging.error(f"有效样本数太少: {len(X_list)}")
            # 使用论文中S&P 500的平均值
            self.a = -0.25
            self.b = -0.40
            self.c = -0.35
            logging.warning("使用论文默认参数")
            return
        
        # 线性回归估计参数
        X = np.array(X_list)
        Y = np.array(Y_list)
        
        # 不需要截距项（论文方程(6)中没有截距）
        model = LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        
        # 保存参数
        self.a = model.coef_[0]
        self.b = model.coef_[1]
        self.c = model.coef_[2]
        
        # R²评估
        r2 = model.score(X, Y)
        
        logging.info(f"训练完成:")
        logging.info(f"  有效样本数: {valid_samples}")
        logging.info(f"  估计参数: a = {self.a:.4f}, b = {self.b:.4f}, c = {self.c:.4f}")
        logging.info(f"  R² = {r2:.4f}")
        logging.info("="*60)
    
    def calculate_mv_delta(self, S, K, T, r, iv, bs_delta=None):
        """
        按照论文方程(5)计算MV Delta
        
        δ_MV = δ_BS + (ν_BS/(S√T)) * (a + b*δ_BS + c*δ_BS²)
        """
        if T <= 1e-10:
            return bs_delta if bs_delta is not None else 1.0
        
        # 计算Greeks
        greeks = self.calculate_bs_greeks(S, K, T, r, iv)
        delta_bs = greeks['delta'] if bs_delta is None else bs_delta
        vega = greeks['vega']
        
        if self.a is None:
            # 如果模型未训练，返回BS Delta
            return delta_bs
        
        # 论文方程(5)的调整项
        adjustment = (vega / (S * np.sqrt(T))) * (
            self.a + self.b * delta_bs + self.c * delta_bs**2
        )
        
        # 最终的MV Delta
        mv_delta = delta_bs + adjustment
        
        # 限制在[0, 1]范围内（对于call option）
        return np.clip(mv_delta, 0, 1)
    
    def predict(self, X, r=0.02):
        """
        预测MV Delta（为了与DKL保持一致的接口）
        
        参数:
            X: 特征矩阵 [Moneyness, Time_to_Expiry, BS_Delta]（需要额外的S, K, IV）
        
        注意：这个方法需要更多信息，通常在evaluate时使用calculate_mv_delta
        """
        if self.a is None:
            raise ValueError("模型未训练")
        
        # 简化版本：直接使用BS Delta作为基础
        # 实际使用时建议直接调用calculate_mv_delta
        bs_delta = X[:, 2] if X.shape[1] > 2 else X[:, -1]
        return bs_delta

