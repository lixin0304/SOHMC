"""
解析Heston模型：用于计算理论最优Delta
"""
import numpy as np
import pandas as pd
import logging
from scipy import integrate
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AnalyticHestonModel:
    """解析Heston模型（用于基准测试）"""
    
    def __init__(self, r=0.02, kappa=1.15, theta=0.04, sigma=0.39, rho=-0.64):
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.a = kappa * theta
    
    @lru_cache(maxsize=50000)
    def compute_delta(self, S, K, T, v0):
        """计算解析Heston Delta"""
        # 舍入以提高缓存命中率
        S = round(S, 2)
        K = round(K, 1)
        T = round(T, 4)
        v0 = round(v0, 4)
        
        if T < 1e-10:
            return 1.0 if S > K else 0.0
        
        # 计算P1（这就是Delta）
        P1 = self._compute_probability(S, K, T, v0, 1)
        
        return np.clip(P1, 0, 1)
    
    def _compute_probability(self, S, K, T, v0, j):
        """计算概率P_j"""
        def integrand(phi):
            return self._integrand(phi, S, K, T, v0, j)
        
        # 使用自适应积分
        integral, _ = integrate.quad(
            integrand, 
            0, 50,
            limit=100,
            epsabs=1e-8,
            epsrel=1e-8
        )
        
        return np.clip(0.5 + integral / np.pi, 0, 1)
    
    def _integrand(self, phi, S, K, T, v0, j):
        """Heston特征函数的被积函数"""
        if j == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa
        
        x = np.log(S / K)
        
        # 复数运算
        rspi = self.rho * self.sigma * phi * 1j
        d = np.sqrt((rspi - b)**2 - self.sigma**2 * (2*u*phi*1j - phi**2))
        g = (b - rspi + d) / (b - rspi - d)
        
        exp_dt = np.exp(d * T)
        
        # 特征函数的组成部分
        C = (self.a / self.sigma**2) * ((b - rspi + d) * T - 2 * np.log((1 - g * exp_dt) / (1 - g)))
        D = ((b - rspi + d) / self.sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))
        
        # 特征函数
        cf = np.exp(C + D * v0 + 1j * phi * x)
        
        # 返回实部
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))


def add_analytic_delta(df, model_params=None, use_true_variance=False):
    """
    为数据集添加解析Heston Delta列
    
    参数:
        df: 数据集
        model_params: Heston模型参数
        use_true_variance: 是否使用真实方差（如果数据中有）
    
    返回:
        DataFrame: 添加了Analytic_Heston_Delta列的数据集
    """
    if model_params is None:
        model_params = {
            'r': 0.02,
            'kappa': 1.15,
            'theta': 0.04,
            'sigma': 0.39,
            'rho': -0.64
        }
    
    # 创建解析Heston模型
    heston_model = AnalyticHestonModel(**model_params)
    
    # 计算Delta
    deltas = []
    total = len(df)
    
    logging.info(f"开始计算{total}个期权的解析Heston Delta...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logging.info(f"进度: {idx}/{total} ({idx*100/total:.1f}%)")
        
        S = row['Stock_Price']
        K = row['Strike_Price'] if 'Strike_Price' in row else row['Stock_Price'] / row['Moneyness']
        T = row['Time_to_Expiry']
        
        # 选择使用哪个方差
        if use_true_variance and 'True_Variance' in df.columns:
            v0 = row['True_Variance']
        elif 'Variance' in df.columns:
            v0 = row['Variance']
        else:
            # 使用隐含波动率的平方作为方差估计
            v0 = row['Implied_Vol'] ** 2 if 'Implied_Vol' in row else 0.04
        
        # 计算解析Delta
        delta = heston_model.compute_delta(S, K, T, v0)
        deltas.append(delta)
    
    df['Analytic_Heston_Delta'] = deltas
    
    logging.info("解析Heston Delta计算完成")
    
    return df

