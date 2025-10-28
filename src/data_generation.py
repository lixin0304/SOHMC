"""
数据生成模块：生成Heston模型下的期权数据
"""
import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from functools import lru_cache
import time
from numba import njit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@njit
def heston_integrand_fast(phi, S, K, T, v0, kappa, theta, sigma, rho, r, j):
    """Numba加速的Heston被积函数"""
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa
    
    a = kappa * theta
    x = np.log(S / K)
    
    # 复数运算
    i = 1j
    rspi = rho * sigma * phi * i
    
    # d的计算
    d_squared = (rspi - b)**2 - sigma**2 * (2*u*phi*i - phi**2)
    d = np.sqrt(d_squared)
    
    # g的计算
    g = (b - rspi + d) / (b - rspi - d)
    
    # 避免溢出
    exp_dt = np.exp(d * T)
    
    # C和D的计算
    C = (a / sigma**2) * ((b - rspi + d) * T - 2 * np.log((1 - g * exp_dt) / (1 - g)))
    D = ((b - rspi + d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))
    
    # 特征函数
    cf = np.exp(C + D * v0 + i * phi * x)
    
    # 返回实部
    integrand = np.exp(-i * phi * np.log(K)) * cf / (i * phi)
    return np.real(integrand)


class FastHestonModelV2:
    """高速Heston模型（用于数据生成）"""
    
    def __init__(self, r=0.02, kappa=1.15, theta=0.04, sigma=0.39, rho=-0.64):
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
        # 预计算积分点（Gauss-Laguerre）
        self.n_quad = 32
        self.setup_quadrature()
    
    def setup_quadrature(self):
        """设置高斯积分点"""
        from numpy.polynomial.laguerre import laggauss
        self.x_lag, self.w_lag = laggauss(self.n_quad)
    
    @lru_cache(maxsize=50000)
    def call_price_and_delta(self, S, K, T, v0):
        """计算期权价格和Delta"""
        # 舍入以提高缓存命中率
        S = round(S, 2)
        K = round(K, 1)
        T = round(T, 4)
        v0 = round(v0, 4)
        
        if T < 1e-10:
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
            return price, delta
        
        P1 = self._compute_probability_fast(S, K, T, v0, 1)
        P2 = self._compute_probability_fast(S, K, T, v0, 2)
        
        price = S * P1 - K * np.exp(-self.r * T) * P2
        delta = P1
        
        return max(price, 0), np.clip(delta, 0, 1)
    
    def _compute_probability_fast(self, S, K, T, v0, j):
        """快速计算概率（使用Gauss-Laguerre积分）"""
        integrals = np.zeros(self.n_quad)
        
        for i in range(self.n_quad):
            x = self.x_lag[i]
            w = self.w_lag[i]
            
            integrand = heston_integrand_fast(
                x, S, K, T, v0, 
                self.kappa, self.theta, self.sigma, self.rho, self.r, j
            )
            
            integrals[i] = w * np.exp(x) * integrand
        
        integral = np.sum(integrals)
        return np.clip(0.5 + integral / np.pi, 0, 1)


@lru_cache(maxsize=50000)
def implied_vol_newton(option_price, S, K, T, r):
    """牛顿法计算隐含波动率"""
    option_price = round(option_price, 3)
    S = round(S, 2)
    K = round(K, 1)
    T = round(T, 4)
    
    if T < 1e-10:
        return 0.2
    
    intrinsic = max(S - K, 0)
    if option_price <= intrinsic:
        return 0.01
    
    # Brenner-Subrahmanyam初始猜测
    sigma = option_price / (0.4 * S * np.sqrt(T))
    sigma = np.clip(sigma, 0.01, 5.0)
    
    # 牛顿迭代
    for _ in range(10):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            break
        
        error = option_price - price
        if abs(error) < 1e-8:
            break
        
        sigma = sigma + error / vega
        sigma = np.clip(sigma, 0.001, 10.0)
    
    return sigma


def generate_heston_paths(S0, v0, T, num_paths, params, dt, seed=None):
    """生成Heston路径"""
    if seed is not None:
        np.random.seed(seed)
    
    N_steps = int(T / dt)
    
    r = params['r']
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    
    # 初始化
    S = np.zeros((N_steps + 1, num_paths))
    v = np.zeros((N_steps + 1, num_paths))
    S[0, :] = S0
    v[0, :] = v0
    
    # 预生成所有随机数
    z1 = np.random.normal(0, 1, (N_steps, num_paths))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (N_steps, num_paths))
    
    # 向量化的路径生成
    for t in range(N_steps):
        v_t = np.maximum(v[t], 1e-8)
        sqrt_v = np.sqrt(v_t)
        
        # Milstein方案
        S[t + 1] = S[t] * np.exp((r - 0.5 * v_t) * dt + sqrt_v * np.sqrt(dt) * z1[t])
        
        # QE方案
        v_temp = v[t] + kappa * (theta - np.maximum(v[t], 0)) * dt + \
                 sigma * sqrt_v * np.sqrt(dt) * z2[t]
        v[t + 1] = np.maximum(v_temp, 0)
    
    return S, v


def process_single_contract(args):
    """处理单个合约（用于并行）"""
    (path_idx, K, T_mat, S_path, v_path, dt, params, 
     true_heston, est_heston, add_noise, noise_level, contract_prefix) = args
    
    r = params['r']
    N_steps = len(S_path) - 1
    
    results = []
    
    for t in range(0, N_steps + 1, 1):  # 每天采样
        time_to_expiry = T_mat - t * dt
        
        if time_to_expiry > dt:
            S_t = S_path[t]
            v_t = v_path[t]
            
            # 真实价格（理论值）
            true_price, _ = true_heston.call_price_and_delta(S_t, K, time_to_expiry, v_t)
            
            # 市场噪声
            if add_noise and true_price > 0:
                moneyness = S_t / K
                noise_factor = 1.0 + 0.5 * abs(moneyness - 1.0)
                time_factor = 1.0 + 0.3 / max(time_to_expiry, 0.1)
                adj_noise = noise_level * noise_factor * time_factor
                noise = np.random.normal(0, adj_noise * true_price)
                market_price = max(true_price + noise, max(S_t - K, 0))
            else:
                market_price = true_price
            
            # 隐含波动率
            iv = implied_vol_newton(market_price, S_t, K, time_to_expiry, r)
            
            # BS Delta
            bs_delta = norm.cdf((np.log(S_t/K) + (r + 0.5*iv**2)*time_to_expiry) / 
                               (iv*np.sqrt(time_to_expiry))) if iv > 0 and time_to_expiry > 0 else (1.0 if S_t > K else 0.0)
            
            # Heston Delta（使用估计模型）
            _, heston_delta = est_heston.call_price_and_delta(S_t, K, time_to_expiry, iv**2)
            
            results.append({
                'Contract_ID': f"{contract_prefix}{path_idx:02d}_K{K:.0f}_T{T_mat:.3f}",
                'Time_Step': t,
                'Stock_Price': S_t,
                'Strike_Price': K,
                'Time_to_Expiry': time_to_expiry,
                'Moneyness': S_t / K,
                'Market_Price': market_price,
                'Heston_Delta': heston_delta,
                'BS_Delta': bs_delta,
                'Implied_Vol': iv
            })
    
    return results


def generate_dataset(S0, v0, T, num_paths, model_params, 
                     moneyness_list, maturities, dt=1/252,
                     seed=None, is_train=True, use_parallel=False):
    """
    生成期权数据集
    
    参数:
        S0: 初始股价
        v0: 初始波动率
        T: 时间范围
        num_paths: 路径数
        model_params: Heston模型参数
        moneyness_list: 行权价比例列表
        maturities: 到期时间列表
        dt: 时间步长
        seed: 随机种子
        is_train: 是否训练集
        use_parallel: 是否使用并行
    
    返回:
        DataFrame: 生成的数据集
    """
    start_time = time.time()
    
    if seed is not None:
        np.random.seed(seed)
    
    # 创建模型
    true_heston = FastHestonModelV2(**model_params)
    
    # 估计模型（参数误差）
    error = 0.15 if is_train else 0.20
    est_params = {
        'r': model_params['r'],
        'kappa': model_params['kappa'] * (1 + np.random.uniform(-error, error)),
        'theta': model_params['theta'] * (1 + np.random.uniform(-error, error)),
        'sigma': model_params['sigma'] * (1 + np.random.uniform(-error, error)),
        'rho': np.clip(model_params['rho'] * (1 + np.random.uniform(-error, error)), -0.99, -0.01)
    }
    est_heston = FastHestonModelV2(**est_params)
    
    dataset_type = "训练集" if is_train else "测试集"
    logging.info(f"生成{dataset_type}: {num_paths}条路径")
    
    # 生成路径
    S, v = generate_heston_paths(S0, v0, T, num_paths, model_params, dt, seed)
    
    # 期权合约
    strikes = S0 * np.array(moneyness_list)
    
    # 准备并行任务
    tasks = []
    contract_prefix = 'TR' if is_train else 'TE'
    
    for path_idx in range(num_paths):
        for K in strikes:
            for T_mat in maturities:
                tasks.append((
                    path_idx, K, T_mat, 
                    S[:, path_idx], v[:, path_idx], dt,
                    model_params, true_heston, est_heston,
                    True, 0.005, contract_prefix
                ))
    
    # 并行处理
    if use_parallel:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            all_results = list(executor.map(process_single_contract, tasks))
    else:
        all_results = [process_single_contract(task) for task in tasks]
    
    # 合并结果
    data = []
    for result in all_results:
        data.extend(result)
    
    df = pd.DataFrame(data)
    
    logging.info(f"{dataset_type}生成完成，耗时 {time.time() - start_time:.2f}秒")
    logging.info(f"数据量: {len(df)}行")
    
    return df

