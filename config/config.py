"""
配置文件：所有模型参数和路径配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# 确保目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Heston模型参数（与论文一致）
HESTON_PARAMS = {
    'r': 0.02,       # 无风险利率
    'kappa': 1.15,   # 均值回归速度
    'theta': 0.04,   # 长期波动率
    'sigma': 0.39,   # 波动率的波动率
    'rho': -0.64     # 相关系数
}

# 数据生成参数
DATA_GEN_PARAMS = {
    'S0': 100,              # 初始股价
    'v0': 0.04,             # 初始波动率
    'T_train': 2.0,         # 训练集时间范围
    'T_test': 0.5,          # 测试集时间范围
    'num_paths_train': 20,  # 训练集路径数
    'num_paths_test': 10,   # 测试集路径数
    'dt': 1/252,            # 时间步长（1天）
    'seed_train': 42,       # 训练集随机种子
    'seed_test': 123        # 测试集随机种子
}

# 期权参数
OPTION_PARAMS = {
    'moneyness': [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],  # 行权价比例
    'maturities': [1/12, 2/12, 3/12, 6/12]  # 到期时间（月）
}

# 对冲频率
HEDGE_FREQUENCIES = [
    (1, 'Daily'),
    (5, 'Weekly'),
    (20, 'Monthly')
]

# DKL模型参数
DKL_PARAMS = {
    'max_samples': 5000,           # 最大训练样本数
    'regularization_type': 'identity',  # 正则化类型：'identity' 或 'kernel'
    'n_calls': 15,                 # 贝叶斯优化迭代次数
    'val_ratio': 0.2               # 验证集比例
}

# 文件路径
FILE_PATHS = {
    'train_data': PROCESSED_DATA_DIR / "heston_train_3features.parquet",
    'test_data': PROCESSED_DATA_DIR / "heston_test_3features.parquet",
    'test_with_analytic': PROCESSED_DATA_DIR / "heston_test_with_analytic.parquet",
    'analytic_metrics': RESULTS_DIR / "analytic_heston_metrics.csv",
    'dkl_metrics': RESULTS_DIR / "dkl_metrics.csv",
    'mv_metrics': RESULTS_DIR / "mv_metrics.csv",
    'comparison_plot': RESULTS_DIR / "hedging_comparison.png"
}

