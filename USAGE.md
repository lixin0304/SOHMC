# 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整流程

最简单的方式是运行完整流程脚本：

```bash
python scripts/run_all.py
```

这将自动执行所有步骤：
1. 生成Heston模型合成数据
2. 计算解析Heston Delta
3. 训练DKL和MV模型
4. 评估所有对冲方法并生成报告

### 3. 分步执行

如果想分步执行或只运行某些步骤：

```bash
# 步骤1: 生成数据
python scripts/01_generate_data.py

# 步骤2: 计算解析Delta（需要先执行步骤1）
python scripts/02_compute_analytic_delta.py

# 步骤3: 训练模型（需要先执行步骤1）
python scripts/03_train_models.py

# 步骤4: 评估（需要先执行步骤1-3）
python scripts/04_evaluate.py
```

## 自定义配置

### 修改模型参数

编辑 `config/config.py` 文件来调整参数：

```python
# Heston模型参数
HESTON_PARAMS = {
    'r': 0.02,       # 调整无风险利率
    'kappa': 1.15,   # 调整均值回归速度
    'theta': 0.04,   # 调整长期波动率
    'sigma': 0.39,   # 调整波动率的波动率
    'rho': -0.64     # 调整相关系数
}

# 数据生成参数
DATA_GEN_PARAMS = {
    'num_paths_train': 20,  # 增加训练路径数以提高模型性能
    'num_paths_test': 10,   # 调整测试路径数
}
```

### 调整DKL超参数

```python
DKL_PARAMS = {
    'max_samples': 5000,           # 增加样本数（需要更多内存和时间）
    'n_calls': 15,                 # 增加优化迭代次数以获得更好的超参数
}
```

## 在Python代码中使用

### 示例1: 生成数据

```python
from src.data_generation import generate_dataset
from config.config import *

# 生成自定义数据集
df = generate_dataset(
    S0=100,
    v0=0.04,
    T=1.0,
    num_paths=10,
    model_params=HESTON_PARAMS,
    moneyness_list=[0.9, 1.0, 1.1],
    maturities=[1/12, 3/12, 6/12],
    seed=42,
    is_train=True
)
```

### 示例2: 训练DKL模型

```python
import pandas as pd
from src.models.dkl import DirectKRRHedging

# 加载数据
train_df = pd.read_parquet('data/processed/heston_train_3features.parquet')

# 训练模型
model = DirectKRRHedging(max_samples=5000)
model.fit(train_df, shift=1, n_calls=15)

# 预测
X_test = [[1.0, 0.25, 0.5]]  # [Moneyness, Time_to_Expiry, BS_Delta]
delta_pred = model.predict(X_test)
```

### 示例3: 训练MV模型

```python
from src.models.mv import PaperMVDeltaHedging

# 训练MV模型
mv_model = PaperMVDeltaHedging()
mv_model.fit(train_df, r=0.02)

# 计算MV Delta
mv_delta = mv_model.calculate_mv_delta(
    S=100,      # 股价
    K=100,      # 行权价
    T=0.25,     # 到期时间
    r=0.02,     # 无风险利率
    iv=0.2      # 隐含波动率
)
```

### 示例4: 评估对冲性能

```python
from src.utils.metrics import evaluate_all_methods, plot_comparison

# 加载测试数据和模型
test_df = pd.read_parquet('data/processed/heston_test_with_analytic.parquet')

models = {
    'dkl': dkl_model,  # 已训练的DKL模型
    'mv': mv_model     # 已训练的MV模型
}

# 评估
results = evaluate_all_methods(test_df, models, freq_days=1, r=0.02)

# 可视化
results_df = pd.DataFrame(results)
plot_comparison(results_df, save_path='my_results.png')
```

## 输出文件说明

### 数据文件

- `data/processed/heston_train_3features.parquet`: 训练集
- `data/processed/heston_test_3features.parquet`: 测试集
- `data/processed/heston_test_with_analytic.parquet`: 带解析Delta的测试集

### 模型文件

- `data/processed/dkl_model.pkl`: 训练好的DKL模型
- `data/processed/mv_model.pkl`: 训练好的MV模型

### 结果文件

- `data/results/hedging_comparison_results.csv`: 详细评估结果CSV
- `data/results/hedging_comparison.png`: 可视化对比图

## 性能优化建议

### 1. 数据生成加速

如果系统支持多进程，可以启用并行处理：

```python
# 在 scripts/01_generate_data.py 中
train_df = generate_dataset(
    ...
    use_parallel=True  # 启用并行处理
)
```

### 2. 减少计算时间

对于快速测试，可以减少数据量：

```python
DATA_GEN_PARAMS = {
    'num_paths_train': 5,   # 减少路径数
    'num_paths_test': 3,
}

DKL_PARAMS = {
    'max_samples': 1000,    # 减少样本数
    'n_calls': 5,           # 减少优化迭代
}
```

### 3. 内存优化

如果遇到内存问题：

```python
DKL_PARAMS = {
    'max_samples': 2000,    # 限制最大样本数
}
```

## 故障排除

### 问题1: 导入错误

```
ModuleNotFoundError: No module named 'src'
```

解决方案：确保在项目根目录运行脚本，或添加：

```python
import sys
sys.path.append('/path/to/Option-Hedging-Comparison')
```

### 问题2: 内存不足

```
MemoryError
```

解决方案：减少 `max_samples` 和 `num_paths`

### 问题3: NumPy/SciPy版本问题

确保安装了正确的依赖版本：

```bash
pip install -r requirements.txt --upgrade
```

## 进阶使用

### 添加自己的对冲方法

1. 在 `src/models/` 创建新的模型文件
2. 实现 `fit()` 和 `predict()` 方法
3. 在 `scripts/03_train_models.py` 添加训练代码
4. 在 `src/utils/metrics.py` 的 `evaluate_all_methods()` 添加评估逻辑

### 使用真实市场数据

修改数据加载部分以读取真实数据：

```python
# 替换数据生成
import pandas as pd

# 加载真实数据
real_data = pd.read_csv('your_real_data.csv')

# 确保数据包含必要的列：
# Contract_ID, Time_Step, Stock_Price, Strike_Price, 
# Time_to_Expiry, Moneyness, Market_Price, BS_Delta, Implied_Vol
```

## 联系与支持

如有问题，请查看：
- GitHub Issues
- 项目文档
- 联系作者

