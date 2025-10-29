# 期权对冲方法比较 | Option Hedging Comparison

基于Heston随机波动率模型的期权对冲策略实证比较研究。

## 📋 项目简介

本项目对比了多种期权Delta对冲策略在Heston随机波动率模型下的性能表现，包括：

- **BS Delta**: Black-Scholes模型Delta（基准方法）
- **Analytic Heston Delta**: 解析Heston Delta（理论最优）
- **Direct KRR (DKL)**: 基于离散直接对冲学习策略
- **MV Delta**: Hull & White (2017) 最小方差Delta对冲策略

## 🎯 主要特性

- ✅ 完整的Heston模型合成数据生成
- ✅ 高效的解析Heston Delta计算（使用缓存和Gauss-Laguerre积分）
- ✅ 基于贝叶斯优化的DKL超参数调优
- ✅ 严格按照论文实现的MV Delta方法
- ✅ 多维度对冲性能评估（Gain, MAE, Std, VaR, CVaR）
- ✅ 可视化对比图表
- ✅ 模块化工程结构，易于扩展

## 📁 项目结构

```
Option-Hedging-Comparison/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config/
│   ├── __init__.py
│   └── config.py            # 配置文件（模型参数、路径等）
├── src/
│   ├── __init__.py
│   ├── data_generation.py   # 数据生成模块
│   ├── models/
│   │   ├── __init__.py
│   │   ├── heston.py        # 解析Heston模型
│   │   ├── dkl.py           # Direct KRR对冲模型
│   │   └── mv.py            # 最小方差Delta对冲模型
│   └── utils/
│       ├── __init__.py
│       └── metrics.py       # 评估指标和可视化
├── scripts/
│   ├── 01_generate_data.py         # 步骤1: 生成数据
│   ├── 02_compute_analytic_delta.py # 步骤2: 计算解析Delta
│   ├── 03_train_models.py          # 步骤3: 训练模型
│   ├── 04_evaluate.py              # 步骤4: 评估
│   └── run_all.py                  # 运行完整流程
├── data/
│   ├── processed/           # 处理后的数据和模型
│   └── results/             # 评估结果
└── tests/                   # 单元测试（可选）
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/Option-Hedging-Comparison.git
cd Option-Hedging-Comparison

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
# 方式1: 运行完整流程（一键执行所有步骤）
python scripts/run_all.py

# 方式2: 分步执行
python scripts/01_generate_data.py              # 生成数据
python scripts/02_compute_analytic_delta.py     # 计算解析Delta
python scripts/03_train_models.py               # 训练DKL和MV模型
python scripts/04_evaluate.py                   # 评估所有方法
```

### 3. 查看结果

运行完成后，结果文件保存在 `data/results/` 目录：

- `hedging_comparison_results.csv`: 详细评估结果
- `hedging_comparison.png`: 对比可视化图表

## ⚙️ 配置说明

所有参数配置在 `config/config.py` 中，可根据需要修改：

### Heston模型参数（与论文一致）

```python
HESTON_PARAMS = {
    'r': 0.02,       # 无风险利率
    'kappa': 1.15,   # 均值回归速度
    'theta': 0.04,   # 长期波动率
    'sigma': 0.39,   # 波动率的波动率
    'rho': -0.64     # 相关系数
}
```

### 数据生成参数

```python
DATA_GEN_PARAMS = {
    'S0': 100,              # 初始股价
    'v0': 0.04,             # 初始波动率
    'T_train': 2.0,         # 训练集时间范围
    'T_test': 0.5,          # 测试集时间范围
    'num_paths_train': 20,  # 训练集路径数
    'num_paths_test': 10,   # 测试集路径数
}
```

### DKL模型参数

```python
DKL_PARAMS = {
    'max_samples': 5000,           # 最大训练样本数
    'regularization_type': 'identity',  # 正则化类型
    'n_calls': 15,                 # 贝叶斯优化迭代次数
}
```

## 📊 评估指标

本项目使用以下指标评估对冲性能：

- **Gain (%)**: 相对于BS Delta的SSE改进百分比
- **E(|ΔV-ΔSf(x)|)**: 平均绝对对冲误差
- **Std**: 对冲误差标准差
- **VaR (95%)**: 95%分位数风险值
- **CVaR (95%)**: 95%条件风险值

## 📚 理论背景

### Heston随机波动率模型

股价 $S_t$ 和波动率 $v_t$ 的随机过程：

$$
dS_t = rS_t dt + \sqrt{v_t}S_t dW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)dt + \sigma\sqrt{v_t}dW_t^v
$$

其中 $dW_t^S$ 和 $dW_t^v$ 的相关系数为 $\rho$。

### Direct KRR方法

最小化目标函数：

$$
\min_{\alpha} \sum_{i=1}^{n} (\Delta V_i - \delta(x_i)\Delta S_i)^2 + \lambda \alpha^T K \alpha
$$

### MV Delta方法

按照Hull & White (2017)论文：

$$
\delta_{MV} = \delta_{BS} + \frac{\nu_{BS}}{S\sqrt{T}}(a + b\delta_{BS} + c\delta_{BS}^2)
$$

## 📖 参考文献

Hull, J. and White, A. (2017). Optimal delta hedging for options. Journal of Banking & Finance, 82:180–190.

Nian, K., Coleman, T. F., and Li, Y. (2018). Learning minimum variance discrete hedging directly from the market. Quantitative Finance, 18(7):1115–1128.

Nian, K., Coleman, T. F., and Li, Y. (2021). Learning sequential option hedging models from market data. Journal of Banking & Finance, 133:106277.

## 📧 联系方式

如有问题，请通过以下方式联系：
- Email: lxxx0304@163.com

---

**注意**: 本项目仅用于研究和教学目的，不构成任何投资建议。

