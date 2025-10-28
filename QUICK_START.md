# 快速开始指南

## ✅ 项目已成功重构！

你的代码已经重构为工程化的项目结构，可以直接上传到GitHub。

## 📁 项目结构

```
Option-Hedging-Comparison/
├── README.md                          # 项目介绍（中英文）
├── USAGE.md                           # 详细使用指南
├── QUICK_START.md                     # 快速开始（本文件）
├── LICENSE                            # MIT开源协议
├── requirements.txt                   # Python依赖包
├── .gitignore                         # Git忽略文件
│
├── config/                            # 配置文件
│   ├── __init__.py
│   └── config.py                      # 所有参数配置（一站式修改）
│
├── src/                               # 源代码
│   ├── __init__.py
│   ├── data_generation.py             # 数据生成模块
│   ├── models/                        # 模型模块
│   │   ├── __init__.py
│   │   ├── heston.py                  # 解析Heston Delta
│   │   ├── dkl.py                     # Direct KRR对冲
│   │   └── mv.py                      # 最小方差Delta对冲
│   └── utils/                         # 工具模块
│       ├── __init__.py
│       └── metrics.py                 # 评估指标和可视化
│
├── scripts/                           # 运行脚本
│   ├── 01_generate_data.py            # 步骤1: 生成数据
│   ├── 02_compute_analytic_delta.py   # 步骤2: 计算解析Delta
│   ├── 03_train_models.py             # 步骤3: 训练模型
│   ├── 04_evaluate.py                 # 步骤4: 评估
│   └── run_all.py                     # 一键运行全部
│
├── data/                              # 数据目录
│   ├── processed/                     # 处理后的数据和模型
│   └── results/                       # 评估结果
│
└── tests/                             # 测试
    ├── __init__.py
    └── test_installation.py           # 安装测试
```

## 🚀 立即开始

### 方式1: 一键运行（推荐）

```bash
cd /Users/lixin/Downloads/Option-Hedging-Comparison
python scripts/run_all.py
```

这将：
1. 生成Heston模型合成数据（训练集+测试集）
2. 计算解析Heston Delta（理论最优）
3. 训练DKL和MV模型
4. 评估所有方法并生成图表

**预计耗时**: 5-10分钟（取决于电脑性能）

### 方式2: 分步执行

```bash
# 步骤1: 生成数据
python scripts/01_generate_data.py

# 步骤2: 计算解析Delta
python scripts/02_compute_analytic_delta.py

# 步骤3: 训练模型
python scripts/03_train_models.py

# 步骤4: 评估
python scripts/04_evaluate.py
```

## 📊 查看结果

运行完成后，结果保存在 `data/results/` 目录：

```bash
# 查看结果CSV
cat data/results/hedging_comparison_results.csv

# 或用Excel打开
open data/results/hedging_comparison_results.csv

# 查看可视化图表
open data/results/hedging_comparison.png
```

## 🔧 测试安装

运行测试脚本确保一切正常：

```bash
python tests/test_installation.py
```

应该看到：
```
✓ config 模块导入成功
✓ data_generation 模块导入成功
✓ models 模块导入成功
✓ utils 模块导入成功
✓ 所有依赖包已安装
✓ 目录结构正确

恭喜！所有测试通过，项目配置正确！
```

## 📤 上传到GitHub

### 1. 初始化Git仓库

```bash
cd /Users/lixin/Downloads/Option-Hedging-Comparison
git init
git add .
git commit -m "Initial commit: Option hedging comparison framework"
```

### 2. 创建GitHub仓库

在GitHub上创建新仓库（不要初始化README）

### 3. 推送代码

```bash
git remote add origin https://github.com/你的用户名/Option-Hedging-Comparison.git
git branch -M main
git push -u origin main
```

## ⚙️ 自定义配置

所有参数都在 `config/config.py` 中，可以轻松修改：

```python
# 修改Heston模型参数
HESTON_PARAMS = {
    'r': 0.02,       # 无风险利率
    'kappa': 1.15,   # 均值回归速度
    'theta': 0.04,   # 长期波动率
    'sigma': 0.39,   # 波动率的波动率
    'rho': -0.64     # 相关系数
}

# 修改数据生成参数
DATA_GEN_PARAMS = {
    'num_paths_train': 20,  # 训练路径数（增加可提高模型性能）
    'num_paths_test': 10,   # 测试路径数
}

# 修改DKL参数
DKL_PARAMS = {
    'max_samples': 5000,    # 最大样本数
    'n_calls': 15,          # 贝叶斯优化迭代次数
}
```

## 📚 文档说明

- **README.md**: 项目完整介绍，包含理论背景、参考文献
- **USAGE.md**: 详细使用指南，包含代码示例、故障排除
- **QUICK_START.md**: 本文件，快速开始指南

## 🎯 核心改进

相比原始代码，新版本：

✅ **模块化**: 代码按功能分模块，易于维护和扩展  
✅ **配置化**: 所有参数集中在config.py，一处修改全局生效  
✅ **标准化**: 遵循Python项目最佳实践  
✅ **文档化**: 完整的README、使用指南和代码注释  
✅ **可测试**: 包含测试框架  
✅ **可复现**: 统一的随机种子和参数设置  

## 📞 获取帮助

如遇到问题：
1. 查看 USAGE.md 的故障排除部分
2. 运行 `python tests/test_installation.py` 检查安装
3. 在GitHub上提Issue

## 🎉 下一步

1. ✅ 运行 `python scripts/run_all.py` 验证流程
2. ✅ 查看生成的结果和图表
3. ✅ 上传到GitHub
4. ✅ 根据需要调整参数重新运行
5. ✅ 添加自己的对冲方法（参考USAGE.md）

祝你研究顺利！🚀

