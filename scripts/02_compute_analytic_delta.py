"""
步骤2: 计算解析Heston Delta（理论最优）
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
from config.config import *
from src.models.heston import add_analytic_delta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """计算解析Heston Delta"""
    
    print("\n" + "="*80)
    print("步骤2: 计算解析Heston Delta")
    print("="*80)
    
    # 加载测试数据
    print(f"\n加载测试数据: {FILE_PATHS['test_data']}")
    test_df = pd.read_parquet(FILE_PATHS['test_data'])
    print(f"数据量: {len(test_df)} 行")
    
    # 计算解析Delta
    print("\n开始计算解析Heston Delta...")
    test_df = add_analytic_delta(
        test_df,
        model_params=HESTON_PARAMS,
        use_true_variance=False  # 使用隐含波动率的平方
    )
    
    # 保存结果
    print(f"\n保存结果: {FILE_PATHS['test_with_analytic']}")
    test_df.to_parquet(FILE_PATHS['test_with_analytic'])
    
    print("\n解析Delta计算完成！")
    print("="*80)
    
    return test_df


if __name__ == "__main__":
    main()

