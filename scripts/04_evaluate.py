"""
步骤4: 评估所有对冲方法
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
import pickle
from config.config import *
from src.utils.metrics import evaluate_all_methods, print_results_table, plot_comparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """评估所有对冲方法"""
    
    print("\n" + "="*80)
    print("步骤4: 评估所有对冲方法")
    print("="*80)
    
    # 加载测试数据
    print(f"\n加载测试数据: {FILE_PATHS['test_with_analytic']}")
    test_df = pd.read_parquet(FILE_PATHS['test_with_analytic'])
    print(f"数据量: {len(test_df)} 行")
    
    # 加载训练好的模型
    print("\n加载训练好的模型...")
    models = {}
    
    try:
        dkl_model_path = PROCESSED_DATA_DIR / "dkl_model.pkl"
        with open(dkl_model_path, 'rb') as f:
            models['dkl'] = pickle.load(f)
        print(f"  - DKL模型已加载")
    except:
        print("  - DKL模型未找到，跳过")
        models['dkl'] = None
    
    try:
        mv_model_path = PROCESSED_DATA_DIR / "mv_model.pkl"
        with open(mv_model_path, 'rb') as f:
            models['mv'] = pickle.load(f)
        print(f"  - MV模型已加载")
    except:
        print("  - MV模型未找到，跳过")
        models['mv'] = None
    
    # 评估不同对冲频率
    all_results = []
    
    for freq_days, freq_name in HEDGE_FREQUENCIES:
        print("\n" + "-"*80)
        print(f"评估 {freq_name} 对冲 (每{freq_days}天)")
        print("-"*80)
        
        results = evaluate_all_methods(
            test_df,
            models,
            freq_days=freq_days,
            r=HESTON_PARAMS['r']
        )
        
        # 添加频率信息
        for result in results:
            result['Frequency'] = freq_name
            all_results.append(result)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 打印结果表格
    print_results_table(results_df)
    
    # 保存结果
    results_csv_path = RESULTS_DIR / "hedging_comparison_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n结果已保存: {results_csv_path}")
    
    # 绘制对比图
    print("\n绘制对比图...")
    plot_comparison(results_df, save_path=FILE_PATHS['comparison_plot'])
    
    print("\n评估完成！")
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    main()

