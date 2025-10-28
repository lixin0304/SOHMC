"""
运行完整的对冲比较流程
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """运行完整流程"""
    
    print("\n" + "="*100)
    print("期权对冲方法比较 - 完整流程")
    print("="*100)
    print("\n本流程将依次执行:")
    print("  1. 生成Heston模型合成数据")
    print("  2. 计算解析Heston Delta（理论最优）")
    print("  3. 训练DKL和MV模型")
    print("  4. 评估所有对冲方法")
    print("\n" + "="*100)
    
    start_time = time.time()
    
    # 步骤1: 生成数据
    print("\n" + ">"*100)
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_data", project_root / "scripts" / "01_generate_data.py")
    generate_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_data)
    train_df, test_df = generate_data.main()
    
    # 步骤2: 计算解析Delta
    print("\n" + ">"*100)
    spec = importlib.util.spec_from_file_location("compute_analytic_delta", project_root / "scripts" / "02_compute_analytic_delta.py")
    compute_analytic_delta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compute_analytic_delta)
    test_df = compute_analytic_delta.main()
    
    # 步骤3: 训练模型
    print("\n" + ">"*100)
    spec = importlib.util.spec_from_file_location("train_models", project_root / "scripts" / "03_train_models.py")
    train_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_models)
    models = train_models.main()
    
    # 步骤4: 评估
    print("\n" + ">"*100)
    spec = importlib.util.spec_from_file_location("evaluate", project_root / "scripts" / "04_evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    results_df = evaluate.main()
    
    # 总结
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*100)
    print("流程执行完成！")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print("="*100)
    
    print("\n结果摘要:")
    print("-"*100)
    
    # 按频率显示最佳方法
    for freq in results_df['Frequency'].unique():
        freq_data = results_df[results_df['Frequency'] == freq]
        best_row = freq_data.loc[freq_data['Gain(%)'].idxmax()]
        print(f"\n{freq} 对冲最佳方法:")
        print(f"  方法: {best_row['Method']}")
        print(f"  Gain: {best_row['Gain(%)']:.2f}%")
        print(f"  MAE: {best_row['E(|ΔV-ΔSf(x)|)']:.6f}")
        print(f"  Std: {best_row['Std']:.6f}")
    
    print("\n" + "="*100)
    print("所有结果文件已保存到 data/results/ 目录")
    print("="*100)


if __name__ == "__main__":
    main()

