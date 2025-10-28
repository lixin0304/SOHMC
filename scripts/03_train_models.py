"""
步骤3: 训练DKL和MV模型
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
from src.models.dkl import DirectKRRHedging
from src.models.mv import PaperMVDeltaHedging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """训练DKL和MV模型"""
    
    print("\n" + "="*80)
    print("步骤3: 训练DKL和MV模型")
    print("="*80)
    
    # 加载训练数据
    print(f"\n加载训练数据: {FILE_PATHS['train_data']}")
    train_df = pd.read_parquet(FILE_PATHS['train_data'])
    print(f"数据量: {len(train_df)} 行")
    
    models = {}
    
    # 训练DKL模型
    print("\n" + "-"*80)
    print("训练Direct KRR模型")
    print("-"*80)
    
    dkl_model = DirectKRRHedging(
        max_samples=DKL_PARAMS['max_samples'],
        regularization_type=DKL_PARAMS['regularization_type']
    )
    
    dkl_model.fit(
        train_df,
        shift=1,  # 日度对冲
        val_ratio=DKL_PARAMS['val_ratio'],
        n_calls=DKL_PARAMS['n_calls']
    )
    
    models['dkl'] = dkl_model
    
    # 保存DKL模型
    dkl_model_path = PROCESSED_DATA_DIR / "dkl_model.pkl"
    with open(dkl_model_path, 'wb') as f:
        pickle.dump(dkl_model, f)
    print(f"DKL模型已保存: {dkl_model_path}")
    
    # 训练MV模型
    print("\n" + "-"*80)
    print("训练MV Delta模型")
    print("-"*80)
    
    mv_model = PaperMVDeltaHedging()
    mv_model.fit(train_df, r=HESTON_PARAMS['r'])
    
    models['mv'] = mv_model
    
    # 保存MV模型
    mv_model_path = PROCESSED_DATA_DIR / "mv_model.pkl"
    with open(mv_model_path, 'wb') as f:
        pickle.dump(mv_model, f)
    print(f"MV模型已保存: {mv_model_path}")
    
    print("\n模型训练完成！")
    print("="*80)
    
    return models


if __name__ == "__main__":
    main()

