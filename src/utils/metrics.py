"""
对冲性能评估指标
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False


def compute_hedge_errors(test_df, delta_col, freq_days=1):
    """
    计算对冲误差
    
    参数:
        test_df: 测试数据集
        delta_col: Delta列名
        freq_days: 对冲频率（天）
    
    返回:
        np.array: 对冲误差数组
    """
    hedge_errors = []
    
    # 按合约分组
    for contract_id in test_df['Contract_ID'].unique():
        contract_df = test_df[test_df['Contract_ID'] == contract_id].sort_values('Time_Step').reset_index(drop=True)
        
        if len(contract_df) <= freq_days:
            continue
        
        # 计算对冲误差
        for i in range(0, len(contract_df) - freq_days, freq_days):
            if i + freq_days < len(contract_df):
                row0 = contract_df.iloc[i]
                row1 = contract_df.iloc[i + freq_days]
                
                dS = row1['Stock_Price'] - row0['Stock_Price']
                dV = row1['Market_Price'] - row0['Market_Price']
                
                delta = row0[delta_col]
                
                # 对冲误差
                hedge_error = dV - delta * dS
                hedge_errors.append(hedge_error)
    
    return np.array(hedge_errors)


def compute_hedge_metrics(errors, method_name='Method'):
    """
    计算完整的对冲性能指标
    
    参数:
        errors: 对冲误差数组
        method_name: 方法名称
    
    返回:
        dict: 包含各项指标的字典
    """
    errors = np.array(errors)
    
    if len(errors) == 0:
        return {
            'Method': method_name,
            'Sample_Size': 0,
            'SSE': np.nan,
            'E(|ΔV-ΔSf(x)|)': np.nan,
            'Std': np.nan,
            'VaR_95': np.nan,
            'CVaR_95': np.nan
        }
    
    abs_errors = np.abs(errors)
    
    return {
        'Method': method_name,
        'Sample_Size': len(errors),
        'SSE': np.sum(errors ** 2),
        'E(|ΔV-ΔSf(x)|)': np.mean(abs_errors),
        'Std': np.std(errors),
        'VaR_95': np.percentile(abs_errors, 95),
        'CVaR_95': np.mean(abs_errors[abs_errors >= np.percentile(abs_errors, 95)])
    }


def calculate_gain(sse_method, sse_baseline):
    """
    计算Gain（相对于基准方法的改进）
    
    Gain = (1 - SSE_method / SSE_baseline) * 100%
    
    参数:
        sse_method: 方法的SSE
        sse_baseline: 基准方法的SSE
    
    返回:
        float: Gain百分比
    """
    if sse_baseline == 0:
        return 0.0
    
    gain = (1 - sse_method / sse_baseline) * 100
    return gain


def evaluate_all_methods(test_df, trained_models, freq_days=1, r=0.02):
    """
    评估所有方法的对冲性能
    
    参数:
        test_df: 测试数据集
        trained_models: 训练好的模型字典 {'dkl': model, 'mv': model}
        freq_days: 对冲频率
        r: 无风险利率
    
    返回:
        list: 包含所有方法指标的列表
    """
    results = []
    
    # 1. BS Delta（基准）
    logging.info("评估BS Delta...")
    errors_bs = compute_hedge_errors(test_df, 'BS_Delta', freq_days)
    metrics_bs = compute_hedge_metrics(errors_bs, 'BS_Delta')
    metrics_bs['Gain(%)'] = 0.0
    results.append(metrics_bs)
    
    baseline_sse = metrics_bs['SSE']
    
    # 2. Heston Delta（如果有）
    if 'Heston_Delta' in test_df.columns:
        logging.info("评估Heston Delta...")
        errors_heston = compute_hedge_errors(test_df, 'Heston_Delta', freq_days)
        metrics_heston = compute_hedge_metrics(errors_heston, 'Heston_Delta')
        metrics_heston['Gain(%)'] = calculate_gain(metrics_heston['SSE'], baseline_sse)
        results.append(metrics_heston)
    
    # 3. Analytic Heston Delta（如果有）
    if 'Analytic_Heston_Delta' in test_df.columns:
        logging.info("评估Analytic Heston Delta...")
        errors_analytic = compute_hedge_errors(test_df, 'Analytic_Heston_Delta', freq_days)
        metrics_analytic = compute_hedge_metrics(errors_analytic, 'Analytic_Heston')
        metrics_analytic['Gain(%)'] = calculate_gain(metrics_analytic['SSE'], baseline_sse)
        results.append(metrics_analytic)
    
    # 4. DKL Delta（如果提供了模型）
    if 'dkl' in trained_models and trained_models['dkl'] is not None:
        logging.info("评估DKL Delta...")
        errors_dkl = []
        
        for contract_id in test_df['Contract_ID'].unique():
            contract_df = test_df[test_df['Contract_ID'] == contract_id].sort_values('Time_Step')
            
            if len(contract_df) <= freq_days:
                continue
            
            for i in range(0, len(contract_df) - freq_days, freq_days):
                if i + freq_days < len(contract_df):
                    current = contract_df.iloc[i]
                    future = contract_df.iloc[i + freq_days]
                    
                    dv = future['Market_Price'] - current['Market_Price']
                    ds = future['Stock_Price'] - current['Stock_Price']
                    
                    # DKL预测
                    X_test = np.array([[
                        current['Moneyness'],
                        current['Time_to_Expiry'],
                        current['BS_Delta']
                    ]])
                    delta_dkl = trained_models['dkl'].predict(X_test)[0]
                    errors_dkl.append(dv - delta_dkl * ds)
        
        metrics_dkl = compute_hedge_metrics(np.array(errors_dkl), 'Direct_KRR')
        metrics_dkl['Gain(%)'] = calculate_gain(metrics_dkl['SSE'], baseline_sse)
        results.append(metrics_dkl)
    
    # 5. MV Delta（如果提供了模型）
    if 'mv' in trained_models and trained_models['mv'] is not None:
        logging.info("评估MV Delta...")
        errors_mv = []
        
        for contract_id in test_df['Contract_ID'].unique():
            contract_df = test_df[test_df['Contract_ID'] == contract_id].sort_values('Time_Step')
            
            if len(contract_df) <= freq_days:
                continue
            
            for i in range(0, len(contract_df) - freq_days, freq_days):
                if i + freq_days >= len(contract_df):
                    break
                    
                row0 = contract_df.iloc[i]
                row1 = contract_df.iloc[i + freq_days]
                
                dS = row1['Stock_Price'] - row0['Stock_Price']
                dV = row1['Market_Price'] - row0['Market_Price']
                
                # MV Delta计算
                S = row0['Stock_Price']
                K = row0['Strike_Price']
                T = row0['Time_to_Expiry']
                iv = row0['Implied_Vol']
                bs_delta = row0['BS_Delta']
                
                mv_delta = trained_models['mv'].calculate_mv_delta(S, K, T, r, iv, bs_delta)
                errors_mv.append(dV - mv_delta * dS)
        
        metrics_mv = compute_hedge_metrics(np.array(errors_mv), 'MV_Delta')
        metrics_mv['Gain(%)'] = calculate_gain(metrics_mv['SSE'], baseline_sse)
        results.append(metrics_mv)
    
    return results


def plot_comparison(results_df, save_path=None):
    """
    绘制对比图
    
    参数:
        results_df: 结果DataFrame
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # 1. Gain对比
    ax = axes[0, 0]
    if 'Frequency' in results_df.columns:
        pivot_gain = results_df.pivot(index='Frequency', columns='Method', values='Gain(%)')
        pivot_gain.plot(kind='bar', ax=ax, color=colors[:len(pivot_gain.columns)])
    else:
        results_df.plot(x='Method', y='Gain(%)', kind='bar', ax=ax, color=colors, legend=False)
    ax.set_title('Gain (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gain (%)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 2. MAE对比
    ax = axes[0, 1]
    if 'Frequency' in results_df.columns:
        pivot_mae = results_df.pivot(index='Frequency', columns='Method', values='E(|ΔV-ΔSf(x)|)')
        pivot_mae.plot(kind='bar', ax=ax, color=colors[:len(pivot_mae.columns)])
    else:
        results_df.plot(x='Method', y='E(|ΔV-ΔSf(x)|)', kind='bar', ax=ax, color=colors, legend=False)
    ax.set_title('E(|ΔV-ΔSf(x)|)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error')
    ax.grid(True, alpha=0.3)
    
    # 3. Std对比
    ax = axes[0, 2]
    if 'Frequency' in results_df.columns:
        pivot_std = results_df.pivot(index='Frequency', columns='Method', values='Std')
        pivot_std.plot(kind='bar', ax=ax, color=colors[:len(pivot_std.columns)])
    else:
        results_df.plot(x='Method', y='Std', kind='bar', ax=ax, color=colors, legend=False)
    ax.set_title('Standard Deviation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Std')
    ax.grid(True, alpha=0.3)
    
    # 4. VaR对比
    ax = axes[1, 0]
    if 'Frequency' in results_df.columns:
        pivot_var = results_df.pivot(index='Frequency', columns='Method', values='VaR_95')
        pivot_var.plot(kind='bar', ax=ax, color=colors[:len(pivot_var.columns)])
    else:
        results_df.plot(x='Method', y='VaR_95', kind='bar', ax=ax, color=colors, legend=False)
    ax.set_title('VaR (95%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value at Risk')
    ax.grid(True, alpha=0.3)
    
    # 5. CVaR对比
    ax = axes[1, 1]
    if 'Frequency' in results_df.columns:
        pivot_cvar = results_df.pivot(index='Frequency', columns='Method', values='CVaR_95')
        pivot_cvar.plot(kind='bar', ax=ax, color=colors[:len(pivot_cvar.columns)])
    else:
        results_df.plot(x='Method', y='CVaR_95', kind='bar', ax=ax, color=colors, legend=False)
    ax.set_title('CVaR (95%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Conditional VaR')
    ax.grid(True, alpha=0.3)
    
    # 6. 汇总表格
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    if 'Frequency' in results_df.columns:
        summary_data = []
        for freq in results_df['Frequency'].unique():
            freq_data = results_df[results_df['Frequency'] == freq]
            best_gain = freq_data['Gain(%)'].max()
            best_method = freq_data.loc[freq_data['Gain(%)'].idxmax(), 'Method']
            summary_data.append([freq, f"{best_gain:.2f}%", best_method])
        
        table = ax.table(cellText=summary_data,
                         colLabels=['Frequency', 'Best Gain', 'Method'],
                         cellLoc='center',
                         loc='center')
    else:
        summary_data = [[row['Method'], f"{row['Gain(%)']:.2f}%"] 
                       for _, row in results_df.iterrows()]
        table = ax.table(cellText=summary_data,
                         colLabels=['Method', 'Gain'],
                         cellLoc='center',
                         loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    plt.suptitle('Hedging Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"图表已保存到: {save_path}")
    
    plt.show()
    
    return fig


def print_results_table(results_df):
    """
    打印格式化的结果表格
    
    参数:
        results_df: 结果DataFrame
    """
    print("\n" + "="*100)
    print("对冲性能评估结果")
    print("="*100)
    
    if 'Frequency' in results_df.columns:
        for freq in results_df['Frequency'].unique():
            freq_data = results_df[results_df['Frequency'] == freq]
            print(f"\n{freq} 对冲:")
            print("-"*100)
            print(f"{'Method':<20} {'Gain(%)':<12} {'E(|ΔV-ΔSf(x)|)':<18} {'Std':<12} {'VaR_95':<12} {'CVaR_95':<12}")
            print("-"*100)
            for _, row in freq_data.iterrows():
                print(f"{row['Method']:<20} {row['Gain(%)']:<12.2f} {row['E(|ΔV-ΔSf(x)|)']:<18.6f} "
                      f"{row['Std']:<12.6f} {row['VaR_95']:<12.6f} {row['CVaR_95']:<12.6f}")
    else:
        print(f"{'Method':<20} {'Gain(%)':<12} {'E(|ΔV-ΔSf(x)|)':<18} {'Std':<12} {'VaR_95':<12} {'CVaR_95':<12}")
        print("-"*100)
        for _, row in results_df.iterrows():
            print(f"{row['Method']:<20} {row['Gain(%)']:<12.2f} {row['E(|ΔV-ΔSf(x)|)']:<18.6f} "
                  f"{row['Std']:<12.6f} {row['VaR_95']:<12.6f} {row['CVaR_95']:<12.6f}")
    
    print("="*100)

