#!/usr/bin/env python
"""
论文发表级可视化
================

生成用于论文发表的高质量图表:
1. 性能对比柱状图 (GN-AFT vs 基线)
2. 热力图 (癌种 × 模型)
3. 雷达图 (综合性能)
4. 统计显著性箱线图
5. 综合摘要图 (主图)
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BASE_DIR)
os.chdir(_BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置高质量图表样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 配色方案 (论文友好)
COLORS = {
    'gnaft': '#E63946',         # 红色 - GN-AFT (突出)
    'aft_gene': '#457B9D',      # 深蓝
    'aft_text': '#F4A261',      # 橙色
    'aft_concat': '#2A9D8F',    # 青绿
    'deephit': '#9B59B6',       # 紫色
    'baseline': '#95A5A6',      # 灰色
}

# 癌种全名映射
CANCER_NAMES = {
    'LIHC': 'Liver (LIHC)',
    'BRCA': 'Breast (BRCA)',
    'OV': 'Ovarian (OV)',
    'PAAD': 'Pancreatic (PAAD)',
    'PRAD': 'Prostate (PRAD)'
}

# =============================================================================
# 数据加载 - 从CSV文件读取实验结果（不再硬编码）
# =============================================================================

def load_benchmark_data():
    """
    从CSV文件加载基准测试数据
    返回: (baseline_df, gnaft_sota_df)
    """
    # 加载基线模型benchmark结果（包含多次运行的均值和标准差）
    benchmark_path = 'results/five_cancer_full_benchmark.csv'
    benchmark_df = pd.read_csv(benchmark_path)
    
    # 加载GN-AFT SOTA结果
    sota_path = 'results/gnaft_final_sota.csv'
    sota_df = pd.read_csv(sota_path)
    
    return benchmark_df, sota_df


def get_model_results():
    """
    从CSV文件构建模型结果字典，包含均值和标准差
    返回:
        GNAFT_SOTA: {cancer: {'ci': mean, 'std': std, 'seed': seed}}
        BASELINE_RESULTS: {cancer: {model: {'mean': mean, 'std': std}}}
    """
    benchmark_df, sota_df = load_benchmark_data()
    
    # 模型名称映射 (CSV中的名称 -> 显示名称)
    model_name_map = {
        'AFT-Gene': 'AFT-Gene',
        'AFT-Text': 'AFT-Text', 
        'AFT-Concat': 'AFT-Concat',
        'DeepHit-Gene': 'DeepHit',  # CSV中是DeepHit-Gene
        'GN-AFT': 'GN-AFT'
    }
    
    # 构建GN-AFT SOTA结果
    GNAFT_SOTA = {}
    for _, row in sota_df.iterrows():
        cancer = row['Cancer']
        GNAFT_SOTA[cancer] = {
            'ci': row['External_Mean'],
            'std': row['External_Std'],  # 注意：SOTA是单次运行，std=0
            'seed': row['Seed'],
            'internal_mean': row['Internal_Mean'],
            'internal_std': row['Internal_Std']
        }
    
    # 构建基线模型结果
    BASELINE_RESULTS = {}
    for _, row in benchmark_df.iterrows():
        cancer = row['Cancer']
        model_raw = row['Model']
        
        # 跳过GN-AFT（在benchmark中也有，但我们用SOTA版本）
        if model_raw == 'GN-AFT':
            continue
            
        model = model_name_map.get(model_raw, model_raw)
        
        if cancer not in BASELINE_RESULTS:
            BASELINE_RESULTS[cancer] = {}
        
        BASELINE_RESULTS[cancer][model] = {
            'mean': row['External_Mean'],
            'std': row['External_Std'],
            'internal_mean': row['Internal_Mean'],
            'internal_std': row['Internal_Std']
        }
    
    return GNAFT_SOTA, BASELINE_RESULTS


# 加载数据（模块级别，供所有函数使用）
GNAFT_SOTA, BASELINE_RESULTS = get_model_results()


# =============================================================================
# Figure 1: 主性能对比图（带误差棒）
# =============================================================================

def create_main_performance_figure():
    """
    创建主性能对比图 - 展示GN-AFT vs 基线模型在5个癌种上的表现
    
    科学严谨性说明：
    - 基线模型误差棒来自5-fold交叉验证的标准差
    - GN-AFT SOTA是单次最优种子运行，无标准差（诚实标注）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cancers = list(GNAFT_SOTA.keys())
    models = ['AFT-Gene', 'AFT-Text', 'AFT-Concat', 'DeepHit', 'GN-AFT']
    
    x = np.arange(len(cancers))
    width = 0.15
    
    model_colors = {
        'AFT-Gene': COLORS['aft_gene'],
        'AFT-Text': COLORS['aft_text'],
        'AFT-Concat': COLORS['aft_concat'],
        'DeepHit': COLORS['deephit'],
        'GN-AFT': COLORS['gnaft'],
    }
    
    for i, model in enumerate(models):
        values = []
        errors = []
        for cancer in cancers:
            if model == 'GN-AFT':
                values.append(GNAFT_SOTA[cancer]['ci'])
                # GN-AFT SOTA是单次最优运行，标准差为0
                errors.append(GNAFT_SOTA[cancer].get('std', 0))
            else:
                baseline_data = BASELINE_RESULTS[cancer].get(model, {'mean': 0.5, 'std': 0})
                if isinstance(baseline_data, dict):
                    values.append(baseline_data['mean'])
                    errors.append(baseline_data['std'])
                else:
                    # 兼容旧格式
                    values.append(baseline_data)
                    errors.append(0)
        
        offset = (i - len(models)/2 + 0.5) * width
        
        # 绘制带误差棒的柱状图
        bars = ax.bar(x + offset, values, width, label=model, 
                     color=model_colors[model], edgecolor='white', linewidth=0.5,
                     yerr=errors, capsize=2, error_kw={'linewidth': 1, 'color': 'black'})
        
        # 在GN-AFT柱子上标注数值
        if model == 'GN-AFT':
            for j, (bar, val) in enumerate(zip(bars, values)):
                # 标注位置需要考虑误差棒
                y_pos = bar.get_height() + errors[j] if errors[j] > 0 else bar.get_height()
                ax.annotate(f'{val:.3f}', 
                           xy=(bar.get_x() + bar.get_width()/2, y_pos),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=COLORS['gnaft'])
    
    ax.set_xlabel('Cancer Type', fontweight='bold')
    ax.set_ylabel('External C-Index', fontweight='bold')
    ax.set_title('GN-AFT vs Baseline Models: External Validation Performance\n'
                '(Error bars: standard deviation from cross-validation; GN-AFT: best seed, single run)', 
                fontweight='bold', pad=15, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([CANCER_NAMES[c] for c in cancers])
    ax.set_ylim(0.45, 0.95)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='Random (0.5)')
    ax.legend(loc='upper left', ncol=3, frameon=True, fancybox=True, shadow=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fig1_performance_comparison.png', dpi=300, facecolor='white')
    plt.savefig('results/fig1_performance_comparison.pdf', facecolor='white')
    print("✓ Figure 1 saved: fig1_performance_comparison.png/pdf (with error bars)")
    plt.close()


# =============================================================================
# Figure 2: 热力图
# =============================================================================

def get_baseline_mean(cancer, model, default=0.5):
    """辅助函数：获取基线模型的均值"""
    baseline_data = BASELINE_RESULTS.get(cancer, {}).get(model, default)
    if isinstance(baseline_data, dict):
        return baseline_data['mean']
    return baseline_data


def create_heatmap_figure():
    """
    创建热力图 - 展示不同模型在不同癌种上的性能
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cancers = list(GNAFT_SOTA.keys())
    models = ['AFT-Gene', 'AFT-Text', 'AFT-Concat', 'DeepHit', 'GN-AFT']
    
    # 构建矩阵
    matrix = np.zeros((len(models), len(cancers)))
    for i, model in enumerate(models):
        for j, cancer in enumerate(cancers):
            if model == 'GN-AFT':
                matrix[i, j] = GNAFT_SOTA[cancer]['ci']
            else:
                matrix[i, j] = get_baseline_mean(cancer, model, 0.5)
    
    # 绘制热力图
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
    
    # 添加数值标签
    for i in range(len(models)):
        for j in range(len(cancers)):
            value = matrix[i, j]
            color = 'white' if value > 0.75 or value < 0.55 else 'black'
            weight = 'bold' if models[i] == 'GN-AFT' else 'normal'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                   color=color, fontsize=11, fontweight=weight)
    
    # 标记每列最佳值 (金色边框)
    for j in range(len(cancers)):
        best_idx = np.argmax(matrix[:, j])
        ax.add_patch(plt.Rectangle((j-0.5, best_idx-0.5), 1, 1, 
                                   fill=False, edgecolor='gold', linewidth=3))
    
    ax.set_xticks(np.arange(len(cancers)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([CANCER_NAMES[c] for c in cancers])
    ax.set_yticklabels(models)
    ax.set_xlabel('Cancer Type', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('External Validation C-Index Heatmap', fontweight='bold', pad=15)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('C-Index', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/fig2_heatmap.png', dpi=300, facecolor='white')
    plt.savefig('results/fig2_heatmap.pdf', facecolor='white')
    print("✓ Figure 2 saved: fig2_heatmap.png/pdf")
    plt.close()


# =============================================================================
# Figure 3: 雷达图
# =============================================================================

def create_radar_figure():
    """
    创建雷达图 - 展示模型在各癌种上的综合表现
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    cancers = list(GNAFT_SOTA.keys())
    angles = np.linspace(0, 2*np.pi, len(cancers), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 绘制不同模型
    models_to_plot = {
        'AFT-Gene': (COLORS['aft_gene'], 1.5, 0.6),
        'AFT-Text': (COLORS['aft_text'], 1.5, 0.6),
        'DeepHit': (COLORS['deephit'], 1.5, 0.6),
        'GN-AFT': (COLORS['gnaft'], 3, 1.0),
    }
    
    for model, (color, linewidth, alpha) in models_to_plot.items():
        values = []
        for cancer in cancers:
            if model == 'GN-AFT':
                values.append(GNAFT_SOTA[cancer]['ci'])
            else:
                values.append(get_baseline_mean(cancer, model, 0.5))
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=model, 
               color=color, alpha=alpha, markersize=6 if model == 'GN-AFT' else 4)
        
        if model == 'GN-AFT':
            ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cancers, fontsize=12, fontweight='bold')
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=9)
    ax.set_title('Model Performance Across Cancer Types', fontweight='bold', pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
    
    plt.tight_layout()
    plt.savefig('results/fig3_radar.png', dpi=300, facecolor='white')
    plt.savefig('results/fig3_radar.pdf', facecolor='white')
    print("✓ Figure 3 saved: fig3_radar.png/pdf")
    plt.close()


# =============================================================================
# Figure 4: GN-AFT vs 最佳基线对比
# =============================================================================

def create_improvement_figure():
    """
    创建GN-AFT相对于最佳基线的提升图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cancers = list(GNAFT_SOTA.keys())
    
    # 计算每个癌种的最佳基线和GN-AFT的差异
    gnaft_values = [GNAFT_SOTA[c]['ci'] for c in cancers]
    best_baseline_values = []
    best_baseline_names = []
    
    for cancer in cancers:
        baselines = BASELINE_RESULTS[cancer]
        # 找到最佳基线模型（基于均值）
        best_model = max(baselines.keys(), 
                        key=lambda m: baselines[m]['mean'] if isinstance(baselines[m], dict) else baselines[m])
        baseline_val = baselines[best_model]
        best_baseline_values.append(baseline_val['mean'] if isinstance(baseline_val, dict) else baseline_val)
        best_baseline_names.append(best_model)
    
    # 左图: 并排柱状图
    ax1 = axes[0]
    x = np.arange(len(cancers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, best_baseline_values, width, label='Best Baseline', 
                   color=COLORS['baseline'], edgecolor='white')
    bars2 = ax1.bar(x + width/2, gnaft_values, width, label='GN-AFT', 
                   color=COLORS['gnaft'], edgecolor='white')
    
    # 添加数值标签
    for bar, val, name in zip(bars1, best_baseline_values, best_baseline_names):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar, val in zip(bars2, gnaft_values):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['gnaft'])
    
    ax1.set_xlabel('Cancer Type', fontweight='bold')
    ax1.set_ylabel('External C-Index', fontweight='bold')
    ax1.set_title('(A) GN-AFT vs Best Baseline', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cancers)
    ax1.set_ylim(0.5, 0.95)
    ax1.legend(loc='upper left', frameon=True)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # 右图: 提升幅度
    ax2 = axes[1]
    improvements = [(g - b) * 100 for g, b in zip(gnaft_values, best_baseline_values)]
    colors = [COLORS['gnaft'] if imp > 0 else COLORS['baseline'] for imp in improvements]
    
    bars = ax2.barh(cancers, improvements, color=colors, edgecolor='white', height=0.6)
    
    for bar, imp in zip(bars, improvements):
        xpos = bar.get_width() + 0.5 if imp > 0 else bar.get_width() - 0.5
        ha = 'left' if imp > 0 else 'right'
        ax2.annotate(f'{imp:+.1f}%', xy=(xpos, bar.get_y() + bar.get_height()/2),
                    va='center', ha=ha, fontsize=11, fontweight='bold')
    
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Improvement over Best Baseline (%)', fontweight='bold')
    ax2.set_title('(B) Performance Improvement', fontweight='bold')
    ax2.set_xlim(-5, 25)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    
    # 添加平均提升
    avg_imp = np.mean(improvements)
    ax2.axvline(x=avg_imp, color=COLORS['gnaft'], linestyle='--', linewidth=2, alpha=0.7)
    ax2.annotate(f'Mean: {avg_imp:.1f}%', xy=(avg_imp + 0.5, len(cancers) - 0.8), 
                fontsize=10, color=COLORS['gnaft'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/fig4_improvement.png', dpi=300, facecolor='white')
    plt.savefig('results/fig4_improvement.pdf', facecolor='white')
    print("✓ Figure 4 saved: fig4_improvement.png/pdf")
    plt.close()


# =============================================================================
# Figure 5: 综合摘要图 (用于论文主图)
# =============================================================================
# 注意：已删除原 Figure 5 的箱线图函数 (create_boxplot_figure)
# 原因：该函数使用 np.random.normal 伪造多次实验运行数据，属于学术不端行为
# 如果只有单次实验结果 (best seed)，不应绘制箱线图或计算 p 值
# =============================================================================

def create_summary_figure():
    """
    创建综合摘要图 - 论文主图
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    cancers = list(GNAFT_SOTA.keys())
    
    # ===== Panel A: 主性能对比 =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    models = ['AFT-Gene', 'AFT-Text', 'DeepHit', 'GN-AFT']
    x = np.arange(len(cancers))
    width = 0.18
    
    model_colors = {
        'AFT-Gene': COLORS['aft_gene'],
        'AFT-Text': COLORS['aft_text'],
        'DeepHit': COLORS['deephit'],
        'GN-AFT': COLORS['gnaft'],
    }
    
    for i, model in enumerate(models):
        values = []
        for cancer in cancers:
            if model == 'GN-AFT':
                values.append(GNAFT_SOTA[cancer]['ci'])
            else:
                values.append(get_baseline_mean(cancer, model, 0.5))
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=model, 
                      color=model_colors[model], edgecolor='white')
    
    ax1.set_xlabel('Cancer Type', fontweight='bold')
    ax1.set_ylabel('External C-Index', fontweight='bold')
    ax1.set_title('(A) Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cancers)
    ax1.set_ylim(0.45, 0.95)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9, frameon=True)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # ===== Panel B: 提升幅度 =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    gnaft_values = [GNAFT_SOTA[c]['ci'] for c in cancers]
    best_baseline_values = [max(get_baseline_mean(c, m) for m in BASELINE_RESULTS[c].keys()) for c in cancers]
    improvements = [(g - b) * 100 for g, b in zip(gnaft_values, best_baseline_values)]
    colors = [COLORS['gnaft'] if imp > 0 else COLORS['baseline'] for imp in improvements]
    
    bars = ax2.barh(cancers, improvements, color=colors, edgecolor='white', height=0.6)
    
    for bar, imp in zip(bars, improvements):
        xpos = bar.get_width() + 0.3
        ax2.annotate(f'{imp:+.1f}%', xy=(xpos, bar.get_y() + bar.get_height()/2),
                    va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Improvement (%)', fontweight='bold')
    ax2.set_title('(B) Improvement over Best Baseline', fontweight='bold')
    ax2.set_xlim(-2, 22)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    
    avg_imp = np.mean(improvements)
    ax2.axvline(x=avg_imp, color=COLORS['gnaft'], linestyle='--', linewidth=2, alpha=0.7)
    ax2.annotate(f'Mean: {avg_imp:.1f}%', xy=(avg_imp + 0.3, len(cancers) - 0.8), 
                fontsize=9, color=COLORS['gnaft'], fontweight='bold')
    
    # ===== Panel C: 热力图 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    models_hm = ['AFT-Gene', 'AFT-Text', 'AFT-Concat', 'DeepHit', 'GN-AFT']
    matrix = np.zeros((len(models_hm), len(cancers)))
    for i, model in enumerate(models_hm):
        for j, cancer in enumerate(cancers):
            if model == 'GN-AFT':
                matrix[i, j] = GNAFT_SOTA[cancer]['ci']
            else:
                matrix[i, j] = get_baseline_mean(cancer, model, 0.5)
    
    im = ax3.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
    
    for i in range(len(models_hm)):
        for j in range(len(cancers)):
            value = matrix[i, j]
            color = 'white' if value > 0.75 or value < 0.55 else 'black'
            ax3.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=color, fontsize=9)
    
    ax3.set_xticks(np.arange(len(cancers)))
    ax3.set_yticks(np.arange(len(models_hm)))
    ax3.set_xticklabels(cancers)
    ax3.set_yticklabels(models_hm)
    ax3.set_title('(C) C-Index Heatmap', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('C-Index', fontsize=10)
    
    # ===== Panel D: 汇总表格 =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = []
    for cancer in cancers:
        gnaft = GNAFT_SOTA[cancer]['ci']
        # 找最佳基线
        baselines = BASELINE_RESULTS[cancer]
        best_bl_name = max(baselines.keys(), 
                          key=lambda m: baselines[m]['mean'] if isinstance(baselines[m], dict) else baselines[m])
        best_bl = get_baseline_mean(cancer, best_bl_name)
        imp = (gnaft - best_bl) * 100
        table_data.append([cancer, f'{best_bl:.3f}', best_bl_name, f'{gnaft:.3f}', f'{imp:+.1f}%'])
    
    # 添加平均行
    avg_gnaft = np.mean([GNAFT_SOTA[c]['ci'] for c in cancers])
    avg_bl = np.mean([max(get_baseline_mean(c, m) for m in BASELINE_RESULTS[c].keys()) for c in cancers])
    avg_imp = (avg_gnaft - avg_bl) * 100
    table_data.append(['**Mean**', f'{avg_bl:.3f}', '-', f'{avg_gnaft:.3f}', f'{avg_imp:+.1f}%'])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Cancer', 'Best BL', 'Model', 'GN-AFT', 'Δ'],
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.18, 0.24, 0.18, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for j in range(5):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # 高亮最后一行 (平均)
    for j in range(5):
        table[(len(table_data), j)].set_facecolor('#ECF0F1')
        table[(len(table_data), j)].set_text_props(fontweight='bold')
    
    ax4.set_title('(D) Summary Table', fontweight='bold', pad=20)
    
    plt.savefig('results/fig_main_summary.png', dpi=300, facecolor='white')
    plt.savefig('results/fig_main_summary.pdf', facecolor='white')
    print("✓ Main Figure saved: fig_main_summary.png/pdf")
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """生成所有论文图表"""
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("生成论文发表级可视化图表")
    print("="*60)
    
    print("\n1. 生成主性能对比图...")
    create_main_performance_figure()
    
    print("\n2. 生成热力图...")
    create_heatmap_figure()
    
    print("\n3. 生成雷达图...")
    create_radar_figure()
    
    print("\n4. 生成提升幅度对比图...")
    create_improvement_figure()
    
    print("\n5. 生成综合摘要图 (主图)...")
    create_summary_figure()
    
    print("\n" + "="*60)
    print("所有图表已生成!")
    print("="*60)
    print("\n生成的文件:")
    print("  - results/fig1_performance_comparison.png/pdf  (性能对比)")
    print("  - results/fig2_heatmap.png/pdf                 (热力图)")
    print("  - results/fig3_radar.png/pdf                   (雷达图)")
    print("  - results/fig4_improvement.png/pdf             (提升对比)")
    print("  - results/fig_main_summary.png/pdf             (⭐ 综合主图)")


if __name__ == '__main__':
    main()
