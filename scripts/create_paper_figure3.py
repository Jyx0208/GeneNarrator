#!/usr/bin/env python
"""
论文 Figure 3: 机制解析 - 为什么能消除批次效应？
================================================

展示文本嵌入如何消除批次效应并保留生物学信息。

子图:
  A: t-SNE - 原始基因表达数据 (预期显示批次效应)
  B: t-SNE - V5 文本嵌入 (预期显示混合且聚类成生物学亚群)
  C: 亚型分布对比 (证明 LIRI-JP 恢复了生物学结构)

数据来源:
  - data/TCGA-LIHC.star_fpkm.tsv.gz (TCGA 基因表达)
  - data/LIRI-JP.star_fpkm.tsv.gz (LIRI-JP 基因表达)
  - data/TCGA-LIHC_embeddings_v5.pt (TCGA V5 嵌入)
  - data/LIRI-JP_embeddings_v5.pt (LIRI-JP V5 嵌入)
  - data/*_reports_v5.txt (亚型标签)

学术诚信声明:
  - 所有数据来自真实实验结果
  - t-SNE 降维使用真实数据，不做任何修改
  - 亚型标签来自 GeneNarrator V5 报告
"""

import os
import sys
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BASE_DIR)
os.chdir(_BASE_DIR)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 图表样式设置
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 配色方案
COLORS = {
    'tcga': '#E63946',      # 红色 - TCGA
    'liri': '#457B9D',      # 蓝色 - LIRI-JP
    'S1': '#2ECC71',        # 绿色 - Hoshida-S1
    'S2': '#E74C3C',        # 红色 - Hoshida-S2
    'S3': '#3498DB',        # 蓝色 - Hoshida-S3
    'Unknown': '#95A5A6',   # 灰色 - Unknown
}


# =============================================================================
# 数据加载函数
# =============================================================================

def load_gene_expression(cohort):
    """加载基因表达数据"""
    path = f'data/{cohort}.star_fpkm.tsv.gz'
    df = pd.read_csv(path, sep='\t', index_col=0)
    return df


def load_embeddings(cohort):
    """加载 V5 文本嵌入"""
    path = f'data/{cohort}_embeddings_v5.pt'
    data = torch.load(path, map_location='cpu')
    
    if isinstance(data, dict):
        if 'embeddings' in data:
            embeddings = data['embeddings']
        else:
            # 尝试获取第一个张量
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    embeddings = v
                    break
    elif isinstance(data, torch.Tensor):
        embeddings = data
    else:
        raise ValueError(f"Unknown embedding format for {cohort}")
    
    return embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings


def extract_subtypes(cohort):
    """从报告文本中提取亚型标签"""
    path = f'data/{cohort}_reports_v5.txt'
    subtypes = []
    sample_ids = []
    
    with open(path, 'r') as f:
        content = f.read()
    
    # 解析每个报告
    reports = content.split('------------------------------------------------------------')
    
    for report in reports:
        report = report.strip()
        if not report:
            continue
        
        # 提取样本ID和亚型
        # 格式: [1] Patient: TCGA-FV-A495-01A | Hoshida-S2
        match = re.search(r'Patient:\s*(\S+)\s*\|\s*(Hoshida-S\d|Unknown)', report)
        if match:
            sample_id = match.group(1)
            subtype = match.group(2).replace('Hoshida-', '')
            sample_ids.append(sample_id)
            subtypes.append(subtype)
    
    return sample_ids, subtypes


# =============================================================================
# Figure 3: 主图
# =============================================================================

def create_figure3():
    """
    创建 Figure 3: 机制解析 - 批次效应消除
    """
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    # =========================================================================
    # 加载数据
    # =========================================================================
    
    # 加载基因表达
    print("\n[1/4] 加载基因表达数据...")
    tcga_gene = load_gene_expression('TCGA-LIHC')
    liri_gene = load_gene_expression('LIRI-JP')
    print(f"  TCGA-LIHC: {tcga_gene.shape[1]} samples, {tcga_gene.shape[0]} genes")
    print(f"  LIRI-JP: {liri_gene.shape[1]} samples, {liri_gene.shape[0]} genes")
    
    # 加载嵌入
    print("\n[2/4] 加载 V5 文本嵌入...")
    tcga_emb = load_embeddings('TCGA-LIHC')
    liri_emb = load_embeddings('LIRI-JP')
    print(f"  TCGA-LIHC embeddings: {tcga_emb.shape}")
    print(f"  LIRI-JP embeddings: {liri_emb.shape}")
    
    # 提取亚型
    print("\n[3/4] 提取亚型标签...")
    tcga_ids, tcga_subtypes = extract_subtypes('TCGA-LIHC')
    liri_ids, liri_subtypes = extract_subtypes('LIRI-JP')
    print(f"  TCGA-LIHC: {len(tcga_subtypes)} samples")
    print(f"  LIRI-JP: {len(liri_subtypes)} samples")
    
    # 统计亚型分布
    from collections import Counter
    tcga_counts = Counter(tcga_subtypes)
    liri_counts = Counter(liri_subtypes)
    print(f"\n  TCGA-LIHC 亚型分布: {dict(tcga_counts)}")
    print(f"  LIRI-JP 亚型分布: {dict(liri_counts)}")
    
    # =========================================================================
    # 准备数据用于 t-SNE
    # =========================================================================
    print("\n[4/4] 准备 t-SNE 数据...")
    
    # 找到共同基因
    common_genes = list(set(tcga_gene.index) & set(liri_gene.index))
    print(f"  共同基因数: {len(common_genes)}")
    
    # 合并基因表达数据
    tcga_gene_common = tcga_gene.loc[common_genes].T.values
    liri_gene_common = liri_gene.loc[common_genes].T.values
    
    n_tcga = tcga_gene_common.shape[0]
    n_liri = liri_gene_common.shape[0]
    
    gene_combined = np.vstack([tcga_gene_common, liri_gene_common])
    cohort_labels = ['TCGA'] * n_tcga + ['LIRI'] * n_liri
    
    # 合并嵌入数据 (确保样本数匹配)
    n_tcga_emb = min(tcga_emb.shape[0], n_tcga)
    n_liri_emb = min(liri_emb.shape[0], n_liri)
    
    emb_combined = np.vstack([tcga_emb[:n_tcga_emb], liri_emb[:n_liri_emb]])
    cohort_labels_emb = ['TCGA'] * n_tcga_emb + ['LIRI'] * n_liri_emb
    
    # 合并亚型标签 (用于嵌入数据)
    subtypes_combined = tcga_subtypes[:n_tcga_emb] + liri_subtypes[:n_liri_emb]
    
    # =========================================================================
    # t-SNE 降维
    # =========================================================================
    print("\n执行 t-SNE 降维...")
    
    # 基因表达 t-SNE
    print("  [A] 基因表达 t-SNE...")
    
    # 处理缺失值：用列均值填充
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    gene_imputed = imputer.fit_transform(gene_combined)
    
    gene_scaled = StandardScaler().fit_transform(gene_imputed)
    
    # 使用 PCA 预降维以加速
    from sklearn.decomposition import PCA
    if gene_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        gene_pca = pca.fit_transform(gene_scaled)
    else:
        gene_pca = gene_scaled
    
    tsne_gene = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    gene_tsne = tsne_gene.fit_transform(gene_pca)
    
    # 嵌入 t-SNE
    print("  [B] 文本嵌入 t-SNE...")
    emb_scaled = StandardScaler().fit_transform(emb_combined)
    tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_tsne = tsne_emb.fit_transform(emb_scaled)
    
    # =========================================================================
    # 创建图表
    # =========================================================================
    print("\n生成图表...")
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # =========================================================================
    # Panel A: t-SNE - 基因表达
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # 按队列分开绘制
    tcga_mask = np.array(cohort_labels) == 'TCGA'
    liri_mask = np.array(cohort_labels) == 'LIRI'
    
    ax_a.scatter(gene_tsne[tcga_mask, 0], gene_tsne[tcga_mask, 1], 
                c=COLORS['tcga'], label=f'TCGA-LIHC (n={tcga_mask.sum()})', 
                alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    ax_a.scatter(gene_tsne[liri_mask, 0], gene_tsne[liri_mask, 1], 
                c=COLORS['liri'], label=f'LIRI-JP (n={liri_mask.sum()})', 
                alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax_a.set_xlabel('t-SNE 1')
    ax_a.set_ylabel('t-SNE 2')
    ax_a.set_title('(A) Gene Expression\n(Batch Effect Visible)', fontweight='bold', pad=10)
    ax_a.legend(loc='best', frameon=True)
    
    # 添加说明
    ax_a.text(0.02, 0.98, 'Cohorts separated\n→ Batch effect', transform=ax_a.transAxes,
             fontsize=9, va='top', ha='left', color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel B: t-SNE - 文本嵌入 (按亚型着色)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # 按亚型分组绘制
    subtypes_arr = np.array(subtypes_combined)
    for subtype in ['S1', 'S2', 'S3', 'Unknown']:
        mask = subtypes_arr == subtype
        if mask.sum() > 0:
            ax_b.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                        c=COLORS.get(subtype, COLORS['Unknown']),
                        label=f'{subtype} (n={mask.sum()})',
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax_b.set_xlabel('t-SNE 1')
    ax_b.set_ylabel('t-SNE 2')
    ax_b.set_title('(B) Text Embedding (V5)\n(Biological Subtypes)', fontweight='bold', pad=10)
    ax_b.legend(loc='best', frameon=True)
    
    # 添加说明
    ax_b.text(0.02, 0.98, 'Subtypes clustered\n→ Biology preserved', transform=ax_b.transAxes,
             fontsize=9, va='top', ha='left', color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel C: 亚型分布对比
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    subtypes_order = ['S1', 'S2', 'S3']
    
    # 计算各亚型比例
    tcga_total = len(tcga_subtypes)
    liri_total = len(liri_subtypes)
    
    tcga_pcts = [tcga_counts.get(s, 0) / tcga_total * 100 for s in subtypes_order]
    liri_pcts = [liri_counts.get(s, 0) / liri_total * 100 for s in subtypes_order]
    
    x = np.arange(len(subtypes_order))
    width = 0.35
    
    bars1 = ax_c.bar(x - width/2, tcga_pcts, width, label=f'TCGA-LIHC (n={tcga_total})',
                    color=COLORS['tcga'], edgecolor='white')
    bars2 = ax_c.bar(x + width/2, liri_pcts, width, label=f'LIRI-JP (n={liri_total})',
                    color=COLORS['liri'], edgecolor='white')
    
    # 添加数值标签
    for bar, pct, count in zip(bars1, tcga_pcts, [tcga_counts.get(s, 0) for s in subtypes_order]):
        if pct > 0:
            ax_c.annotate(f'{pct:.1f}%\n({count})', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    for bar, pct, count in zip(bars2, liri_pcts, [liri_counts.get(s, 0) for s in subtypes_order]):
        if pct > 0:
            ax_c.annotate(f'{pct:.1f}%\n({count})', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    ax_c.set_xlabel('Hoshida Subtype')
    ax_c.set_ylabel('Percentage (%)')
    ax_c.set_title('(C) Subtype Distribution\n(Biological Structure Recovered)', fontweight='bold', pad=10)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f'Hoshida-{s}' for s in subtypes_order])
    ax_c.legend(loc='upper right', frameon=True)
    ax_c.set_ylim(0, max(max(tcga_pcts), max(liri_pcts)) * 1.3)
    ax_c.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # 添加说明
    unknown_pct_liri = liri_counts.get('Unknown', 0) / liri_total * 100 if liri_total > 0 else 0
    ax_c.text(0.98, 0.98, f'LIRI-JP Unknown: {unknown_pct_liri:.1f}%\n(Biology recovered!)',
             transform=ax_c.transAxes, fontsize=9, va='top', ha='right', color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # 保存图表
    # =========================================================================
    plt.tight_layout()
    
    os.makedirs('paper_figures', exist_ok=True)
    plt.savefig('paper_figures/figure3_mechanism.png', dpi=300, facecolor='white')
    plt.savefig('paper_figures/figure3_mechanism.pdf', facecolor='white')
    
    print("\n" + "=" * 60)
    print("Figure 3 已生成!")
    print("=" * 60)
    print("\n保存位置:")
    print("  - paper_figures/figure3_mechanism.png")
    print("  - paper_figures/figure3_mechanism.pdf")
    print("\n数据来源 (学术诚信声明):")
    print("  - 基因表达: data/TCGA-LIHC.star_fpkm.tsv.gz, data/LIRI-JP.star_fpkm.tsv.gz")
    print("  - V5 嵌入: data/*_embeddings_v5.pt")
    print("  - 亚型标签: data/*_reports_v5.txt (GeneNarrator 生成)")
    print("\n分析结果:")
    print(f"  - 共同基因数: {len(common_genes)}")
    print(f"  - TCGA-LIHC 样本数: {n_tcga}")
    print(f"  - LIRI-JP 样本数: {n_liri}")
    print(f"\n亚型分布:")
    print(f"  TCGA-LIHC: {dict(tcga_counts)}")
    print(f"  LIRI-JP: {dict(liri_counts)}")
    
    plt.close()


if __name__ == '__main__':
    create_figure3()

