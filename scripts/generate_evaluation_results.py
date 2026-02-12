#!/usr/bin/env python
"""
生成可追溯的评估结果
=====================

这个脚本从保存的模型中生成评估结果，并保存到 CSV 文件。
所有图表脚本应该从这个 CSV 文件读取数据，而不是硬编码。

学术诚信保证：
- 结果直接来自模型评估
- 使用固定随机种子，100% 可复现
- 输出保存到 results/improved_gnaft_evaluation.csv

使用方法：
    python scripts/generate_evaluation_results.py
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BASE_DIR)
os.chdir(_BASE_DIR)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from unified_data import load_cancer_data, CANCER_PAIRS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 模型定义 (必须与保存时一致)
# =============================================================================

class ImprovedGNAFT(nn.Module):
    """改进版 GN-AFT - 必须与训练时的定义完全一致"""
    
    def __init__(self, gene_dim=1000, text_dim=1024, hidden_dim=256, dropout=0.35):
        super().__init__()
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(384, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        self.gene_quality = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.text_quality = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.g2t_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.t2g_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        
        self.ln_g = nn.LayerNorm(hidden_dim)
        self.ln_t = nn.LayerNorm(hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        self.aft_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        
        nn.init.constant_(self.aft_head[-1].bias[0], 6.9)
        nn.init.constant_(self.aft_head[-1].bias[1], 0.4)
    
    def forward(self, gene, text):
        g = self.gene_encoder(gene)
        t = self.text_encoder(text)
        
        q_g = self.gene_quality(g)
        q_t = self.text_quality(t)
        
        q_sum = q_g + q_t + 1e-8
        w_g = q_g / q_sum
        w_t = q_t / q_sum
        
        g_seq = g.unsqueeze(1)
        t_seq = t.unsqueeze(1)
        
        g2t, _ = self.g2t_attn(g_seq, t_seq, t_seq)
        t2g, _ = self.t2g_attn(t_seq, g_seq, g_seq)
        
        g_enhanced = self.ln_g((g_seq + g2t).squeeze(1))
        t_enhanced = self.ln_t((t_seq + t2g).squeeze(1))
        
        weighted = w_g * g_enhanced + w_t * t_enhanced
        
        concat = torch.cat([weighted, g_enhanced, t_enhanced], dim=-1)
        fused = self.fusion(concat) + weighted
        
        params = self.aft_head(fused)
        scale = torch.exp(torch.clamp(params[:, 0], 3.5, 8.5))
        shape = 0.5 + 3.0 * torch.sigmoid(params[:, 1])
        
        return scale, shape, (w_g.squeeze(), w_t.squeeze())
    
    def predict_median(self, gene, text):
        scale, shape, _ = self.forward(gene, text)
        ln2 = torch.log(torch.tensor(2.0, device=scale.device))
        return scale * (ln2 ** (1.0 / shape))


def evaluate_single_cancer(cancer_type):
    """评估单个癌种的模型"""
    
    model_path = f'models/improved_gnaft_{cancer_type.lower()}_sota.pt'
    
    if not os.path.exists(model_path):
        print(f"  ⚠ 模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    
    model = ImprovedGNAFT(
        gene_dim=config['gene_dim'],
        text_dim=config['text_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据
    train_data, test_data, info = load_cancer_data(cancer_type, n_genes=config['gene_dim'])
    
    # 评估外部验证集
    test_gene = torch.FloatTensor(test_data['gene']).to(DEVICE)
    test_text = torch.FloatTensor(test_data['text']).to(DEVICE)
    
    with torch.no_grad():
        pred = model.predict_median(test_gene, test_text)
        risk = 1.0 / (pred.cpu().numpy() + 1e-8)
    
    external_ci = concordance_index(test_data['time'], -risk, test_data['event'])
    
    # 评估内部验证集 (训练集)
    train_gene = torch.FloatTensor(train_data['gene']).to(DEVICE)
    train_text = torch.FloatTensor(train_data['text']).to(DEVICE)
    
    with torch.no_grad():
        pred_train = model.predict_median(train_gene, train_text)
        risk_train = 1.0 / (pred_train.cpu().numpy() + 1e-8)
    
    internal_ci = concordance_index(train_data['time'], -risk_train, train_data['event'])
    
    return {
        'cancer': cancer_type,
        'model': 'Improved-GN-AFT',
        'model_file': model_path,
        'seed': config['seed'],
        'train_dataset': info['train_name'],
        'test_dataset': info['test_name'],
        'train_samples': len(train_data['gene']),
        'test_samples': len(test_data['gene']),
        'internal_ci': internal_ci,
        'external_ci': external_ci,
        'expected_ci': checkpoint['performance']['external_ci'],
        'ci_match': abs(external_ci - checkpoint['performance']['external_ci']) < 0.001
    }


def main():
    """生成所有评估结果并保存"""
    
    print("=" * 70)
    print("生成可追溯的评估结果")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    for cancer in CANCER_PAIRS.keys():
        print(f"\n评估 {cancer}...")
        result = evaluate_single_cancer(cancer)
        if result:
            results.append(result)
            match_str = "✓" if result['ci_match'] else "✗"
            print(f"  External CI: {result['external_ci']:.4f} (期望: {result['expected_ci']:.4f}) {match_str}")
            print(f"  Internal CI: {result['internal_ci']:.4f}")
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果
    output_path = 'results/improved_gnaft_evaluation.csv'
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    print(f"\n结果已保存到: {output_path}")
    print("\n结果摘要:")
    print(df[['cancer', 'external_ci', 'internal_ci', 'seed', 'ci_match']].to_string(index=False))
    
    # 验证所有结果是否匹配
    all_match = all(r['ci_match'] for r in results)
    if all_match:
        print("\n✓ 所有结果与期望值匹配，数据可信！")
    else:
        print("\n⚠ 警告：部分结果与期望值不匹配，请检查！")
    
    return df


if __name__ == '__main__':
    main()

