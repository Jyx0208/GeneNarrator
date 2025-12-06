#!/usr/bin/env python
"""
加载并评估改进版SOTA模型
========================

用于验证保存的模型是否能复现SOTA结果
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
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

from unified_data import load_cancer_data, set_random_seed, CANCER_PAIRS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 模型定义 (必须与保存时一致)
# =============================================================================

class ImprovedGNAFT(nn.Module):
    """改进版 GN-AFT"""
    
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


def load_and_evaluate(cancer_type, model_dir='models'):
    """加载并评估单个癌种的模型"""
    
    # 尝试加载改进版模型
    model_path = os.path.join(model_dir, f'improved_gnaft_{cancer_type.lower()}_sota.pt')
    
    if not os.path.exists(model_path):
        print(f"  ⚠ 模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    config = checkpoint['config']
    
    # 创建模型
    model = ImprovedGNAFT(
        gene_dim=config['gene_dim'],
        text_dim=config['text_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    train_data, test_data, info = load_cancer_data(cancer_type, n_genes=config['gene_dim'])
    
    # 评估
    test_gene = torch.FloatTensor(test_data['gene']).to(DEVICE)
    test_text = torch.FloatTensor(test_data['text']).to(DEVICE)
    
    with torch.no_grad():
        pred = model.predict_median(test_gene, test_text)
        risk = 1.0 / (pred.cpu().numpy() + 1e-8)
    
    ci = concordance_index(test_data['time'], -risk, test_data['event'])
    
    return {
        'ci': ci,
        'expected': checkpoint['performance']['external_ci'],
        'old_sota': checkpoint['performance']['old_sota'],
        'seed': config['seed']
    }


def main():
    print("="*70)
    print("加载并评估改进版 SOTA 模型")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    results = {}
    
    for cancer in CANCER_PAIRS.keys():
        print(f"\n{cancer}:")
        
        result = load_and_evaluate(cancer)
        
        if result:
            results[cancer] = result
            match = "✓" if abs(result['ci'] - result['expected']) < 0.01 else "≈"
            improvement = (result['ci'] - result['old_sota']) * 100
            
            print(f"  Seed: {result['seed']}")
            print(f"  实际 CI: {result['ci']:.4f}")
            print(f"  期望 CI: {result['expected']:.4f} {match}")
            print(f"  旧 SOTA: {result['old_sota']:.4f}")
            print(f"  提升: {improvement:+.2f}%")
    
    # 汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)
    print(f"{'Cancer':<10} {'实际CI':<12} {'期望CI':<12} {'旧SOTA':<12} {'提升':<10}")
    print("-"*55)
    
    for cancer, r in results.items():
        improvement = (r['ci'] - r['old_sota']) * 100
        print(f"{cancer:<10} {r['ci']:.4f}       {r['expected']:.4f}       {r['old_sota']:.4f}       {improvement:+.2f}%")
    
    if results:
        avg_improvement = np.mean([(r['ci'] - r['old_sota']) * 100 for r in results.values()])
        print("-"*55)
        print(f"平均提升: {avg_improvement:+.2f}%")


if __name__ == '__main__':
    main()

