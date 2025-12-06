#!/usr/bin/env python
"""
GN-AFT 模型评估脚本
==================

加载保存的模型并评估，结果100%可复现。

使用方法:
    python scripts/evaluate.py              # 评估所有癌症
    python scripts/evaluate.py --cancer LIHC  # 评估单个癌症
"""

import os
import sys
import json
import argparse

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from unified_data import load_cancer_data, CANCER_PAIRS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNAFTModel(nn.Module):
    """GN-AFT 模型"""
    
    def __init__(self, gene_dim, text_dim=1024, hidden_dim=128, dropout=0.2, gene_weight=0.5):
        super().__init__()
        self.gene_weight = gene_weight
        
        self.gene_enc = nn.Sequential(
            nn.Linear(gene_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.text_enc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        nn.init.constant_(self.head[-1].bias[0], 6.9)
        nn.init.constant_(self.head[-1].bias[1], 0.4)
    
    def forward(self, gene, text):
        g = self.gene_enc(gene)
        t = self.text_enc(text)
        h = self.gene_weight * g + (1 - self.gene_weight) * t
        p = self.head(h)
        scale = torch.exp(torch.clamp(p[:, 0], 3.5, 8.5))
        shape = 0.5 + 3.0 * torch.sigmoid(p[:, 1])
        return scale, shape
    
    def predict_median(self, gene, text):
        scale, shape = self.forward(gene, text)
        return scale * (torch.log(torch.tensor(2.0, device=scale.device)) ** (1.0 / shape))


def evaluate_model(model, test_data):
    """评估模型"""
    model.eval()
    
    test_g = torch.FloatTensor(test_data['gene']).to(DEVICE)
    test_t = torch.FloatTensor(test_data['text']).to(DEVICE)
    test_time = test_data['time']
    test_event = test_data['event']
    
    with torch.no_grad():
        pred = model.predict_median(test_g, test_t)
        risk = 1.0 / (pred.cpu().numpy() + 1e-8)
    
    try:
        ci = concordance_index(test_time, -risk, test_event)
    except:
        ci = 0.5
    
    return ci


def main():
    parser = argparse.ArgumentParser(description='GN-AFT 模型评估')
    parser.add_argument('--cancer', type=str, default=None, 
                        help='癌症类型 (LIHC, BRCA, OV, PAAD, PRAD)')
    args = parser.parse_args()
    
    print("="*60)
    print("GN-AFT 模型评估")
    print("="*60)
    print(f"设备: {DEVICE}")
    
    # 加载配置
    with open('configs/sota_config.json', 'r') as f:
        config = json.load(f)
    
    # 确定要评估的癌症
    cancers = [args.cancer.upper()] if args.cancer else list(CANCER_PAIRS.keys())
    
    results = []
    
    for cancer in cancers:
        print(f"\n{'='*60}")
        print(f"癌症: {cancer}")
        print("="*60)
        
        model_path = f'models/gnaft_{cancer.lower()}_sota.pt'
        
        if not os.path.exists(model_path):
            print(f"  ⚠️ 模型文件不存在: {model_path}")
            continue
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        cancer_config = checkpoint['config']
        gene_dim = checkpoint['gene_dim']
        expected_ci = checkpoint['external_ci']
        
        print(f"  Seed: {checkpoint['seed']}")
        print(f"  预期 C-Index: {expected_ci:.4f}")
        
        # 创建模型
        model = GNAFTModel(
            gene_dim=gene_dim,
            dropout=cancer_config['dropout'],
            gene_weight=cancer_config['gene_weight']
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载测试数据
        try:
            _, test_data, _ = load_cancer_data(cancer, n_genes=1000)
        except Exception as e:
            print(f"  数据加载失败: {e}")
            continue
        
        # 评估
        actual_ci = evaluate_model(model, test_data)
        
        print(f"\n  结果:")
        print(f"    External C-Index: {actual_ci:.4f}")
        
        diff = abs(actual_ci - expected_ci)
        match = diff < 0.0001
        
        if match:
            print(f"    ✅ 完全匹配!")
        else:
            print(f"    差异: {diff:.4f}")
        
        results.append({
            'Cancer': cancer,
            'Expected': expected_ci,
            'Actual': actual_ci,
            'Match': match
        })
    
    # 汇总
    print("\n" + "="*60)
    print("汇总")
    print("="*60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 保存结果
    df.to_csv('results/evaluation_results.csv', index=False)
    print(f"\n结果已保存: results/evaluation_results.csv")
    
    if all(df['Match']):
        print("\n✅ 所有结果完全可复现!")
    else:
        print("\n⚠️ 部分结果有差异")


if __name__ == '__main__':
    main()

