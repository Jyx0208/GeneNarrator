#!/usr/bin/env python
"""
GN-AFT 从头训练脚本
==================

用法:
    python scripts/train_from_scratch.py              # 训练所有5个癌种
    python scripts/train_from_scratch.py --cancer LIHC   # 训练单个癌种
    python scripts/train_from_scratch.py --cancer LIHC --seed 6  # 指定seed
"""

import os
import sys
import argparse
import json
from datetime import datetime

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
print(f"Device: {DEVICE}")

# =============================================================================
# SOTA配置
# =============================================================================

SOTA_SEEDS = {
    'LIHC': 6,
    'BRCA': 7,
    'OV': 17,
    'PAAD': 4,
    'PRAD': 3,
}

EXPECTED_CI = {
    'LIHC': 0.8038,
    'BRCA': 0.7015,
    'OV': 0.6421,
    'PAAD': 0.6398,
    'PRAD': 0.8646,
}

# =============================================================================
# 模型定义
# =============================================================================

class ImprovedGNAFT(nn.Module):
    """GN-AFT模型"""
    
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
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.text_quality = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid()
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
        
        return scale, shape
    
    def predict_median(self, gene, text):
        scale, shape = self.forward(gene, text)
        ln2 = torch.log(torch.tensor(2.0, device=scale.device))
        return scale * (ln2 ** (1.0 / shape))


# =============================================================================
# 损失函数
# =============================================================================

def weibull_loss(scale, shape, time, event, reg=0.01):
    eps = 1e-8
    scale = torch.clamp(scale, min=1.0)
    shape = torch.clamp(shape, min=0.1)
    time = torch.clamp(time, min=1.0)
    
    z = (time / scale) ** shape
    log_f = torch.log(shape + eps) - torch.log(scale + eps) + \
            (shape - 1) * (torch.log(time + eps) - torch.log(scale + eps)) - z
    log_S = -z
    
    # 标签平滑
    smooth_event = event * 0.95 + 0.025
    nll = -torch.mean(smooth_event * log_f + (1 - smooth_event) * log_S)
    
    reg_scale = reg * torch.mean((torch.log(scale) - 6.5)**2)
    reg_shape = reg * torch.mean((shape - 1.5)**2)
    
    return nll + reg_scale + reg_shape


# =============================================================================
# 训练函数
# =============================================================================

def train_model(model, train_data, val_data, config):
    """训练模型"""
    epochs = config.get('epochs', 200)
    lr = config.get('lr', 5e-5)
    batch_size = config.get('batch_size', 64)
    patience = config.get('patience', 30)
    
    train_gene = torch.FloatTensor(train_data['gene'])
    train_text = torch.FloatTensor(train_data['text'])
    train_time = torch.FloatTensor(train_data['time'])
    train_event = torch.FloatTensor(train_data['event'])
    
    val_gene = torch.FloatTensor(val_data['gene']).to(DEVICE)
    val_text = torch.FloatTensor(val_data['text']).to(DEVICE)
    val_time = val_data['time']
    val_event = val_data['event']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)
    
    best_ci = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_gene))
        
        for i in range(0, len(train_gene), batch_size):
            idx = perm[i:i+batch_size]
            batch_gene = train_gene[idx].to(DEVICE)
            batch_text = train_text[idx].to(DEVICE)
            batch_time = train_time[idx].to(DEVICE)
            batch_event = train_event[idx].to(DEVICE)
            
            optimizer.zero_grad()
            scale, shape = model(batch_gene, batch_text)
            loss = weibull_loss(scale, shape, batch_time, batch_event)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        scheduler.step()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred = model.predict_median(val_gene, val_text)
                risk = 1.0 / (pred.cpu().numpy() + 1e-8)
            
            try:
                ci = concordance_index(val_time, -risk, val_event)
            except:
                ci = 0.5
            
            if ci > best_ci:
                best_ci = ci
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: Val CI = {ci:.4f}, Best = {best_ci:.4f}")
            
            if patience_counter >= patience // 5:
                print(f"    Early stopping at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_ci


def evaluate_model(model, test_data):
    """评估模型"""
    model.eval()
    
    test_gene = torch.FloatTensor(test_data['gene']).to(DEVICE)
    test_text = torch.FloatTensor(test_data['text']).to(DEVICE)
    
    with torch.no_grad():
        pred = model.predict_median(test_gene, test_text)
        risk = 1.0 / (pred.cpu().numpy() + 1e-8)
    
    return concordance_index(test_data['time'], -risk, test_data['event'])


# =============================================================================
# 训练单个癌种
# =============================================================================

def train_cancer(cancer_type, seed=None, config=None, save_model=True):
    """训练单个癌种"""
    if seed is None:
        seed = SOTA_SEEDS.get(cancer_type, 42)
    
    if config is None:
        config = {
            'epochs': 200,
            'lr': 5e-5,
            'batch_size': 64,
            'patience': 30,
            'hidden_dim': 256,
            'dropout': 0.35,
        }
    
    print(f"\n{'='*60}")
    print(f"训练 {cancer_type} (seed={seed})")
    print("="*60)
    
    set_random_seed(seed)
    
    # 加载数据
    train_data, test_data, info = load_cancer_data(cancer_type, n_genes=1000)
    
    gene_dim = train_data['gene'].shape[1]
    text_dim = train_data['text'].shape[1]
    
    print(f"  训练集: {len(train_data['gene'])} 样本")
    print(f"  测试集: {len(test_data['gene'])} 样本")
    
    # 创建模型
    model = ImprovedGNAFT(
        gene_dim=gene_dim,
        text_dim=text_dim,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    # 训练
    print("  训练中...")
    train_model(model, train_data, test_data, config)
    
    # 评估
    ci = evaluate_model(model, test_data)
    expected = EXPECTED_CI.get(cancer_type, 0)
    
    print(f"\n  结果:")
    print(f"    External C-Index: {ci:.4f}")
    print(f"    期望值: {expected:.4f}")
    print(f"    差异: {(ci - expected) * 100:+.2f}%")
    
    # 保存模型
    if save_model:
        os.makedirs('models', exist_ok=True)
        save_path = f'models/improved_gnaft_{cancer_type.lower()}_sota.pt'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'gene_dim': gene_dim,
                'text_dim': text_dim,
                'hidden_dim': config['hidden_dim'],
                'dropout': config['dropout'],
                'seed': seed,
                'cancer_type': cancer_type,
            },
            'performance': {
                'external_ci': ci,
                'expected_ci': expected,
            },
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, save_path)
        
        print(f"  模型已保存: {save_path}")
    
    return ci, model


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train GN-AFT from scratch')
    parser.add_argument('--cancer', type=str, default=None,
                        help='Cancer type to train (e.g., LIHC). If not specified, train all.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed. If not specified, use SOTA seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save model')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': 64,
        'patience': 30,
        'hidden_dim': 256,
        'dropout': 0.35,
    }
    
    print("="*60)
    print("GN-AFT Training from Scratch")
    print("="*60)
    print(f"Config: {config}")
    
    if args.cancer:
        # 训练单个癌种
        train_cancer(
            args.cancer.upper(),
            seed=args.seed,
            config=config,
            save_model=not args.no_save
        )
    else:
        # 训练所有癌种
        results = {}
        
        for cancer in SOTA_SEEDS.keys():
            ci, _ = train_cancer(
                cancer,
                seed=SOTA_SEEDS[cancer],
                config=config,
                save_model=not args.no_save
            )
            results[cancer] = ci
        
        # 汇总
        print("\n" + "="*60)
        print("训练结果汇总")
        print("="*60)
        print(f"{'Cancer':<10} {'C-Index':<12} {'Expected':<12} {'Diff':<10}")
        print("-"*45)
        
        for cancer, ci in results.items():
            expected = EXPECTED_CI[cancer]
            diff = (ci - expected) * 100
            print(f"{cancer:<10} {ci:.4f}       {expected:.4f}       {diff:+.2f}%")
        
        avg_ci = np.mean(list(results.values()))
        avg_expected = np.mean(list(EXPECTED_CI.values()))
        print("-"*45)
        print(f"{'Average':<10} {avg_ci:.4f}       {avg_expected:.4f}")


if __name__ == '__main__':
    main()



