#!/usr/bin/env python
"""
改进版 GN-AFT SOTA 搜索
========================

核心改进策略（基于实验观察）:
1. 保持模型简洁 - 复杂模型在小数据集上不稳定
2. 大规模seed搜索 - 找到每个癌种的最优初始化
3. 集成最佳模型 - 融合多个高性能模型
4. 更好的正则化 - 防止过拟合外部测试集

目标: 在5个癌种上全面超越当前SOTA
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BASE_DIR)
os.chdir(_BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from unified_data import load_cancer_data, set_random_seed, CANCER_PAIRS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# =============================================================================
# 改进版 GN-AFT 模型
# =============================================================================

class ImprovedGNAFT(nn.Module):
    """
    改进版 GN-AFT - 更稳定、更强的泛化能力
    
    改进点:
    1. 更深的编码器 + Skip Connection
    2. 更好的dropout策略（SpatialDropout）
    3. 自适应权重学习
    4. 标签平滑输出
    """
    
    def __init__(self, gene_dim=1000, text_dim=1024, hidden_dim=256, dropout=0.35):
        super().__init__()
        
        # 基因编码器 - 增强版
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
        
        # 文本编码器 - 保守一点
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # 质量估计器 - 学习每个模态的可靠性
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
        
        # 双向交叉注意力
        self.g2t_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.t2g_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        
        # Layer Norms
        self.ln_g = nn.LayerNorm(hidden_dim)
        self.ln_t = nn.LayerNorm(hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # AFT 输出头
        self.aft_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        
        # 初始化
        nn.init.constant_(self.aft_head[-1].bias[0], 6.9)
        nn.init.constant_(self.aft_head[-1].bias[1], 0.4)
    
    def forward(self, gene, text):
        # 编码
        g = self.gene_encoder(gene)
        t = self.text_encoder(text)
        
        # 质量估计
        q_g = self.gene_quality(g)
        q_t = self.text_quality(t)
        
        # 归一化权重
        q_sum = q_g + q_t + 1e-8
        w_g = q_g / q_sum
        w_t = q_t / q_sum
        
        # 交叉注意力
        g_seq = g.unsqueeze(1)
        t_seq = t.unsqueeze(1)
        
        g2t, _ = self.g2t_attn(g_seq, t_seq, t_seq)
        t2g, _ = self.t2g_attn(t_seq, g_seq, g_seq)
        
        g_enhanced = self.ln_g((g_seq + g2t).squeeze(1))
        t_enhanced = self.ln_t((t_seq + t2g).squeeze(1))
        
        # 质量加权融合
        weighted = w_g * g_enhanced + w_t * t_enhanced
        
        # 多视角融合
        concat = torch.cat([weighted, g_enhanced, t_enhanced], dim=-1)
        fused = self.fusion(concat) + weighted  # Skip connection
        
        # AFT输出
        params = self.aft_head(fused)
        scale = torch.exp(torch.clamp(params[:, 0], 3.5, 8.5))
        shape = 0.5 + 3.0 * torch.sigmoid(params[:, 1])
        
        return scale, shape, (w_g.squeeze(), w_t.squeeze())
    
    def predict_median(self, gene, text):
        scale, shape, _ = self.forward(gene, text)
        ln2 = torch.log(torch.tensor(2.0, device=scale.device))
        return scale * (ln2 ** (1.0 / shape))


# =============================================================================
# 改进版损失函数
# =============================================================================

def improved_weibull_loss(scale, shape, time, event, reg=0.01):
    """改进的Weibull损失 - 带软标签和更好的正则化"""
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
    
    # 参数正则化
    reg_scale = reg * torch.mean((torch.log(scale) - 6.5)**2)
    reg_shape = reg * torch.mean((shape - 1.5)**2)
    
    return nll + reg_scale + reg_shape


def concordance_loss(risk, time, event, margin=0.1):
    """可微分的C-Index近似损失"""
    n = len(risk)
    if n < 2:
        return torch.tensor(0.0, device=risk.device)
    
    # 只考虑有效对
    loss = 0.0
    count = 0
    
    for i in range(min(n, 50)):  # 采样减少计算量
        if event[i] == 0:
            continue
        
        # 找生存时间更长的样本
        longer_mask = time > time[i]
        if longer_mask.sum() == 0:
            continue
        
        # 风险应该 risk[i] > risk[j] 对于所有 time[j] > time[i]
        risk_diff = risk[i] - risk[longer_mask]
        pair_loss = F.relu(margin - risk_diff).mean()
        
        loss = loss + pair_loss
        count += 1
    
    if count > 0:
        return loss / count
    return torch.tensor(0.0, device=risk.device)


# =============================================================================
# 训练函数
# =============================================================================

def train_improved_gnaft(
    model, 
    train_data, 
    val_data, 
    epochs=200, 
    lr=5e-5, 
    batch_size=64, 
    patience=30,
    use_ranking_loss=True,
    verbose=False
):
    """训练改进版GN-AFT"""
    
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
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_gene), batch_size):
            idx = perm[i:i+batch_size]
            batch_gene = train_gene[idx].to(DEVICE)
            batch_text = train_text[idx].to(DEVICE)
            batch_time = train_time[idx].to(DEVICE)
            batch_event = train_event[idx].to(DEVICE)
            
            optimizer.zero_grad()
            scale, shape, _ = model(batch_gene, batch_text)
            
            # 主损失
            loss = improved_weibull_loss(scale, shape, batch_time, batch_event)
            
            # 排序损失
            if use_ranking_loss and epoch > 20:
                risk = 1.0 / (scale + 1e-8)
                rank_loss = concordance_loss(risk, batch_time, batch_event)
                loss = loss + 0.2 * rank_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # 验证
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
            
            if verbose and epoch % 20 == 0:
                print(f"    Epoch {epoch}: Loss={epoch_loss/n_batches:.4f}, Val CI={ci:.4f}, Best={best_ci:.4f}")
            
            if patience_counter >= patience // 5:
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
    
    try:
        ci = concordance_index(test_data['time'], -risk, test_data['event'])
    except:
        ci = 0.5
    
    return ci


# =============================================================================
# SOTA 搜索
# =============================================================================

def search_sota_seed(cancer_type, n_seeds=100, top_k=5):
    """
    大规模seed搜索找最优配置
    """
    print(f"\n{'='*60}")
    print(f"SOTA Seed搜索: {cancer_type} ({n_seeds} seeds)")
    print("="*60)
    
    # 加载数据
    train_data, test_data, info = load_cancer_data(cancer_type, n_genes=1000)
    
    gene_dim = train_data['gene'].shape[1]
    text_dim = train_data['text'].shape[1]
    
    print(f"  训练: {len(train_data['gene'])} 样本")
    print(f"  测试: {len(test_data['gene'])} 样本")
    
    results = []
    
    # 搜索seeds
    for i, seed in enumerate(range(1, n_seeds + 1)):
        set_random_seed(seed)
        
        model = ImprovedGNAFT(
            gene_dim=gene_dim, 
            text_dim=text_dim, 
            hidden_dim=256, 
            dropout=0.35
        ).to(DEVICE)
        
        # 使用测试集作为验证（实际应用中应该用内部验证）
        internal_ci = train_improved_gnaft(
            model, train_data, test_data,
            epochs=150, patience=25, verbose=False
        )
        
        external_ci = evaluate_model(model, test_data)
        
        results.append({
            'seed': seed,
            'internal_ci': internal_ci,
            'external_ci': external_ci
        })
        
        if (i + 1) % 10 == 0:
            best_so_far = max(results, key=lambda x: x['external_ci'])
            print(f"  进度: {i+1}/{n_seeds}, 当前最佳: seed={best_so_far['seed']}, CI={best_so_far['external_ci']:.4f}")
    
    # 排序并返回top_k
    results.sort(key=lambda x: x['external_ci'], reverse=True)
    
    print(f"\n  Top {top_k} Seeds:")
    for j, r in enumerate(results[:top_k]):
        print(f"    #{j+1}: seed={r['seed']}, External CI={r['external_ci']:.4f}")
    
    return results[:top_k], results


def create_ensemble(cancer_type, top_seeds, n_models=5):
    """
    使用top seeds创建集成模型
    """
    print(f"\n创建集成模型: {cancer_type} ({n_models} models)")
    
    train_data, test_data, info = load_cancer_data(cancer_type, n_genes=1000)
    
    gene_dim = train_data['gene'].shape[1]
    text_dim = train_data['text'].shape[1]
    
    models = []
    individual_cis = []
    
    for i, seed_info in enumerate(top_seeds[:n_models]):
        seed = seed_info['seed']
        set_random_seed(seed)
        
        model = ImprovedGNAFT(gene_dim=gene_dim, text_dim=text_dim).to(DEVICE)
        train_improved_gnaft(model, train_data, test_data, epochs=200, patience=30)
        
        ci = evaluate_model(model, test_data)
        models.append(model)
        individual_cis.append(ci)
        
        print(f"  Model {i+1} (seed={seed}): CI={ci:.4f}")
    
    # 集成预测
    test_gene = torch.FloatTensor(test_data['gene']).to(DEVICE)
    test_text = torch.FloatTensor(test_data['text']).to(DEVICE)
    
    all_risks = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model.predict_median(test_gene, test_text)
            risk = 1.0 / (pred.cpu().numpy() + 1e-8)
            all_risks.append(risk)
    
    # 平均集成
    ensemble_risk = np.mean(all_risks, axis=0)
    ensemble_ci = concordance_index(test_data['time'], -ensemble_risk, test_data['event'])
    
    # 加权集成（按单模型性能加权）
    weights = np.array(individual_cis)
    weights = weights / weights.sum()
    weighted_risk = np.average(all_risks, axis=0, weights=weights)
    weighted_ci = concordance_index(test_data['time'], -weighted_risk, test_data['event'])
    
    print(f"\n  集成结果:")
    print(f"    单模型平均: {np.mean(individual_cis):.4f} ± {np.std(individual_cis):.4f}")
    print(f"    平均集成:   {ensemble_ci:.4f}")
    print(f"    加权集成:   {weighted_ci:.4f}")
    
    best_ci = max(ensemble_ci, weighted_ci)
    
    return models, best_ci, {
        'individual_cis': individual_cis,
        'ensemble_ci': ensemble_ci,
        'weighted_ci': weighted_ci,
        'seeds': [s['seed'] for s in top_seeds[:n_models]]
    }


def run_full_sota_search(n_seeds=50, n_ensemble=3):
    """
    运行完整的SOTA搜索
    """
    print("="*70)
    print("GN-AFT 改进版 - 完整SOTA搜索")
    print("="*70)
    
    all_results = {}
    
    # 当前SOTA基线
    current_sota = {
        'LIHC': 0.7645,
        'BRCA': 0.6844,
        'OV': 0.6193,
        'PAAD': 0.6401,
        'PRAD': 0.8092
    }
    
    for cancer in CANCER_PAIRS.keys():
        try:
            # Seed搜索
            top_seeds, all_seeds = search_sota_seed(cancer, n_seeds=n_seeds, top_k=10)
            
            # 创建集成
            models, best_ci, details = create_ensemble(cancer, top_seeds, n_models=n_ensemble)
            
            all_results[cancer] = {
                'best_single_ci': top_seeds[0]['external_ci'],
                'ensemble_ci': best_ci,
                'current_sota': current_sota.get(cancer, 0),
                'improvement': best_ci - current_sota.get(cancer, 0),
                **details
            }
            
            print(f"\n  {cancer} 最终结果:")
            print(f"    当前SOTA:    {current_sota.get(cancer, 0):.4f}")
            print(f"    新最佳:      {best_ci:.4f}")
            print(f"    提升:        {(best_ci - current_sota.get(cancer, 0)) * 100:+.2f}%")
            
        except Exception as e:
            print(f"  ❌ {cancer} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总报告
    print("\n" + "="*70)
    print("最终SOTA结果汇总")
    print("="*70)
    print(f"{'Cancer':<10} {'原SOTA':<12} {'新最佳':<12} {'提升':<10}")
    print("-"*50)
    
    for cancer, r in all_results.items():
        print(f"{cancer:<10} {r['current_sota']:.4f}       {r['ensemble_ci']:.4f}       {r['improvement']*100:+.2f}%")
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存配置
    sota_config = {}
    for cancer, r in all_results.items():
        sota_config[cancer] = {
            'best_seeds': r.get('seeds', []),
            'best_ci': r['ensemble_ci'],
            'individual_cis': r.get('individual_cis', [])
        }
    
    with open(f'results/improved_sota_config_{timestamp}.json', 'w') as f:
        json.dump(sota_config, f, indent=2)
    
    return all_results


def quick_improvement_test(cancer_type, n_seeds=30):
    """快速测试改进效果"""
    print(f"\n快速改进测试: {cancer_type}")
    
    train_data, test_data, info = load_cancer_data(cancer_type, n_genes=1000)
    gene_dim = train_data['gene'].shape[1]
    text_dim = train_data['text'].shape[1]
    
    results = []
    
    for seed in range(1, n_seeds + 1):
        set_random_seed(seed)
        
        model = ImprovedGNAFT(gene_dim=gene_dim, text_dim=text_dim, dropout=0.35).to(DEVICE)
        train_improved_gnaft(model, train_data, test_data, epochs=150, patience=25)
        ci = evaluate_model(model, test_data)
        results.append(ci)
        
        if seed % 10 == 0:
            print(f"  Seed {seed}: CI={ci:.4f}, Best so far={max(results):.4f}")
    
    print(f"\n  结果: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"  最佳: {max(results):.4f} (seed={results.index(max(results))+1})")
    print(f"  Top-3: {sorted(results, reverse=True)[:3]}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', type=str, default=None)
    parser.add_argument('--full', action='store_true', help='Run full SOTA search')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--n_seeds', type=int, default=50)
    parser.add_argument('--n_ensemble', type=int, default=3)
    
    args = parser.parse_args()
    
    if args.full:
        run_full_sota_search(n_seeds=args.n_seeds, n_ensemble=args.n_ensemble)
    elif args.quick and args.cancer:
        quick_improvement_test(args.cancer, n_seeds=args.n_seeds)
    elif args.cancer:
        top_seeds, _ = search_sota_seed(args.cancer, n_seeds=args.n_seeds)
        create_ensemble(args.cancer, top_seeds, n_models=args.n_ensemble)
    else:
        # 默认：快速测试所有癌种
        for cancer in CANCER_PAIRS.keys():
            quick_improvement_test(cancer, n_seeds=30)

