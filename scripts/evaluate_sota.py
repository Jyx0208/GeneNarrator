#!/usr/bin/env python
"""
GN-AFT SOTA 模型评估脚本
========================

用法:
    python scripts/evaluate_sota.py              # 评估所有5个癌种
    python scripts/evaluate_sota.py --cancer LIHC   # 评估单个癌种
    python scripts/evaluate_sota.py --verbose       # 详细输出
"""

import os
import sys
import argparse

# 设置路径
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

# =============================================================================
# 配置
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SOTA配置
SOTA_CONFIG = {
    'LIHC': {'seed': 6, 'expected_ci': 0.8038},
    'BRCA': {'seed': 7, 'expected_ci': 0.7015},
    'OV': {'seed': 17, 'expected_ci': 0.6421},
    'PAAD': {'seed': 4, 'expected_ci': 0.6398},
    'PRAD': {'seed': 3, 'expected_ci': 0.8646},
}

# =============================================================================
# 模型定义
# =============================================================================

class ImprovedGNAFT(nn.Module):
    """GN-AFT模型 - 多模态生存预测"""
    
    def __init__(self, gene_dim=1000, text_dim=1024, hidden_dim=256, dropout=0.35):
        super().__init__()
        
        # 基因编码器
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
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # 质量估计器
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
        
        # 交叉注意力
        self.g2t_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.t2g_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        
        self.ln_g = nn.LayerNorm(hidden_dim)
        self.ln_t = nn.LayerNorm(hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # 输出头
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
        """预测中位生存时间"""
        scale, shape = self.forward(gene, text)
        ln2 = torch.log(torch.tensor(2.0, device=scale.device))
        return scale * (ln2 ** (1.0 / shape))


# =============================================================================
# 数据加载
# =============================================================================

def load_cancer_data_simple(cancer_type, data_dir='data'):
    """简化的数据加载函数"""
    try:
        from unified_data import load_cancer_data
        return load_cancer_data(cancer_type, data_dir=data_dir, n_genes=1000)
    except ImportError:
        raise ImportError("请确保 unified_data.py 在当前目录")


# =============================================================================
# 评估函数
# =============================================================================

def load_model(cancer_type, model_dir='models'):
    """加载预训练模型"""
    model_path = os.path.join(model_dir, f'improved_gnaft_{cancer_type.lower()}_sota.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    config = checkpoint['config']
    model = ImprovedGNAFT(
        gene_dim=config['gene_dim'],
        text_dim=config['text_dim'],
        hidden_dim=config.get('hidden_dim', 256),
        dropout=config.get('dropout', 0.35)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_cancer(cancer_type, model_dir='models', data_dir='data', verbose=False):
    """评估单个癌种"""
    if verbose:
        print(f"\n评估 {cancer_type}...")
    
    # 加载模型
    model, checkpoint = load_model(cancer_type, model_dir)
    
    if verbose:
        print(f"  模型加载成功: {checkpoint['config']}")
    
    # 加载数据
    train_data, test_data, info = load_cancer_data_simple(cancer_type, data_dir)
    
    if verbose:
        print(f"  测试集样本数: {len(test_data['gene'])}")
    
    # 评估
    test_gene = torch.FloatTensor(test_data['gene']).to(DEVICE)
    test_text = torch.FloatTensor(test_data['text']).to(DEVICE)
    
    with torch.no_grad():
        pred = model.predict_median(test_gene, test_text)
        risk = 1.0 / (pred.cpu().numpy() + 1e-8)
    
    ci = concordance_index(test_data['time'], -risk, test_data['event'])
    
    expected = SOTA_CONFIG[cancer_type]['expected_ci']
    match = abs(ci - expected) < 0.01
    
    return {
        'cancer': cancer_type,
        'c_index': ci,
        'expected': expected,
        'match': match,
        'n_samples': len(test_data['gene'])
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate GN-AFT SOTA models')
    parser.add_argument('--cancer', type=str, default=None, 
                        help='Single cancer to evaluate (e.g., LIHC)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing model files')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GN-AFT SOTA Evaluation")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # 确定要评估的癌种
    if args.cancer:
        cancers = [args.cancer.upper()]
    else:
        cancers = list(SOTA_CONFIG.keys())
    
    results = []
    
    for cancer in cancers:
        try:
            result = evaluate_cancer(
                cancer, 
                model_dir=args.model_dir,
                data_dir=args.data_dir,
                verbose=args.verbose
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ {cancer} 评估失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"{'Cancer':<10} {'C-Index':<12} {'Expected':<12} {'Match':<8}")
    print("-"*45)
    
    all_match = True
    for r in results:
        match_str = "✓" if r['match'] else "✗"
        print(f"{r['cancer']:<10} {r['c_index']:.4f}       {r['expected']:.4f}       {match_str}")
        if not r['match']:
            all_match = False
    
    print("-"*45)
    
    if all_match and len(results) == len(cancers):
        print("\n✅ 所有结果验证通过!")
    elif len(results) < len(cancers):
        print(f"\n⚠️ 部分癌种评估失败 ({len(results)}/{len(cancers)})")
    else:
        print("\n⚠️ 部分结果与预期不符 (可能由于CUDA非确定性)")
    
    # 统计
    if results:
        avg_ci = np.mean([r['c_index'] for r in results])
        print(f"\n平均 C-Index: {avg_ci:.4f}")


if __name__ == '__main__':
    main()



