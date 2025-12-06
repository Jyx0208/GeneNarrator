"""
GN-AFT 数据加载模块
==================

用于加载预处理好的数据。
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 癌症配对
CANCER_PAIRS = {
    'LIHC': ('TCGA-LIHC', 'LIRI-JP'),
    'BRCA': ('TCGA-BRCA', 'GSE20685'),
    'OV': ('TCGA-OV', 'OV-AU'),
    'PAAD': ('TCGA-PAAD', 'PACA-CA'),
    'PRAD': ('TCGA-PRAD', 'PRAD-CA'),
}

# 数据集配置
DATASET_CONFIGS = {
    'TCGA-LIHC': {'expr': 'TCGA-LIHC.star_fpkm.tsv.gz', 'surv': 'TCGA-LIHC.survival.tsv.gz', 'embed': 'TCGA-LIHC_embeddings_v5.pt'},
    'LIRI-JP': {'expr': 'LIRI-JP.star_fpkm.tsv.gz', 'surv': 'LIRI-JP.survival.tsv.gz', 'embed': 'LIRI-JP_embeddings_v5.pt'},
    'TCGA-BRCA': {'expr': 'TCGA-BRCA_symbols.star_fpkm.tsv.gz', 'surv': 'TCGA-BRCA.survival.tsv.gz', 'embed': 'TCGA-BRCA_embeddings_v5.pt'},
    'GSE20685': {'expr': 'GSE20685.expression.tsv.gz', 'surv': 'GSE20685.survival.tsv.gz', 'embed': 'GSE20685_embeddings_v5.pt'},
    'TCGA-OV': {'expr': 'TCGA-OV.star_fpkm.tsv.gz', 'surv': 'TCGA-OV.survival.tsv.gz', 'embed': 'TCGA-OV_embeddings_v5.pt'},
    'OV-AU': {'expr': 'OV-AU.star_fpkm.tsv.gz', 'surv': 'OV-AU.survival.tsv.gz', 'embed': 'OV-AU_embeddings_v5.pt'},
    'TCGA-PAAD': {'expr': 'TCGA-PAAD.star_fpkm.tsv.gz', 'surv': 'TCGA-PAAD.survival.tsv.gz', 'embed': 'TCGA-PAAD_embeddings_v5.pt'},
    'PACA-CA': {'expr': 'PACA-CA.star_fpkm.tsv.gz', 'surv': 'PACA-CA.survival.tsv.gz', 'embed': 'PACA-CA_embeddings_v5.pt'},
    'TCGA-PRAD': {'expr': 'TCGA-PRAD.star_fpkm.tsv.gz', 'surv': 'TCGA-PRAD.survival.tsv.gz', 'embed': 'TCGA-PRAD_embeddings_v5.pt'},
    'PRAD-CA': {'expr': 'PRAD-CA.star_fpkm.tsv.gz', 'surv': 'PRAD-CA.survival.tsv.gz', 'embed': 'PRAD-CA_embeddings_v5.pt'},
}


def strip_ensembl_version(gene_id: str) -> str:
    """去除Ensembl版本号"""
    if isinstance(gene_id, str) and gene_id.startswith('ENSG') and '.' in gene_id:
        return gene_id.split('.')[0]
    return str(gene_id)


def load_expression(filepath: str) -> pd.DataFrame:
    """加载表达数据"""
    df = pd.read_csv(filepath, sep='\t', index_col=0, compression='gzip')
    if df.shape[0] < df.shape[1]:
        df = df.T
    return df.fillna(0)


def load_survival(filepath: str) -> pd.DataFrame:
    """加载生存数据"""
    return pd.read_csv(filepath, sep='\t', compression='gzip')


def load_embeddings(filepath: str) -> Tuple[torch.Tensor, list]:
    """加载嵌入数据"""
    data = torch.load(filepath, weights_only=False)
    if isinstance(data, dict):
        return data.get('embeddings', data.get('embedding')), data.get('patient_ids', [])
    return data, []


def load_gene_mapping() -> Dict[str, str]:
    """加载基因ID映射"""
    mapping_file = os.path.join(DATA_DIR, "gene_id_mapping.csv")
    if not os.path.exists(mapping_file):
        return {}
    mapping = {}
    df = pd.read_csv(mapping_file)
    for _, row in df.iterrows():
        ensembl_id = str(row.get('ensembl_id', ''))
        symbol = str(row.get('gene_symbol', ''))
        if ensembl_id and symbol and symbol != 'nan':
            mapping[ensembl_id] = symbol
    return mapping


def load_dataset(dataset_name: str, genes: list = None, norm_stats: dict = None, n_genes: int = 1000) -> Dict:
    """
    加载单个数据集
    
    Args:
        dataset_name: 数据集名称
        genes: 指定基因列表
        norm_stats: 标准化统计量
        n_genes: 高变异基因数量
    
    Returns:
        包含 gene, text, time, event 的字典
    """
    config = DATASET_CONFIGS[dataset_name]
    
    # 加载数据
    expr_df = load_expression(os.path.join(DATA_DIR, config['expr']))
    surv_df = load_survival(os.path.join(DATA_DIR, config['surv']))
    surv_df = surv_df.set_index('sample')
    embeddings, patient_ids = load_embeddings(os.path.join(DATA_DIR, config['embed']))
    
    # 确定基因
    if genes is None:
        var_series = expr_df.var(axis=1).sort_values(ascending=False)
        genes = list(var_series.head(n_genes).index)
    
    # 找共同样本
    expr_samples = set(expr_df.columns)
    surv_samples = set(surv_df.index)
    embed_samples = set(patient_ids) if patient_ids else expr_samples
    common_samples = list(expr_samples & surv_samples & embed_samples)
    
    sample_to_idx = {s: i for i, s in enumerate(patient_ids)} if patient_ids else {}
    
    # 构建数据
    expr_list, text_list, time_list, event_list = [], [], [], []
    
    for sample in common_samples:
        if patient_ids and sample in sample_to_idx:
            idx = sample_to_idx[sample]
        else:
            continue
        
        # 获取可用基因
        available_genes = [g for g in genes if g in expr_df.index]
        if len(available_genes) < len(genes) * 0.5:
            # 尝试去版本号匹配
            gene_map = {strip_ensembl_version(g): g for g in expr_df.index}
            available_genes = []
            for g in genes:
                stripped = strip_ensembl_version(g)
                if stripped in gene_map:
                    available_genes.append(gene_map[stripped])
        
        if len(available_genes) < 100:
            continue
        
        expr_vec = expr_df.loc[available_genes, sample].fillna(0).values
        expr_list.append(expr_vec)
        text_list.append(embeddings[idx].numpy() if hasattr(embeddings[idx], 'numpy') else embeddings[idx])
        time_list.append(max(1.0, surv_df.loc[sample, 'OS.time']))
        event_list.append(surv_df.loc[sample, 'OS'])
    
    # 转换为数组
    gene_data = np.array(expr_list, dtype=np.float32)
    text_data = np.array(text_list, dtype=np.float32)
    time_data = np.array(time_list, dtype=np.float32)
    event_data = np.array(event_list, dtype=np.float32)
    
    # 标准化
    gene_log = np.log1p(np.abs(gene_data))
    if norm_stats:
        gene_normalized = (gene_log - norm_stats['mean']) / norm_stats['std']
    else:
        mean = gene_log.mean(axis=0)
        std = gene_log.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        gene_normalized = (gene_log - mean) / std
        norm_stats = {'mean': mean, 'std': std}
    
    return {
        'gene': gene_normalized,
        'text': text_data,
        'time': time_data,
        'event': event_data,
        'genes': genes,
        'norm_stats': norm_stats
    }


def load_cancer_data(cancer_type: str, n_genes: int = 1000) -> Tuple[Dict, Dict]:
    """
    加载配对的训练和测试数据
    
    Args:
        cancer_type: 癌症类型 (LIHC, BRCA, OV, PAAD, PRAD)
        n_genes: 基因数量
    
    Returns:
        (train_data, test_data)
    """
    if cancer_type not in CANCER_PAIRS:
        raise ValueError(f"未知癌症类型: {cancer_type}")
    
    train_name, test_name = CANCER_PAIRS[cancer_type]
    
    print(f"加载 {cancer_type} 数据...")
    print(f"  训练集: {train_name}")
    print(f"  测试集: {test_name}")
    
    # 加载训练数据
    train_data = load_dataset(train_name, n_genes=n_genes)
    
    # 加载测试数据（使用训练集的基因和标准化参数）
    test_data = load_dataset(test_name, genes=train_data['genes'], norm_stats=train_data['norm_stats'])
    
    print(f"  训练样本: {len(train_data['gene'])}")
    print(f"  测试样本: {len(test_data['gene'])}")
    
    return train_data, test_data


if __name__ == '__main__':
    # 测试
    for cancer in CANCER_PAIRS.keys():
        try:
            train, test = load_cancer_data(cancer)
            print(f"✓ {cancer}: train={len(train['gene'])}, test={len(test['gene'])}")
        except Exception as e:
            print(f"✗ {cancer}: {e}")




