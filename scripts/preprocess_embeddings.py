import os
import torch
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from tqdm import tqdm
import time
import gseapy as gp
from io import StringIO

# 配置 API
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
CHAT_MODEL = "qwen3-max"          # 用于生成报告 (Reasoning)
EMBEDDING_MODEL = "text-embedding-v4" # 用于编码报告 (Embedding)

# MSigDB Hallmark Gene Sets - 50个癌症核心通路
HALLMARK_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/h.all.v7.5.1.symbols.gmt"

# 基因 ID 映射缓存
_GENE_MAPPING_CACHE = {}
_HALLMARK_GENE_SETS = None


# =============================================================================
# 基因 ID 映射功能 (Ensembl ID -> Gene Symbol)
# =============================================================================

def load_or_create_gene_mapping(data_dir="./data"):
    """
    加载或创建 Ensembl ID -> Gene Symbol 映射表
    优先从本地缓存加载，否则从 MyGene.info API 查询
    """
    global _GENE_MAPPING_CACHE
    
    if _GENE_MAPPING_CACHE:
        return _GENE_MAPPING_CACHE
    
    mapping_file = os.path.join(data_dir, "gene_id_mapping.csv")
    
    if os.path.exists(mapping_file):
        print(f"Loading gene mapping from {mapping_file}...")
        df = pd.read_csv(mapping_file)
        for _, row in df.iterrows():
            ensembl_id = str(row.get('ensembl_id', ''))
            symbol = str(row.get('gene_symbol', ''))
            if ensembl_id and symbol and symbol != 'nan':
                _GENE_MAPPING_CACHE[ensembl_id] = symbol
        print(f"Loaded {len(_GENE_MAPPING_CACHE)} gene mappings.")
    
    return _GENE_MAPPING_CACHE


def query_gene_symbols_batch(ensembl_ids, batch_size=100):
    """使用 MyGene.info API 批量查询 Ensembl ID 对应的 Gene Symbol"""
    clean_ids = [eid.split('.')[0] for eid in ensembl_ids]
    unique_clean_ids = list(set(clean_ids))
    
    url = "https://mygene.info/v3/query"
    id_to_symbol = {}
    
    print(f"Querying {len(unique_clean_ids)} unique gene IDs from MyGene.info...")
    
    for i in tqdm(range(0, len(unique_clean_ids), batch_size), desc="Fetching gene symbols"):
        batch = unique_clean_ids[i:i+batch_size]
        params = {
            'q': ','.join(batch),
            'scopes': 'ensembl.gene',
            'fields': 'symbol',
            'species': 'human'
        }
        try:
            response = requests.post(url, data=params, timeout=30)
            if response.status_code == 200:
                results = response.json()
                for result in results:
                    if isinstance(result, dict) and 'symbol' in result:
                        id_to_symbol[result.get('query', '')] = result['symbol']
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Error querying batch: {e}")
            continue
    
    result = {}
    for orig_id in ensembl_ids:
        clean_id = orig_id.split('.')[0]
        if clean_id in id_to_symbol:
            result[orig_id] = id_to_symbol[clean_id]
    
    print(f"Successfully mapped {len(result)}/{len(ensembl_ids)} gene IDs to symbols.")
    return result


def get_gene_symbol(ensembl_id, mapping_cache):
    """将单个 Ensembl ID 转换为 Gene Symbol"""
    clean_id = ensembl_id.split('.')[0]
    if ensembl_id in mapping_cache:
        return mapping_cache[ensembl_id]
    if clean_id in mapping_cache:
        return mapping_cache[clean_id]
    return clean_id


def convert_expr_to_symbols(expr_df, gene_mapping):
    """
    将表达矩阵的列名从 Ensembl ID 转换为 Gene Symbol
    返回: 以 Gene Symbol 为列名的 DataFrame
    """
    new_columns = []
    for col in expr_df.columns:
        symbol = get_gene_symbol(col, gene_mapping)
        new_columns.append(symbol)
    
    expr_df_symbols = expr_df.copy()
    expr_df_symbols.columns = new_columns
    
    # 处理重复的 Gene Symbol (保留表达量最高的)
    expr_df_symbols = expr_df_symbols.T.groupby(level=0).max().T
    
    return expr_df_symbols


# =============================================================================
# MSigDB Hallmark Gene Sets 加载
# =============================================================================

def load_hallmark_gene_sets(data_dir="./data"):
    """
    加载 MSigDB Hallmark Gene Sets (50个癌症核心通路)
    这些通路概括了癌症的核心特征，极其稳健
    """
    global _HALLMARK_GENE_SETS
    
    if _HALLMARK_GENE_SETS is not None:
        return _HALLMARK_GENE_SETS
    
    gmt_file = os.path.join(data_dir, "h.all.v7.5.1.symbols.gmt")
    
    # 如果本地没有，从 MSigDB 下载
    if not os.path.exists(gmt_file):
        print("Downloading MSigDB Hallmark Gene Sets...")
        try:
            response = requests.get(HALLMARK_URL, timeout=60)
            if response.status_code == 200:
                with open(gmt_file, 'w') as f:
                    f.write(response.text)
                print(f"Downloaded to {gmt_file}")
            else:
                print(f"Failed to download: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading Hallmark gene sets: {e}")
            return None
    
    # 解析 GMT 文件
    _HALLMARK_GENE_SETS = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pathway_name = parts[0].replace("HALLMARK_", "")  # 简化名称
                genes = parts[2:]  # 跳过 URL 描述
                _HALLMARK_GENE_SETS[pathway_name] = set(genes)
    
    print(f"Loaded {len(_HALLMARK_GENE_SETS)} Hallmark pathways.")
    return _HALLMARK_GENE_SETS


# =============================================================================
# ssGSEA (Single-Sample Gene Set Enrichment Analysis)
# =============================================================================

def compute_ssgsea(expr_df_symbols, gene_sets, n_jobs=4):
    """
    计算单样本 GSEA (ssGSEA) 得分
    
    Args:
        expr_df_symbols: 表达矩阵 (行=样本, 列=Gene Symbol)
        gene_sets: 通路基因集字典 {pathway_name: set(genes)}
        n_jobs: 并行线程数
    
    Returns:
        DataFrame: 通路活性矩阵 (行=样本, 列=通路)
    """
    print(f"Computing ssGSEA scores for {len(expr_df_symbols)} samples...")
    
    # gseapy 需要 (genes x samples) 格式
    expr_T = expr_df_symbols.T
    
    try:
        # 使用 gseapy 的 ssgsea 方法
        ss = gp.ssgsea(
            data=expr_T,
            gene_sets=gene_sets,
            outdir=None,  # 不保存文件
            sample_norm_method='rank',  # 排序标准化
            threads=n_jobs,
            no_plot=True,
            verbose=False
        )
        
        # 获取 NES (Normalized Enrichment Score) 矩阵
        # res2d 格式: Name(样本), Term(通路), NES, ...
        # 需要 pivot: 行=样本(Name), 列=通路(Term)
        pathway_scores = ss.res2d.pivot(index='Name', columns='Term', values='NES')
        
        # 确保行是样本，列是通路
        # pivot 后: index=样本名, columns=通路名
        print(f"ssGSEA complete: {pathway_scores.shape[0]} samples × {pathway_scores.shape[1]} pathways")
        return pathway_scores
        
    except Exception as e:
        print(f"Error in ssGSEA: {e}")
        print("Falling back to simple mean-based pathway scoring...")
        return compute_simple_pathway_scores(expr_df_symbols, gene_sets)


def compute_simple_pathway_scores(expr_df_symbols, gene_sets):
    """
    简单的通路活性计算 (备选方案)
    使用通路内基因的平均 z-score
    """
    # Z-score 标准化
    expr_z = (expr_df_symbols - expr_df_symbols.mean()) / (expr_df_symbols.std() + 1e-8)
    
    pathway_scores = pd.DataFrame(index=expr_df_symbols.index)
    
    for pathway_name, genes in gene_sets.items():
        # 找到表达矩阵中存在的基因
        available_genes = [g for g in genes if g in expr_z.columns]
        if len(available_genes) > 0:
            # 通路得分 = 通路内基因的平均 z-score
            pathway_scores[pathway_name] = expr_z[available_genes].mean(axis=1)
        else:
            pathway_scores[pathway_name] = 0.0
    
    return pathway_scores


# =============================================================================
# 通路活性分析与解读
# =============================================================================

def analyze_pathway_profile(pathway_scores_row):
    """
    分析单个样本的通路活性谱
    
    Args:
        pathway_scores_row: Series, 单个样本的通路得分
    
    Returns:
        dict: 包含激活/抑制通路及其生物学解读
    """
    # 按得分排序
    sorted_pathways = pathway_scores_row.sort_values(ascending=False)
    
    # 获取显著激活的通路 (NES > 1.0)
    activated = sorted_pathways[sorted_pathways > 1.0]
    # 获取显著抑制的通路 (NES < -1.0)
    suppressed = sorted_pathways[sorted_pathways < -1.0]
    
    # 如果没有显著变化，取 Top 5
    if len(activated) == 0:
        activated = sorted_pathways.head(5)
    if len(suppressed) == 0:
        suppressed = sorted_pathways.tail(5)
    
    return {
        'activated': activated.head(5),
        'suppressed': suppressed.head(5),
        'top_pathway': sorted_pathways.index[0],
        'top_score': sorted_pathways.iloc[0]
    }


# Hallmark 通路的简洁描述 (用于 Prompt 构建)
PATHWAY_DESCRIPTIONS = {
    # 增殖相关
    "MYC_TARGETS_V1": "MYC-driven proliferation",
    "MYC_TARGETS_V2": "MYC activation",
    "E2F_TARGETS": "cell cycle progression",
    "G2M_CHECKPOINT": "G2/M checkpoint activation",
    "MITOTIC_SPINDLE": "mitotic activity",
    
    # 代谢相关
    "OXIDATIVE_PHOSPHORYLATION": "enhanced OXPHOS metabolism",
    "GLYCOLYSIS": "glycolytic shift (Warburg)",
    "FATTY_ACID_METABOLISM": "lipid metabolic reprogramming",
    "CHOLESTEROL_HOMEOSTASIS": "altered cholesterol metabolism",
    "BILE_ACID_METABOLISM": "bile acid dysregulation",
    "XENOBIOTIC_METABOLISM": "drug metabolism alteration",
    
    # 免疫相关
    "INFLAMMATORY_RESPONSE": "inflammatory activation",
    "INTERFERON_ALPHA_RESPONSE": "IFN-α response (antiviral)",
    "INTERFERON_GAMMA_RESPONSE": "IFN-γ response (immune activation)",
    "IL6_JAK_STAT3_SIGNALING": "IL6/JAK/STAT3 signaling",
    "IL2_STAT5_SIGNALING": "IL2/STAT5 T-cell signaling",
    "TNFA_SIGNALING_VIA_NFKB": "TNF-α/NF-κB inflammation",
    "COMPLEMENT": "complement activation",
    "ALLOGRAFT_REJECTION": "immune rejection signature",
    
    # EMT/侵袭
    "EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT (invasion potential)",
    "ANGIOGENESIS": "angiogenic signaling",
    "COAGULATION": "coagulation cascade activation",
    
    # 信号通路
    "PI3K_AKT_MTOR_SIGNALING": "PI3K/AKT/mTOR activation",
    "MTORC1_SIGNALING": "mTORC1 metabolic signaling",
    "KRAS_SIGNALING_UP": "KRAS activation",
    "KRAS_SIGNALING_DN": "KRAS suppression",
    "NOTCH_SIGNALING": "Notch pathway activation",
    "WNT_BETA_CATENIN_SIGNALING": "Wnt/β-catenin signaling",
    "HEDGEHOG_SIGNALING": "Hedgehog pathway",
    "TGF_BETA_SIGNALING": "TGF-β signaling",
    
    # 应激/凋亡
    "P53_PATHWAY": "p53 activation (tumor suppression)",
    "APOPTOSIS": "apoptotic signaling",
    "HYPOXIA": "hypoxic response",
    "REACTIVE_OXYGEN_SPECIES_PATHWAY": "oxidative stress response",
    "UV_RESPONSE_UP": "UV damage response",
    "UV_RESPONSE_DN": "UV damage suppression",
    "DNA_REPAIR": "DNA repair activation",
    "UNFOLDED_PROTEIN_RESPONSE": "ER stress/UPR",
    
    # 分化/发育
    "MYOGENESIS": "muscle differentiation",
    "ADIPOGENESIS": "adipocyte differentiation",
    "SPERMATOGENESIS": "spermatogenic markers",
    "PANCREAS_BETA_CELLS": "pancreatic β-cell markers",
    
    # 激素相关
    "ESTROGEN_RESPONSE_EARLY": "early estrogen response",
    "ESTROGEN_RESPONSE_LATE": "late estrogen response",
    "ANDROGEN_RESPONSE": "androgen signaling",
    
    # 其他
    "HEME_METABOLISM": "heme/iron metabolism",
    "PROTEIN_SECRETION": "protein secretion",
    "PEROXISOME": "peroxisome function"
}


def get_pathway_description(pathway_name):
    """获取通路的简洁描述"""
    # 清理名称
    clean_name = pathway_name.replace("HALLMARK_", "").upper()
    return PATHWAY_DESCRIPTIONS.get(clean_name, pathway_name.lower().replace("_", " "))


# =============================================================================
# 基于通路的 Prompt 生成 (高信息密度)
# =============================================================================

def generate_pathway_prompt(patient_id, pathway_profile, cancer_type=None):
    """
    生成基于通路活性的高信息密度 Prompt
    
    这个 Prompt 设计原则：
    1. 使用通路状态而非基因列表 (跨平台稳定)
    2. 直接给出生物学解读 (减少 LLM 推理负担)
    3. 要求结构化输出 (提高 embedding 质量)
    """
    activated = pathway_profile['activated']
    suppressed = pathway_profile['suppressed']
    
    # 构建激活通路描述
    activated_desc = []
    for pathway, score in activated.items():
        desc = get_pathway_description(pathway)
        activated_desc.append(f"{desc}(+{score:.1f})")
    
    # 构建抑制通路描述
    suppressed_desc = []
    for pathway, score in suppressed.items():
        desc = get_pathway_description(pathway)
        suppressed_desc.append(f"{desc}({score:.1f})")
    
    # 癌种上下文
    cancer_context = f" in {cancer_type} patient" if cancer_type else ""
    
    # 构建高密度 Prompt
    prompt = f"""Pathway activity profile{cancer_context}:
ACTIVATED: {', '.join(activated_desc[:5])}
SUPPRESSED: {', '.join(suppressed_desc[:3])}

Based on this hallmark pathway signature, provide a concise clinical narrative (max 80 words) covering:
1. Molecular subtype inference
2. Key biological processes driving disease
3. Prognostic risk assessment (favorable/intermediate/poor)

Be specific and clinically actionable. Use domain terminology."""
    
    return prompt


def generate_pathway_prompt_v2(patient_id, pathway_profile, top_genes=None, cancer_type=None):
    """
    增强版 Prompt：结合通路 + 关键基因
    适用于需要更详细解读的场景
    """
    activated = pathway_profile['activated']
    
    # 只取最显著的通路
    top_pathways = []
    for pathway, score in activated.head(3).items():
        desc = get_pathway_description(pathway)
        top_pathways.append(f"{desc}(NES={score:.1f})")
    
    # 简洁格式
    pathway_str = "; ".join(top_pathways)
    
    # 可选：添加关键基因
    genes_str = ""
    if top_genes:
        genes_str = f"\nKey genes: {', '.join(top_genes[:5])}"
    
    cancer_context = f"[{cancer_type}] " if cancer_type else ""
    
    prompt = f"""{cancer_context}Hallmark signature: {pathway_str}{genes_str}

Generate a clinical molecular report (≤60 words):
- Subtype & pathway dependencies
- Risk stratification with rationale"""
    
    return prompt


# =============================================================================
# 通路/基因功能分组定义 (用于 V3 Prompt)
# =============================================================================

# 通路功能分组
PATHWAY_FUNCTIONAL_GROUPS = {
    'Proliferation_Stress': [
        'MYC_TARGETS_V1', 'MYC_TARGETS_V2', 'E2F_TARGETS', 
        'G2M_CHECKPOINT', 'MITOTIC_SPINDLE', 'MTORC1_SIGNALING',
        'UNFOLDED_PROTEIN_RESPONSE', 'P53_PATHWAY', 'DNA_REPAIR'
    ],
    'Metabolism': [
        'OXIDATIVE_PHOSPHORYLATION', 'GLYCOLYSIS', 'FATTY_ACID_METABOLISM',
        'CHOLESTEROL_HOMEOSTASIS', 'BILE_ACID_METABOLISM', 'XENOBIOTIC_METABOLISM',
        'HEME_METABOLISM', 'PEROXISOME'
    ],
    'Immunity': [
        'INFLAMMATORY_RESPONSE', 'INTERFERON_ALPHA_RESPONSE', 'INTERFERON_GAMMA_RESPONSE',
        'IL6_JAK_STAT3_SIGNALING', 'IL2_STAT5_SIGNALING', 'TNFA_SIGNALING_VIA_NFKB',
        'COMPLEMENT', 'ALLOGRAFT_REJECTION'
    ],
    'EMT_Invasion': [
        'EPITHELIAL_MESENCHYMAL_TRANSITION', 'ANGIOGENESIS', 'COAGULATION',
        'TGF_BETA_SIGNALING', 'HYPOXIA'
    ],
    'Signaling': [
        'PI3K_AKT_MTOR_SIGNALING', 'KRAS_SIGNALING_UP', 'KRAS_SIGNALING_DN',
        'NOTCH_SIGNALING', 'WNT_BETA_CATENIN_SIGNALING', 'HEDGEHOG_SIGNALING'
    ]
}

# 基因功能分组 (肝癌特异性)
GENE_FUNCTIONAL_GROUPS = {
    'Differentiation_Markers': [
        'ALB', 'APOA1', 'APOA2', 'APOC3', 'APOE', 'APOB',
        'AMBP', 'AFP', 'TTR', 'TF', 'FGA', 'FGB', 'FGG',
        'SERPINA1', 'SERPINC1', 'F2', 'F9', 'F10', 'PLG',
        'HPX', 'HP', 'GC', 'AHSG', 'ORM1', 'ORM2'
    ],
    'Mitochondrial_Function': [
        'MT-CO1', 'MT-CO2', 'MT-CO3', 'MT-ND1', 'MT-ND2', 'MT-ND3',
        'MT-ND4', 'MT-ND5', 'MT-ND6', 'MT-CYB', 'MT-ATP6', 'MT-ATP8'
    ],
    'Proliferation_Markers': [
        'MKI67', 'PCNA', 'TOP2A', 'MCM2', 'MCM6', 'CDK1', 'CDK2',
        'CCNA2', 'CCNB1', 'CCND1', 'CCNE1', 'E2F1', 'MYC', 'MYCN'
    ],
    'Stress_Response': [
        'HSPA1A', 'HSPA1B', 'HSP90AA1', 'HSPH1', 'DNAJB1',
        'ATF4', 'ATF6', 'XBP1', 'DDIT3', 'GADD45A', 'GADD45B'
    ],
    'Iron_Metabolism': [
        'FTL', 'FTH1', 'TFRC', 'SLC40A1', 'HAMP', 'HFE'
    ]
}


def classify_pathways_by_function(pathway_scores, use_percentile=True):
    """
    将通路按功能分组，返回每组的激活/抑制状态
    
    优化 A: 使用相对阈值（百分位数）而非绝对阈值
    这样可以适应不同平台的数据分布
    """
    result = {}
    
    # 计算全局百分位数阈值 (相对阈值)
    all_scores = pathway_scores.values
    if use_percentile and len(all_scores) > 0:
        p75 = np.percentile(all_scores, 75)  # 上四分位
        p50 = np.percentile(all_scores, 50)  # 中位数
        p25 = np.percentile(all_scores, 25)  # 下四分位
        high_threshold = p75
        mid_threshold = p50
        low_threshold = p25
    else:
        # 降级到绝对阈值
        high_threshold = 1.2
        mid_threshold = 1.0
        low_threshold = 0.8
    
    for group_name, pathways in PATHWAY_FUNCTIONAL_GROUPS.items():
        group_scores = []
        for pathway in pathways:
            if pathway in pathway_scores.index:
                group_scores.append((pathway, pathway_scores[pathway]))
        
        if group_scores:
            # 按得分排序
            group_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 使用相对阈值分级
            highly_activated = [(p, s) for p, s in group_scores if s > high_threshold]
            mildly_activated = [(p, s) for p, s in group_scores if mid_threshold < s <= high_threshold]
            suppressed = [(p, s) for p, s in group_scores if s < low_threshold]
            
            result[group_name] = {
                'highly_activated': highly_activated[:2],
                'mildly_activated': mildly_activated[:2],
                'activated': highly_activated[:2] or mildly_activated[:2],  # 兼容旧代码
                'suppressed': suppressed[:2],
                'top_pathway': group_scores[0] if group_scores else None,
                'all_scores': group_scores  # 保留全部用于调试
            }
    
    return result


def classify_genes_by_function(gene_series, top_n=30):
    """
    将高表达基因按功能分组
    """
    # 获取 Top N 高表达基因
    top_genes = gene_series.sort_values(ascending=False).head(top_n)
    top_gene_set = set(top_genes.index)
    
    result = {}
    
    for group_name, genes in GENE_FUNCTIONAL_GROUPS.items():
        # 找到在 Top N 中的基因
        found = [(g, top_genes[g]) for g in genes if g in top_gene_set]
        if found:
            found.sort(key=lambda x: x[1], reverse=True)
            result[group_name] = found[:5]  # 每组最多5个
    
    return result


def generate_pathway_prompt_v3(patient_id, pathway_scores, gene_series, cancer_type='HCC'):
    """
    V3 Prompt: 功能分组 + 冲突解析 + 亚型推理
    
    核心改进:
    1. 按功能分组通路和基因
    2. 明确要求 LLM 解决"高增殖 vs 高分化"的冲突
    3. 引导 Hoshida 亚型分类推理
    4. 使用专业术语
    """
    # 1. 通路功能分组
    pathway_groups = classify_pathways_by_function(pathway_scores)
    
    # 2. 基因功能分组
    gene_groups = classify_genes_by_function(gene_series)
    
    # 3. 构建通路活性描述
    pathway_sections = []
    
    # Proliferation & Stress
    if 'Proliferation_Stress' in pathway_groups:
        group = pathway_groups['Proliferation_Stress']
        items = []
        for p, s in group['activated']:
            items.append(f"{p.replace('_', ' ')} (Activated, NES={s:.1f})")
        if items:
            pathway_sections.append(f"**Proliferation & Stress**: {', '.join(items)}")
    
    # Metabolism
    if 'Metabolism' in pathway_groups:
        group = pathway_groups['Metabolism']
        items = []
        for p, s in group['activated']:
            items.append(f"{p.replace('_', ' ')} (Activated)")
        for p, s in group['suppressed']:
            items.append(f"{p.replace('_', ' ')} (Suppressed)")
        if items:
            pathway_sections.append(f"**Metabolism**: {', '.join(items)}")
    
    # Immunity
    if 'Immunity' in pathway_groups:
        group = pathway_groups['Immunity']
        items = []
        for p, s in group['activated']:
            items.append(f"{p.replace('_', ' ')} (Activated)")
        for p, s in group['suppressed']:
            items.append(f"{p.replace('_', ' ')} (Suppressed)")
        if items:
            pathway_sections.append(f"**Immunity**: {', '.join(items)}")
    
    # 4. 构建基因描述
    gene_sections = []
    
    if 'Differentiation_Markers' in gene_groups:
        genes = [f"{g}" for g, _ in gene_groups['Differentiation_Markers'][:4]]
        gene_sections.append(f"**Differentiation Markers**: {', '.join(genes)} (High)")
    
    if 'Mitochondrial_Function' in gene_groups:
        genes = [f"{g}" for g, _ in gene_groups['Mitochondrial_Function'][:3]]
        gene_sections.append(f"**Mitochondrial Function**: {', '.join(genes)} (High)")
    
    if 'Proliferation_Markers' in gene_groups:
        genes = [f"{g}" for g, _ in gene_groups['Proliferation_Markers'][:3]]
        gene_sections.append(f"**Proliferation Markers**: {', '.join(genes)} (High)")
    
    # 5. 检测潜在冲突
    has_differentiation = 'Differentiation_Markers' in gene_groups and len(gene_groups['Differentiation_Markers']) >= 2
    has_proliferation = 'Proliferation_Stress' in pathway_groups and len(pathway_groups['Proliferation_Stress']['activated']) >= 1
    
    conflict_instruction = ""
    if has_differentiation and has_proliferation:
        conflict_instruction = """
**[Reasoning Requirements]**
1. **Resolve the Conflict**: The patient exhibits high Oncogenic Signaling (MYC/mTOR) despite retaining strong Hepatocellular Differentiation (ALB/APO). Explain this phenotype. Is this a metabolically active, well-differentiated tumor?
2. **Subtype Inference**: Does this align with Hoshida S1/S2 (Proliferative) or S3 (Well-differentiated)? Or a transitional state?
3. **Risk Assessment**: Synthesize the metabolic burden with the retained liver function."""
    else:
        conflict_instruction = """
**[Reasoning Requirements]**
1. Infer the molecular subtype based on the pathway signature.
2. Assess prognosis considering the dominant biological processes.
3. Identify potential therapeutic vulnerabilities."""
    
    # 6. 组装完整 Prompt
    prompt = f"""[Patient: {cancer_type}]

**[Pathway Activity (ssGSEA)]**
{chr(10).join(pathway_sections) if pathway_sections else "No significant pathway alterations detected."}

**[Key Driver Genes]**
{chr(10).join(gene_sections) if gene_sections else "No distinctive gene markers."}
{conflict_instruction}

**[Output]**
Generate a Clinico-Genomic Narrative (~100 words). Use terms: "Metabolic Reprogramming," "Differentiation Status," "Proteostatic Stress." Be specific, not vague."""
    
    return prompt


def get_v3_system_prompt(cancer_type='HCC'):
    """
    V3 版本的 System Prompt - 专科医生角色
    """
    specialty_map = {
        'LIHC': 'hepatobiliary oncology',
        'HCC': 'hepatobiliary oncology',
        'BRCA': 'breast oncology',
        'LUAD': 'thoracic oncology',
        'LUSC': 'thoracic oncology', 
        'COAD': 'gastrointestinal oncology',
        'READ': 'gastrointestinal oncology',
        'STAD': 'gastrointestinal oncology',
        'ESCA': 'gastrointestinal oncology',
        'PAAD': 'pancreatic oncology',
        'OV': 'gynecologic oncology',
        'UCEC': 'gynecologic oncology',
        'PRAD': 'urologic oncology',
        'KIRC': 'urologic oncology',
        'KICH': 'urologic oncology',
        'KIRP': 'urologic oncology',
        'BLCA': 'urologic oncology',
        'HNSC': 'head and neck oncology',
        'GBM': 'neuro-oncology',
        'LGG': 'neuro-oncology',
        'SKCM': 'dermatologic oncology',
        'THCA': 'endocrine oncology',
    }
    
    specialty = specialty_map.get(cancer_type, 'molecular oncology')
    
    return f"""You are a specialist in {specialty}. Your task is to generate a Clinico-Genomic Narrative based on the patient's transcriptomic signature.

Key principles:
- Use precise medical terminology (Hoshida classification for HCC, PAM50 for breast cancer, etc.)
- Synthesize conflicting signals (e.g., high proliferation + high differentiation)
- Provide specific, actionable insights
- Avoid vague statements like "intermediate risk" without justification
- Reference known molecular subtypes when applicable"""


# =============================================================================
# V4 Prompt: 军事化管理 - 消除语义抖动 (Deterministic Generation)
# =============================================================================

# 控制术语词库 (Controlled Vocabulary) - V5 增强版：增加中间态
CONTROLLED_VOCABULARY = {
    # 代谢状态 - 增加中间态
    'Metabolic_Status': [
        'Hyper-metabolic',            # 高代谢 (>P75)
        'Metabolically-active',       # 中高代谢 (P50-P75) [新增]
        'Normo-metabolic',            # 正常代谢 (P25-P50)
        'Hypo-metabolic',             # 低代谢 (<P25)
        'Warburg-dominant',           # 糖酵解为主
        'OXPHOS-dominant',            # 氧化磷酸化为主
        'Lipid-metabolic',            # 脂代谢为主 [新增]
        'Metabolically-dysregulated'  # 代谢紊乱
    ],
    
    # 分化状态
    'Differentiation': [
        'Well-differentiated',        # 高分化
        'Moderately-differentiated',  # 中分化
        'Poorly-differentiated',      # 低分化
        'Dedifferentiated',           # 去分化
        'Progenitor-like',            # 干/祖细胞样
        'Lineage-preserved'           # 谱系保留 [新增]
    ],
    
    # 增殖状态 - 增加中间态
    'Proliferation': [
        'Highly-proliferative',       # 高增殖 (>P75)
        'Mildly-proliferative',       # 轻度增殖 (P50-P75) [新增]
        'Slow-cycling',               # 慢周期 (P25-P50) [新增]
        'Quiescent',                  # 静止期 (<P25)
        'Cell-cycle-arrested'         # 细胞周期停滞
    ],
    
    # 免疫状态 - 增加中间态
    'Immune_Status': [
        'Immune-Hot',                 # 免疫热 (高炎症)
        'Immune-Warm',                # 免疫温和 [新增]
        'Immune-Cold',                # 免疫冷
        'Immune-Excluded',            # 免疫排斥
        'Immune-Suppressed'           # 免疫抑制
    ],
    
    # 应激状态 - 增加中间态
    'Stress_Status': [
        'Proteostatic-stress',        # 蛋白稳态应激
        'Oxidative-stress',           # 氧化应激
        'Hypoxia-driven',             # 缺氧驱动
        'DNA-damage-response',        # DNA 损伤
        'Mild-stress',                # 轻度应激 [新增]
        'Stress-adapted',             # 应激适应
        'Stress-free'                 # 无应激
    ],
    
    # 侵袭/转移潜能
    'Invasion_Potential': [
        'Highly-invasive',
        'Moderately-invasive',        # [新增]
        'Locally-invasive',
        'Non-invasive',
        'Metastatic-primed'
    ],
    
    # 风险等级 - 细化
    'Risk_Level': [
        'Very-High-Risk',             # [新增]
        'High-Risk',
        'Intermediate-High-Risk',
        'Intermediate-Risk',
        'Intermediate-Low-Risk',
        'Low-Risk',
        'Very-Low-Risk'               # [新增]
    ],
    
    # 主效通路 (用于第二修饰语) [优化 C]
    'Dominant_Pathway': [
        'OXPHOS', 'Glycolysis', 'Fatty-Acid-Metabolism', 'Cholesterol-Homeostasis',
        'Bile-Acid-Metabolism', 'MYC-Targets', 'E2F-Targets', 'mTORC1-Signaling',
        'Wnt/β-catenin', 'TGF-β-Signaling', 'EMT', 'Hypoxia', 'P53-Pathway',
        'Interferon-Response', 'Inflammatory-Response', 'Complement'
    ],
    
    # 分子亚型 (癌种特异性)
    'HCC_Subtype': ['Hoshida-S1', 'Hoshida-S2', 'Hoshida-S3', 'S2-transitional', 'S3-indolent'],
    'BRCA_Subtype': ['Luminal-A', 'Luminal-B', 'HER2-enriched', 'Basal-like', 'Normal-like'],
    'LUAD_Subtype': ['TRU', 'proximal-inflammatory', 'proximal-proliferative'],
    'COAD_Subtype': ['CMS1', 'CMS2', 'CMS3', 'CMS4'],
    # ESCA 食管癌分子亚型 (TCGA 2017)
    'ESCA_Subtype': ['ESCC-Classical', 'ESCC-Basal', 'ESCC-Secretory', 'EAC-CIN', 'EAC-GS', 'EAC-MSI'],
}


# Few-Shot 示例库 (自然语言风格)
FEW_SHOT_EXAMPLES = {
    'LIHC': [
        {
            'input': 'MYC_TARGETS (High), OXPHOS (High), ALB (High), APOA1 (High), INFLAMMATORY_RESPONSE (Low)',
            'output': 'The tumor phenotype is Hyper-metabolic, OXPHOS-dominant, and Well-differentiated with a Highly-proliferative profile. The microenvironment is Immune-Cold. The tissue shows Proteostatic-stress. This aligns with Hoshida-S2 subtype. The patient is Intermediate-High-Risk due to MYC-driven proliferation balanced by retained differentiation and metabolic vulnerability.'
        },
        {
            'input': 'EMT (High), GLYCOLYSIS (High), TGF_BETA (High), ALB (Low), HYPOXIA (High)',
            'output': 'The tumor phenotype is Warburg-dominant, Poorly-differentiated, with a Highly-invasive profile. The microenvironment is Immune-Excluded. The tissue shows Hypoxia-driven stress. This aligns with Hoshida-S1 subtype. The patient is High-Risk due to EMT activation and dedifferentiation indicating therapeutic resistance.'
        },
        {
            'input': 'WNT_BETA_CATENIN (High), BILE_ACID_METABOLISM (High), ALB (High), MYC_TARGETS (Low)',
            'output': 'The tumor phenotype is Normo-metabolic, Well-differentiated, with a Quiescent profile. The microenvironment is Immune-Cold. The tissue is Stress-free. This aligns with Hoshida-S3 subtype. The patient is Low-Risk due to retained hepatocyte differentiation and absence of proliferative signaling.'
        }
    ],
    'BRCA': [
        {
            'input': 'ESTROGEN_RESPONSE (High), MYC_TARGETS (Low), E2F_TARGETS (Low), IMMUNE (Low)',
            'output': 'The tumor phenotype is Normo-metabolic, Well-differentiated, with a Quiescent profile. The microenvironment is Immune-Cold. The tissue is Stress-free. This aligns with Luminal-A subtype. The patient is Low-Risk due to hormone-responsiveness and low proliferative activity.'
        },
        {
            'input': 'MYC_TARGETS (High), E2F_TARGETS (High), DNA_REPAIR (High), ESTROGEN (Low)',
            'output': 'The tumor phenotype is Hyper-metabolic, Poorly-differentiated, with a Highly-proliferative profile. The microenvironment is Immune-Hot. The tissue shows DNA-damage-response stress. This aligns with Basal-like subtype. The patient is High-Risk due to aggressive proliferation but responsive to immunotherapy.'
        }
    ],
    'ESCA': [
        {
            'input': 'MYC_TARGETS (High), E2F_TARGETS (High), DNA_REPAIR (High), KERATINIZATION (High)',
            'output': 'The tumor phenotype is Hyper-metabolic, Squamous-like, with a Highly-proliferative profile. The microenvironment is Immune-Cold. The tissue shows DNA-damage-response stress. This aligns with ESCC-Classical subtype. The patient is Intermediate-High-Risk due to high proliferation with intact DNA repair.'
        },
        {
            'input': 'EMT (High), HYPOXIA (High), INFLAMMATORY_RESPONSE (High), PI3K_AKT (High)',
            'output': 'The tumor phenotype is Warburg-dominant, Poorly-differentiated, with a Highly-invasive profile. The microenvironment is Immune-Hot. The tissue shows Hypoxia-driven stress. This aligns with EAC-CIN subtype. The patient is High-Risk due to EMT activation and invasive phenotype.'
        },
        {
            'input': 'WNT_BETA_CATENIN (High), NOTCH (High), INTERFERON (Low), PROLIFERATION (Low)',
            'output': 'The tumor phenotype is Normo-metabolic, Moderately-differentiated, with a Slow-cycling profile. The microenvironment is Immune-Excluded. The tissue shows Mild-stress. This aligns with ESCC-Secretory subtype. The patient is Intermediate-Risk due to developmental pathway activation.'
        }
    ],
    'default': [
        {
            'input': 'MYC_TARGETS (High), OXPHOS (High), INFLAMMATORY_RESPONSE (Low)',
            'output': 'The tumor phenotype is Hyper-metabolic, Moderately-differentiated, with a Highly-proliferative profile. The microenvironment is Immune-Cold. The tissue shows Proteostatic-stress. The patient is Intermediate-High-Risk due to MYC-driven proliferation and immune evasion.'
        },
        {
            'input': 'EMT (High), HYPOXIA (High), P53_PATHWAY (High)',
            'output': 'The tumor phenotype is Metabolically-dysregulated, Poorly-differentiated, with a Highly-invasive profile. The microenvironment is Immune-Excluded. The tissue shows Hypoxia-driven stress. The patient is High-Risk due to EMT activation and p53 dysfunction.'
        }
    ]
}


def generate_pathway_prompt_v4(patient_id, pathway_scores, gene_series, cancer_type='LIHC'):
    """
    V4 Prompt: 军事化管理版本
    
    核心策略:
    1. 强制降温 (temperature=0, 在调用时设置)
    2. 菜单式术语 (Controlled Vocabulary)
    3. Few-Shot 锚点 (标准答案示例)
    4. 结构化输出格式
    """
    
    # 1. 通路功能分组
    pathway_groups = classify_pathways_by_function(pathway_scores)
    
    # 2. 基因功能分组
    gene_groups = classify_genes_by_function(gene_series)
    
    # 3. 构建简化的输入描述
    input_features = []
    
    # 增殖/应激
    if 'Proliferation_Stress' in pathway_groups:
        for p, s in pathway_groups['Proliferation_Stress']['activated'][:2]:
            status = 'High' if s > 1.3 else 'Moderate'
            input_features.append(f"{p.replace('_', ' ')} ({status})")
    
    # 代谢
    if 'Metabolism' in pathway_groups:
        for p, s in pathway_groups['Metabolism']['activated'][:1]:
            input_features.append(f"{p.replace('_', ' ')} (High)")
    
    # 免疫
    if 'Immunity' in pathway_groups:
        grp = pathway_groups['Immunity']
        if grp['activated']:
            input_features.append(f"INFLAMMATORY ({grp['activated'][0][0].split('_')[0]} High)")
        elif grp['suppressed']:
            input_features.append(f"INFLAMMATORY (Low)")
    
    # 分化标记
    if 'Differentiation_Markers' in gene_groups:
        genes = [g for g, _ in gene_groups['Differentiation_Markers'][:2]]
        input_features.append(f"{', '.join(genes)} (High)")
    
    input_str = ', '.join(input_features) if input_features else 'No distinctive features'
    
    # 4. 获取 Few-Shot 示例
    examples = FEW_SHOT_EXAMPLES.get(cancer_type, FEW_SHOT_EXAMPLES['default'])
    
    few_shot_text = ""
    for i, ex in enumerate(examples[:2], 1):
        few_shot_text += f"""
[Example {i}]
Input: {ex['input']}
Output: {ex['output']}
"""
    
    # 5. 构建词库约束
    vocab_constraint = """
[TERMINOLOGY CONSTRAINT - MANDATORY]
You MUST select terms ONLY from these lists. Do NOT invent synonyms or alternative phrases.

• Metabolic: Hyper-metabolic | Hypo-metabolic | Warburg-dominant | OXPHOS-dominant | Metabolically-dysregulated | Normo-metabolic
• Differentiation: Well-differentiated | Moderately-differentiated | Poorly-differentiated | Dedifferentiated | Progenitor-like
• Proliferation: Highly-proliferative | Moderately-proliferative | Quiescent | Cell-cycle-arrested
• Immune: Immune-Hot | Immune-Cold | Immune-Excluded | Immunosuppressed
• Stress: Proteostatic-stress | Oxidative-stress | Hypoxia-driven | DNA-damage-response | Stress-adapted | Stress-free
• Risk: High-Risk | Intermediate-High-Risk | Intermediate-Risk | Intermediate-Low-Risk | Low-Risk
"""
    
    # 6. 添加亚型词库
    if cancer_type in ['LIHC', 'HCC']:
        vocab_constraint += "• HCC Subtype: Hoshida-S1 | Hoshida-S2 | Hoshida-S3\n"
    elif cancer_type == 'BRCA':
        vocab_constraint += "• BRCA Subtype: Luminal-A | Luminal-B | HER2-enriched | Basal-like\n"
    elif cancer_type in ['COAD', 'READ']:
        vocab_constraint += "• CRC Subtype: CMS1 | CMS2 | CMS3 | CMS4\n"
    elif cancer_type == 'STAD':
        vocab_constraint += "• STAD Subtype: TCGA-MSI-H | TCGA-EBV+ | TCGA-CIN | TCGA-GS\n"
        vocab_constraint += "• Lauren Type: Intestinal | Diffuse | Mixed\n"
    elif cancer_type == 'PAAD':
        vocab_constraint += "• PAAD Subtype: PAAD-Squamous | PAAD-Progenitor | PAAD-ADEX | PAAD-Immunogenic\n"
    
    # 7. 构建机制短语 (基于检测到的特征)
    mechanism_parts = []
    if 'Proliferation_Stress' in pathway_groups and pathway_groups['Proliferation_Stress']['activated']:
        mechanism_parts.append("MYC-driven proliferation")
    if 'Metabolism' in pathway_groups and pathway_groups['Metabolism']['activated']:
        mechanism_parts.append("metabolic reprogramming")
    if 'Differentiation_Markers' in gene_groups:
        mechanism_parts.append("retained differentiation")
    if not mechanism_parts:
        mechanism_parts.append("pathway dysregulation")
    
    mechanism_str = " and ".join(mechanism_parts[:2])  # 最多2个机制
    
    # 8. 输出格式 - 自然语言风格 (便于 Embedding)
    output_format = f"""
[OUTPUT FORMAT - Exact Template]
Generate exactly this sentence, filling in the bracketed terms:
"The tumor phenotype is [Metabolic], [Differentiation], with a [Proliferation] profile. The microenvironment is [Immune]. The tissue shows [Stress]. This aligns with [Subtype] subtype. The patient is [Risk] due to {mechanism_str}."

Use ONLY terms from the vocabulary lists above. Output the single sentence only, nothing else."""
    
    # 8. 组装完整 Prompt
    prompt = f"""[Cancer Type: {cancer_type}]

{vocab_constraint}
{few_shot_text}
[Current Patient Data]
{input_str}
{output_format}
Generate the clinical narrative:"""
    
    return prompt


def get_v4_system_prompt():
    """
    V4 System Prompt: 自然语言风格，严格词库控制
    """
    return """You are a molecular oncology classifier generating standardized clinical narratives.

RULES:
1. Use ONLY terms from the provided vocabulary lists.
2. Write as natural flowing prose, not a labeled list.
3. Keep output to exactly ONE paragraph (~50-60 words).
4. Do not add hedging words like "possibly", "may", "suggests".
5. Be direct and definitive in your classifications."""


# =============================================================================
# Hoshida 亚型判定逻辑 (基于样本内相对排序)
# =============================================================================

# Hoshida 分类生物学特征定义 (调整权重使分布更均衡)
HOSHIDA_SIGNATURES = {
    'S1': {
        # S1: TGF-β/Wnt 激活, EMT 高, 侵袭性强, 预后最差
        # 增加更多 S1 特征通路，提高检出率
        'positive_pathways': ['TGF_BETA_SIGNALING', 'WNT_BETA_CATENIN_SIGNALING', 
                              'EPITHELIAL_MESENCHYMAL_TRANSITION', 'ANGIOGENESIS',
                              'HYPOXIA', 'COAGULATION', 'KRAS_SIGNALING_UP'],
        'negative_pathways': ['OXIDATIVE_PHOSPHORYLATION', 'BILE_ACID_METABOLISM',
                              'MYC_TARGETS_V1', 'E2F_TARGETS'],  # S1 增殖不如 S2 高
        'description': 'Invasive/EMT-high',
        'weight': 1.2,  # 增加 S1 权重
    },
    'S2': {
        # S2: MYC/AKT 激活, 增殖活跃, AFP 高
        'positive_pathways': ['MYC_TARGETS_V1', 'MYC_TARGETS_V2', 'E2F_TARGETS',
                              'G2M_CHECKPOINT', 'MTORC1_SIGNALING', 'PI3K_AKT_MTOR_SIGNALING'],
        'negative_pathways': ['BILE_ACID_METABOLISM', 'XENOBIOTIC_METABOLISM',
                              'TGF_BETA_SIGNALING'],  # S2 不同于 S1 的 TGF-β
        'description': 'Proliferative/MYC-driven',
        'weight': 1.0,
    },
    'S3': {
        # S3: 肝细胞分化保留, 代谢正常, 预后最好
        'positive_pathways': ['BILE_ACID_METABOLISM', 'XENOBIOTIC_METABOLISM',
                              'FATTY_ACID_METABOLISM', 'CHOLESTEROL_HOMEOSTASIS',
                              'PEROXISOME', 'ADIPOGENESIS'],
        'negative_pathways': ['MYC_TARGETS_V1', 'E2F_TARGETS', 'G2M_CHECKPOINT',
                              'EPITHELIAL_MESENCHYMAL_TRANSITION', 'HYPOXIA'],
        'description': 'Well-differentiated/Hepatocyte-like',
        'weight': 0.9,  # 降低 S3 权重以平衡分布
    }
}


def classify_hoshida_subtype(pathway_scores, gene_series=None):
    """
    基于样本内相对排序的 Hoshida 亚型判定 (加权竞争法)
    
    核心改进: 
    1. 使用样本内 rank percentile (消除跨平台偏差)
    2. 加权放大 EMT 维度 (解决 S1 检出率过低问题)
    3. 增加抑制交互项 (增殖高则抑制 S3)
    """
    ranks = pathway_scores.rank(pct=True)
    
    # 三大核心通路组
    emt_paths = ['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_TGF_BETA_SIGNALING',
                 'HALLMARK_ANGIOGENESIS', 'HALLMARK_HYPOXIA', 'HALLMARK_COAGULATION',
                 'HALLMARK_KRAS_SIGNALING_UP']  # 增加 KRAS
    prolif_paths = ['HALLMARK_MYC_TARGETS_V1', 'HALLMARK_E2F_TARGETS', 
                    'HALLMARK_G2M_CHECKPOINT', 'HALLMARK_MTORC1_SIGNALING']
    metab_paths = ['HALLMARK_BILE_ACID_METABOLISM', 'HALLMARK_XENOBIOTIC_METABOLISM',
                   'HALLMARK_FATTY_ACID_METABOLISM', 'HALLMARK_CHOLESTEROL_HOMEOSTASIS']
    
    emt_avg = np.mean([ranks.get(p, 0.5) for p in emt_paths])
    prolif_avg = np.mean([ranks.get(p, 0.5) for p in prolif_paths])
    metab_avg = np.mean([ranks.get(p, 0.5) for p in metab_paths])
    
    # 关键单通路排名
    emt_single = ranks.get('HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 0.5)
    myc_single = ranks.get('HALLMARK_MYC_TARGETS_V1', 0.5)
    bile_single = ranks.get('HALLMARK_BILE_ACID_METABOLISM', 0.5)
    
    # 加权得分计算
    # S1: EMT 维度得分 (加权 1.5x 以提高检出率)
    s1_score = emt_avg * 1.5
    # 如果增殖比代谢高，降低 S1 (S1 应该增殖不是最高)
    if prolif_avg > metab_avg + 0.05:
        s1_score -= 0.1
    # 单通路加成
    if emt_single > 0.7:
        s1_score += 0.15
    
    # S2: 增殖维度得分
    s2_score = prolif_avg
    # 增殖必须明显高于代谢
    if prolif_avg > metab_avg:
        s2_score += 0.08
    if myc_single > 0.75:
        s2_score += 0.08
    
    # S3: 代谢维度得分 (微调)
    s3_score = metab_avg * 0.95
    # 如果增殖很高，惩罚 S3
    if prolif_avg > 0.65:
        s3_score -= 0.12
    if bile_single > 0.75:
        s3_score += 0.1
    
    scores = {'S1': s1_score, 'S2': s2_score, 'S3': s3_score}
    best_subtype = max(scores, key=scores.get)
    
    # 生成推理依据
    if best_subtype == 'S1':
        key_paths = []
        if emt_single > 0.5: key_paths.append('EMT')
        if ranks.get('HALLMARK_TGF_BETA_SIGNALING', 0) > 0.5: key_paths.append('TGF-β')
        if ranks.get('HALLMARK_HYPOXIA', 0) > 0.5: key_paths.append('Hypoxia')
        reasoning = f"mesenchymal ({', '.join(key_paths[:2]) if key_paths else 'stromal-active'})"
    elif best_subtype == 'S2':
        key_paths = []
        if myc_single > 0.5: key_paths.append('MYC')
        if ranks.get('HALLMARK_E2F_TARGETS', 0) > 0.5: key_paths.append('E2F')
        reasoning = f"proliferative ({', '.join(key_paths[:2]) if key_paths else 'cell-cycle'})"
    else:
        key_paths = []
        if bile_single > 0.5: key_paths.append('Bile-Acid')
        if ranks.get('HALLMARK_FATTY_ACID_METABOLISM', 0) > 0.5: key_paths.append('Lipid')
        reasoning = f"hepatocyte ({', '.join(key_paths[:2]) if key_paths else 'metabolic'})"
    
    # 置信度
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence_gap = sorted_scores[0] - sorted_scores[1]
    conf_level = 'High' if confidence_gap > 0.1 else 'Moderate' if confidence_gap > 0.05 else 'Low'
    
    return f"Hoshida-{best_subtype}", conf_level, reasoning


def classify_esca_subtype(pathway_scores, gene_series=None):
    """
    食管癌 (ESCA) 分子亚型判定
    
    ESCC (鳞状细胞癌): 
    - Classical: 高增殖、高角化
    - Basal: 干细胞样、EMT 高
    - Secretory: 分泌特征、Notch 高
    
    EAC (腺癌):
    - CIN: 染色体不稳定、TP53 突变
    - GS: 基因组稳定
    - MSI: 微卫星不稳定
    """
    ranks = pathway_scores.rank(pct=True)
    
    # 核心通路
    prolif_avg = np.mean([ranks.get(f'HALLMARK_{p}', 0.5) for p in 
                          ['MYC_TARGETS_V1', 'E2F_TARGETS', 'G2M_CHECKPOINT']])
    emt_avg = np.mean([ranks.get(f'HALLMARK_{p}', 0.5) for p in 
                       ['EPITHELIAL_MESENCHYMAL_TRANSITION', 'TGF_BETA_SIGNALING']])
    immune_avg = np.mean([ranks.get(f'HALLMARK_{p}', 0.5) for p in 
                          ['INFLAMMATORY_RESPONSE', 'INTERFERON_GAMMA_RESPONSE']])
    hypoxia = ranks.get('HALLMARK_HYPOXIA', 0.5)
    
    # ESCC vs EAC 判定 (基于代谢特征)
    acid_metab = ranks.get('HALLMARK_BILE_ACID_METABOLISM', 0.5)
    
    # 亚型评分
    scores = {}
    
    # ESCC-Classical: 高增殖
    scores['ESCC-Classical'] = prolif_avg * 1.2
    if prolif_avg > 0.6:
        scores['ESCC-Classical'] += 0.1
    
    # ESCC-Basal: EMT + 干细胞样
    scores['ESCC-Basal'] = emt_avg * 1.3
    if hypoxia > 0.6:
        scores['ESCC-Basal'] += 0.1
    
    # EAC-CIN: 高增殖 + 高代谢
    scores['EAC-CIN'] = (prolif_avg + acid_metab) / 2
    if acid_metab > 0.5:
        scores['EAC-CIN'] += 0.15
    
    # 选择最高分
    best_subtype = max(scores, key=scores.get)
    
    # 生成推理依据
    if 'Classical' in best_subtype:
        reasoning = f"proliferative (MYC/E2F)"
    elif 'Basal' in best_subtype:
        reasoning = f"EMT-active (stem-like)"
    else:
        reasoning = f"metabolically-active (CIN)"
    
    # 置信度
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence_gap = sorted_scores[0] - sorted_scores[1]
    conf_level = 'High' if confidence_gap > 0.1 else 'Moderate' if confidence_gap > 0.05 else 'Low'
    
    return best_subtype, conf_level, reasoning


def classify_stad_subtype(pathway_scores, gene_series=None):
    """
    胃腺癌 (STAD) TCGA分子亚型判定
    
    基于TCGA 2014 Nature的四种分子亚型:
    - MSI-H: 微卫星不稳定，高突变负荷，免疫浸润高，预后最好
    - EBV+: EB病毒阳性，PIK3CA突变，DNA甲基化高
    - CIN: 染色体不稳定，TP53突变多，RTK/RAS通路激活
    - GS: 基因组稳定，CDH1/RHOA突变，弥漫型，预后最差
    """
    ranks = pathway_scores.rank(pct=True)
    
    # 核心通路组
    # MSI特征: DNA修复缺陷 + 免疫高
    dna_repair = ranks.get('HALLMARK_DNA_REPAIR', 0.5)
    immune_paths = ['HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_INTERFERON_GAMMA_RESPONSE',
                    'HALLMARK_INTERFERON_ALPHA_RESPONSE', 'HALLMARK_IL6_JAK_STAT3_SIGNALING']
    immune_avg = np.mean([ranks.get(p, 0.5) for p in immune_paths])
    
    # CIN特征: 增殖高 + 染色体不稳定相关通路
    prolif_paths = ['HALLMARK_MYC_TARGETS_V1', 'HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT']
    prolif_avg = np.mean([ranks.get(p, 0.5) for p in prolif_paths])
    p53_pathway = ranks.get('HALLMARK_P53_PATHWAY', 0.5)
    
    # GS特征: EMT高 + CDH1相关 (弥漫型)
    emt_paths = ['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_TGF_BETA_SIGNALING']
    emt_avg = np.mean([ranks.get(p, 0.5) for p in emt_paths])
    apoptosis = ranks.get('HALLMARK_APOPTOSIS', 0.5)
    
    # EBV特征: PI3K/mTOR通路 + 甲基化相关
    pi3k = ranks.get('HALLMARK_PI3K_AKT_MTOR_SIGNALING', 0.5)
    mtorc1 = ranks.get('HALLMARK_MTORC1_SIGNALING', 0.5)
    hypoxia = ranks.get('HALLMARK_HYPOXIA', 0.5)
    
    # 评分系统
    scores = {}
    
    # MSI-H: 免疫浸润高 + DNA修复可能有缺陷
    scores['MSI-H'] = immune_avg * 1.3
    if immune_avg > 0.65:
        scores['MSI-H'] += 0.15
    # MSI往往增殖不是最高的
    if prolif_avg < 0.55:
        scores['MSI-H'] += 0.08
    
    # EBV+: PI3K/mTOR激活 + 免疫中等
    scores['EBV+'] = (pi3k + mtorc1) / 2 * 1.2
    if pi3k > 0.6 and mtorc1 > 0.5:
        scores['EBV+'] += 0.12
    if immune_avg > 0.45 and immune_avg < 0.7:  # 中等免疫
        scores['EBV+'] += 0.05
    
    # CIN: 高增殖 + P53通路异常
    scores['CIN'] = prolif_avg * 1.15
    if prolif_avg > 0.65:
        scores['CIN'] += 0.1
    # CIN往往EMT不是最高
    if emt_avg < 0.55:
        scores['CIN'] += 0.05
    
    # GS: EMT高 (弥漫型特征)
    scores['GS'] = emt_avg * 1.25
    if emt_avg > 0.6:
        scores['GS'] += 0.12
    # GS增殖往往较低
    if prolif_avg < 0.5:
        scores['GS'] += 0.08
    
    # 选择最高分
    best_subtype = max(scores, key=scores.get)
    
    # 生成推理依据
    if best_subtype == 'MSI-H':
        key_features = []
        if immune_avg > 0.55: key_features.append('immune-active')
        if dna_repair < 0.45: key_features.append('DNA-repair-deficient')
        reasoning = f"hypermutated, {', '.join(key_features) if key_features else 'high immune infiltration'}"
        risk = "Favorable"
    elif best_subtype == 'EBV+':
        key_features = []
        if pi3k > 0.5: key_features.append('PI3K-activated')
        if mtorc1 > 0.5: key_features.append('mTOR-high')
        reasoning = f"viral-associated, {', '.join(key_features) if key_features else 'PIK3CA-mutated'}"
        risk = "Intermediate"
    elif best_subtype == 'CIN':
        key_features = []
        if prolif_avg > 0.55: key_features.append('high-proliferation')
        if p53_pathway > 0.5: key_features.append('P53-pathway-active')
        reasoning = f"chromosomally-unstable, {', '.join(key_features) if key_features else 'RTK/RAS-driven'}"
        risk = "Intermediate-Poor"
    else:  # GS
        key_features = []
        if emt_avg > 0.55: key_features.append('EMT-active')
        key_features.append('diffuse-type')
        reasoning = f"CDH1-loss-like, {', '.join(key_features)}"
        risk = "Poor"
    
    # 置信度
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence_gap = sorted_scores[0] - sorted_scores[1]
    conf_level = 'High' if confidence_gap > 0.1 else 'Moderate' if confidence_gap > 0.05 else 'Low'
    
    return f"TCGA-{best_subtype}", conf_level, reasoning


def classify_paad_subtype(pathway_scores, gene_series=None):
    """
    胰腺腺癌 (PAAD) TCGA分子亚型判定
    
    基于TCGA 2017 Cell的四种分子亚型:
    - Squamous（鳞状样）: TP63/KRT高表达，EMT高，预后最差
    - Progenitor（祖细胞样）: 分化好，代谢活跃，预后较好
    - ADEX（异常分化内分泌外分泌）: 内外分泌标记共表达
    - Immunogenic（免疫原性）: 免疫浸润高，预后中等
    """
    ranks = pathway_scores.rank(pct=True)
    
    # 核心通路组
    # Squamous特征: EMT高 + 增殖高
    emt_paths = ['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_TGF_BETA_SIGNALING']
    emt_avg = np.mean([ranks.get(p, 0.5) for p in emt_paths])
    
    # Progenitor特征: 代谢活跃 + 分化好
    metab_paths = ['HALLMARK_OXIDATIVE_PHOSPHORYLATION', 'HALLMARK_FATTY_ACID_METABOLISM',
                   'HALLMARK_BILE_ACID_METABOLISM', 'HALLMARK_XENOBIOTIC_METABOLISM']
    metab_avg = np.mean([ranks.get(p, 0.5) for p in metab_paths])
    
    # Immunogenic特征: 免疫高
    immune_paths = ['HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_INTERFERON_GAMMA_RESPONSE',
                    'HALLMARK_INTERFERON_ALPHA_RESPONSE', 'HALLMARK_IL6_JAK_STAT3_SIGNALING']
    immune_avg = np.mean([ranks.get(p, 0.5) for p in immune_paths])
    
    # 增殖特征
    prolif_paths = ['HALLMARK_MYC_TARGETS_V1', 'HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT']
    prolif_avg = np.mean([ranks.get(p, 0.5) for p in prolif_paths])
    
    # ADEX特征: 分泌通路
    secre_paths = ['HALLMARK_PROTEIN_SECRETION', 'HALLMARK_UNFOLDED_PROTEIN_RESPONSE']
    secre_avg = np.mean([ranks.get(p, 0.5) for p in secre_paths])
    
    # 评分系统
    scores = {}
    
    # Squamous: EMT高 + 增殖高
    scores['Squamous'] = (emt_avg + prolif_avg) / 2 * 1.2
    if emt_avg > 0.6 and prolif_avg > 0.5:
        scores['Squamous'] += 0.15
    
    # Progenitor: 代谢高 + EMT低
    scores['Progenitor'] = metab_avg * 1.15
    if metab_avg > 0.55 and emt_avg < 0.5:
        scores['Progenitor'] += 0.1
    
    # ADEX: 分泌高 + 代谢中等
    scores['ADEX'] = (secre_avg + metab_avg * 0.5) * 1.1
    if secre_avg > 0.55:
        scores['ADEX'] += 0.08
    
    # Immunogenic: 免疫高
    scores['Immunogenic'] = immune_avg * 1.2
    if immune_avg > 0.6:
        scores['Immunogenic'] += 0.12
    
    # 选择最高分
    best_subtype = max(scores, key=scores.get)
    
    # 生成推理依据
    if best_subtype == 'Squamous':
        key_features = ['EMT-active', 'proliferative']
        reasoning = f"basal-like, {', '.join(key_features)}"
        risk = "Poor"
    elif best_subtype == 'Progenitor':
        key_features = ['metabolically-active', 'well-differentiated']
        reasoning = f"pancreatic-progenitor, {', '.join(key_features)}"
        risk = "Favorable"
    elif best_subtype == 'ADEX':
        key_features = ['exocrine-like', 'endocrine-like']
        reasoning = f"aberrantly-differentiated, {', '.join(key_features)}"
        risk = "Intermediate"
    else:  # Immunogenic
        key_features = ['immune-active', 'inflamed']
        reasoning = f"immune-infiltrated, {', '.join(key_features)}"
        risk = "Intermediate"
    
    # 置信度
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence_gap = sorted_scores[0] - sorted_scores[1]
    conf_level = 'High' if confidence_gap > 0.1 else 'Moderate' if confidence_gap > 0.05 else 'Low'
    
    return f"PAAD-{best_subtype}", conf_level, reasoning


def classify_prad_subtype(pathway_scores, gene_series=None):
    """
    前列腺腺癌 (PRAD) TCGA分子亚型判定
    
    基于TCGA 2015 Cell的分子亚型及AR信号状态:
    - ERG融合阳性 (ERG+): TMPRSS2-ERG融合，最常见(~50%)
    - SPOP突变型: SPOP突变，预后较好
    - FOXA1突变型: FOXA1高表达/突变
    - IDH1突变型: 代谢异常
    - 其他/Mixed: 无明确驱动
    
    主要通路特征:
    - AR信号通路活性
    - 雄激素响应
    - 细胞周期/增殖
    - DNA损伤修复
    """
    ranks = pathway_scores.rank(pct=True)
    
    # 核心通路评分
    # AR/雄激素响应 (PRAD核心)
    ar_paths = ['HALLMARK_ANDROGEN_RESPONSE']
    ar_avg = np.mean([ranks.get(p, 0.5) for p in ar_paths])
    
    # 增殖特征
    prolif_paths = ['HALLMARK_MYC_TARGETS_V1', 'HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT']
    prolif_avg = np.mean([ranks.get(p, 0.5) for p in prolif_paths])
    
    # EMT/间质特征
    emt_paths = ['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']
    emt_avg = np.mean([ranks.get(p, 0.5) for p in emt_paths])
    
    # DNA修复
    dna_paths = ['HALLMARK_DNA_REPAIR', 'HALLMARK_P53_PATHWAY']
    dna_avg = np.mean([ranks.get(p, 0.5) for p in dna_paths])
    
    # 代谢特征
    metab_paths = ['HALLMARK_OXIDATIVE_PHOSPHORYLATION', 'HALLMARK_FATTY_ACID_METABOLISM']
    metab_avg = np.mean([ranks.get(p, 0.5) for p in metab_paths])
    
    # 免疫特征
    immune_paths = ['HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_INTERFERON_GAMMA_RESPONSE']
    immune_avg = np.mean([ranks.get(p, 0.5) for p in immune_paths])
    
    # 分类逻辑 (基于TCGA分子特征)
    scores = {}
    reasoning_parts = []
    
    # 类型1: Luminal-like (AR高，预后好)
    scores['Luminal'] = ar_avg * 1.3
    if ar_avg > 0.6 and prolif_avg < 0.5:
        scores['Luminal'] += 0.15
        reasoning_parts.append(f"AR高({ar_avg:.2f})且增殖低")
    
    # 类型2: Basal-like (EMT高，AR低，预后差)
    scores['Basal'] = (emt_avg * 0.6 + (1 - ar_avg) * 0.4) * 1.2
    if emt_avg > 0.55 and ar_avg < 0.45:
        scores['Basal'] += 0.15
        reasoning_parts.append(f"EMT高({emt_avg:.2f})且AR低")
    
    # 类型3: Proliferative (高增殖)
    scores['Proliferative'] = prolif_avg * 1.2
    if prolif_avg > 0.6:
        scores['Proliferative'] += 0.1
        reasoning_parts.append(f"高增殖({prolif_avg:.2f})")
    
    # 类型4: Immune-enriched (免疫浸润高)
    scores['Immune'] = immune_avg * 1.15
    if immune_avg > 0.6:
        scores['Immune'] += 0.1
        reasoning_parts.append(f"免疫浸润高({immune_avg:.2f})")
    
    # 类型5: Metabolic (代谢活跃，与IDH相关)
    scores['Metabolic'] = metab_avg * 1.1
    if metab_avg > 0.6 and dna_avg > 0.5:
        scores['Metabolic'] += 0.1
        reasoning_parts.append(f"代谢活跃({metab_avg:.2f})")
    
    # 确定最佳亚型
    best_subtype = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    
    # 置信度
    confidence_gap = sorted_scores[0] - sorted_scores[1]
    conf_level = 'High' if confidence_gap > 0.1 else 'Moderate' if confidence_gap > 0.05 else 'Low'
    
    # 生成理由
    reasoning = f"PRAD亚型: {best_subtype} | AR={ar_avg:.2f}, 增殖={prolif_avg:.2f}, EMT={emt_avg:.2f}"
    if reasoning_parts:
        reasoning += " | " + "; ".join(reasoning_parts[:2])
    
    return f"PRAD-{best_subtype}", conf_level, reasoning


def classify_cancer_subtype(pathway_scores, gene_series=None, cancer_type='LIHC'):
    """
    通用癌症亚型分类入口函数
    根据癌症类型调用不同的分类函数
    """
    cancer_type_upper = cancer_type.upper().replace('TCGA-', '')
    
    if cancer_type_upper in ['LIHC', 'HCC', 'LIRI-JP']:
        return classify_hoshida_subtype(pathway_scores, gene_series)
    elif cancer_type_upper in ['ESCA', 'GSE53624']:
        return classify_esca_subtype(pathway_scores, gene_series)
    elif cancer_type_upper in ['BRCA']:
        return classify_pam50_subtype(pathway_scores, gene_series)
    elif cancer_type_upper in ['STAD']:
        return classify_stad_subtype(pathway_scores, gene_series)
    elif cancer_type_upper in ['PAAD']:
        return classify_paad_subtype(pathway_scores, gene_series)
    elif cancer_type_upper in ['PRAD', 'PRAD-CA']:
        return classify_prad_subtype(pathway_scores, gene_series)
    else:
        # 默认使用通用增殖/代谢/免疫分类
        return classify_generic_subtype(pathway_scores, gene_series)


def classify_pam50_subtype(pathway_scores, gene_series=None):
    """
    PAM50 分类 (乳腺癌) - 基于200样本分析的优化阈值
    
    目标分布: Luminal-A ~33%, Luminal-B ~27%, HER2 ~6%, Basal ~10%, Normal ~25%
    """
    ranks = pathway_scores.rank(pct=True)
    
    # 计算关键评分
    er_early = ranks.get('HALLMARK_ESTROGEN_RESPONSE_EARLY', 0.5)
    er_late = ranks.get('HALLMARK_ESTROGEN_RESPONSE_LATE', 0.5)
    e2f = ranks.get('HALLMARK_E2F_TARGETS', 0.5)
    myc = ranks.get('HALLMARK_MYC_TARGETS_V1', 0.5)
    g2m = ranks.get('HALLMARK_G2M_CHECKPOINT', 0.5)
    emt = ranks.get('HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 0.5)
    
    # 综合评分
    er_rank = (er_early + er_late) / 2
    prolif_rank = (e2f + myc + g2m) / 3
    
    # PAM50 分类 (基于200样本分析优化的阈值)
    # 1. Basal-like: ER很低(<P30) + EMT高(>P60)
    if er_rank < 0.30 and emt > 0.60:
        subtype = 'Basal-like'
        reasoning = 'triple-negative with high EMT activation'
        conf = 'High' if er_rank < 0.20 and emt > 0.70 else 'Moderate'
    
    # 2. HER2-enriched: ER中低(<P45) + 增殖很高(>P70) + 不是Basal
    elif er_rank < 0.45 and prolif_rank > 0.70 and er_rank >= 0.30:
        subtype = 'HER2-enriched'
        reasoning = 'high proliferation without ER dominance'
        conf = 'High' if prolif_rank > 0.80 else 'Moderate'
    
    # 3. Luminal-A: ER高(>P40) + 增殖低(<P45)
    elif er_rank > 0.40 and prolif_rank < 0.45:
        subtype = 'Luminal-A'
        reasoning = 'ER-positive with low proliferation'
        conf = 'High' if er_rank > 0.60 and prolif_rank < 0.35 else 'Moderate'
    
    # 4. Luminal-B: ER高(>P40) + 增殖高(>=P45)
    elif er_rank > 0.40 and prolif_rank >= 0.45:
        subtype = 'Luminal-B'
        reasoning = 'ER-positive with elevated proliferation'
        conf = 'High' if prolif_rank > 0.60 else 'Moderate'
    
    # 5. Normal-like: 其他情况
    else:
        subtype = 'Normal-like'
        reasoning = 'features resembling normal breast tissue'
        conf = 'Low'
    
    return subtype, conf, reasoning


def classify_generic_subtype(pathway_scores, gene_series=None):
    """
    通用亚型分类 (增殖型 vs 代谢型 vs 免疫型)
    """
    ranks = pathway_scores.rank(pct=True)
    
    prolif_avg = np.mean([ranks.get(f'HALLMARK_{p}', 0.5) for p in 
                          ['MYC_TARGETS_V1', 'E2F_TARGETS', 'G2M_CHECKPOINT']])
    metab_avg = np.mean([ranks.get(f'HALLMARK_{p}', 0.5) for p in 
                         ['OXIDATIVE_PHOSPHORYLATION', 'FATTY_ACID_METABOLISM']])
    immune_avg = np.mean([ranks.get(f'HALLMARK_{p}', 0.5) for p in 
                          ['INFLAMMATORY_RESPONSE', 'INTERFERON_GAMMA_RESPONSE']])
    
    scores = {
        'Proliferative': prolif_avg,
        'Metabolic': metab_avg,
        'Immune-Active': immune_avg
    }
    
    best_subtype = max(scores, key=scores.get)
    
    if best_subtype == 'Proliferative':
        reasoning = "MYC/E2F-driven"
    elif best_subtype == 'Metabolic':
        reasoning = "OXPHOS/lipid-active"
    else:
        reasoning = "immune-infiltrated"
    
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence_gap = sorted_scores[0] - sorted_scores[1]
    conf_level = 'High' if confidence_gap > 0.1 else 'Moderate' if confidence_gap > 0.05 else 'Low'
    
    return best_subtype, conf_level, reasoning


def get_sample_pathway_percentiles(pathway_scores):
    """
    计算样本内通路的百分位数，返回分级
    """
    ranks = pathway_scores.rank(pct=True)
    
    result = {}
    for pathway, rank_val in ranks.items():
        clean_name = pathway.replace('HALLMARK_', '')
        if rank_val > 0.8:
            level = 'Very-High'
        elif rank_val > 0.6:
            level = 'High'
        elif rank_val > 0.4:
            level = 'Moderate'
        elif rank_val > 0.2:
            level = 'Low'
        else:
            level = 'Very-Low'
        result[clean_name] = (level, rank_val)
    
    return result


# =============================================================================
# V5: 增强语义分辨率版本 (含 Hoshida 强制判定)
# =============================================================================

def generate_pathway_prompt_v5(patient_id, pathway_scores, gene_series, cancer_type='LIHC'):
    """
    V5 Prompt: 增强语义分辨率 + 强制亚型判定版本
    
    核心优化:
    A. 相对阈值 (样本内百分位排序) - 适应不同平台
    B. 中间态词汇 - 增加区分度  
    C. 主效通路修饰语 - 打破同质化
    D. 强制亚型判定 - 基于生物学特征，不让 LLM 瞎猜
    """
    
    # 1. 强制亚型判定 (使用通用分类函数)
    subtype, confidence, subtype_reason = classify_cancer_subtype(
        pathway_scores, gene_series, cancer_type
    )
    
    # 2. 获取样本内通路百分位排名
    pathway_percentiles = get_sample_pathway_percentiles(pathway_scores)
    
    # 3. 使用相对阈值的通路分组
    pathway_groups = classify_pathways_by_function(pathway_scores, use_percentile=True)
    
    # 4. 基因功能分组
    gene_groups = classify_genes_by_function(gene_series)
    
    # 5. 识别主效通路 (基于样本内排名最高)
    ranks = pathway_scores.rank(pct=True)
    top_pathway_idx = ranks.idxmax()
    dominant_pathway_name = top_pathway_idx.replace('HALLMARK_', '').replace('_', '-')
    dominant_pathway_rank = ranks[top_pathway_idx]
    
    # 6. 基于亚型确定主要特征描述
    cancer_upper = cancer_type.upper().replace('TCGA-', '')
    if cancer_upper in ['LIHC', 'HCC', 'LIRI-JP']:
        # HCC 亚型
        if 'S1' in subtype:
            primary_features = "EMT-active, TGF-β-driven, invasive"
            risk_tendency = "High-Risk to Intermediate-High-Risk"
        elif 'S2' in subtype:
            primary_features = "MYC/mTOR-driven, proliferative"
            risk_tendency = "Intermediate-High-Risk to Intermediate-Risk"
        else:
            primary_features = "hepatocyte-like, metabolism-preserved"
            risk_tendency = "Low-Risk to Intermediate-Low-Risk"
    elif cancer_upper in ['ESCA', 'GSE53624']:
        # ESCA 亚型
        if 'Classical' in subtype:
            primary_features = "squamous, proliferative"
            risk_tendency = "Intermediate-High-Risk to Intermediate-Risk"
        elif 'Basal' in subtype:
            primary_features = "EMT-active, stem-like"
            risk_tendency = "High-Risk to Intermediate-High-Risk"
        else:
            primary_features = "metabolically-active"
            risk_tendency = "Intermediate-Risk to Low-Risk"
    elif cancer_upper in ['BRCA']:
        # BRCA PAM50 亚型
        if 'Luminal-A' in subtype:
            primary_features = "ER-positive, low proliferation, hormone-responsive"
            risk_tendency = "Low-Risk to Intermediate-Low-Risk"
        elif 'Luminal-B' in subtype:
            primary_features = "ER-positive, high proliferation"
            risk_tendency = "Intermediate-Risk to Intermediate-High-Risk"
        elif 'HER2-enriched' in subtype:
            primary_features = "HER2-driven, highly proliferative"
            risk_tendency = "High-Risk to Intermediate-High-Risk"
        elif 'Basal-like' in subtype:
            primary_features = "triple-negative, EMT-active, aggressive"
            risk_tendency = "High-Risk"
        else:  # Normal-like
            primary_features = "low tumor cellularity, normal-like expression"
            risk_tendency = "Intermediate-Low-Risk to Intermediate-Risk"
    elif cancer_upper in ['STAD']:
        # STAD TCGA分子亚型 (MSI-H, EBV+, CIN, GS)
        if 'MSI-H' in subtype:
            primary_features = "hypermutated, immune-active, DNA-repair-deficient"
            risk_tendency = "Low-Risk to Intermediate-Low-Risk"  # 最好预后
        elif 'EBV' in subtype:
            primary_features = "viral-associated, PIK3CA-mutated, DNA-hypermethylated"
            risk_tendency = "Intermediate-Low-Risk to Intermediate-Risk"
        elif 'CIN' in subtype:
            primary_features = "chromosomally-unstable, TP53-mutated, RTK/RAS-activated"
            risk_tendency = "Intermediate-Risk to Intermediate-High-Risk"
        else:  # GS
            primary_features = "genomically-stable, CDH1-loss, diffuse-type"
            risk_tendency = "High-Risk to Intermediate-High-Risk"  # 最差预后
    elif cancer_upper in ['PAAD']:
        # PAAD TCGA分子亚型 (Squamous, Progenitor, ADEX, Immunogenic)
        if 'Squamous' in subtype:
            primary_features = "basal-like, EMT-active, TP63-high"
            risk_tendency = "High-Risk"  # 最差预后
        elif 'Progenitor' in subtype:
            primary_features = "well-differentiated, metabolically-active, PDX1-high"
            risk_tendency = "Low-Risk to Intermediate-Low-Risk"  # 最好预后
        elif 'ADEX' in subtype:
            primary_features = "aberrantly-differentiated, exocrine-endocrine"
            risk_tendency = "Intermediate-Risk"
        else:  # Immunogenic
            primary_features = "immune-infiltrated, inflamed-microenvironment"
            risk_tendency = "Intermediate-Low-Risk to Intermediate-Risk"
    elif cancer_upper in ['PRAD', 'PRAD-CA']:
        # PRAD 前列腺癌分子亚型 (基于AR信号和分子特征)
        if 'Luminal' in subtype:
            primary_features = "AR-high, hormone-responsive, well-differentiated"
            risk_tendency = "Low-Risk to Intermediate-Low-Risk"  # 最好预后
        elif 'Basal' in subtype:
            primary_features = "AR-low, EMT-active, stem-like, aggressive"
            risk_tendency = "High-Risk to Intermediate-High-Risk"  # 最差预后
        elif 'Proliferative' in subtype:
            primary_features = "high-proliferation, MYC-driven, cell-cycle-active"
            risk_tendency = "Intermediate-High-Risk to Intermediate-Risk"
        elif 'Immune' in subtype:
            primary_features = "immune-infiltrated, T-cell-enriched"
            risk_tendency = "Intermediate-Risk to Intermediate-Low-Risk"
        else:  # Metabolic
            primary_features = "metabolically-active, IDH-associated"
            risk_tendency = "Intermediate-Risk"
    else:
        # 通用
        if 'Proliferative' in subtype:
            primary_features = "proliferative, MYC-driven"
            risk_tendency = "Intermediate-High-Risk"
        elif 'Immune' in subtype:
            primary_features = "immune-active"
            risk_tendency = "Intermediate-Risk (immunotherapy-responsive)"
        else:
            primary_features = "metabolic"
            risk_tendency = "Intermediate-Low-Risk"
    
    # 7. 构建输入特征 (相对分级)
    input_features = []
    
    # 增殖信号 (基于排名)
    prolif_pathways = ['MYC_TARGETS_V1', 'E2F_TARGETS', 'G2M_CHECKPOINT']
    prolif_ranks = [ranks.get(f'HALLMARK_{p}', 0) for p in prolif_pathways]
    avg_prolif = np.mean(prolif_ranks)
    if avg_prolif > 0.7:
        input_features.append(f"Proliferation: Very-High (P{int(avg_prolif*100)})")
    elif avg_prolif > 0.5:
        input_features.append(f"Proliferation: High (P{int(avg_prolif*100)})")
    elif avg_prolif > 0.3:
        input_features.append(f"Proliferation: Moderate (P{int(avg_prolif*100)})")
    else:
        input_features.append(f"Proliferation: Low (P{int(avg_prolif*100)})")
    
    # 代谢信号 (基于排名)
    metab_pathways = ['OXIDATIVE_PHOSPHORYLATION', 'FATTY_ACID_METABOLISM', 'BILE_ACID_METABOLISM']
    metab_ranks = [ranks.get(f'HALLMARK_{p}', 0) for p in metab_pathways]
    avg_metab = np.mean(metab_ranks)
    if avg_metab > 0.7:
        input_features.append(f"Metabolism: Hyper-active (P{int(avg_metab*100)})")
    elif avg_metab > 0.5:
        input_features.append(f"Metabolism: Active (P{int(avg_metab*100)})")
    elif avg_metab > 0.3:
        input_features.append(f"Metabolism: Normal (P{int(avg_metab*100)})")
    else:
        input_features.append(f"Metabolism: Suppressed (P{int(avg_metab*100)})")
    
    # 免疫信号 (基于排名)
    immune_pathways = ['INFLAMMATORY_RESPONSE', 'INTERFERON_GAMMA_RESPONSE']
    immune_ranks = [ranks.get(f'HALLMARK_{p}', 0) for p in immune_pathways]
    avg_immune = np.mean(immune_ranks)
    if avg_immune > 0.6:
        input_features.append("Immune: Hot")
    elif avg_immune > 0.4:
        input_features.append("Immune: Warm")
    else:
        input_features.append("Immune: Cold")
    
    # EMT/侵袭信号
    emt_rank = ranks.get('HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 0.5)
    if emt_rank > 0.7:
        input_features.append("EMT: Active")
    elif emt_rank > 0.4:
        input_features.append("EMT: Partial")
    else:
        input_features.append("EMT: Inactive")
    
    input_str = '; '.join(input_features)
    
    # 8. 词库约束 - V5 增强版 (含 Hoshida 强制判定)
    vocab_constraint = f"""
[TERMINOLOGY CONSTRAINT V5 - MANDATORY]
Select terms ONLY from these lists. Match intensity to the percentile ranks.

• Metabolic Status:
  P>70: Hyper-metabolic | Warburg-dominant | OXPHOS-dominant | Lipid-metabolic
  P50-70: Metabolically-active
  P<50: Normo-metabolic | Hypo-metabolic

• Differentiation:
  Well-differentiated | Lineage-preserved | Moderately-differentiated | Poorly-differentiated | Dedifferentiated

• Proliferation:
  P>70: Highly-proliferative
  P40-70: Mildly-proliferative | Slow-cycling  
  P<40: Quiescent | Cell-cycle-arrested

• Immune Status:
  P>60: Immune-Hot
  P40-60: Immune-Warm
  P<40: Immune-Cold | Immune-Excluded

• Risk Level (MUST align with {subtype}):
  High-proliferative → High-Risk or Intermediate-High-Risk
  Moderate → Intermediate-Risk
  Low-proliferative → Intermediate-Low-Risk or Low-Risk
"""
    
    # 9. V5 输出格式 - 强制亚型
    output_format = f"""
[OUTPUT FORMAT V5 - Exact Template]
Generate exactly this sentence:

"The tumor phenotype is [Metabolic], [Differentiation], with a [Proliferation] profile. The microenvironment is [Immune]. The tissue shows [Stress]. This aligns with {subtype} subtype ({subtype_reason}). The patient is [Risk] due to {primary_features} driven by {dominant_pathway_name}."

CRITICAL RULES:
1. The subtype is FIXED as {subtype} - do NOT change it.
2. Risk level MUST be within {risk_tendency} range.
3. The ending MUST include "driven by {dominant_pathway_name}".
4. Output the single sentence only."""
    
    # 10. 组装完整 Prompt
    prompt = f"""[Cancer Type: {cancer_type}]
[Patient: {patient_id}]

[PRE-COMPUTED MOLECULAR SUBTYPE: {subtype}]
[Confidence: {confidence}]
[Key Evidence: {subtype_reason}]

{vocab_constraint}
[Current Patient Data]
{input_str}

[Dominant Pathway: {dominant_pathway_name}]
{output_format}

Generate the clinical narrative:"""
    
    return prompt, dominant_pathway_name, subtype


def get_v5_system_prompt():
    """
    V5 System Prompt: 高语义分辨率版本
    """
    return """You are a molecular oncology classifier generating standardized clinical narratives with HIGH SEMANTIC RESOLUTION.

RULES:
1. Use ONLY terms from the provided vocabulary lists.
2. Match risk level precisely to the phenotype intensity.
3. ALWAYS include the dominant pathway name at the end.
4. Differentiate between STRONG, MODERATE, and MILD phenotypes - avoid defaulting to extremes.
5. Output exactly ONE paragraph (~50-60 words), no labels or headers."""


def generate_report_v5(client, prompt, system_prompt=None):
    """
    V5 报告生成: temperature=0, 增强语义分辨率
    """
    if system_prompt is None:
        system_prompt = get_v5_system_prompt()
    
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0,
            max_tokens=200,
            top_p=1.0,
        )
        
        report = completion.choices[0].message.content
        
        # 清理思考标签
        if report and '</think>' in report:
            report = report.split('</think>')[-1].strip()
        
        return report.strip()
        
    except Exception as e:
        print(f"Error in V5 generation: {e}")
        return "Phenotype: Unknown. Risk: Unknown."


def generate_report_v4(client, prompt, system_prompt=None):
    """
    V4 报告生成: temperature=0 确保确定性输出
    """
    if system_prompt is None:
        system_prompt = get_v4_system_prompt()
    
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0,  # 强制降温：消除随机性
            max_tokens=200,
            top_p=1.0,      # 不使用 nucleus sampling
        )
        
        report = completion.choices[0].message.content
        
        # 清理思考标签
        if report and '</think>' in report:
            report = report.split('</think>')[-1].strip()
        
        return report.strip()
        
    except Exception as e:
        print(f"Error in V4 generation: {e}")
        return "Phenotype: Unknown. Microenvironment: Unknown. Risk: Unknown."


# =============================================================================
# LLM 调用
# =============================================================================

def generate_reasoning_report(client, prompt, enable_reasoning=True):
    """
    步骤 2: LLM Reasoning (生成微型报告)
    """
    try:
        # System prompt 优化：强调简洁和临床价值
        system_prompt = """You are an expert molecular oncologist. Generate concise, clinically actionable reports.
Rules:
- Use precise medical terminology
- Be specific about molecular subtypes and pathways
- Include prognostic implications
- Maximum 80 words
- No hedging or unnecessary qualifiers"""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        
        # 使用较低的 temperature 确保一致性
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,  # 降低随机性
            max_tokens=200,   # 限制输出长度
        )
        
        report = completion.choices[0].message.content
        
        # 清理可能的思考标签
        if report and '</think>' in report:
            report = report.split('</think>')[-1].strip()
        
        return report
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Error generating report."


def get_embedding(client, text):
    """步骤 3: Embedding"""
    try:
        completion = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return completion.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0.0] * 768


# =============================================================================
# 主流程
# =============================================================================

def generate_embeddings(cancer_type, data_dir, use_pathway=True, prompt_version='v1'):
    """
    生成病人 embedding 的主函数
    
    Args:
        cancer_type: 癌症类型 (e.g., 'LIHC', 'BRCA')
        data_dir: 数据目录
        use_pathway: 是否使用通路分析 (推荐 True)
        prompt_version: 'v1' (通路为主) 或 'v2' (通路+基因)
    """
    print(f"="*60)
    print(f"GeneNarrator Embedding Pipeline for {cancer_type}")
    print(f"Mode: {'Pathway-based (ssGSEA)' if use_pathway else 'Gene-based'}")
    print(f"="*60)
    
    if not API_KEY:
        print("Error: DASHSCOPE_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 路径设置
    expr_file = os.path.join(data_dir, f"TCGA-{cancer_type}.star_fpkm.tsv.gz")
    surv_file = os.path.join(data_dir, f"TCGA-{cancer_type}.survival.tsv.gz")
    output_path = os.path.join(data_dir, f"TCGA-{cancer_type}_embeddings.pt")
    report_path = os.path.join(data_dir, f"TCGA-{cancer_type}_reports.txt")
    pathway_scores_path = os.path.join(data_dir, f"TCGA-{cancer_type}_pathway_scores.csv")

    if os.path.exists(output_path):
        print(f"Embeddings already exist at {output_path}. Skipping generation.")
        return

    # ===== 数据加载 =====
    print(f"\n[1/5] Loading data...")
    survival_df = pd.read_csv(surv_file, sep='\t')
    expr_df = pd.read_csv(expr_file, sep='\t', index_col=0).T
    
    # 样本对齐
    common_samples = survival_df['sample'].isin(expr_df.index)
    survival_df = survival_df[common_samples]
    common_samples_ids = survival_df['sample'].values
    expr_df = expr_df.loc[common_samples_ids]
    
    print(f"Matched {len(expr_df)} samples.")
    
    # ===== 基因 ID 映射 =====
    print(f"\n[2/5] Converting Ensembl IDs to Gene Symbols...")
    gene_mapping = load_or_create_gene_mapping(data_dir)
    
    # 补充缺失映射
    all_gene_ids = expr_df.columns.tolist()
    missing_ids = [gid for gid in all_gene_ids if get_gene_symbol(gid, gene_mapping) == gid.split('.')[0]]
    
    if len(missing_ids) > len(all_gene_ids) * 0.3:
        print(f"Fetching {len(missing_ids)} missing gene symbols...")
        new_mappings = query_gene_symbols_batch(missing_ids)
        gene_mapping.update(new_mappings)
        
        # 保存映射
        mapping_file = os.path.join(data_dir, "gene_id_mapping.csv")
        if new_mappings:
            mapping_df = pd.DataFrame([
                {'ensembl_id': k.split('.')[0], 'gene_symbol': v} 
                for k, v in gene_mapping.items()
            ])
            mapping_df.to_csv(mapping_file, index=False)
    
    # 转换表达矩阵
    expr_df_symbols = convert_expr_to_symbols(expr_df, gene_mapping)
    print(f"Expression matrix: {expr_df_symbols.shape[0]} samples × {expr_df_symbols.shape[1]} genes (symbols)")
    
    # ===== ssGSEA 通路分析 =====
    pathway_scores = None
    if use_pathway:
        print(f"\n[3/5] Computing ssGSEA pathway scores...")
        hallmark_sets = load_hallmark_gene_sets(data_dir)
        
        if hallmark_sets:
            pathway_scores = compute_ssgsea(expr_df_symbols, hallmark_sets)
            
            # 保存通路得分
            if pathway_scores is not None:
                pathway_scores.to_csv(pathway_scores_path)
                print(f"Saved pathway scores to {pathway_scores_path}")
    else:
        print(f"\n[3/5] Skipping pathway analysis (gene-based mode)")
    
    # ===== 生成报告和 Embedding =====
    print(f"\n[4/5] Generating reports and embeddings...")
    
    embeddings = []
    reports = []
    
    for i in tqdm(range(len(expr_df)), desc="Processing patients"):
        patient_id = common_samples_ids[i]
        
        # 生成 Prompt
        if use_pathway and pathway_scores is not None:
            # 基于通路的 Prompt
            pathway_profile = analyze_pathway_profile(pathway_scores.iloc[i])
            
            if prompt_version == 'v2':
                # 增强版：通路 + Top 基因
                top_genes = expr_df_symbols.iloc[i].sort_values(ascending=False).head(5).index.tolist()
                prompt = generate_pathway_prompt_v2(patient_id, pathway_profile, top_genes, cancer_type)
            else:
                # 标准版：纯通路
                prompt = generate_pathway_prompt(patient_id, pathway_profile, cancer_type)
        else:
            # 降级：基于基因的 Prompt (旧方法)
            prompt = generate_gene_based_prompt(patient_id, expr_df_symbols.iloc[i])
        
        # LLM Reasoning
        report = generate_reasoning_report(client, prompt)
        reports.append(f"ID: {patient_id}\nPrompt: {prompt}\nReport: {report}\n{'-'*60}")
        
        # Embedding
        emb = get_embedding(client, report)
        embeddings.append(emb)
        
        # API 速率限制
        time.sleep(0.05)
            
    # ===== 保存结果 =====
    print(f"\n[5/5] Saving results...")
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    torch.save(embeddings_tensor, output_path)
    
    with open(report_path, "w") as f:
        for r in reports:
            f.write(r + "\n")
            
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"  Embeddings: {output_path}")
    print(f"  Reports: {report_path}")
    if pathway_scores is not None:
        print(f"  Pathway scores: {pathway_scores_path}")
    print(f"{'='*60}")


def generate_gene_based_prompt(patient_id, gene_series, top_n=10):
    """降级方案：基于基因的 Prompt (保持向后兼容)"""
    sorted_genes = gene_series.sort_values(ascending=False)
    top_genes = sorted_genes.head(top_n).index.tolist()
    genes_str = ", ".join(top_genes)
    
    prompt = (
        f"Patient ID: {patient_id}. "
        f"Highly expressed genes: {genes_str}. "
        "Based on these key genes, summarize the potential molecular subtype, "
        "biological pathways involved, and potential prognosis risk description. "
        "Keep the response concise (under 100 words)."
    )
    return prompt


# =============================================================================
# 外部验证数据处理
# =============================================================================

def generate_external_embeddings(dataset_name, data_dir, cancer_type=None):
    """
    为外部验证数据集生成 embedding
    使用相同的通路分析流程确保特征空间一致
    """
    print(f"Processing external dataset: {dataset_name}")
    
    expr_file = os.path.join(data_dir, f"{dataset_name}.star_fpkm.tsv.gz")
    surv_file = os.path.join(data_dir, f"{dataset_name}.survival.tsv.gz")
    output_path = os.path.join(data_dir, f"{dataset_name}_embeddings.pt")
    
    if os.path.exists(output_path):
        print(f"Embeddings already exist: {output_path}")
        return
    
    # 使用相同的流程
    generate_embeddings(
        cancer_type=dataset_name,  # 使用 dataset_name 作为标识
        data_dir=data_dir,
        use_pathway=True,
        prompt_version='v1'
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GeneNarrator Embedding Pipeline')
    parser.add_argument('--cancer', type=str, default='LIHC', help='Cancer type (e.g., LIHC, BRCA)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--no_pathway', action='store_true', help='Disable pathway analysis')
    parser.add_argument('--prompt_version', type=str, default='v1', choices=['v1', 'v2'])
    
    args = parser.parse_args()
    
    generate_embeddings(
        cancer_type=args.cancer,
        data_dir=args.data_dir,
        use_pathway=not args.no_pathway,
        prompt_version=args.prompt_version
    )
