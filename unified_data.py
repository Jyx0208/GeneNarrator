"""
统一数据处理模块 - GeneNarrator-AFT
====================================

提供一致的数据加载、基因对齐、标准化流程，确保所有实验结果可比较。

核心原则：
1. 强制基因对齐 - 如果无法对齐则报错，不允许退化策略
2. 统一标准化流程 - 所有数据使用相同的 log1p + z-score 标准化
3. 统一文件命名 - 嵌入文件统一使用 _embeddings_v5.pt 格式
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import warnings

# 获取数据目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str                    # 数据集名称
    expr_file: str               # 表达数据文件
    surv_file: str               # 生存数据文件
    embed_file: str              # 嵌入文件
    report_file: str             # 报告文件
    is_training: bool = True     # 是否为训练集
    gene_id_type: str = 'ensembl'  # 基因 ID 类型: 'ensembl', 'symbol', 'illumina'
    
    
# 预定义的数据集配置
DATASET_CONFIGS = {
    # ===== 训练集 (TCGA) =====
    'TCGA-LIHC': DatasetConfig(
        name='TCGA-LIHC',
        expr_file='TCGA-LIHC.star_fpkm.tsv.gz',
        surv_file='TCGA-LIHC.survival.tsv.gz',
        embed_file='TCGA-LIHC_embeddings_v5.pt',
        report_file='TCGA-LIHC_reports_v5.txt',
        is_training=True,
        gene_id_type='ensembl'
    ),
    'TCGA-BRCA': DatasetConfig(
        name='TCGA-BRCA',
        expr_file='TCGA-BRCA_symbols.star_fpkm.tsv.gz',  # 使用 symbol 版本
        surv_file='TCGA-BRCA.survival.tsv.gz',
        embed_file='TCGA-BRCA_embeddings_v5.pt',  # 统一命名
        report_file='TCGA-BRCA_reports_v5.txt',
        is_training=True,
        gene_id_type='symbol'
    ),
    'TCGA-OV': DatasetConfig(
        name='TCGA-OV',
        expr_file='TCGA-OV.star_fpkm.tsv.gz',
        surv_file='TCGA-OV.survival.tsv.gz',
        embed_file='TCGA-OV_embeddings_v5.pt',
        report_file='TCGA-OV_reports_v5.txt',
        is_training=True,
        gene_id_type='ensembl'
    ),
    'TCGA-PAAD': DatasetConfig(
        name='TCGA-PAAD',
        expr_file='TCGA-PAAD.star_fpkm.tsv.gz',
        surv_file='TCGA-PAAD.survival.tsv.gz',
        embed_file='TCGA-PAAD_embeddings_v5.pt',
        report_file='TCGA-PAAD_reports_v5.txt',
        is_training=True,
        gene_id_type='ensembl'
    ),
    'TCGA-PRAD': DatasetConfig(
        name='TCGA-PRAD',
        expr_file='TCGA-PRAD.star_fpkm.tsv.gz',
        surv_file='TCGA-PRAD.survival.tsv.gz',
        embed_file='TCGA-PRAD_embeddings_v5.pt',
        report_file='TCGA-PRAD_reports_v5.txt',
        is_training=True,
        gene_id_type='ensembl'
    ),
    
    # ===== 外部验证集 =====
    'LIRI-JP': DatasetConfig(
        name='LIRI-JP',
        expr_file='LIRI-JP.star_fpkm.tsv.gz',
        surv_file='LIRI-JP.survival.tsv.gz',
        embed_file='LIRI-JP_embeddings_v5.pt',
        report_file='LIRI-JP_reports_v5.txt',
        is_training=False,
        gene_id_type='ensembl'
    ),
    'GSE20685': DatasetConfig(
        name='GSE20685',
        expr_file='GSE20685.expression.tsv.gz',
        surv_file='GSE20685.survival.tsv.gz',
        embed_file='GSE20685_embeddings_v5.pt',  # 统一命名
        report_file='GSE20685_reports_v5.txt',
        is_training=False,
        gene_id_type='symbol'
    ),
    'OV-AU': DatasetConfig(
        name='OV-AU',
        expr_file='OV-AU.star_fpkm.tsv.gz',
        surv_file='OV-AU.survival.tsv.gz',
        embed_file='OV-AU_embeddings_v5.pt',
        report_file='OV-AU_reports_v5.txt',
        is_training=False,
        gene_id_type='ensembl'
    ),
    'PACA-CA': DatasetConfig(
        name='PACA-CA',
        expr_file='PACA-CA.star_fpkm.tsv.gz',
        surv_file='PACA-CA.survival.tsv.gz',
        embed_file='PACA-CA_embeddings_v5.pt',
        report_file='PACA-CA_reports_v5.txt',
        is_training=False,
        gene_id_type='ensembl'
    ),
    'PRAD-CA': DatasetConfig(
        name='PRAD-CA',
        expr_file='PRAD-CA.star_fpkm.tsv.gz',
        surv_file='PRAD-CA.survival.tsv.gz',
        embed_file='PRAD-CA_embeddings_v5.pt',
        report_file='PRAD-CA_reports_v5.txt',
        is_training=False,
        gene_id_type='symbol'  # PRAD-CA 使用 Gene Symbol
    ),
}

# 不兼容的基因 ID 类型组合（需要转换）
INCOMPATIBLE_ID_PAIRS = {
    ('ensembl', 'illumina'),  # STAD: Ensembl vs Illumina 探针
    ('illumina', 'ensembl'),
}

# 癌种配对关系
CANCER_PAIRS = {
    'LIHC': ('TCGA-LIHC', 'LIRI-JP'),
    'BRCA': ('TCGA-BRCA', 'GSE20685'),
    'OV': ('TCGA-OV', 'OV-AU'),  # 卵巢癌：TCGA 训练，ICGC 澳大利亚验证
    'PAAD': ('TCGA-PAAD', 'PACA-CA'),
    'PRAD': ('TCGA-PRAD', 'PRAD-CA'),
}


class GeneAlignmentError(Exception):
    """基因对齐失败异常"""
    pass


class DataLoadingError(Exception):
    """数据加载失败异常"""
    pass


def strip_ensembl_version(gene_id: str) -> str:
    """
    去除 Ensembl 基因 ID 的版本号后缀
    例如: ENSG00000000003.15 -> ENSG00000000003
    """
    if isinstance(gene_id, str) and gene_id.startswith('ENSG') and '.' in gene_id:
        return gene_id.split('.')[0]
    return str(gene_id)


def create_gene_id_mapping(gene_ids: List[str]) -> Dict[str, str]:
    """
    创建基因 ID 映射表（去版本号 -> 原始 ID）
    """
    mapping = {}
    for gid in gene_ids:
        stripped = strip_ensembl_version(gid)
        if stripped not in mapping:  # 保留第一个遇到的
            mapping[stripped] = gid
    return mapping


def load_gene_mapping(data_dir: str) -> Dict[str, str]:
    """
    加载 Ensembl ID -> Gene Symbol 映射表
    """
    mapping_file = os.path.join(data_dir, "gene_id_mapping.csv")
    
    if not os.path.exists(mapping_file):
        return {}
    
    mapping = {}
    df = pd.read_csv(mapping_file)
    for _, row in df.iterrows():
        ensembl_id = str(row.get('ensembl_id', ''))
        symbol = str(row.get('gene_symbol', ''))
        if ensembl_id and symbol and symbol != 'nan':
            # 存储双向映射
            mapping[ensembl_id] = symbol
            # 也存储带版本号的映射
            if not ensembl_id.startswith('ENSG'):
                continue
    
    return mapping


def convert_ensembl_to_symbol(
    expr_df: pd.DataFrame, 
    gene_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    将 Ensembl ID 转换为 Gene Symbol
    
    Args:
        expr_df: 表达矩阵 (基因 x 样本)
        gene_mapping: Ensembl ID -> Symbol 映射
        
    Returns:
        转换后的表达矩阵
    """
    new_index = []
    for gene_id in expr_df.index:
        stripped = strip_ensembl_version(gene_id)
        if stripped in gene_mapping:
            new_index.append(gene_mapping[stripped])
        elif gene_id in gene_mapping:
            new_index.append(gene_mapping[gene_id])
        else:
            # 保留原始 ID
            new_index.append(stripped)
    
    result = expr_df.copy()
    result.index = new_index
    
    # 处理重复的 Symbol (保留方差最大的)
    if result.index.duplicated().any():
        # 计算每行方差
        var_series = result.var(axis=1)
        result['_var'] = var_series
        result = result.sort_values('_var', ascending=False)
        result = result[~result.index.duplicated(keep='first')]
        result = result.drop('_var', axis=1)
    
    return result


def align_genes(
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    n_genes: int = 1000,
    min_common_genes: int = 500,
    strict: bool = True,
    train_id_type: str = 'ensembl',
    test_id_type: str = 'ensembl',
    gene_mapping: Dict[str, str] = None
) -> Tuple[List[str], List[str]]:
    """
    对齐训练集和测试集的基因
    
    Args:
        train_expr: 训练集表达矩阵 (基因 x 样本)
        test_expr: 测试集表达矩阵 (基因 x 样本)
        n_genes: 选择的高变异基因数量
        min_common_genes: 最少共同基因数量（低于此值则报错）
        strict: 是否启用严格模式（失败时报错而非降级）
        train_id_type: 训练集基因 ID 类型
        test_id_type: 测试集基因 ID 类型
        gene_mapping: Ensembl ID -> Symbol 映射
        
    Returns:
        (train_genes, test_genes): 对齐后的基因列表
        
    Raises:
        GeneAlignmentError: 当共同基因数量不足时
    """
    # 检查是否需要 ID 转换
    if (train_id_type, test_id_type) in INCOMPATIBLE_ID_PAIRS:
        error_msg = (
            f"基因 ID 类型不兼容！\n"
            f"  训练集 ID 类型: {train_id_type}\n"
            f"  测试集 ID 类型: {test_id_type}\n"
            f"  这种组合需要特殊的探针注释文件进行转换。\n"
            f"  建议:\n"
            f"    - 对于 Illumina 微阵列数据，需要使用探针注释文件转换为 Gene Symbol\n"
            f"    - 或者使用相同平台的数据进行验证"
        )
        if strict:
            raise GeneAlignmentError(error_msg)
        else:
            warnings.warn(error_msg)
            return [], []
    
    # 如果一个是 Ensembl，另一个是 Symbol，尝试转换
    train_expr_conv = train_expr
    test_expr_conv = test_expr
    
    if train_id_type == 'ensembl' and test_id_type == 'symbol' and gene_mapping:
        print(f"    转换训练集 Ensembl ID -> Gene Symbol...")
        train_expr_conv = convert_ensembl_to_symbol(train_expr, gene_mapping)
        
    if train_id_type == 'symbol' and test_id_type == 'ensembl' and gene_mapping:
        print(f"    转换测试集 Ensembl ID -> Gene Symbol...")
        test_expr_conv = convert_ensembl_to_symbol(test_expr, gene_mapping)
    
    train_genes = set(train_expr_conv.index)
    test_genes = set(test_expr_conv.index)
    
    # Step 1: 直接匹配
    common_genes = train_genes & test_genes
    
    # Step 2: 如果直接匹配不够，尝试去版本号匹配
    if len(common_genes) < min_common_genes:
        print(f"    直接匹配: {len(common_genes)} 基因，尝试去版本号匹配...")
        
        # 创建映射
        train_mapping = create_gene_id_mapping(train_genes)
        test_mapping = create_gene_id_mapping(test_genes)
        
        # 找共同的去版本号 ID
        common_stripped = set(train_mapping.keys()) & set(test_mapping.keys())
        
        if len(common_stripped) >= min_common_genes:
            print(f"    去版本号匹配: {len(common_stripped)} 基因")
            
            # 获取训练集的原始 ID（用于方差排序）
            train_common_genes = [train_mapping[g] for g in common_stripped]
            
            # 按训练集方差排序
            train_common_expr = train_expr_conv.loc[train_common_genes]
            var_series = train_common_expr.var(axis=1).sort_values(ascending=False)
            top_train_genes = list(var_series.head(n_genes).index)
            
            # 转换为测试集的对应基因
            top_test_genes = []
            for tg in top_train_genes:
                stripped = strip_ensembl_version(tg)
                if stripped in test_mapping:
                    top_test_genes.append(test_mapping[stripped])
                else:
                    # 这不应该发生，但以防万一
                    raise GeneAlignmentError(f"基因 {tg} 无法映射到测试集")
            
            return top_train_genes, top_test_genes
        else:
            common_genes = common_stripped
    
    # Step 3: 检查是否满足最低要求
    if len(common_genes) < min_common_genes:
        error_msg = (
            f"基因对齐失败！\n"
            f"  训练集基因数: {len(train_genes)}\n"
            f"  测试集基因数: {len(test_genes)}\n"
            f"  共同基因数: {len(common_genes)}\n"
            f"  最低要求: {min_common_genes}\n"
            f"  训练集 ID 类型: {train_id_type}\n"
            f"  测试集 ID 类型: {test_id_type}\n"
            f"  可能原因:\n"
            f"    - 不同平台 (RNA-seq vs 微阵列)\n"
            f"    - 不同基因 ID 格式 (Ensembl vs Gene Symbol)\n"
            f"  建议:\n"
            f"    - 将基因 ID 统一转换为 Gene Symbol\n"
            f"    - 或使用相同平台的数据"
        )
        if strict:
            raise GeneAlignmentError(error_msg)
        else:
            warnings.warn(error_msg)
            # 非严格模式：返回空列表
            return [], []
    
    # Step 4: 对于直接匹配成功的情况
    common_genes_list = list(common_genes)
    train_common_expr = train_expr_conv.loc[common_genes_list]
    var_series = train_common_expr.var(axis=1).sort_values(ascending=False)
    top_genes = list(var_series.head(n_genes).index)
    
    print(f"    基因对齐成功: {len(top_genes)} 高变异基因 (共同基因: {len(common_genes_list)})")
    
    return top_genes, top_genes  # 直接匹配时两者相同


def parse_sample_ids_from_report(report_file: str) -> List[str]:
    """
    从报告文件解析样本 ID
    
    支持的格式:
    - [1] Patient: TCGA-XX-XXXX | Hoshida-S1
    - [1] Patient: TCGA-XX-XXXX
    """
    sample_ids = []
    
    if not os.path.exists(report_file):
        raise DataLoadingError(f"报告文件不存在: {report_file}")
    
    with open(report_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 'Patient:' in line:
                try:
                    # 提取 Patient: 后面的内容
                    parts = line.split('Patient:')
                    if len(parts) > 1:
                        # 处理可能的分隔符: | 或换行
                        sid_part = parts[1].strip()
                        # 检查是否有 | 分隔符
                        if '|' in sid_part:
                            sid = sid_part.split('|')[0].strip()
                        else:
                            # 没有 | 时，取整个内容或到下一个空格
                            sid = sid_part.split()[0].strip() if ' ' in sid_part else sid_part
                        
                        if sid:
                            sample_ids.append(sid)
                except Exception as e:
                    warnings.warn(f"解析样本 ID 失败: {line}, 错误: {e}")
                    continue
    
    if len(sample_ids) == 0:
        raise DataLoadingError(f"未能从报告文件解析出任何样本 ID: {report_file}")
    
    return sample_ids


def load_expression_data(expr_file: str) -> pd.DataFrame:
    """
    加载表达数据，统一为 基因 x 样本 格式
    """
    if not os.path.exists(expr_file):
        raise DataLoadingError(f"表达数据文件不存在: {expr_file}")
    
    # 自动检测压缩格式
    compression = 'gzip' if expr_file.endswith('.gz') else None
    
    # 读取数据
    df = pd.read_csv(expr_file, sep='\t', index_col=0, compression=compression)
    
    # 判断格式：如果行数远大于列数，可能是 基因 x 样本
    # 如果列数远大于行数，可能是 样本 x 基因，需要转置
    # 通常基因数量在 20000-60000，样本数量在 100-2000
    
    if df.shape[0] < df.shape[1]:
        # 行数 < 列数，可能是 样本 x 基因，需要转置
        print(f"    检测到 样本 x 基因 格式 ({df.shape[0]} x {df.shape[1]})，转置中...")
        df = df.T
    
    # 处理缺失值
    df = df.fillna(0)
    
    return df


def load_survival_data(surv_file: str) -> pd.DataFrame:
    """
    加载生存数据
    """
    if not os.path.exists(surv_file):
        raise DataLoadingError(f"生存数据文件不存在: {surv_file}")
    
    compression = 'gzip' if surv_file.endswith('.gz') else None
    
    df = pd.read_csv(surv_file, sep='\t', compression=compression)
    
    # 确保必要的列存在
    required_cols = ['sample', 'OS.time', 'OS']
    for col in required_cols:
        if col not in df.columns:
            raise DataLoadingError(f"生存数据缺少必要列: {col}")
    
    return df


def load_embeddings(embed_file: str, fallback_files: List[str] = None) -> Tuple[torch.Tensor, List[str]]:
    """
    加载嵌入数据，支持回退到备选文件
    
    Returns:
        (embeddings_tensor, patient_ids_list)
    """
    def _load_embed_file(path):
        data = torch.load(path, weights_only=False)
        if isinstance(data, dict):
            embeddings = data.get('embeddings', data.get('embedding'))
            patient_ids = data.get('patient_ids', data.get('sample_ids', []))
            return embeddings, patient_ids
        else:
            # 旧格式：只有张量
            return data, []
    
    # 尝试主文件
    if os.path.exists(embed_file):
        return _load_embed_file(embed_file)
    
    # 尝试回退文件
    if fallback_files:
        for fallback in fallback_files:
            if os.path.exists(fallback):
                warnings.warn(f"使用回退嵌入文件: {fallback}")
                return _load_embed_file(fallback)
    
    raise DataLoadingError(f"嵌入文件不存在: {embed_file}")


def standardize_expression(
    expr_data: np.ndarray,
    fit_data: np.ndarray = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    统一的表达数据标准化流程: log1p + z-score
    
    Args:
        expr_data: 表达数据 (样本 x 基因)
        fit_data: 用于计算均值和标准差的数据（通常是训练集）
                  如果为 None，使用 expr_data 自身
    
    Returns:
        (标准化后的数据, {'mean': 均值, 'std': 标准差})
    """
    # Step 1: log1p 变换（处理可能的负值）
    expr_log = np.log1p(np.abs(expr_data).astype(np.float32))
    
    # Step 2: 计算统计量
    if fit_data is not None:
        fit_log = np.log1p(np.abs(fit_data).astype(np.float32))
        mean = fit_log.mean(axis=0)
        std = fit_log.std(axis=0)
    else:
        mean = expr_log.mean(axis=0)
        std = expr_log.std(axis=0)
    
    # 避免除零
    std = np.where(std < 1e-8, 1.0, std)
    
    # Step 3: Z-score 标准化
    expr_normalized = (expr_log - mean) / std
    
    return expr_normalized, {'mean': mean, 'std': std}


class UnifiedDataLoader:
    """
    统一数据加载器
    
    使用示例:
    ```python
    loader = UnifiedDataLoader(data_dir='./data')
    
    # 加载单个数据集
    train_data = loader.load_dataset('TCGA-LIHC')
    
    # 加载配对的训练-测试数据集（自动基因对齐）
    train_data, test_data, gene_info = loader.load_paired_datasets('LIHC')
    ```
    """
    
    def __init__(self, data_dir: str = None, n_genes: int = 1000, strict_alignment: bool = True):
        """
        Args:
            data_dir: 数据目录
            n_genes: 选择的高变异基因数量
            strict_alignment: 是否启用严格基因对齐模式
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.n_genes = n_genes
        self.strict_alignment = strict_alignment
        
    def _get_full_path(self, filename: str) -> str:
        """获取完整路径"""
        return os.path.join(self.data_dir, filename)
    
    def _get_embed_fallbacks(self, config: DatasetConfig) -> List[str]:
        """获取嵌入文件的回退路径列表"""
        base_name = config.name
        fallbacks = [
            self._get_full_path(f"{base_name}_embeddingsv5.pt"),  # 无下划线版本
            self._get_full_path(f"{base_name}_embeddings.pt"),    # 无版本号
        ]
        return fallbacks
    
    def load_dataset(
        self,
        dataset_name: str,
        genes: List[str] = None,
        norm_stats: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        加载单个数据集
        
        Args:
            dataset_name: 数据集名称 (如 'TCGA-LIHC', 'LIRI-JP')
            genes: 指定的基因列表（用于测试集对齐）
            norm_stats: 标准化统计量（用于测试集）
            
        Returns:
            {
                'gene': 基因表达数据 (n_samples, n_genes),
                'text': 文本嵌入 (n_samples, embed_dim),
                'time': 生存时间 (n_samples,),
                'event': 事件标志 (n_samples,),
                'sample_ids': 样本 ID 列表,
                'genes': 使用的基因列表,
                'norm_stats': 标准化统计量
            }
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset_name}. 可用: {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[dataset_name]
        
        print(f"  加载数据集: {dataset_name}")
        
        # 1. 加载表达数据
        expr_path = self._get_full_path(config.expr_file)
        print(f"    表达数据: {config.expr_file}")
        expr_df = load_expression_data(expr_path)
        
        # 2. 加载生存数据
        surv_path = self._get_full_path(config.surv_file)
        print(f"    生存数据: {config.surv_file}")
        surv_df = load_survival_data(surv_path)
        surv_df = surv_df.set_index('sample')
        
        # 3. 加载嵌入
        embed_path = self._get_full_path(config.embed_file)
        fallbacks = self._get_embed_fallbacks(config)
        print(f"    嵌入数据: {config.embed_file}")
        embeddings, embed_patient_ids = load_embeddings(embed_path, fallbacks)
        
        # 4. 获取样本 ID（优先使用嵌入文件中的 patient_ids）
        if embed_patient_ids and len(embed_patient_ids) == len(embeddings):
            sample_ids = embed_patient_ids
        else:
            # 回退到报告文件解析
            report_path = self._get_full_path(config.report_file)
            sample_ids = parse_sample_ids_from_report(report_path)
        
        # 5. 确定使用的基因
        if genes is not None:
            # 使用指定的基因（对齐模式）
            available_genes = [g for g in genes if g in expr_df.index]
            
            if len(available_genes) < len(genes) * 0.5:
                # 尝试去版本号匹配
                gene_mapping = create_gene_id_mapping(expr_df.index)
                genes_stripped = {strip_ensembl_version(g): g for g in genes}
                
                available_genes = []
                for g_strip, g_orig in genes_stripped.items():
                    if g_strip in gene_mapping:
                        available_genes.append(gene_mapping[g_strip])
                
                if len(available_genes) < len(genes) * 0.5:
                    if self.strict_alignment:
                        raise GeneAlignmentError(
                            f"数据集 {dataset_name} 无法与指定基因对齐\n"
                            f"  指定基因: {len(genes)}\n"
                            f"  可用基因: {len(available_genes)}"
                        )
                    else:
                        warnings.warn(f"基因对齐不完整: {len(available_genes)}/{len(genes)}")
            
            use_genes = available_genes
        else:
            # 使用高变异基因
            var_series = expr_df.var(axis=1).sort_values(ascending=False)
            use_genes = list(var_series.head(self.n_genes).index)
        
        # 6. 找共同样本
        expr_samples = set(expr_df.columns)
        surv_samples = set(surv_df.index)
        embed_samples = set(sample_ids)
        
        common_samples = list(expr_samples & surv_samples & embed_samples)
        
        if len(common_samples) < 10:
            raise DataLoadingError(
                f"共同样本不足: {len(common_samples)}\n"
                f"  表达数据样本: {len(expr_samples)}\n"
                f"  生存数据样本: {len(surv_samples)}\n"
                f"  嵌入数据样本: {len(embed_samples)}"
            )
        
        # 7. 构建数据
        sample_to_idx = {s: i for i, s in enumerate(sample_ids)}
        
        expr_list, text_list, time_list, event_list = [], [], [], []
        valid_samples = []
        
        for sample in common_samples:
            if sample in sample_to_idx:
                idx = sample_to_idx[sample]
                
                # 提取表达数据
                expr_vec = expr_df.loc[use_genes, sample].fillna(0).values
                expr_list.append(expr_vec)
                
                # 提取嵌入
                text_list.append(embeddings[idx].numpy())
                
                # 提取生存信息
                time_val = max(1.0, surv_df.loc[sample, 'OS.time'])
                time_list.append(time_val)
                event_list.append(surv_df.loc[sample, 'OS'])
                
                valid_samples.append(sample)
        
        # 8. 转换为数组
        gene_data = np.array(expr_list, dtype=np.float32)
        text_data = np.array(text_list, dtype=np.float32)
        time_data = np.array(time_list, dtype=np.float32)
        event_data = np.array(event_list, dtype=np.float32)
        
        # 9. 标准化
        if norm_stats is not None:
            # 使用提供的统计量（测试集）
            gene_log = np.log1p(np.abs(gene_data))
            gene_normalized = (gene_log - norm_stats['mean']) / norm_stats['std']
            stats = norm_stats
        else:
            # 计算并保存统计量（训练集）
            gene_normalized, stats = standardize_expression(gene_data)
        
        print(f"    样本数: {len(valid_samples)}")
        print(f"    基因数: {len(use_genes)}")
        print(f"    事件率: {event_data.mean()*100:.1f}%")
        
        return {
            'gene': gene_normalized,
            'text': text_data,
            'time': time_data,
            'event': event_data,
            'sample_ids': valid_samples,
            'genes': use_genes,
            'norm_stats': stats
        }
    
    def load_paired_datasets(
        self,
        cancer_type: str
    ) -> Tuple[Dict, Dict, Dict]:
        """
        加载配对的训练-测试数据集，自动进行基因对齐
        
        Args:
            cancer_type: 癌种代码 (如 'LIHC', 'BRCA')
            
        Returns:
            (train_data, test_data, info)
            
            info 包含:
            - train_genes: 训练集基因列表
            - test_genes: 测试集基因列表  
            - n_common_genes: 共同基因数量
        """
        if cancer_type not in CANCER_PAIRS:
            raise ValueError(f"未知癌种: {cancer_type}. 可用: {list(CANCER_PAIRS.keys())}")
        
        train_name, test_name = CANCER_PAIRS[cancer_type]
        train_config = DATASET_CONFIGS[train_name]
        test_config = DATASET_CONFIGS[test_name]
        
        print(f"\n{'='*60}")
        print(f"加载配对数据集: {cancer_type}")
        print(f"  训练集: {train_name} (ID类型: {train_config.gene_id_type})")
        print(f"  测试集: {test_name} (ID类型: {test_config.gene_id_type})")
        print(f"{'='*60}")
        
        # 1. 加载基因 ID 映射（如果需要转换）
        gene_mapping = None
        need_conversion = train_config.gene_id_type != test_config.gene_id_type
        if need_conversion:
            gene_mapping = load_gene_mapping(self.data_dir)
            if gene_mapping:
                print(f"  已加载 {len(gene_mapping)} 个基因 ID 映射")
        
        # 2. 先加载表达数据用于基因对齐
        train_expr = load_expression_data(self._get_full_path(train_config.expr_file))
        test_expr = load_expression_data(self._get_full_path(test_config.expr_file))
        
        # 3. 基因对齐
        print("\n[Step 1] 基因对齐...")
        
        # 对于需要转换的情况，我们需要跟踪原始基因 ID
        train_genes_original = None
        
        if need_conversion and gene_mapping:
            # 转换训练集（如果需要）
            if train_config.gene_id_type == 'ensembl' and test_config.gene_id_type == 'symbol':
                print(f"    转换训练集 Ensembl ID -> Gene Symbol...")
                train_expr_conv = convert_ensembl_to_symbol(train_expr, gene_mapping)
                
                # 找共同基因（使用转换后的 Symbol）
                common_genes = set(train_expr_conv.index) & set(test_expr.index)
                
                if len(common_genes) >= 500:
                    # 按方差排序
                    var_series = train_expr_conv.loc[list(common_genes)].var(axis=1).sort_values(ascending=False)
                    top_genes_symbol = list(var_series.head(self.n_genes).index)
                    
                    # 反向映射回原始 Ensembl ID
                    symbol_to_ensembl = {v: k for k, v in gene_mapping.items()}
                    train_genes_original = []
                    for symbol in top_genes_symbol:
                        if symbol in symbol_to_ensembl:
                            train_genes_original.append(symbol_to_ensembl[symbol])
                        else:
                            # 如果找不到反向映射，尝试在原始数据中搜索
                            for orig_id in train_expr.index:
                                stripped = strip_ensembl_version(orig_id)
                                if stripped in gene_mapping and gene_mapping[stripped] == symbol:
                                    train_genes_original.append(orig_id)
                                    break
                    
                    train_genes = train_genes_original
                    test_genes = top_genes_symbol
                    print(f"    基因对齐成功: {len(train_genes)} 高变异基因 (共同基因: {len(common_genes)})")
                else:
                    raise GeneAlignmentError(
                        f"基因对齐失败！\n"
                        f"  转换后共同基因数: {len(common_genes)}\n"
                        f"  最低要求: 500"
                    )
            else:
                # 其他转换情况，使用标准流程
                train_genes, test_genes = align_genes(
                    train_expr, test_expr,
                    n_genes=self.n_genes,
                    min_common_genes=500,
                    strict=self.strict_alignment,
                    train_id_type=train_config.gene_id_type,
                    test_id_type=test_config.gene_id_type,
                    gene_mapping=gene_mapping
                )
        else:
            # 不需要转换，使用标准对齐
            train_genes, test_genes = align_genes(
                train_expr, test_expr,
                n_genes=self.n_genes,
                min_common_genes=500,
                strict=self.strict_alignment,
                train_id_type=train_config.gene_id_type,
                test_id_type=test_config.gene_id_type,
                gene_mapping=gene_mapping
            )
        
        # 4. 加载训练数据
        print("\n[Step 2] 加载训练数据...")
        train_data = self.load_dataset(train_name, genes=train_genes)
        
        # 5. 加载测试数据（使用训练集的统计量）
        print("\n[Step 3] 加载测试数据...")
        test_data = self.load_dataset(
            test_name,
            genes=test_genes,
            norm_stats=train_data['norm_stats']
        )
        
        info = {
            'train_genes': train_genes,
            'test_genes': test_genes,
            'n_common_genes': len(train_genes),
            'train_name': train_name,
            'test_name': test_name,
            'train_id_type': train_config.gene_id_type,
            'test_id_type': test_config.gene_id_type
        }
        
        print(f"\n[完成] 数据加载成功")
        print(f"  训练集: {len(train_data['gene'])} 样本")
        print(f"  测试集: {len(test_data['gene'])} 样本")
        print(f"  基因数: {len(train_genes)}")
        
        return train_data, test_data, info


def set_random_seed(seed: int = 42):
    """
    设置全局随机种子，确保可重复性
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# 便捷函数
# =============================================================================

def load_cancer_data(
    cancer_type: str,
    data_dir: str = None,
    n_genes: int = 1000,
    strict: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    便捷函数：加载指定癌种的配对数据
    
    示例:
    ```python
    train_data, test_data, info = load_cancer_data('LIHC')
    ```
    """
    loader = UnifiedDataLoader(
        data_dir=data_dir,
        n_genes=n_genes,
        strict_alignment=strict
    )
    return loader.load_paired_datasets(cancer_type)


if __name__ == '__main__':
    # 测试数据加载
    print("测试统一数据加载模块\n")
    
    try:
        train_data, test_data, info = load_cancer_data('LIHC', strict=True)
        print("\n✓ LIHC 数据加载成功")
    except Exception as e:
        print(f"\n✗ LIHC 数据加载失败: {e}")

