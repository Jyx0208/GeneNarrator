"""Data loading and preprocessing (Methods): log2(x+1), Z-score, gene alignment."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Training (TCGA) -> external (ICGC/GEO) cohort pairs.
CANCER_PAIRS: Dict[str, Tuple[str, str]] = {
    "LIHC": ("TCGA-LIHC", "LIRI-JP"),
    "BRCA": ("TCGA-BRCA", "GSE20685"),
    "OV": ("TCGA-OV", "OV-AU"),
    "PAAD": ("TCGA-PAAD", "PACA-CA"),
}


def load_expression(path: str) -> pd.DataFrame:
    """Load a genes x samples expression matrix (tsv, optionally gzipped)."""
    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(path, sep="\t", index_col=0, compression=compression)
    if df.shape[0] < df.shape[1]:
        df = df.T
    return df.fillna(0.0)


def load_survival(path: str) -> pd.DataFrame:
    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(path, sep="\t", compression=compression)
    required = {"sample", "OS.time", "OS"}
    if not required.issubset(df.columns):
        raise ValueError(f"survival file {path} must contain columns {sorted(required)}")
    return df.set_index("sample")


def strip_ensembl_version(gene_id: str) -> str:
    """ENSG00000000003.15 -> ENSG00000000003."""
    if isinstance(gene_id, str) and gene_id.startswith("ENSG") and "." in gene_id:
        return gene_id.split(".")[0]
    return str(gene_id)


def select_high_variance_genes(
    train_expr: pd.DataFrame, n_genes: int = 1000
) -> List[str]:
    """Top-N genes by expression variance in the training cohort (Methods)."""
    var = train_expr.var(axis=1).sort_values(ascending=False)
    return list(var.head(n_genes).index)


def align_genes(
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    train_genes: List[str],
    symbol_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], int]:
    """Map the frozen training gene manifest onto the target cohort.

    ID formats are reconciled by exact match, version stripping, or an
    Ensembl-ID -> symbol mapping. Genes unavailable in the target cohort are
    kept in order as sentinels and set to the development mean (zero after
    standardisation) during loading. Returns (target gene list, n_missing).
    """
    test_exact = set(test_expr.index)
    test_by_stripped = {strip_ensembl_version(g): g for g in test_expr.index}

    target_genes: List[str] = []
    for gene in train_genes:
        mapped: Optional[str] = None
        if gene in test_exact:
            mapped = gene
        elif strip_ensembl_version(gene) in test_by_stripped:
            mapped = test_by_stripped[strip_ensembl_version(gene)]
        elif symbol_mapping and symbol_mapping.get(strip_ensembl_version(gene)) in test_exact:
            mapped = symbol_mapping[strip_ensembl_version(gene)]
        target_genes.append(mapped if mapped is not None else f"__MISSING_{gene}__")

    n_missing = sum(g.startswith("__MISSING_") for g in target_genes)
    return target_genes, n_missing


def standardize(
    expr: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log2(x + 1) followed by Z-score standardisation (Methods).

    Statistics must come from the training cohort only; external cohorts use
    the training-derived mean/std verbatim.
    """
    logged = np.log2(np.maximum(np.asarray(expr, dtype=np.float32), 0.0) + 1.0)
    if mean is None:
        mean = logged.mean(axis=0)
        std = logged.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (logged - mean) / std, mean, std


def build_dataset(
    expr: pd.DataFrame,
    surv: pd.DataFrame,
    genes: List[str],
    embeddings: np.ndarray,
    sample_ids: List[str],
    norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Assemble (samples x genes) gene features, text embeddings, OS data.

    Missing frozen features are set to the development mean (zero after
    standardisation), preserving input dimension and feature order.
    """
    common = sorted(set(expr.columns) & set(surv.index) & set(sample_ids))
    id_to_idx = {s: i for i, s in enumerate(sample_ids)}
    missing_mask = np.array([g.startswith("__MISSING_") for g in genes], dtype=bool)

    gene_rows, text_rows, time_list, event_list = [], [], [], []
    for sample in common:
        gene_rows.append(expr.reindex(genes)[sample].fillna(0.0).values)
        text_rows.append(embeddings[id_to_idx[sample]])
        time_list.append(max(1.0, float(surv.loc[sample, "OS.time"])))
        event_list.append(float(surv.loc[sample, "OS"]))

    gene = np.asarray(gene_rows, dtype=np.float32)
    if norm_stats is not None:
        mean, std = norm_stats
        gene = (np.log2(np.maximum(gene, 0.0) + 1.0) - mean) / std
    else:
        gene, mean, std = standardize(gene)
        norm_stats = (mean, std)
    if missing_mask.any():
        gene[:, missing_mask] = 0.0

    return {
        "gene": gene,
        "text": np.asarray(text_rows, dtype=np.float32),
        "time": np.asarray(time_list, dtype=np.float32),
        "event": np.asarray(event_list, dtype=np.float32),
        "sample_ids": common,
        "genes": genes,
        "norm_stats": norm_stats,
    }


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        emb = data.get("embeddings", data.get("embedding"))
        ids = data.get("patient_ids", data.get("sample_ids", []))
        return emb.numpy(), list(ids)
    return data.numpy(), []


def load_cancer_pair(
    cancer_type: str,
    data_dir: str = "data",
    n_genes: int = 1000,
    symbol_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict]:
    """Load the training/external pair for one cancer type (Methods)."""
    train_name, test_name = CANCER_PAIRS[cancer_type.upper()]
    train_expr = load_expression(os.path.join(data_dir, f"{train_name}.star_fpkm.tsv.gz"))
    test_expr = load_expression(os.path.join(data_dir, f"{test_name}.expression.tsv.gz"))

    train_genes = select_high_variance_genes(train_expr, n_genes=n_genes)
    test_genes, n_missing = align_genes(train_expr, test_expr, train_genes, symbol_mapping)

    train_surv = load_survival(os.path.join(data_dir, f"{train_name}.survival.tsv.gz"))
    test_surv = load_survival(os.path.join(data_dir, f"{test_name}.survival.tsv.gz"))

    train_emb, train_ids = load_embeddings(
        os.path.join(data_dir, f"{train_name}_embeddings_v5.pt")
    )
    test_emb, test_ids = load_embeddings(
        os.path.join(data_dir, f"{test_name}_embeddings_v5.pt")
    )

    train_data = build_dataset(train_expr, train_surv, train_genes, train_emb, train_ids)
    test_data = build_dataset(
        test_expr,
        test_surv,
        test_genes,
        test_emb,
        test_ids,
        norm_stats=train_data["norm_stats"],
    )
    info = {
        "train_name": train_name,
        "test_name": test_name,
        "n_genes": len(train_genes),
        "n_missing_external_features": n_missing,
    }
    return train_data, test_data, info
