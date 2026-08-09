"""Semantic narrative generation (Methods): ssGSEA, V5 prompt, embeddings.

LLM calls require the DashScope-compatible OpenAI endpoint and a valid
``DASHSCOPE_API_KEY`` environment variable. No API credentials are stored
in this repository.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CHAT_MODEL = "qwen3-max"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
HALLMARK_RELEASE = "7.5.1"

_CLIENT = None


def _client():
    from openai import OpenAI

    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _CLIENT


# ---------------------------------------------------------------------------
# ssGSEA
# ---------------------------------------------------------------------------


def load_hallmark_gene_sets(
    gmt_path: str = "data/h.all.v7.5.1.symbols.gmt",
) -> Dict[str, set]:
    """Parse the MSigDB HALLMARK v7.5.1 GMT file into {pathway: genes}."""
    gene_sets: Dict[str, set] = {}
    with open(gmt_path, "r") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0].replace("HALLMARK_", "")] = set(parts[2:])
    return gene_sets


def compute_ssgsea(
    expr: pd.DataFrame, gene_sets: Dict[str, set], n_jobs: int = 4
) -> pd.DataFrame:
    """Rank-normalized single-sample GSEA scores (gseapy).

    ``expr``: genes x samples DataFrame. Returns a samples x pathways
    DataFrame of NES values (Methods).
    """
    import gseapy as gp

    result = gp.ssgsea(
        data=expr,
        gene_sets=gene_sets,
        outdir=None,
        sample_norm_method="rank",
        threads=n_jobs,
        no_plot=True,
        verbose=False,
    )
    return result.res2d.pivot(index="Name", columns="Term", values="NES")


# ---------------------------------------------------------------------------
# V5 prompt protocol
# ---------------------------------------------------------------------------

SUBTYPE_CLASSIFIERS = {
    "LIHC": "Hoshida S1/S2/S3 (EMT-active, MYC/mTOR-driven, hepatocyte-like)",
    "BRCA": "PAM50 (Luminal A/B, HER2-enriched, Basal-like, Normal-like)",
    "PAAD": "Squamous, Progenitor, ADEX, Immunogenic",
}


def classify_subtype(
    pathway_scores: pd.Series, gene_series: pd.Series, cancer_type: str
) -> Tuple[str, str]:
    """Deterministic subtype assignment from pathway percentile ranks.

    Full subtype rules for the analysed cancer types are described in the
    manuscript supplement; the generic rule below is applied otherwise.
    """
    ranks = pathway_scores.rank(pct=True)
    if ranks.get("EPITHELIAL_MESENCHYMAL_TRANSITION", 0.5) > 0.6:
        subtype, reason = "EMT-active", "dominant epithelial-mesenchymal transition signal"
    elif ranks.get("MYC_TARGETS_V1", 0.5) > 0.6:
        subtype, reason = "MYC-driven", "dominant MYC target activation"
    elif ranks.get("INFLAMMATORY_RESPONSE", 0.5) > 0.6:
        subtype, reason = "Immune-active", "dominant inflammatory response signal"
    else:
        subtype, reason = "Metabolic", "preserved metabolism-related signal"
    return subtype, reason


def build_v5_prompt(
    patient_id: str,
    pathway_scores: pd.Series,
    gene_series: pd.Series,
    cancer_type: str = "LIHC",
) -> str:
    """Assemble the V5 prompt (Methods): subtype + vocabulary + output template.

    The LLM is instructed to emit a single standardized paragraph (~50-60
    words) covering phenotype, differentiation, proliferation, immune status,
    subtype, dominant pathway, and qualitative risk, using only the provided
    controlled vocabulary.
    """
    subtype, reason = classify_subtype(pathway_scores, gene_series, cancer_type)
    ranks = pathway_scores.rank(pct=True)
    dominant = ranks.idxmax()

    vocab = (
        "Use ONLY terms from these lists, matched to the percentile ranks:\n"
        "  Metabolic: Hyper-metabolic | Metabolically-active | Normo-metabolic\n"
        "  Differentiation: Well-differentiated | Moderately-differentiated | Poorly-differentiated\n"
        "  Proliferation (P>70 highly / P40-70 mildly / P<40 quiescent)\n"
        "  Immune (P>60 hot / P40-60 warm / P<40 cold)\n"
        "  Risk: Low | Intermediate-Low | Intermediate | Intermediate-High | High"
    )
    template = (
        'Output exactly one sentence: "The tumor phenotype is [Metabolic], '
        '[Differentiation], with a [Proliferation] profile. The microenvironment '
        'is [Immune]. This aligns with the {subtype} subtype ({reason}). The '
        'patient is [Risk] driven by {dominant}."'
    ).format(subtype=subtype, reason=reason, dominant=dominant)

    return (
        f"[Cancer Type: {cancer_type}]\n[Patient: {patient_id}]\n\n"
        f"[PRE-COMPUTED MOLECULAR SUBTYPE: {subtype}]\n{vocab}\n\n"
        f"[Dominant Pathway: {dominant}]\n{template}\n\n"
        "Generate the clinical narrative:"
    )


SYSTEM_PROMPT = (
    "You are a molecular oncology classifier generating standardized clinical "
    "narratives. Rules: use only the provided vocabulary; match risk level to "
    "phenotype intensity; always include the dominant pathway at the end; "
    "output exactly one paragraph (~50-60 words), no labels or headers."
)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def generate_report(prompt: str) -> str:
    """Single Qwen3-Max call (temperature 0)."""
    response = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return str(response.choices[0].message.content)


def get_embedding(text: str) -> np.ndarray:
    """text-embedding-v4 embedding (1,024-dim)."""
    response = _client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    vector = response.data[0].embedding
    return np.asarray(vector, dtype=np.float32)


def generate_embeddings(
    pathway_scores: pd.DataFrame,
    gene_expr: pd.DataFrame,
    cancer_type: str,
    *,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Generate narrative embeddings for all samples of one cohort."""
    vectors: List[np.ndarray] = []
    for patient_id in pathway_scores.index:
        prompt = build_v5_prompt(
            patient_id, pathway_scores.loc[patient_id], gene_expr.loc[patient_id], cancer_type
        )
        report = generate_report(prompt)
        vectors.append(get_embedding(report))
    embeddings = np.stack(vectors) if vectors else np.zeros((0, EMBEDDING_DIM))
    if cache_path is not None:
        import torch

        torch.save(
            {"embeddings": torch.as_tensor(embeddings), "patient_ids": list(pathway_scores.index)},
            cache_path,
        )
    return embeddings, list(pathway_scores.index)
