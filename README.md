# GeneNarrator

**Bridging Genomic and Semantic Spaces: A Multimodal Framework for Cross-Cohort Cancer Survival Prediction**

## Overview

GeneNarrator (GN-AFT) integrates gene expression data with LLM-generated semantic embeddings via an adaptive gating mechanism for cancer survival prediction. By converting patient-level transcriptomic profiles into structured natural language reports using large language models, GN-AFT captures biological knowledge that complements raw expression features, enabling robust cross-cohort generalization without batch-correction preprocessing.

## Architecture

GN-AFT consists of three components:

1. **Gene Expression Encoder** — Processes top-1000 variance genes through fully connected layers
2. **Semantic Encoder** — Processes 1024-dim text embeddings from LLM-generated patient reports
3. **Adaptive Gating Mechanism** — Learns cancer-type-specific fusion weights between genomic and semantic features

The model uses an Accelerated Failure Time (AFT) loss for survival time prediction, combined with a ranking loss for concordance optimization.

## Results

External validation C-index (trained on TCGA, tested on independent cohorts):

| Cancer | GN-AFT | Cox-PH | DeepSurv | DeepHit | AFT-Gene | ComBat+AFT |
|--------|--------|--------|----------|---------|----------|------------|
| LIHC   | **0.762** | 0.679 | 0.644 | 0.582 | 0.540 | 0.536 |
| BRCA   | **0.657** | 0.584 | 0.638 | 0.564 | 0.639 | 0.593 |
| OV     | **0.679** | 0.577 | 0.506 | 0.506 | 0.508 | 0.492 |
| PAAD   | **0.611** | 0.566 | 0.567 | 0.503 | 0.575 | 0.560 |
| **Mean** | **0.677** | 0.602 | 0.589 | 0.539 | 0.566 | 0.545 |

## Repository Structure

```
GeneNarrator/
├── unified_data.py                    # Core data loading module
├── configs/sota_config.json           # Model configuration & hyperparameters
├── scripts/
│   └── preprocess_embeddings.py       # LLM report generation & embedding pipeline
├── requirements.txt                   # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/GeneNarrator.git
cd GeneNarrator
pip install -r requirements.txt
```

## Data Preparation

Each cancer type requires three files in the `data/` directory:

| File | Description |
|------|-------------|
| `{dataset}.star_fpkm.tsv.gz` | Gene expression matrix (genes × samples) |
| `{dataset}.survival.tsv.gz` | Survival data (`sample`, `OS.time`, `OS`) |
| `{dataset}_embeddings_v5.pt` | Pre-computed semantic embeddings (1024-dim) |

| Cancer | Training (TCGA) | External Validation |
|--------|----------------|---------------------|
| LIHC   | TCGA-LIHC (n=433) | LIRI-JP (n=232) |
| BRCA   | TCGA-BRCA (n=1,232) | GSE20685 (n=327) |
| OV     | TCGA-OV (n=609) | OV-AU (n=93) |
| PAAD   | TCGA-PAAD (n=195) | PACA-CA (n=247) |

## Generating Semantic Embeddings

```bash
export DASHSCOPE_API_KEY="your-api-key"
python scripts/preprocess_embeddings.py
```

This performs: (1) GSEA pathway enrichment per patient, (2) structured report generation via Qwen3-Max, (3) text embedding via text-embedding-v4.

## Citation

```bibtex
@article{jiang2026genenarrrator,
  title={Bridging Genomic and Semantic Spaces: A Multimodal Framework for
         Cross-Cohort Cancer Survival Prediction},
  author={Jiang, Yuxuan and Zhou, Xuancheng and Jiang, Lai and Huang, Gang
          and Gu, Yuheng and Yin, Shiyang and Wang, Lexin and Xu, Ke
          and Li, Xiaosong and Chi, Hao and Deng, Youping},
  journal={npj Digital Medicine},
  year={2026}
}
```

## License

MIT License
