# GeneNarrator

**Bridging Genomic and Semantic Spaces: A Multimodal Framework for Cross-Cohort Cancer Survival Prediction**

This repository contains the code for GeneNarrator (GN-AFT), a multimodal survival model that fuses gene expression data and LLM-derived semantic embeddings for cross-cohort cancer survival prediction.

## Model overview

GN-AFT consists of three components:

1. **Gene encoder** -- processes transcriptomic features via a three-layer MLP
2. **Semantic encoder** -- processes 1024-d text embeddings from LLM-generated patient reports
3. **Adaptive fusion gate** -- learns modality-specific quality weights via cross-attention

The model is optimised with a Weibull AFT objective and evaluated on external validation cohorts.

## External validation performance (C-index)

| Cancer | GN-AFT | Cox-PH | DeepSurv | DeepHit | AFT-Gene | ComBat+AFT |
|--------|--------|--------|----------|---------|----------|------------|
| LIHC   | **0.762** | 0.679 | 0.644 | 0.582 | 0.540 | 0.536 |
| BRCA   | **0.657** | 0.584 | 0.638 | 0.564 | 0.639 | 0.593 |
| OV     | **0.679** | 0.577 | 0.506 | 0.506 | 0.508 | 0.492 |
| PAAD   | **0.611** | 0.566 | 0.567 | 0.503 | 0.575 | 0.560 |
| **Mean** | **0.677** | 0.602 | 0.589 | 0.539 | 0.566 | 0.545 |

## Repository structure

```
GeneNarrator/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run_paper_experiments.py          # orchestrates the full workflow
├── unified_data.py                   # data loading & gene alignment
└── scripts/
    ├── preprocess_embeddings.py      # LLM report generation & embedding
    ├── setup_data.py                 # copy data files into data/
    ├── train_from_scratch.py         # model training
    └── generate_evaluation_results.py  # evaluate saved models
```

## Installation

```bash
git clone https://github.com/Jyx0208/GeneNarrator.git
cd GeneNarrator
pip install -r requirements.txt
```

## Data

Place data files under `data/` (directory is git-ignored).  Required files per cohort:

- `{cohort}.star_fpkm.tsv.gz` -- gene expression matrix
- `{cohort}.survival.tsv.gz` -- survival annotation
- `{cohort}_embeddings_v5.pt` -- pre-computed text embeddings
- `{cohort}_reports_v5.txt` -- LLM-generated reports

Training cohorts: TCGA-LIHC, TCGA-BRCA, TCGA-OV, TCGA-PAAD
Validation cohorts: LIRI-JP, GSE20685, OV-AU, PACA-CA

## Workflow

```bash
# Full pipeline: data setup -> training -> evaluation
python run_paper_experiments.py --full

# Train and evaluate
python run_paper_experiments.py

# Evaluate pre-trained models only
python run_paper_experiments.py --evaluate-only
```

## Embedding generation

Generating text embeddings requires a DashScope API key:

```bash
export DASHSCOPE_API_KEY="your-api-key"
python scripts/preprocess_embeddings.py
```

## License

GPL-3.0 License
