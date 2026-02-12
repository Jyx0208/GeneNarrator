# GeneNarrator

**Bridging Genomic and Semantic Spaces: A Multimodal Framework for Cross-Cohort Cancer Survival Prediction**

This repository contains the code and reproducible workflow for GeneNarrator (GN-AFT), a multimodal survival model that fuses gene expression and LLM-derived semantic embeddings.

## What is included

- Core data loading and preprocessing modules
- Training, model saving, and evaluation scripts
- Reproducible benchmark result CSV files

## Model overview

GN-AFT has three components:

1. **Gene encoder** for transcriptomic features
2. **Semantic encoder** for 1024-d text embeddings from LLM-generated reports
3. **Adaptive gate** to fuse genomic and semantic representations

The model is optimized with an AFT objective and ranking-aware supervision.

## External validation performance (C-index)

| Cancer | GN-AFT | Cox-PH | DeepSurv | DeepHit | AFT-Gene | ComBat+AFT |
|--------|--------|--------|----------|---------|----------|------------|
| LIHC   | **0.762** | 0.679 | 0.644 | 0.582 | 0.540 | 0.536 |
| BRCA   | **0.657** | 0.584 | 0.638 | 0.564 | 0.639 | 0.593 |
| OV     | **0.679** | 0.577 | 0.506 | 0.506 | 0.508 | 0.492 |
| PAAD   | **0.611** | 0.566 | 0.567 | 0.503 | 0.575 | 0.560 |
| **Mean** | **0.677** | 0.602 | 0.589 | 0.539 | 0.566 | 0.545 |

## Repository structure

```text
GeneNarrator/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run_paper_experiments.py
├── unified_data.py
├── configs/
│   └── sota_config.json
├── results/
│   ├── five_cancer_full_benchmark.csv
│   └── improved_gnaft_evaluation.csv
└── scripts/
    ├── preprocess_embeddings.py
    ├── setup_data.py
    ├── train_from_scratch.py
    ├── save_improved_sota.py
    ├── evaluate_sota.py
    ├── generate_evaluation_results.py
    ├── load_improved_models.py
    └── improved_gnaft_sota.py
```

## Installation

```bash
git clone https://github.com/Jyx0208/GeneNarrator.git
cd GeneNarrator
pip install -r requirements.txt
```

## Data requirements

Place data under `data/` (directory is gitignored):

- `{dataset}.star_fpkm.tsv.gz`
- `{dataset}.survival.tsv.gz`
- `{dataset}_embeddings_v5.pt`

Supported cohorts in this release focus on LIHC, BRCA, OV, and PAAD.

## Typical workflow

```bash
# Default: evaluate
python run_paper_experiments.py

# Full pipeline: setup + training + evaluation
python run_paper_experiments.py --full
```

## Embedding generation (LLM)

```bash
export DASHSCOPE_API_KEY="your-api-key"
python scripts/preprocess_embeddings.py
```

## Citation

```bibtex
@article{jiang2026genenarrrator,
  title={Bridging Genomic and Semantic Spaces: A Multimodal Framework for Cross-Cohort Cancer Survival Prediction},
  author={Jiang, Yuxuan and Zhou, Xuancheng and Jiang, Lai and Huang, Gang and Gu, Yuheng and Yin, Shiyang and Wang, Lexin and Xu, Ke and Li, Xiaosong and Chi, Hao and Deng, Youping},
  journal={npj Digital Medicine},
  year={2026}
}
```

## License

MIT License.
