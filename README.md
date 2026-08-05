# GeneNarrator

GeneNarrator (GN-AFT) is a research implementation for combining gene-expression features with embeddings of pathway-derived narratives in Weibull accelerated-failure-time modelling.

## Important results note

The historical manuscript values were produced by an earlier workflow in which target-cohort outcomes were used for early stopping and target-platform feature availability affected gene alignment. They are therefore exploratory target-guided cross-cohort estimates, not independent external-validation results. The current code corrects those two issues, but it has not been rerun and does not claim to reproduce the historical values.

The corrected workflow uses a patient-level TCGA development split for model selection, freezes the development-derived gene manifest before target evaluation, and contains no expected-result constants or C-index matching checks.

## Model

- Gene encoder: 1,000→512→384→256
- Semantic encoder: 1,024→512→256
- Learned sample-level gates and single-token cross-modal projections
- Residual fusion and a Weibull AFT head: 256→128→64→2

The gates are task-optimized coefficients, not calibrated measurements of assay quality or batch effect.

## Installation

```bash
git clone https://github.com/Jyx0208/GeneNarrator.git
cd GeneNarrator
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Data are not redistributed. Expected cohorts are TCGA-LIHC/LIRI-JP, TCGA-BRCA/GSE20685, TCGA-OV/OV-AU, and TCGA-PAAD/PACA-CA. GSE20685 used the Affymetrix Human Genome U133 Plus 2.0 Array (GPL570).

Sources: [GDC](https://portal.gdc.cancer.gov/), [GSE20685](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685), and [ICGC legacy data](https://docs.icgc-argo.org/docs/data-access/icgc-25k-data). Follow all source-repository access and redistribution conditions. Never commit patient-level controlled data or API keys.

Place required files in `data/` as described by `scripts/setup_data.py`.

## Embeddings

V5 is the default prompt path. Generation uses DashScope `qwen3-max` and `text-embedding-v4`.

```bash
export DASHSCOPE_API_KEY="..."
python scripts/preprocess_embeddings.py --cancer LIHC --prompt_version v5
```

For a reported analysis, record the exact hosted-model identifier/date and hashes of cached reports and embeddings.

## Corrected workflow

```bash
python scripts/setup_data.py --check
python scripts/train_from_scratch.py
python scripts/generate_evaluation_results.py
```

Corrected checkpoints are saved as `models/gnaft_<cancer>_corrected.pt`. The evaluator refuses legacy checkpoints that do not declare the corrected protocol. A fixed seed supports repeatability within a recorded environment but does not guarantee bitwise identity across hardware or package versions.

This software is for research use and is not a clinical decision-support device.

## License

GPL-3.0. See [LICENSE](LICENSE).
