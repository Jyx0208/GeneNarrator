# GeneNarrator

GeneNarrator (GN-AFT) is a research implementation of a multimodal survival
prediction framework that fuses gene-expression features with LLM-derived
semantic embeddings of pathway-based clinical narratives, using
quality-adaptive cross-modal fusion and a parameterized Weibull
accelerated-failure-time head.

## Features

- Dual-tower encoders: gene 1,000 -> 512 -> 256; semantic 1,024 -> 512 -> 256
- Quality-Adaptive Cross-Modal Fusion: sample-level quality gates, bidirectional
  cross-attention, and adaptive weighted residual fusion
- Parameterized Weibull AFT survival head (256 -> 128 -> 64 -> 2) with
  prior-informed initialization
- Rank-normalized ssGSEA (gseapy) and the V5 narrative prompt protocol
  (Qwen3-Max + text-embedding-v4)
- 5-fold internal cross-validation and development-cohort model selection
- C-index with bootstrap 95% CI, log-rank stratification, and silhouette-based
  cohort-separability quantification

## Install

Requires Python 3.9 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
import torch
from genenarrator.model import GNAFT, weibull_loss

model = GNAFT()
gene = torch.randn(64, 1000)
text = torch.randn(64, 1024)
scale, shape = model(gene, text)          # Weibull parameters per sample
loss = weibull_loss(scale, shape, torch.full((64,), 400.0), torch.ones(64))
```

Narrative generation requires a DashScope-compatible API key:

```bash
export DASHSCOPE_API_KEY="..."
python - <<'PY'
import pandas as pd
from genenarrator.narrative import compute_ssgsea, generate_embeddings
scores = compute_ssgsea(expr, load_hallmark_gene_sets())
emb, ids = generate_embeddings(scores, expr, "LIHC", cache_path="data/out_embeddings_v5.pt")
PY
```

The examples use synthetic or locally prepared data.

## Data

Third-party datasets and API credentials are not included. The analysed
cohorts are TCGA-LIHC/LIRI-JP, TCGA-BRCA/GSE20685, TCGA-OV/OV-AU, and
TCGA-PAAD/PACA-CA. GSE20685 used the Affymetrix Human Genome U133 Plus 2.0
Array (GPL570). Data can be obtained from the [GDC Data Portal](https://portal.gdc.cancer.gov/),
[GEO](https://www.ncbi.nlm.nih.gov/geo/), and
[ICGC legacy data](https://docs.icgc-argo.org/docs/data-access/icgc-25k-data),
subject to each repository's access and redistribution terms. MSigDB HALLMARK
gene sets (release 7.5.1) are from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/).

## License

GPL-3.0. See [LICENSE](LICENSE).
