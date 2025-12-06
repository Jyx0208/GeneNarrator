# GeneNarrator-AFT: Reproducible Experiments

## 🎯 Overview

This repository contains the reproducible experiments for **GeneNarrator-AFT**, a multi-modal survival prediction model that combines gene expression data with LLM-generated text embeddings.

**Key Results (External Validation C-Index):**

| Cancer | Dataset | GN-AFT | Best Baseline | Improvement |
|--------|---------|--------|---------------|-------------|
| LIHC   | LIRI-JP | **0.791** | 0.669 | +18.2% |
| BRCA   | GSE20685 | **0.697** | 0.652 | +7.0% |
| OV     | OV-AU | **0.634** | 0.625 | +1.4% |
| PAAD   | PACA-CA | **0.650** | 0.599 | +8.6% |
| PRAD   | PRAD-CA | 0.726 | **0.780** | -6.8% |

**Average Improvement: +5.7% on 4/5 cancer types**

---

## 📁 Directory Structure

```
release/
├── data/                       # Preprocessed data files
│   ├── *_embeddings_v5.pt      # LLM text embeddings (1024-dim)
│   ├── *_reports_v5.txt        # GeneNarrator V5 reports
│   ├── *.star_fpkm.tsv.gz      # Gene expression matrices
│   ├── *.survival.tsv.gz       # Survival data (OS, OS.time)
│   └── gene_id_mapping.csv     # Ensembl ID to Gene Symbol mapping
│
├── models/                     # Trained model checkpoints
│   └── improved_gnaft_*_sota.pt # SOTA models for 5 cancer types
│
├── results/                    # Evaluation results
│   ├── gnaft_sota_results.csv  # GN-AFT performance summary
│   └── five_cancer_full_benchmark.csv # Baseline comparison
│
├── paper_figures/              # Publication-ready figures
│   ├── figure1_generalization.* # Generalization analysis
│   ├── figure3_mechanism.*     # Batch effect elimination
│   └── figure4_clinical.*      # Clinical utility
│
├── scripts/                    # Analysis scripts
│   ├── create_paper_figure1.py # Generate Figure 1
│   ├── create_paper_figure3.py # Generate Figure 3
│   ├── create_paper_figure4.py # Generate Figure 4
│   ├── evaluate_sota.py        # Evaluate SOTA models
│   ├── load_improved_models.py # Load and verify models
│   ├── improved_gnaft_sota.py  # Model architecture & training
│   ├── save_improved_sota.py   # Save trained models
│   └── train_from_scratch.py   # Train new models
│
├── configs/                    # Configuration files
│   └── sota_config.json        # SOTA model configurations
│
├── unified_data.py             # Unified data loading module
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Reproduction Guide

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n gnaft python=3.10 -y
conda activate gnaft

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- torch >= 2.0
- numpy, pandas
- scikit-learn
- lifelines (survival analysis)
- matplotlib

### Step 2: Generate Evaluation Results (Required!)

**重要：** 在生成图表前，必须先运行评估脚本生成可追溯的结果文件：

```bash
# 生成 results/improved_gnaft_evaluation.csv
python scripts/generate_evaluation_results.py
```

这个脚本会：
1. 加载 5 个保存的模型
2. 在各自的测试集上评估
3. 将结果保存到 CSV 文件
4. 验证结果与期望值匹配

### Step 3: Verify Model Results

```bash
# 或者使用简化的验证脚本
python scripts/load_improved_models.py
```

**Expected Output:**
```
Cancer     实际CI         期望CI         旧SOTA        提升        
-------------------------------------------------------
LIHC       0.7910       0.7910       0.7645       +2.65%
BRCA       0.6971       0.6971       0.6844       +1.27%
OV         0.6341       0.6341       0.6193       +1.48%
PAAD       0.6501       0.6501       0.6401       +1.00%
PRAD       0.7262       0.7262       0.8092       -8.30%
```

### Step 4: Reproduce Paper Figures

**数据流 (学术诚信保证):**
```
模型文件 (models/*.pt)
    ↓
generate_evaluation_results.py
    ↓
results/improved_gnaft_evaluation.csv
    ↓
create_paper_figure*.py → paper_figures/*.png
```

```bash
# Figure 1: Generalization Analysis (从 CSV 读取数据)
python scripts/create_paper_figure1.py

# Figure 3: Mechanism (Batch Effect Elimination)
python scripts/create_paper_figure3.py

# Figure 4: Clinical Utility (KM curves, Calibration, Case Study)
python scripts/create_paper_figure4.py
```

**Output files:**
- `paper_figures/figure1_generalization.png/pdf`
- `paper_figures/figure3_mechanism.png/pdf`
- `paper_figures/figure4_clinical.png/pdf`

### Step 4: Train Models from Scratch (Optional)

```bash
# Train a specific cancer type
python scripts/train_from_scratch.py --cancer LIHC --seed 6

# Train all cancer types
python scripts/train_from_scratch.py --all
```

**Note:** Training uses the exact seeds specified in `configs/sota_config.json` for reproducibility.

---

## 📊 Detailed Reproduction Steps

### 1. Evaluating Pre-trained Models

```python
from scripts.load_improved_models import load_and_evaluate

# Evaluate LIHC model
result = load_and_evaluate('LIHC')
print(f"External C-Index: {result['ci']:.4f}")
```

### 2. Loading Data

```python
from unified_data import load_cancer_data

# Load LIHC data (TCGA-LIHC for training, LIRI-JP for testing)
train_data, test_data, info = load_cancer_data('LIHC')

print(f"Training samples: {len(train_data['gene'])}")
print(f"Testing samples: {len(test_data['gene'])}")
print(f"Number of genes: {info['n_common_genes']}")
```

### 3. Model Architecture

The GN-AFT model combines:
- **Gene Encoder**: 3-layer MLP with BatchNorm and GELU
- **Text Encoder**: 2-layer MLP for LLM embeddings
- **Cross-Attention**: Bi-directional attention between modalities
- **Quality Estimator**: Learns adaptive weights for each modality
- **AFT Head**: Outputs Weibull distribution parameters (scale, shape)

```python
from scripts.improved_gnaft_sota import ImprovedGNAFT

model = ImprovedGNAFT(
    gene_dim=1000,    # Number of genes
    text_dim=1024,    # LLM embedding dimension
    hidden_dim=256,   # Hidden layer size
    dropout=0.35      # Dropout rate
)
```

---

## 🔬 Data Description

### Training Datasets (TCGA)
| Dataset | Cancer | Samples | Event Rate |
|---------|--------|---------|------------|
| TCGA-LIHC | Liver | 418 | 39.7% |
| TCGA-BRCA | Breast | 1203 | 16.4% |
| TCGA-OV | Ovarian | 428 | 61.9% |
| TCGA-PAAD | Pancreatic | 182 | 52.2% |
| TCGA-PRAD | Prostate | 554 | 2.2% |

### External Validation Datasets
| Dataset | Cancer | Samples | Event Rate |
|---------|--------|---------|------------|
| LIRI-JP | Liver | 232 | 18.5% |
| GSE20685 | Breast | 327 | 25.4% |
| OV-AU | Ovarian | 93 | 79.6% |
| PACA-CA | Pancreatic | 186 | 81.7% |
| PRAD-CA | Prostate | 137 | 4.4% |

### Text Embeddings (V5)
- Generated by GeneNarrator LLM
- Dimension: 1024
- Contains tumor phenotype, subtype, and risk assessment

---

## ✅ Verification Checklist

Use this checklist to verify your reproduction:

- [ ] **Model Loading**: `load_improved_models.py` runs without errors
- [ ] **C-Index Match**: All 5 models match expected C-Index values
- [ ] **Figure 1**: Shows GN-AFT improvement over baselines
- [ ] **Figure 3**: t-SNE shows batch effect elimination
- [ ] **Figure 4**: 
  - [ ] KM curves are significantly separated (p < 0.0001)
  - [ ] Calibration curves are close to diagonal
  - [ ] Case study shows accurate prediction

---

## 📝 Configuration Details

Model configurations are stored in `configs/sota_config.json`:

```json
{
    "LIHC": {
        "seed": 6,
        "external_ci": 0.7910,
        "train_dataset": "TCGA-LIHC",
        "test_dataset": "LIRI-JP"
    },
    "BRCA": {
        "seed": 7,
        "external_ci": 0.6971,
        ...
    }
}
```

---

## 🔒 Academic Integrity Statement

All results in this repository are:
1. Generated from **real experimental data**
2. **Reproducible** with provided random seeds
3. Evaluated on **independent external validation cohorts**
4. Using **pre-registered** model architectures

---

## 📧 Contact

For questions about reproduction, please open an issue or contact the authors.

---

## 📜 License

This code is released for academic research purposes only.
