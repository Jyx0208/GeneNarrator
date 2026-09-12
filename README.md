# GeneNarrator

GeneNarrator is a transcriptomic survival-prediction model combining gene-level expression and pathway-level representations. This repository is the minimal analysis release accompanying the manuscript. The public repository is https://github.com/Jyx0208/GeneNarrator.

## Contents

- `src/`: preprocessing, model fitting, evaluation and figure-generation code used by the analysis.
- `configs/`: protocol and run metadata needed to identify the reported configuration.
- Model checkpoints are distributed separately with the versioned release record because they are large binary files.

## Data

Expression and survival records are not redistributed here. Obtain them from the public GEO, TCGA/GDC and ICGC resources listed in the manuscript Data availability statement. Patient-level figure source values are supplied with the manuscript rather than this software repository.

## Reproducibility

The final tagged release will add the environment specification, exact input-axis checksums, model-checkpoint checksums and one-command inference example. Checkpoint binaries are withheld from this initial source commit until the final model version is frozen; they will be attached to the corresponding tagged release.

## Scope

This release contains analysis code and protocol metadata only; it does not include patient records, API keys, private credentials, intermediate experiment history or alternative trial directories.
