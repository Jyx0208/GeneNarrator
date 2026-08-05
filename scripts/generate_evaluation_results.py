#!/usr/bin/env python
"""Evaluate saved GeneNarrator checkpoints without target-guided selection.

This script computes metrics from a checkpoint and the supplied cohort files.
It contains no hard-coded expected C-index values and does not select a model,
seed, epoch, threshold, or feature set from target outcomes.
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BASE_DIR))
os.chdir(_BASE_DIR)

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index

from unified_data import CANCER_PAIRS, load_cancer_data
from scripts.train_from_scratch import ImprovedGNAFT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def c_index(model, data) -> float:
    gene = torch.as_tensor(data["gene"], dtype=torch.float32, device=DEVICE)
    text = torch.as_tensor(data["text"], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        median = model.predict_median(gene, text).cpu().numpy()
    risk = 1.0 / (median + 1e-8)
    return float(concordance_index(data["time"], -risk, data["event"]))


def evaluate_single_cancer(cancer_type: str):
    checkpoint_path = Path("models") / f"gnaft_{cancer_type.lower()}_corrected.pt"
    if not checkpoint_path.exists():
        print(f"[SKIP] Missing corrected checkpoint: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    protocol = checkpoint.get("protocol", {})
    if protocol.get("target_outcomes_used_for_model_selection") is not False:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} lacks the corrected protocol declaration. "
            "Legacy target-guided checkpoints are intentionally not evaluated by this script."
        )

    config = checkpoint["config"]
    model = ImprovedGNAFT(
        gene_dim=config["gene_dim"],
        text_dim=config["text_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source_data, target_data, info = load_cancer_data(
        cancer_type, n_genes=config["gene_dim"]
    )
    return {
        "cancer": cancer_type,
        "model": "GN-AFT",
        "protocol_version": protocol.get("version", "unknown"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": config.get("seed"),
        "source_dataset": info["train_name"],
        "target_dataset": info["test_name"],
        "source_samples": len(source_data["gene"]),
        "target_samples": len(target_data["gene"]),
        "source_apparent_ci": c_index(model, source_data),
        "target_cohort_ci": c_index(model, target_data),
        "target_missing_features": info.get("n_missing_external_features"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate corrected GeneNarrator checkpoints"
    )
    parser.add_argument(
        "--cancer", choices=sorted(CANCER_PAIRS), help="Evaluate one cancer type"
    )
    args = parser.parse_args()

    cancers = [args.cancer] if args.cancer else list(CANCER_PAIRS)
    results = []
    for cancer in cancers:
        print(f"[EVALUATE] {cancer}")
        result = evaluate_single_cancer(cancer)
        if result is not None:
            results.append(result)
            print(f"  target-cohort C-index: {result['target_cohort_ci']:.4f}")

    if not results:
        print("No corrected checkpoints were evaluated.")
        return 1

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gnaft_corrected_evaluation.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
