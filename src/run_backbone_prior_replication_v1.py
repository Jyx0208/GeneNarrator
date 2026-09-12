"""Five-seed replication of the fixed-prior mechanism; no new hyperparameters."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from datetime import datetime, timezone
from importlib.metadata import version
import shutil
import torch

from run_backbone_objective_v1 import ModelCallback, additional_metrics, EXTERNAL, GRID, ROOT
from experiment_base import atomic_json, sha256_file
from unified_experiment_base import UnifiedProtocol, UnifiedExperimentRunner

OUT = ROOT / 'cohort_trial28' / 'backbone_prior_replication_v1'
SEEDS = (17, 29, 43, 101, 151)
CONFIGS = {
    'LegacyPrior': ('WeibullLegacy', True),
    'LegacyNoPrior': ('WeibullLegacy', False),
    'FullPrior': ('WeibullFull', True),
    'FullNoPrior': ('WeibullFull', False),
}


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    files = ('run_backbone_prior_replication_v1.py', 'run_backbone_objective_v1.py',
             'backbone_objective_model.py', 'graph_model14.py', 'original_weibull_loss_corrected.py',
             'original_round10/model_snapshot_0.py', 'unified_experiment_base.py',
             'experiment_base.py', 'run_activity22_coxnet.py')
    protocol = UnifiedProtocol('backbone_prior_replication_v1',
        feature_version='common9703-p50-original-reference-variance1000',
        model_version='fixed-prior-removal-five-seed-replication', grid_version='0-1095-61',
        extra={'arms': list(CONFIGS), 'configs': CONFIGS, 'seeds': SEEDS, 'external_os': EXTERNAL,
               'external_status': 'all development-exposed; pilot outcomes already inspected',
               'hypothesis': 'replicate parameter-prior removal with two more fixed seeds; no model/hyperparameter changes',
               'comparisons': ['LegacyNoPrior - LegacyPrior', 'FullNoPrior - FullPrior'],
               'max_epochs': 160, 'patience': 25, 'lr': .0003, 'weight_decay': .01,
               'early_stop': 'Weibull NLL; local cohort-event20%seed17; median of five seed epochs; refit whole training block',
               'LLM_scope': 'primary backbone mechanism only; no new LLM features',
               'packages': {n: version(n) for n in ('torch', 'torchsurv', 'scikit-survival')},
               'code_hashes': {f: sha256_file(ROOT / f) for f in files}})
    runner = UnifiedExperimentRunner.from_standard_registry(
        ROOT, output_dir=OUT, protocol=protocol, grid=GRID, arms=list(CONFIGS),
        external_names=EXTERNAL).prepare()
    for name in files:
        dest = OUT / 'code_at_run' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(ROOT / name, dest)
    started = datetime.now(timezone.utc).isoformat()
    for label, (arm, prior) in CONFIGS.items():
        runner.run_arm(label, ModelCallback(arm, protocol, output_dir=OUT, label=label,
            seeds=SEEDS, fit_options={'distribution_prior': prior}))
    metrics = runner.score()
    additional_metrics(runner)
    atomic_json(OUT / 'analysis_complete.json', {'started_utc': started,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol.sha256, 'metrics_sha256': sha256_file(OUT / 'metrics.csv')})
    print(metrics[['cohort', 'arm', 'uno_c_3y', 'harrell_c', 'brier_3y']].to_string(index=False))


if __name__ == '__main__':
    main()
