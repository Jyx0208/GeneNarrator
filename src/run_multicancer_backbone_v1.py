"""Apply the SAME declared backbone mechanism to three existing cancer pairs.

Each cancer uses its own measured common gene background, recomputed ranks
and all 50 Hallmark readouts. No historical pathway CSV values are reused.
The original 1000-gene encoder and entire fusion core are retained.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
from datetime import datetime, timezone
import shutil
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from experiment_base import (ExpressionData, Outcomes, CohortBundle, atomic_npz,
                             atomic_json, sha256_file, sha256_array)
from unified_experiment_base import UnifiedProtocol, UnifiedExperimentRunner
from run_backbone_objective_v1 import ModelCallback, additional_metrics, ROOT, GRID

OUT = ROOT / 'cohort_trial28' / 'multicancer_backbone_v1'
PANELS = {'LIHC': ('TCGA-LIHC', 'LIRI-JP', 'GSE76427'),
          'OV': ('TCGA-OV', 'OV-AU', 'GSE32062'),
          'PAAD': ('TCGA-PAAD', 'PACA-CA', 'GSE57495')}
CONFIGS = {'Original': {'distribution_prior': True},
           'NoDistributionPrior': {'distribution_prior': False}}
SEEDS = (17, 29, 43, 101, 151)


def prepare_inputs(cancer, names):
    directory = OUT / cancer / 'inputs'
    directory.mkdir(parents=True, exist_ok=True)
    rows, labels, hashes = {}, {}, {}
    for name in names:
        xp = ROOT / 'fixed_panel_inputs' / f'{name}_gene_ranks.npz'
        yp = ROOT / 'clinical_round5' / f'{name}_clinical.csv'
        with np.load(xp) as z:
            ids, genes, values = z['ids'].astype(str), z['genes'].astype(str), z['x'].astype(float)
        frame = pd.read_csv(yp).set_index('sample')
        frame.index = frame.index.astype(str)
        assert frame.index.is_unique and len(set(ids)) == len(ids)
        qualified = frame.reindex(ids)
        time = pd.to_numeric(qualified.time, errors='coerce').to_numpy(float)
        event = pd.to_numeric(qualified.event, errors='coerce').to_numpy(float)
        keep = np.isfinite(time) & (time > 0) & np.isin(event, [0, 1])
        rows[name] = (ids[keep], genes, values[keep])
        labels[name] = Outcomes(name, ids[keep], time[keep], event[keep].astype(bool), 'OS', str(yp))
        hashes[name] = {'expression': sha256_file(xp), 'outcomes': sha256_file(yp),
                        'expression_n': len(ids), 'qualified_n': int(keep.sum())}
    axis = rows[names[0]][1]
    for name in names[1:]:
        np.testing.assert_array_equal(rows[name][1], axis)
    measured = np.logical_and.reduce([np.isfinite(rows[n][2]).all(axis=0) for n in names])
    genes = axis[measured]
    if len(genes) < 1000:
        raise ValueError('insufficient shared measured genes')
    gmt = ROOT / 'knowledge_trial23/resources/h.all.v7.5.1.symbols.gmt'
    sets = {}
    for line in gmt.read_text(encoding='utf-8').splitlines():
        name, _, *members = line.split('\t')
        sets[name.replace('HALLMARK_', '')] = set(members)
    pathways = np.asarray(sorted(sets))
    assert len(pathways) == 50
    members = [np.flatnonzero(np.isin(genes, list(sets[name]))) for name in pathways]
    if min(map(len, members)) < 15:
        raise ValueError('a common-panel Hallmark has fewer than 15 measured genes')
    expressions = {}
    for name in names:
        ids, _, old = rows[name]
        x = (rankdata(old[:, measured], method='average', axis=1) / len(genes)).astype(np.float32)
        p = np.column_stack([2*x[:, ix].mean(axis=1)-1 for ix in members]).astype(np.float32)
        one = rankdata(old[:1, measured], method='average', axis=1) / len(genes)
        np.testing.assert_allclose(one, x[:1], atol=4e-8, rtol=0)
        path = directory / f'{name}.npz'
        if path.exists():
            with np.load(path) as z:
                for key, val in {'ids': ids, 'genes': genes, 'x': x, 'p': p}.items():
                    np.testing.assert_array_equal(z[key], val)
        else:
            atomic_npz(path, ids=ids, genes=genes, x=x, p=p, pathways=pathways)
        expressions[name] = ExpressionData(name, ids, genes, x, p, str(path))
    meta = {'cancer': cancer, 'source_hashes': hashes, 'common_genes': len(genes),
            'gene_sha256': sha256_array(genes), 'gmt_sha256': sha256_file(gmt),
            'pathways': pathways.tolist(), 'measured_members': [len(v) for v in members],
            'endpoint': 'OS', 'source': names[0], 'external': list(names[1:]),
            'qualification': 'external labels used for valid-ID/time/event eligibility; never supplied to model callback',
            'ranking': 'all cohorts reranked on cancer-specific intersection; no neutral padding',
            'scope': '4383 Hallmark-union candidate panel; these three cancer panels differ from BRCA9703'}
    atomic_json(directory / 'preparation.json', meta)
    return expressions, labels, meta


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    files = ('run_multicancer_backbone_v1.py', 'run_backbone_objective_v1.py',
             'backbone_objective_model.py', 'graph_model14.py', 'original_weibull_loss_corrected.py',
             'original_round10/model_snapshot_0.py', 'unified_experiment_base.py',
             'experiment_base.py', 'run_activity22_coxnet.py')
    metrics = []
    for cancer, names in PANELS.items():
        expressions, labels, meta = prepare_inputs(cancer, names)
        source = names[0]
        ext_labels = {n: labels[n] for n in names[1:]}
        protocol = UnifiedProtocol('multicancer_backbone_v1_' + cancer,
            development_cohorts=(source,), feature_version='per-cancer-measured-common-panel-h50-variance1000',
            model_version='original-fusion-fixed-prior-mechanism', grid_version='0-1095-61',
            extra={'arms': list(CONFIGS), 'configs': CONFIGS, 'seeds': SEEDS, 'inputs': meta,
                   'external_status': 'all existing development-exposed cohorts; no result-based replacement',
                   'max_epochs': 160, 'patience': 25, 'lr': .0003, 'weight_decay': .01,
                   'batch': 128, 'early_stop': 'cohort-event20%seed17 inside each train block; five-seed median epoch refit',
                   'scope': 'same original architecture and prior-removal change; no new LLM input',
                   'code_hashes': {f: sha256_file(ROOT/f) for f in files}})
        directory = OUT / cancer
        runner = UnifiedExperimentRunner(output_dir=directory, protocol=protocol,
            development=[CohortBundle(expressions[source], labels[source])],
            external={n: expressions[n] for n in names[1:]}, grid=GRID, arms=list(CONFIGS),
            outcome_loader=lambda: ext_labels,
            outcome_sources={n: labels[n].source_path for n in names[1:]}).prepare()
        for name in files:
            dest = directory / 'code_at_run' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(ROOT/name, dest)
        print('CANCER', cancer, 'n', runner.data.n, 'events', int(runner.data.event.sum()),
              'common_genes', len(runner.data.genes), flush=True)
        for label, options in CONFIGS.items():
            runner.run_arm(label, ModelCallback('WeibullLegacy', protocol, output_dir=directory,
                label=label, seeds=SEEDS, fit_options=options))
        frame = runner.score()
        frame['cancer'] = cancer
        additional_metrics(runner, ext_labels)
        metrics.append(frame)
        print(frame[['cohort','arm','uno_c_3y','harrell_c','brier_3y']].to_string(index=False), flush=True)
    pd.concat(metrics).to_csv(OUT / 'metrics.csv', index=False)
    atomic_json(OUT/'complete.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS', 'cancers': list(PANELS), 'metrics_sha256': sha256_file(OUT/'metrics.csv')})


if __name__ == '__main__':
    main()
