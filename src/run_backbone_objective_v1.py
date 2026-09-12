"""One fixed experiment: retrain the original fusion under survival objectives.

No LLM increment is claimed by this experiment. It tests the primary backbone
bottleneck before spending more compute on an additional knowledge branch.
"""
from __future__ import annotations

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time as clock

import numpy as np
import pandas as pd
import torch
from importlib.metadata import version

import graph_model14 as gm
import backbone_objective_model as bm
from experiment_base import (atomic_json, atomic_npz, inner_split, sha256_file,
                             save_prediction, load_prediction, load_standard_outcomes)
from unified_experiment_base import UnifiedProtocol, UnifiedExperimentRunner
from run_activity22_coxnet import GRID, survival_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cohort_trial28' / 'backbone_objective_v1'
EXTERNAL = ('GSE7390', 'GSE1456', 'GSE42568', 'GSE199633')
MAX_EPOCHS = 160


class ModelCallback:
    def __init__(self, arm, protocol):
        self.arm, self.protocol = arm, protocol
        self.cache = {}

    def __call__(self, context, train, test):
        directory = OUT / self.arm / 'models' / context.name
        directory.mkdir(parents=True, exist_ok=True)
        if context.name not in self.cache:
            start = clock.perf_counter()
            it, iv = inner_split(np.arange(train.n), train.cohort, train.event, .2, 17)
            pp0 = gm.fit_preprocess(train.x[it], train.p[it], train.genes)
            gi, pi = gm.transform(train.x[it], train.p[it], pp0)
            gv, pv = gm.transform(train.x[iv], train.p[iv], pp0)
            validation = (gv, pv, train.time[iv], train.event[iv], train.cohort[iv])
            stopping = []
            for seed in bm.SEEDS:
                model, epochs, history = bm.fit_model(
                    gi, pi, train.time[it], train.event[it], train.cohort[it],
                    self.arm, seed, epochs=MAX_EPOCHS, validation=validation)
                pd.DataFrame(history).to_csv(directory / f'stop_seed{seed}.csv', index=False)
                torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                            'preprocess': pp0, 'arm': self.arm, 'seed': seed,
                            'epochs': epochs, 'fit_ids': train.ids[it], 'stop_ids': train.ids[iv],
                            'protocol_sha256': self.protocol.sha256},
                           directory / f'stop_seed{seed}.pt')
                stopping.append(epochs)
                del model
            epochs = int(round(float(np.median(stopping))))
            pp = gm.fit_preprocess(train.x, train.p, train.genes)
            gt, pt = gm.transform(train.x, train.p, pp)
            members, verification = [], []
            for seed in bm.SEEDS:
                model, _, history = bm.fit_model(
                    gt, pt, train.time, train.event, train.cohort,
                    self.arm, seed, epochs=epochs)
                pd.DataFrame(history).to_csv(directory / f'refit_seed{seed}.csv', index=False)
                parameters = bm.predict_parameters(model, gt, pt)
                baseline = (bm.fit_baselines(parameters, train.time, train.event,
                                             train.cohort, self.arm == 'CoxStratified')
                            if self.arm.startswith('Cox') else None)
                state = {k: v.cpu() for k, v in model.state_dict().items()}
                checkpoint = {'state_dict': state, 'preprocess': pp, 'baselines': baseline,
                              'fit_ids': train.ids, 'stop_train_ids': train.ids[it],
                              'stop_valid_ids': train.ids[iv], 'arm': self.arm,
                              'seed': seed, 'epochs': epochs, 'seed_stop_epochs': stopping,
                              'protocol_sha256': self.protocol.sha256}
                path = directory / f'seed{seed}.pt'
                torch.save(checkpoint, path)
                # Check the saved model, not only the in-memory predictor.
                saved = torch.load(path, map_location='cpu', weights_only=False)
                reloaded = bm.BackboneModel(self.arm).cuda()
                reloaded.load_state_dict(saved['state_dict'])
                expected = bm.predict_parameters(model, gt[:7], pt[:7])
                actual = bm.predict_parameters(reloaded, gt[:7], pt[:7])
                one = bm.predict_parameters(reloaded, gt[:1], pt[:1])
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(one, expected[:1], rtol=1e-5, atol=1e-5)
                verification.append({'seed': seed, 'checkpoint_sha256': sha256_file(path),
                                     'reload_max_error': float(abs(actual-expected).max()),
                                     'singleton_max_error': float(abs(one-expected[:1]).max())})
                members.append((state, baseline, sha256_file(path)))
                del model, reloaded
            self.cache[context.name] = (pp, members)
            atomic_json(directory / 'fit_complete.json', {
                'protocol_sha256': self.protocol.sha256, 'fit_ids': train.ids.tolist(),
                'epochs': epochs, 'seed_stop_epochs': stopping, 'verification': verification,
                'parameter_count': sum(v.numel() for v in bm.BackboneModel(self.arm).parameters()),
                'seconds': clock.perf_counter()-start})
            print(json.dumps({'arm': self.arm, 'context': context.name, 'epochs': epochs,
                              'seed_stop_epochs': stopping, 'seconds': round(clock.perf_counter()-start, 2)}), flush=True)
        pp, members = self.cache[context.name]
        g, p = gm.transform(test.x, test.p, pp)
        predictions = []
        for seed, (state, baseline, digest) in zip(bm.SEEDS, members):
            model = bm.BackboneModel(self.arm).cuda()
            model.load_state_dict(state)
            parameters = bm.predict_parameters(model, g, p)
            curves = bm.curves_from_parameters(parameters, self.arm, GRID, baseline)
            save_prediction(directory / f'{test.cohort}_seed{seed}.npz', test.ids, curves, GRID)
            atomic_npz(directory / f'{test.cohort}_seed{seed}_parameters.npz',
                       ids=test.ids, parameters=parameters, grid=GRID)
            predictions.append(curves)
            del model
        return np.mean(predictions, axis=0)


def additional_metrics(runner):
    rows = []
    for arm in bm.ARMS:
        ids, curves, _ = load_prediction(OUT / arm / 'source_oof.npz')
        np.testing.assert_array_equal(ids, runner.data.ids)
        for cohort in np.unique(runner.data.cohort):
            ix = runner.data.cohort == cohort
            met, _ = survival_metrics(runner.data.time[ix], runner.data.event[ix], curves[ix])
            rows.append({'cohort': cohort, 'role': 'source_oof', 'arm': arm,
                         'seed': 'ensemble', **met})
    labels = load_standard_outcomes(ROOT, list(EXTERNAL))
    for arm in bm.ARMS:
        for name, label in labels.items():
            for seed in bm.SEEDS:
                ids, curves, _ = load_prediction(OUT / arm / 'models' / 'full' / f'{name}_seed{seed}.npz')
                ix = pd.Index(label.ids).get_indexer(ids)
                assert (ix >= 0).all() and len(np.unique(ix)) == len(ix)
                met, _ = survival_metrics(label.time[ix], label.event[ix], curves)
                rows.append({'cohort': name, 'role': 'external_development_exposed',
                             'arm': arm, 'seed': seed, **met})
    pd.DataFrame(rows).to_csv(OUT / 'detailed_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checks = bm.verify_numerics()
    files = ('run_backbone_objective_v1.py', 'backbone_objective_model.py',
             'graph_model14.py', 'original_weibull_loss_corrected.py',
             'original_round10/model_snapshot_0.py', 'unified_experiment_base.py',
             'experiment_base.py', 'run_activity22_coxnet.py')
    protocol = UnifiedProtocol(
        'backbone_objective_v1', feature_version='common9703-p50-original-reference-variance1000',
        model_version='original-fusion-retrained-objective-comparison', grid_version='0-1095-61',
        extra={'arms': bm.ARMS, 'seeds': bm.SEEDS, 'external_os': EXTERNAL,
               'external_status': 'all previously used in development',
               'hypothesis': 'End-to-end risk-set learning improves beyond frozen Weibull backbone',
               'max_epochs': MAX_EPOCHS, 'patience': 25, 'lr': .0003, 'weight_decay': .01,
               'early_stop': 'each objective validation loss on cohort-event 20% seed17; median seed epoch refit',
               'Cox': 'exact full-risk-set Breslow; stratification has equal event weights',
               'external_baseline': 'pooled Breslow for CoxPooled; equal mixture of two source strata for CoxStratified',
               'Weibull': 'original parameter regularizers; legacy minibatch128 vs fullbatch matched objectives',
               'LLM_scope': 'no new knowledge input; cannot establish LLM contribution',
               'numerical_checks': checks,
               'packages': {n: version(n) for n in ('torch', 'torchsurv', 'scikit-survival', 'numpy')},
               'code_hashes': {f: sha256_file(ROOT / f) for f in files}})
    runner = UnifiedExperimentRunner.from_standard_registry(
        ROOT, output_dir=OUT, protocol=protocol, grid=GRID, arms=bm.ARMS,
        external_names=EXTERNAL).prepare()
    if args.preflight:
        print(json.dumps({'PASS': True, 'numerical_checks': checks, 'n': runner.data.n,
                          'events': int(runner.data.event.sum()), 'genes': len(runner.data.genes),
                          'external': {k: v.n for k, v in runner.external.items()}}, indent=2))
        return
    started = datetime.now(timezone.utc).isoformat()
    for arm in bm.ARMS:
        runner.run_arm(arm, ModelCallback(arm, protocol))
    metrics = runner.score()
    additional_metrics(runner)
    atomic_json(OUT / 'analysis_complete.json', {'started_utc': started,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol.sha256,
        'metrics_sha256': sha256_file(OUT / 'metrics.csv'),
        'detailed_metrics_sha256': sha256_file(OUT / 'detailed_metrics.csv')})
    print(metrics[['cohort', 'arm', 'uno_c_3y', 'harrell_c', 'brier_3y']].to_string(index=False))


if __name__ == '__main__':
    main()
