"""Matched, nested source-CV Coxnet comparisons with decoupler activity inputs."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score
from sksurv.util import Surv
import decoupler as dc
import sksurv
from activity22_features import MolecularTransform, mask_arm, O, P, sha
from verify_activity22_features import load_molecular

ARMS = ['G', 'H', 'PD', 'GH', 'GP', 'GD', 'GPD']
L1 = [.1, .5, .9]
RATIOS = np.geomspace(1., .03, 12)
GRID = np.linspace(0., 1095., 61)

def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def survival_metrics(time, event, curves):
    assert np.isfinite(curves).all() and ((curves >= 0) & (curves <= 1)).all()
    assert (np.diff(curves, axis=1) <= 1e-8).all()
    risk = 1. - np.sum((curves[:, 1:] + curves[:, :-1]) * (np.diff(GRID)[None, :] / 2), axis=1) / 1095.
    y = Surv.from_arrays(event.astype(bool), time.astype(float))
    return {'uno_c_3y': float(concordance_index_ipcw(y, y, risk, tau=1095.)[0]), 'harrell_c': float(concordance_index_censored(event, time, risk)[0]), 'brier_3y': float(brier_score(y, y, curves[:, -1:], [1095.])[1][0]), 'n': len(time), 'events': int(event.sum())}, risk

def fit_path(x, y, l1, baseline=False):
    # The regularization scale is inferred using this training set only.
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        initial = CoxnetSurvivalAnalysis(l1_ratio=l1, n_alphas=2, alpha_min_ratio=.99, max_iter=100000).fit(x, y)
        alpha_max = float(initial.alphas_[0])
        model = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=alpha_max * RATIOS, tol=1e-7, max_iter=100000, fit_baseline_model=baseline).fit(x, y)
    if any(issubclass(note.category, ConvergenceWarning) for note in notes):
        raise RuntimeError('Coxnet path did not converge')
    assert np.isfinite(model.coef_).all()
    assert np.allclose(model.alphas_, alpha_max * RATIOS)
    return model, alpha_max

def refit(x, y, l1, ratio):
    model, alpha_max = fit_path(x, y, l1, baseline=True)
    alpha = alpha_max * ratio
    assert np.isfinite(model.coef_).all()
    model.selected_alpha_ = alpha
    return model, alpha

def predict(model, x):
    curves = np.stack([np.where(GRID < fn.x[0], 1., fn(GRID)) for fn in model.predict_survival_function(x, alpha=model.selected_alpha_)])
    assert np.isfinite(curves).all()
    assert (curves[:, 0] == 1.).all()
    return curves

def cached_transform(folder, fit, evaluate, src, sx, genes, net, ex=None, eh=None):
    folder.mkdir(parents=True, exist_ok=True)
    model_file = folder / 'preprocess.joblib'
    data_file = folder / 'transformed.npz'
    if model_file.exists() and data_file.exists():
        obj = joblib.load(model_file)
        assert obj['fit_ids'] == src['ids'][fit].tolist()
        assert obj['code_hash'] == sha(P / 'activity22_features.py')
        z = np.load(data_file)
        assert obj['fingerprint'] == obj['preprocess'].fingerprint()
        return z['train'], z['test'], z['names'], obj['fingerprint']
    tf = MolecularTransform(genes, net, src['pathways']).fit(sx[fit], src['p'][fit])
    train, names = tf.transform(sx[fit], src['p'][fit])
    test, test_names = tf.transform(sx[evaluate], src['p'][evaluate]) if evaluate is not None else tf.transform(ex, eh)
    assert np.array_equal(names, test_names)
    obj = {'preprocess': tf, 'fit_ids': src['ids'][fit].tolist(), 'fingerprint': tf.fingerprint(), 'code_hash': sha(P / 'activity22_features.py')}
    joblib.dump(obj, model_file)
    np.savez_compressed(data_file, train=train, test=test, names=names)
    write(folder / 'preprocessing_manifest.json', {'fit_ids': obj['fit_ids'], 'fingerprint': obj['fingerprint'], 'artifact_sha256': sha(model_file), 'transformed_sha256': sha(data_file)})
    return train, test, names, obj['fingerprint']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    out = O / 'coxnet'; out.mkdir(exist_ok=True)
    feature_check = json.loads((O / 'feature_verification.json').read_text())
    assert feature_check['passed']
    for path, expected in feature_check['input_hashes'].items():
        assert sha(P / path) == expected
    src, ext, genes, sx, ex, eh = load_molecular()
    assert np.isfinite(src['time']).all() and (src['time'] > 0).all()
    assert set(np.unique(src['event'])) <= {0, 1}
    net = pd.concat([pd.read_csv(O / f'{k}_network.tsv', sep='\t') for k in ['P', 'D']], ignore_index=True)
    protocol = {'created_for': 'Development iteration; SCAN-B already exposed', 'arms': ARMS, 'model': 'scikit-survival CoxnetSurvivalAnalysis (elastic net)', 'sksurv': sksurv.__version__, 'decoupler': dc.__version__, 'inner_cv': '3 event-stratified folds seed202 within each outer training set; all transforms refitted', 'outer_cv': '3 event-stratified folds seed101, matching trial16', 'l1_ratios': L1, 'alpha_ratios_to_training_max': RATIOS.tolist(), 'selection': 'Largest mean inner 3-year Uno C; exact ties favor stronger shrinkage, then smaller l1 ratio', 'primary_metric': '3y Uno C; 1-normalized RMST for saved survival curves', 'censoring_estimator': 'KM in evaluated population for legacy comparability; never used to fit molecular features or model', 'refit': 'All outer-training patients; all-source model uses its own nested inner selection', 'external_use': 'Report all arms for development transfer; no independent validation claim', 'inputs': '13254 measured genes, identical within-patient reranking; source-centered inverse-normal ranks for official ULM; no clinical, image, GenePT or LLM branch', 'scoring_note': 'ULM score is a transcriptomic proxy, not measured protein activation or patient risk', 'source_hashes': {f: sha(P / f) for f in ['run_activity22_coxnet.py', 'activity22_features.py', 'verify_activity22_features.py', 'fusion_trial16/source_inputs.npz', 'activity_trial22/P_network.tsv', 'activity_trial22/D_network.tsv']}}
    protocol_file = out / 'protocol.json'
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text()) == protocol, 'Do not mix code/parameters in a locked run'
    else:
        write(protocol_file, protocol)
    ph = sha(protocol_file)
    y = Surv.from_arrays(src['event'].astype(bool), src['time'])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=101).split(sx, src['event']))
    contexts = [(f'outer{k}', tr, te) for k, (tr, te) in enumerate(folds)] + [('full', np.arange(len(sx)), None)]
    selected_arms = [args.arm] if args.arm else ARMS
    for context, fit, test in contexts:
        inner_data = []
        for j, (it, iv) in enumerate(StratifiedKFold(3, shuffle=True, random_state=202).split(fit, src['event'][fit])):
            it, iv = fit[it], fit[iv]
            xi, xv, names, fh = cached_transform(out / 'features' / context / f'inner{j}', it, iv, src, sx, genes, net)
            inner_data.append((xi, xv, names, y[it], y[iv], fh))
        xf, xt, names, fh = cached_transform(out / 'features' / context / 'refit', fit, test, src, sx, genes, net, ex, eh)
        for arm in selected_arms:
            arm_dir = out / arm; arm_dir.mkdir(exist_ok=True)
            done_path = arm_dir / f'{context}_complete.json'
            checkpoint = arm_dir / f'{context}.joblib'
            if done_path.exists():
                old = json.loads(done_path.read_text())
                assert old['protocol_sha256'] == ph and old['model_sha256'] == sha(checkpoint)
                assert old['prediction_sha256'] == sha(arm_dir / f'{context}_predictions.npz')
                continue
            start = time.time(); scores = []
            for j, (xi, xv, innames, yi, yv, inner_hash) in enumerate(inner_data):
                mask = mask_arm(innames, arm)
                for l1 in L1:
                    path, alpha_max = fit_path(xi[:, mask], yi, l1)
                    for ratio in RATIOS:
                        risk = path.predict(xv[:, mask], alpha=alpha_max * ratio)
                        score = concordance_index_ipcw(yv, yv, risk, tau=1095.)[0]
                        scores.append({'inner': j, 'l1_ratio': l1, 'alpha_ratio': float(ratio), 'alpha': alpha_max * ratio, 'uno_c_3y': float(score), 'preprocess_fingerprint': inner_hash})
            score_df = pd.DataFrame(scores)
            means = score_df.groupby(['l1_ratio', 'alpha_ratio'], as_index=False).uno_c_3y.mean().sort_values(['uno_c_3y', 'alpha_ratio', 'l1_ratio'], ascending=[False, False, True])
            choice = means.iloc[0]
            score_df.to_csv(arm_dir / f'{context}_inner_scores.csv', index=False)
            mask = mask_arm(names, arm)
            model, alpha = refit(xf[:, mask], y[fit], float(choice.l1_ratio), float(choice.alpha_ratio))
            ck = {'model': model, 'mask': mask, 'feature_names': names, 'fit_ids': src['ids'][fit].tolist(), 'preprocess_fingerprint': fh, 'protocol_sha256': ph}
            joblib.dump(ck, checkpoint)
            curves = predict(model, xt[:, mask])
            ids = ext['ids'] if test is None else src['ids'][test]
            np.savez_compressed(arm_dir / f'{context}_predictions.npz', ids=ids, survival=curves, grid=GRID)
            selected_index = int(np.argmin(np.abs(model.alphas_ - alpha)))
            row = {'context': context, 'arm': arm, 'selected_l1': float(choice.l1_ratio), 'selected_alpha_ratio': float(choice.alpha_ratio), 'alpha': alpha, 'inner_uno': float(choice.uno_c_3y), 'nonzero_coefficients': int(np.count_nonzero(model.coef_[:, selected_index])), 'features': int(mask.sum()), 'seconds': round(time.time() - start, 2), 'protocol_sha256': ph, 'model_sha256': sha(checkpoint), 'prediction_sha256': sha(arm_dir / f'{context}_predictions.npz')}
            if test is not None:
                row.update(survival_metrics(src['time'][test], src['event'][test], curves)[0])
            write(done_path, row)
            pd.DataFrame({'feature': names[mask], 'coefficient': model.coef_[:, selected_index]}).to_csv(arm_dir / f'{context}_coefficients.csv', index=False)
            print(json.dumps(row), flush=True)
    if not all((out / arm / 'full_complete.json').exists() for arm in ARMS):
        return
    write(out / 'source_complete.json', {'protocol_sha256': ph, 'full_models': {arm: sha(out / arm / 'full.joblib') for arm in ARMS}, 'at_utc': datetime.now(timezone.utc).isoformat()})
    # No external outcome table is opened until after all source choices are saved.
    target = pd.read_csv(P / 'clinical_round9/BRCA_target_population.csv')
    assert target['sample'].is_unique and target['donor'].is_unique
    ext_ids = target['sample'].to_numpy(str)
    assert set(ext_ids) == set(ext['ids'])
    rows = []
    for arm in ARMS:
        for context, fit, test in contexts:
            pred = np.load(out / arm / f'{context}_predictions.npz')
            if test is None:
                lookup = {value: j for j, value in enumerate(pred['ids'])}
                order = [lookup[value] for value in ext_ids]
                t, e = target.time.to_numpy(float), target.event.to_numpy(bool)
                curves = pred['survival'][order]
                met, risk = survival_metrics(t, e, curves)
                np.savez_compressed(out / arm / 'external_ensemble.npz', ids=ext_ids, time=t, event=e, survival=curves, risk=risk, grid=GRID)
                rows.append({'cohort': 'SCAN-B', 'arm': arm, 'fold': 'full', **met})
            else:
                assert np.array_equal(pred['ids'], src['ids'][test])
                met, risk = survival_metrics(src['time'][test], src['event'][test], pred['survival'])
                rows.append({'cohort': 'TCGA', 'arm': arm, 'fold': context, **met})
    frame = pd.DataFrame(rows); frame.to_csv(out / 'fold_metrics.csv', index=False)
    summary = frame.groupby(['cohort', 'arm'])[['uno_c_3y', 'harrell_c', 'brier_3y']].mean().reset_index()
    summary.to_csv(out / 'summary_metrics.csv', index=False)
    print(summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
