"""Shared experiment infrastructure for GeneNarrator survival studies.

Model scripts should implement only feature construction and model fitting.
This module owns data contracts, endpoint separation, nested splits, artifact
validation and deterministic, atomic output.  It deliberately keeps outcome
loading separate from expression loading so an external label cannot be read
accidentally during prediction generation.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class ContractError(ValueError):
    """Raised when an experiment artifact violates a declared contract."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(values: Any) -> str:
    """Stable hash for a NumPy array, including dtype and shape metadata."""
    arr = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(canonical_json(list(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _strings(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values).astype(str)
    if arr.ndim != 1:
        raise ContractError(f"{name} must be one-dimensional")
    if np.any(arr == "") or np.any(arr == "None"):
        raise ContractError(f"{name} contains empty identifiers")
    if len(np.unique(arr)) != len(arr):
        raise ContractError(f"{name} contains duplicate identifiers")
    return arr


@dataclass(frozen=True)
class ExpressionData:
    cohort: str
    ids: np.ndarray
    genes: np.ndarray
    x: np.ndarray
    p: np.ndarray
    source_path: str

    @property
    def n(self) -> int:
        return int(len(self.ids))


@dataclass(frozen=True)
class Outcomes:
    cohort: str
    ids: np.ndarray
    time: np.ndarray
    event: np.ndarray
    endpoint: str
    source_path: str

    @property
    def n(self) -> int:
        return int(len(self.ids))


@dataclass(frozen=True)
class CohortSpec:
    """Declarative description of one expression/outcome source."""
    name: str
    expression_path: Path
    endpoint: str
    label_path: Path | None = None
    label_mode: str = "embedded"  # embedded, npz, csv
    id_column: str = "ids"
    time_column: str = "time"
    event_column: str = "event"


@dataclass(frozen=True)
class CohortBundle:
    expression: ExpressionData
    outcomes: Outcomes


def load_cohort(spec: CohortSpec) -> CohortBundle:
    """Load one registered cohort using the declared label boundary."""
    expression = load_expression_npz(spec.expression_path, cohort=spec.name)
    mode = spec.label_mode.lower()
    if mode == "embedded":
        outcomes = load_embedded_outcomes_npz(expression, spec.expression_path, spec.endpoint)
    elif mode == "npz":
        if spec.label_path is None:
            raise ContractError(f"{spec.name}: npz label_path is required")
        outcomes = load_outcomes_npz(expression, spec.label_path, spec.endpoint)
    elif mode == "csv":
        if spec.label_path is None:
            raise ContractError(f"{spec.name}: csv label_path is required")
        frame = pd.read_csv(spec.label_path)
        outcomes = load_outcomes_frame(
            expression, frame, spec.id_column, spec.time_column,
            spec.event_column, spec.endpoint, str(spec.label_path),
        )
    else:
        raise ContractError(f"{spec.name}: unsupported label_mode {spec.label_mode!r}")
    return CohortBundle(expression, outcomes)


def standard_cohort_specs(root: str | Path) -> dict[str, CohortSpec]:
    """Return the single registry used by all current GeneNarrator trials."""
    root = Path(root)
    cand = root / "cohort_trial28" / "confirmation_candidates"
    scanb_labels = root / "clinical_round8" / "SCANB_outcomes_original.csv"
    return {
        "development": CohortSpec(
            "development", cand / "development_model_inputs.npz", "OS", label_mode="embedded"),
        "scanb": CohortSpec(
            "scanb", cand / "scanb_model_inputs.npz", "OS", scanb_labels,
            label_mode="csv", id_column="gsm", time_column="overall survival days",
            event_column="overall survival event"),
        "GSE7390": CohortSpec(
            "GSE7390", cand / "GSE7390_model_inputs.npz", "OS", label_mode="embedded"),
        "GSE1456": CohortSpec(
            "GSE1456", cand / "GSE1456_model_inputs.npz", "OS", label_mode="embedded"),
        "GSE42568": CohortSpec(
            "GSE42568", cand / "GSE42568_model_inputs.npz", "OS",
            cand / "GSE42568_labels.npz", label_mode="npz"),
        "GSE199633": CohortSpec(
            "GSE199633", cand / "GSE199633_model_inputs.npz", "OS",
            cand / "GSE199633_labels.npz", label_mode="npz"),
        "GSE3494": CohortSpec(
            "GSE3494", cand / "GSE3494_model_inputs.npz", "DSS",
            cand / "GSE3494_labels.npz", label_mode="npz"),
    }


def load_standard_cohorts(root: str | Path,
                          names: Sequence[str] | None = None) -> dict[str, CohortBundle]:
    specs = standard_cohort_specs(root)
    selected = list(specs) if names is None else list(names)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise ContractError(f"unknown registered cohorts: {unknown}")
    bundles = {name: load_cohort(specs[name]) for name in selected}
    assert_same_gene_axis([b.expression for b in bundles.values()])
    return bundles


def load_standard_expressions(root: str | Path,
                              names: Sequence[str] | None = None) -> dict[str, ExpressionData]:
    """Load only expression inputs from the registry (no outcome access)."""
    specs = standard_cohort_specs(root)
    selected = list(specs) if names is None else list(names)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise ContractError(f"unknown registered cohorts: {unknown}")
    data = {name: load_expression_npz(specs[name].expression_path, cohort=name)
            for name in selected}
    assert_same_gene_axis(list(data.values()))
    return data


def load_standard_outcomes(root: str | Path,
                           names: Sequence[str] | None = None) -> dict[str, Outcomes]:
    """Load labels explicitly from the registry, after predictions are frozen."""
    specs = standard_cohort_specs(root)
    selected = list(specs) if names is None else list(names)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise ContractError(f"unknown registered cohorts: {unknown}")
    outcomes = {}
    for name in selected:
        expression = load_expression_npz(specs[name].expression_path, cohort=name)
        outcomes[name] = load_cohort(specs[name]).outcomes
        if not np.array_equal(outcomes[name].ids, expression.ids):
            raise ContractError(f"{name}: outcome IDs changed between registry loads")
    return outcomes


def load_expression_npz(path: str | Path, cohort: str | None = None) -> ExpressionData:
    """Read and validate expression/pathway arrays without reading outcomes."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        required = {"ids", "genes", "x", "p"}
        missing = required.difference(z.files)
        if missing:
            raise ContractError(f"{path}: missing arrays {sorted(missing)}")
        ids = _strings(z["ids"], f"{path}:ids")
        genes = _strings(z["genes"], f"{path}:genes")
        x = np.asarray(z["x"], dtype=np.float32)
        p = np.asarray(z["p"], dtype=np.float32)
    if x.ndim != 2 or x.shape != (len(ids), len(genes)):
        raise ContractError(f"{path}: x shape {x.shape} != ({len(ids)}, {len(genes)})")
    if p.ndim != 2 or p.shape[0] != len(ids):
        raise ContractError(f"{path}: p shape {p.shape} has wrong patient axis")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ContractError(f"{path}: non-finite expression/pathway values")
    return ExpressionData(cohort or path.stem, ids, genes, x, p, str(path))


def _align_outcomes(ids: np.ndarray, label_ids: np.ndarray, time: np.ndarray,
                    event: np.ndarray, cohort: str, endpoint: str,
                    source_path: str) -> Outcomes:
    ids = _strings(ids, f"{cohort}:input_ids")
    label_ids = _strings(label_ids, f"{source_path}:label_ids")
    if set(ids) != set(label_ids):
        only_input = sorted(set(ids) - set(label_ids))[:5]
        only_label = sorted(set(label_ids) - set(ids))[:5]
        raise ContractError(f"{cohort}: ID set mismatch; input_only={only_input}, label_only={only_label}")
    pos = {v: i for i, v in enumerate(label_ids)}
    order = np.asarray([pos[v] for v in ids], dtype=int)
    t = np.asarray(time, dtype=float)[order]
    e = np.asarray(event).astype(bool)[order]
    if t.shape != ids.shape or e.shape != ids.shape:
        raise ContractError(f"{cohort}: outcome lengths do not match IDs")
    if not np.isfinite(t).all() or not (t > 0).all():
        raise ContractError(f"{cohort}: times must be finite and > 0")
    if not np.isin(e, [False, True]).all():
        raise ContractError(f"{cohort}: event must be binary")
    endpoint = str(endpoint).upper()
    if endpoint not in {"OS", "DSS", "DFS", "PFS"}:
        raise ContractError(f"{cohort}: unknown endpoint {endpoint}")
    return Outcomes(cohort, ids, t, e, endpoint, source_path)


def load_outcomes_npz(expression: ExpressionData, path: str | Path,
                      endpoint: str) -> Outcomes:
    """Load a label archive only when the caller explicitly requests outcomes."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        required = {"ids", "time", "event"}
        missing = required.difference(z.files)
        if missing:
            raise ContractError(f"{path}: missing arrays {sorted(missing)}")
        return _align_outcomes(expression.ids, z["ids"], z["time"], z["event"],
                               expression.cohort, endpoint, str(path))


def load_embedded_outcomes_npz(expression: ExpressionData, path: str | Path,
                               endpoint: str) -> Outcomes:
    """Read outcomes explicitly from an expression archive that embeds labels.

    Several legacy model-input archives contain ``ids``, ``x`` and ``p`` together
    with ``time`` and ``event``.  The normal expression loader deliberately
    ignores labels.  Callers must use this named function when the archive is
    intentionally being opened for training/evaluation, which makes the label
    boundary visible in every runner and audit log.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        required = {"ids", "time", "event"}
        missing = required.difference(z.files)
        if missing:
            raise ContractError(f"{path}: missing embedded outcome arrays {sorted(missing)}")
        return _align_outcomes(expression.ids, z["ids"], z["time"], z["event"],
                               expression.cohort, endpoint, str(path) + "::embedded_outcomes")


def load_outcomes_frame(expression: ExpressionData, frame: pd.DataFrame,
                        id_column: str, time_column: str, event_column: str,
                        endpoint: str, source_path: str) -> Outcomes:
    if id_column not in frame or time_column not in frame or event_column not in frame:
        raise ContractError(f"{source_path}: missing outcome columns")
    if frame[id_column].astype(str).duplicated().any():
        raise ContractError(f"{source_path}: duplicate label IDs")
    return _align_outcomes(expression.ids, frame[id_column].astype(str).to_numpy(),
                           frame[time_column].to_numpy(), frame[event_column].to_numpy(),
                           expression.cohort, endpoint, source_path)


def assert_same_gene_axis(data: Sequence[ExpressionData]) -> np.ndarray:
    if not data:
        raise ContractError("no expression cohorts supplied")
    genes = data[0].genes
    for d in data[1:]:
        if not np.array_equal(d.genes, genes):
            raise ContractError(f"gene axis mismatch: {data[0].cohort} vs {d.cohort}")
    return genes.copy()


def combine_expression(data: Sequence[ExpressionData], outcomes: Sequence[Outcomes] | None = None):
    """Concatenate expression cohorts after checking axes and optional outcomes."""
    genes = assert_same_gene_axis(data)
    ids = np.concatenate([d.ids for d in data]).astype(str)
    if len(np.unique(ids)) != len(ids):
        raise ContractError("duplicate patient IDs across development cohorts")
    x = np.concatenate([d.x for d in data], axis=0)
    p = np.concatenate([d.p for d in data], axis=0)
    if outcomes is None:
        return ids, genes, x, p
    if len(outcomes) != len(data):
        raise ContractError("one outcome table is required for each cohort")
    for d, o in zip(data, outcomes):
        if d.cohort != o.cohort or not np.array_equal(d.ids, o.ids):
            raise ContractError(f"outcome alignment mismatch for {d.cohort}")
    time = np.concatenate([o.time for o in outcomes])
    event = np.concatenate([o.event for o in outcomes])
    cohort = np.concatenate([np.repeat(d.cohort, d.n) for d in data]).astype(str)
    endpoint = {o.endpoint for o in outcomes}
    if len(endpoint) != 1:
        raise ContractError(f"cannot combine different endpoints: {sorted(endpoint)}")
    return ids, genes, x, p, time, event, cohort, next(iter(endpoint))


def cohort_event_strata(cohort: Sequence[str], event: Sequence[bool]) -> np.ndarray:
    c = np.asarray(cohort).astype(str)
    e = np.asarray(event).astype(bool)
    if c.ndim != 1 or e.ndim != 1 or len(c) != len(e):
        raise ContractError("cohort/event arrays have incompatible shapes")
    return np.asarray([f"{a}|{int(b)}" for a, b in zip(c, e)], dtype=str)


def outer_folds(n: int, cohort: Sequence[str], event: Sequence[bool],
                n_splits: int = 3, seed: int = 101):
    strata = cohort_event_strata(cohort, event)
    counts = pd.Series(strata).value_counts()
    if (counts < n_splits).any():
        raise ContractError(f"not enough observations in cohort-event stratum: {counts.to_dict()}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(splitter.split(np.arange(n), strata)):
        if set(tr) & set(te):
            raise ContractError(f"outer fold {k} overlaps")
        folds.append((tr.astype(int), te.astype(int)))
    if len(np.concatenate([te for _, te in folds])) != n or len(np.unique(np.concatenate([te for _, te in folds]))) != n:
        raise ContractError("outer folds do not partition patients exactly once")
    return folds


def inner_split(indices: Sequence[int], cohort: Sequence[str], event: Sequence[bool],
                test_size: float = .2, seed: int = 17):
    indices = np.asarray(indices, dtype=int)
    strata = cohort_event_strata(np.asarray(cohort)[indices], np.asarray(event)[indices])
    counts = pd.Series(strata).value_counts()
    if (counts < 2).any():
        raise ContractError(f"not enough observations for inner split: {counts.to_dict()}")
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                       random_state=seed).split(indices, strata))
    train, valid = indices[a], indices[b]
    if set(train) & set(valid) or len(train) + len(valid) != len(indices):
        raise ContractError("inner split overlap or loss")
    return train.astype(int), valid.astype(int)


def inner_folds(indices: Sequence[int], cohort: Sequence[str], event: Sequence[bool],
                n_splits: int = 3, seed: int = 202):
    """Stratified inner CV over only an outer-training index set."""
    indices = np.asarray(indices, dtype=int)
    strata = cohort_event_strata(np.asarray(cohort)[indices], np.asarray(event)[indices])
    counts = pd.Series(strata).value_counts()
    if (counts < n_splits).any():
        raise ContractError(f"not enough observations for inner folds: {counts.to_dict()}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (a, b) in enumerate(splitter.split(indices, strata)):
        train, valid = indices[a].astype(int), indices[b].astype(int)
        if set(train) & set(valid) or len(train) + len(valid) != len(indices):
            raise ContractError(f"inner fold {k} overlap or loss")
        folds.append((train, valid))
    return folds


def development_arrays(bundles: Mapping[str, CohortBundle],
                       names: Sequence[str] = ("development", "scanb"),
                       endpoint: str = "OS"):
    """Concatenate the declared development cohorts after endpoint checks."""
    selected = [bundles[name] for name in names]
    assert_endpoint([b.outcomes for b in selected], endpoint)
    return combine_expression(
        [b.expression for b in selected], [b.outcomes for b in selected]
    )


def development_contexts(cohort: Sequence[str], event: Sequence[bool],
                         n_splits: int = 3, seed: int = 101,
                         include_cohort_holdouts: bool = True,
                         include_full: bool = True):
    """Build the fixed source contexts used for nested training.

    Every returned tuple is ``(name, train_indices, test_indices_or_None)``.
    The outer folds are a true patient partition.  Cohort holdouts and the
    full fit are additional contexts and are never mixed into the pooled OOF
    estimate.
    """
    cohort = np.asarray(cohort).astype(str)
    event = np.asarray(event).astype(bool)
    n = len(cohort)
    folds = outer_folds(n, cohort, event, n_splits=n_splits, seed=seed)
    contexts = [(f"outer{k}", tr, te) for k, (tr, te) in enumerate(folds)]
    if include_cohort_holdouts:
        for held in np.unique(cohort):
            tr = np.flatnonzero(cohort != held).astype(int)
            te = np.flatnonzero(cohort == held).astype(int)
            if len(tr) == 0 or len(te) == 0:
                raise ContractError(f"invalid cohort holdout {held!r}")
            contexts.append((f"holdout_{held}", tr, te))
    if include_full:
        contexts.append(("full", np.arange(n, dtype=int), None))
    return contexts


def validate_curves(ids: Sequence[str], curves: np.ndarray, expected_grid: np.ndarray | None = None):
    ids = _strings(ids, "prediction_ids")
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] != len(ids):
        raise ContractError(f"prediction shape {curves.shape} does not match IDs {len(ids)}")
    if not np.isfinite(curves).all() or ((curves < 0) | (curves > 1)).any():
        raise ContractError("prediction curves contain non-finite or out-of-range values")
    if not np.allclose(curves[:, 0], 1.0, atol=1e-7):
        raise ContractError("survival curves must start at one")
    if (np.diff(curves, axis=1) > 1e-7).any():
        raise ContractError("survival curves are not non-increasing")
    if expected_grid is not None:
        grid = np.asarray(expected_grid, dtype=float)
        if len(grid) != curves.shape[1] or not np.isclose(grid[0], 0):
            raise ContractError("prediction grid mismatch")
    return curves


def validate_prediction_ids(expected_ids: Sequence[str], predicted_ids: Sequence[str]):
    """Require an exact one-to-one ID match before an artifact is scored."""
    expected = _strings(expected_ids, "expected_ids")
    predicted = _strings(predicted_ids, "predicted_ids")
    if set(expected) != set(predicted):
        raise ContractError(
            f"prediction ID mismatch; expected_only={sorted(set(expected)-set(predicted))[:5]}, "
            f"predicted_only={sorted(set(predicted)-set(expected))[:5]}"
        )
    return predicted


def save_prediction(path: str | Path, ids: Sequence[str], survival: np.ndarray,
                    grid: np.ndarray):
    """Validate and atomically save a prediction artifact."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) != np.asarray(survival).shape[1] or len(grid) == 0:
        raise ContractError("prediction grid has incompatible shape")
    if np.any(np.diff(grid) < 0) or not np.isclose(grid[0], 0):
        raise ContractError("prediction grid must be sorted and start at zero")
    curves = validate_curves(ids, survival, grid)
    atomic_npz(path, ids=np.asarray(ids).astype(str), survival=curves.astype(np.float32),
               grid=grid.astype(np.float32))


def load_prediction(path: str | Path):
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        required = {"ids", "survival", "grid"}
        missing = required.difference(z.files)
        if missing:
            raise ContractError(f"{path}: missing prediction arrays {sorted(missing)}")
        ids = _strings(z["ids"], f"{path}:ids")
        grid = np.asarray(z["grid"], dtype=float)
        curves = validate_curves(ids, z["survival"], grid)
    return ids, curves, grid


def assert_endpoint(outcomes: Sequence[Outcomes], endpoint: str):
    endpoint = str(endpoint).upper()
    bad = [(o.cohort, o.endpoint) for o in outcomes if o.endpoint != endpoint]
    if bad:
        raise ContractError(f"endpoint mixing prohibited; expected {endpoint}, got {bad}")


def equal_weight_endpoint_summary(rows: pd.DataFrame, endpoint: str,
                                  cohort_column: str = "cohort") -> pd.DataFrame:
    """Average cohort metrics only after enforcing one endpoint and one row/cohort."""
    endpoint = str(endpoint).upper()
    if "endpoint" not in rows:
        raise ContractError("summary requires an endpoint column")
    if cohort_column not in rows:
        raise ContractError(f"summary requires cohort column {cohort_column!r}")
    q = rows[rows.endpoint.astype(str).str.upper() == endpoint].copy()
    if q.empty:
        raise ContractError(f"no rows for endpoint {endpoint}")
    if q[cohort_column].duplicated().any():
        raise ContractError("summary input has duplicate cohort rows")
    metric_cols = [c for c in ["uno_c_3y", "harrell_c", "brier_3y"] if c in q]
    return q[metric_cols].mean().to_frame().T.assign(endpoint=endpoint, cohorts=len(q))


class RunLock:
    """Prevent two processes from writing one run directory concurrently."""
    def __init__(self, directory: str | Path, name: str = ".running.lock"):
        self.directory = Path(directory); self.path = self.directory / name; self.acquired = False
    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"run already locked: {self.path}") from exc
        os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd); self.acquired = True
        atexit.register(self.release)
        return self
    def release(self):
        if self.acquired:
            self.path.unlink(missing_ok=True); self.acquired = False
    def __exit__(self, *_): self.release()


def atomic_json(path: str | Path, obj: Any):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False); fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path: str | Path, **arrays: Any):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)); os.close(fd)
    try:
        # Passing an open handle prevents NumPy from appending a second '.npz'
        # suffix to the temporary filename.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch_save(path: str | Path, payload: Any):
    """Atomically save a checkpoint; import Torch only for neural experiments."""
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as fh:
            torch.save(payload, fh); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_joblib_save(path: str | Path, payload: Any):
    """Atomically save a fitted sklearn pipeline without partial cache files."""
    import joblib
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as fh:
            joblib.dump(payload, fh); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def artifact_record(path: str | Path, **metadata: Any) -> dict[str, Any]:
    path = Path(path)
    if not path.exists(): raise ContractError(f"artifact does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path), **metadata}


def write_run_manifest(path: str | Path, protocol: Mapping[str, Any], artifacts: Iterable[str | Path]):
    rows = [artifact_record(a) for a in artifacts]
    obj = {"protocol": dict(protocol), "artifacts": rows,
           "protocol_sha256": hashlib.sha256(canonical_json(protocol).encode()).hexdigest()}
    atomic_json(path, obj)
    return obj
