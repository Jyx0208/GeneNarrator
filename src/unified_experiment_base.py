"""Unified, auditable experiment base for GeneNarrator survival studies.

This module is intentionally additive: legacy runners continue to import
``experiment_base`` unchanged.  New runners can use this module as the single
place for protocol locking, cohort-axis checks, nested split plans, prediction
completeness checks and the historical three-metric survival score.

The base does not fit a model.  It only makes the parts around model fitting
deterministic and checkable, so a feature or model change cannot silently alter
the endpoint, patient partition, gene axis, or scoring convention.

A valid fold plan proves index separation only.  It does not prove that a
loaded checkpoint or an upstream stacking model was trained without the held
out patients.  Honest split ensembles must additionally call
``validate_honest_split_provenance`` with the actual training-ID dependency
records saved by every upstream base fit.  Reconstructing those records from a
desired fold plan after training is not evidence of model training lineage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment_base import (
    CohortBundle,
    ContractError,
    ExpressionData,
    Outcomes,
    RunLock,
    artifact_record,
    assert_endpoint,
    atomic_json,
    atomic_npz,
    canonical_json,
    combine_expression,
    inner_folds,
    load_prediction,
    load_standard_cohorts,
    load_standard_expressions,
    load_standard_outcomes,
    outer_folds,
    save_prediction,
    sha256_array,
    sha256_file,
    standard_cohort_specs,
    validate_curves,
    validate_prediction_ids,
    write_run_manifest,
)


UNIFIED_BASE_VERSION = "unified-experiment-base-v1"
METRIC_CONVENTION = "rmst-risk-ipcw-1095-v1"
RUNNER_VERSION = "unified-runner-v1"


def _as_strings(values: Sequence[str], name: str) -> np.ndarray:
    arr = np.asarray(values).astype(str)
    if arr.ndim != 1 or len(arr) == 0:
        raise ContractError(f"{name} must be a non-empty one-dimensional array")
    if np.any(arr == "") or np.any(arr == "None"):
        raise ContractError(f"{name} contains empty identifiers")
    if len(np.unique(arr)) != len(arr):
        raise ContractError(f"{name} contains duplicate identifiers")
    return arr


@dataclass(frozen=True)
class UnifiedProtocol:
    """Immutable protocol declaration shared by one complete experiment.

    ``development_cohorts`` are the only cohorts used to fit a model.  External
    cohorts are evaluated by the caller after predictions are frozen; they are
    deliberately not part of the split generator.
    """

    name: str
    endpoint: str = "OS"
    horizon: float = 1095.0
    outer_splits: int = 3
    outer_seed: int = 101
    inner_splits: int = 3
    inner_seed: int = 202
    inner_test_size: float = 0.2
    development_cohorts: tuple[str, ...] = ("development", "scanb")
    allow_cohort_holdouts: bool = False
    grid_version: str = "grid-defined-by-runner"
    feature_version: str = "declared-by-runner"
    model_version: str = "declared-by-runner"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint).upper()
        if endpoint not in {"OS", "DSS", "DFS", "PFS"}:
            raise ContractError(f"unsupported protocol endpoint {self.endpoint!r}")
        object.__setattr__(self, "endpoint", endpoint)
        if not self.name or not str(self.name).strip():
            raise ContractError("protocol name must be non-empty")
        if not np.isfinite(float(self.horizon)) or float(self.horizon) <= 0:
            raise ContractError("protocol horizon must be positive and finite")
        if int(self.outer_splits) < 2 or int(self.inner_splits) < 2:
            raise ContractError("outer_splits and inner_splits must be at least two")
        if not 0 < float(self.inner_test_size) < 1:
            raise ContractError("inner_test_size must lie strictly between zero and one")
        if not self.development_cohorts:
            raise ContractError("at least one development cohort is required")
        if len(set(self.development_cohorts)) != len(self.development_cohorts):
            raise ContractError("development_cohorts contains duplicates")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["development_cohorts"] = list(self.development_cohorts)
        out["extra"] = dict(self.extra)
        out["base_version"] = UNIFIED_BASE_VERSION
        out["metric_convention"] = METRIC_CONVENTION
        return out

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()


class LockedProtocol:
    """Run-directory lock plus an immutable protocol declaration.

    The first process writes ``protocol.lock.json``.  Later invocations must
    provide the exact same protocol hash; this prevents an accidental change to
    endpoint, split seed or feature/model version from reusing a run directory.
    """

    def __init__(self, directory: str | Path, protocol: UnifiedProtocol):
        self.directory = Path(directory)
        self.protocol = protocol
        self._run_lock: RunLock | None = None

    @property
    def path(self) -> Path:
        return self.directory / "protocol.lock.json"

    def __enter__(self) -> "LockedProtocol":
        self._run_lock = RunLock(self.directory)
        self._run_lock.__enter__()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if existing.get("protocol_sha256") != self.protocol.sha256:
                    raise ContractError(
                        f"protocol lock mismatch at {self.path}; use a new run directory"
                    )
            else:
                atomic_json(self.path, {
                    "protocol": self.protocol.as_dict(),
                    "protocol_sha256": self.protocol.sha256,
                })
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        if self._run_lock is not None:
            self._run_lock.__exit__(exc_type, exc, tb)
            self._run_lock = None


def lock_protocol(directory: str | Path, protocol: UnifiedProtocol) -> LockedProtocol:
    """Return a context manager that acquires the run lock and freezes protocol."""
    return LockedProtocol(directory, protocol)


def audit_expression_axes(data: Sequence[ExpressionData], *, require_same_pathways: bool = True) -> dict[str, Any]:
    """Validate common gene/patient axes and return hashes for a manifest."""
    if not data:
        raise ContractError("no expression cohorts supplied")
    genes = _as_strings(data[0].genes, f"{data[0].cohort}:genes")
    for d in data[1:]:
        d_genes = _as_strings(d.genes, f"{d.cohort}:genes")
        if not np.array_equal(d_genes, genes):
            raise ContractError(f"gene axis mismatch: {data[0].cohort} vs {d.cohort}")
    rows: dict[str, Any] = {}
    p_dims = {int(d.p.shape[1]) for d in data}
    if require_same_pathways and len(p_dims) != 1:
        raise ContractError(f"pathway axis mismatch: {sorted(p_dims)}")
    all_ids: list[str] = []
    for d in data:
        ids = _as_strings(d.ids, f"{d.cohort}:ids")
        if d.x.shape != (d.n, len(genes)):
            raise ContractError(f"{d.cohort}: expression shape does not match axes")
        if d.p.ndim != 2 or d.p.shape[0] != d.n:
            raise ContractError(f"{d.cohort}: pathway shape does not match patients")
        if not np.isfinite(d.x).all() or not np.isfinite(d.p).all():
            raise ContractError(f"{d.cohort}: non-finite expression/pathway values")
        all_ids.extend(ids.tolist())
        rows[d.cohort] = {
            "n": d.n,
            "gene_count": int(len(genes)),
            "pathway_count": int(d.p.shape[1]),
            "ids_sha256": sha256_array(ids),
            "gene_sha256": sha256_array(d.genes),
        }
    if len(all_ids) != len(set(all_ids)):
        raise ContractError("duplicate patient IDs across supplied cohorts")
    return {
        "gene_count": int(len(genes)),
        "pathway_count": int(next(iter(p_dims))),
        "gene_sha256": sha256_array(genes),
        "cohorts": rows,
    }


def _check_partition(indices: Sequence[int], n: int, name: str) -> None:
    arr = np.asarray(indices, dtype=int)
    if arr.ndim != 1 or len(arr) == 0 or np.any(arr < 0) or np.any(arr >= n):
        raise ContractError(f"{name} contains invalid indices")
    if len(np.unique(arr)) != len(arr):
        raise ContractError(f"{name} contains duplicate indices")


def build_split_plan(
    cohort: Sequence[str],
    event: Sequence[bool],
    protocol: UnifiedProtocol,
    ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create and validate one nested split plan.

    Inner folds are generated only inside each outer-training set.  The returned
    JSON-compatible object stores indices and, when supplied, patient IDs so a
    later runner can prove that it used exactly this plan.

    This is an index contract, not a checkpoint provenance check.  In particular,
    ordinary global OOF predictions can have correct row/fold indices while the
    upstream models used to train a stacker include its current evaluation fold.
    """
    c = np.asarray(cohort).astype(str)
    e_raw = np.asarray(event)
    if e_raw.ndim != 1 or e_raw.dtype.kind not in "biuf" or not np.isfinite(e_raw).all() or not np.isin(e_raw, [0, 1]).all():
        raise ContractError("split event must contain only numeric 0/1 or boolean labels")
    e = e_raw.astype(bool)
    if c.ndim != 1 or e.ndim != 1 or len(c) != len(e) or len(c) == 0:
        raise ContractError("cohort and event must be aligned non-empty vectors")
    n = len(c)
    patient_ids = _as_strings(ids, "split_ids") if ids is not None else None
    if patient_ids is not None and len(patient_ids) != n:
        raise ContractError("split_ids length does not match cohort/event")
    outer = outer_folds(n, c, e, protocol.outer_splits, protocol.outer_seed)
    rows: list[dict[str, Any]] = []
    covered: list[int] = []
    for k, (tr, te) in enumerate(outer):
        _check_partition(tr, n, f"outer{k}.train")
        _check_partition(te, n, f"outer{k}.test")
        if set(tr) & set(te):
            raise ContractError(f"outer{k} train/test overlap")
        inner = inner_folds(tr, c, e, protocol.inner_splits, protocol.inner_seed + k)
        inner_rows = []
        inner_covered: list[int] = []
        for j, (itr, iva) in enumerate(inner):
            _check_partition(itr, n, f"outer{k}.inner{j}.train")
            _check_partition(iva, n, f"outer{k}.inner{j}.valid")
            if set(itr) & set(iva) or not set(itr) | set(iva) <= set(tr):
                raise ContractError(f"outer{k}.inner{j} is not contained in outer training set")
            inner_rows.append({"fold": j, "train": itr.tolist(), "valid": iva.tolist()})
            inner_covered.extend(iva.tolist())
        if len(inner_covered) != len(set(inner_covered)) or set(inner_covered) != set(tr):
            raise ContractError(f"outer{k} inner validation folds do not partition training set")
        rows.append({"fold": k, "train": tr.tolist(), "test": te.tolist(), "inner": inner_rows})
        covered.extend(te.tolist())
    if len(covered) != n or len(set(covered)) != n:
        raise ContractError("outer test folds do not partition all patients exactly once")
    plan: dict[str, Any] = {
        "base_version": UNIFIED_BASE_VERSION,
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.sha256,
        "n": int(n),
        "cohort_counts": {str(k): int(v) for k, v in pd.Series(c).value_counts().sort_index().items()},
        "event_count": int(e.sum()),
        "cohort_sha256": sha256_array(c),
        "event_sha256": sha256_array(e.astype(np.uint8)),
        "folds": rows,
    }
    if patient_ids is not None:
        plan["ids_sha256"] = sha256_array(patient_ids)
        plan["ids"] = patient_ids.tolist()
    validate_split_plan(plan, protocol=protocol, cohort=c, event=e, ids=patient_ids)
    return plan


def save_split_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    """Save a previously validated split plan as JSON."""
    validate_split_plan(plan)
    atomic_json(path, dict(plan))


def validate_split_plan(
    plan: Mapping[str, Any],
    *,
    protocol: UnifiedProtocol | None = None,
    cohort: Sequence[str] | None = None,
    event: Sequence[bool] | None = None,
    ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a stored split plan before a runner consumes it.

    Optional arrays make the check stronger: their hashes must match the plan,
    which catches accidentally reordering patients or changing labels after the
    split file was created.  The returned plain dictionary is safe to serialize.

    Passing this check does not establish safe stacking: the function does not
    inspect the training ancestry of any fitted estimator or prediction file.
    Use ``validate_honest_split_provenance`` for declared upstream training IDs.
    """
    if not isinstance(plan, Mapping) or "folds" not in plan or "protocol_sha256" not in plan or "n" not in plan:
        raise ContractError("invalid split plan")
    n = int(plan["n"])
    if n <= 0:
        raise ContractError("split plan n must be positive")
    if protocol is not None and str(plan["protocol_sha256"]) != protocol.sha256:
        raise ContractError("split plan protocol hash does not match the requested protocol")
    if cohort is not None or event is not None:
        if cohort is None or event is None:
            raise ContractError("cohort and event must be supplied together")
        c = np.asarray(cohort).astype(str)
        e_raw = np.asarray(event)
        if e_raw.ndim != 1 or e_raw.dtype.kind not in "biuf" or not np.isfinite(e_raw).all() or not np.isin(e_raw, [0, 1]).all():
            raise ContractError("split event must contain only numeric 0/1 or boolean labels")
        e = e_raw.astype(bool)
        if len(c) != n or len(e) != n:
            raise ContractError("split plan arrays have the wrong length")
        if "cohort_sha256" in plan and sha256_array(c) != plan["cohort_sha256"]:
            raise ContractError("split plan cohort hash mismatch")
        if "event_sha256" in plan and sha256_array(e.astype(np.uint8)) != plan["event_sha256"]:
            raise ContractError("split plan event hash mismatch")
    if ids is not None:
        pids = _as_strings(ids, "split_ids")
        if len(pids) != n or ("ids_sha256" in plan and sha256_array(pids) != plan["ids_sha256"]):
            raise ContractError("split plan ID hash mismatch")
    folds = plan["folds"]
    if not isinstance(folds, list) or not folds:
        raise ContractError("split plan has no folds")
    covered: list[int] = []
    for i, row in enumerate(folds):
        if not isinstance(row, Mapping) or "train" not in row or "test" not in row or "inner" not in row:
            raise ContractError(f"split fold {i} is incomplete")
        tr = np.asarray(row["train"], dtype=int)
        te = np.asarray(row["test"], dtype=int)
        _check_partition(tr, n, f"outer{i}.train")
        _check_partition(te, n, f"outer{i}.test")
        if set(tr) & set(te):
            raise ContractError(f"outer{i} train/test overlap")
        covered.extend(te.tolist())
        inner_rows = row["inner"]
        if not isinstance(inner_rows, list) or not inner_rows:
            raise ContractError(f"outer{i} has no inner folds")
        valid_covered: list[int] = []
        for j, inner_row in enumerate(inner_rows):
            if not isinstance(inner_row, Mapping) or "train" not in inner_row or "valid" not in inner_row:
                raise ContractError(f"outer{i}.inner{j} is incomplete")
            itr = np.asarray(inner_row["train"], dtype=int)
            iva = np.asarray(inner_row["valid"], dtype=int)
            _check_partition(itr, n, f"outer{i}.inner{j}.train")
            _check_partition(iva, n, f"outer{i}.inner{j}.valid")
            if set(itr) & set(iva) or not (set(itr) | set(iva)) <= set(tr):
                raise ContractError(f"outer{i}.inner{j} is not contained in outer training set")
            valid_covered.extend(iva.tolist())
        if len(valid_covered) != len(set(valid_covered)) or set(valid_covered) != set(tr):
            raise ContractError(f"outer{i} inner validation folds do not partition training set")
    if len(covered) != n or len(set(covered)) != n:
        raise ContractError("outer test folds do not partition all patients exactly once")
    return dict(plan)


def validate_honest_split_provenance(
    *,
    base_fit_ids: Sequence[str],
    residual_calibration_ids: Sequence[str],
    eval_ids: Sequence[str],
    residual_fit_ids: Sequence[str],
    residual_validation_ids: Sequence[str],
    upstream_base_fit_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Check the declared training ancestry of one honest split ensemble.

    The base-training, residual-calibration and evaluation patient blocks must
    be mutually disjoint.  Residual fitting/validation must partition only the
    calibration block.  An empty validation list is allowed for a final residual
    refit on the complete calibration block after its hyperparameters are fixed.

    ``upstream_base_fit_ids`` maps each actual ensemble member/checkpoint name to
    its training dependency closure: patients used by model fitting, learned
    preprocessing, feature selection, early stopping, hyperparameter selection
    and any upstream fitted models.  Every dependency must remain inside the
    base-training block.  These records must come from training-time artifacts;
    this validator verifies their set relationships but cannot inspect model
    weights or establish that a caller's claimed metadata is truthful/complete.

    Do not fill upstream records from the desired split plan after training.
    Correct indices alone cannot detect nested-OOF stacking contamination.
    """
    blocks = {
        "base_fit": _as_strings(base_fit_ids, "base_fit_ids"),
        "residual_calibration": _as_strings(residual_calibration_ids, "residual_calibration_ids"),
        "eval": _as_strings(eval_ids, "eval_ids"),
        "residual_fit": _as_strings(residual_fit_ids, "residual_fit_ids"),
    }
    validation = np.asarray(residual_validation_ids).astype(str)
    if validation.ndim != 1:
        raise ContractError("residual_validation_ids must be one-dimensional")
    blocks["residual_validation"] = (
        _as_strings(validation, "residual_validation_ids") if len(validation) else validation
    )
    sets = {name: set(values.tolist()) for name, values in blocks.items()}

    for left, right in (("base_fit", "residual_calibration"), ("base_fit", "eval"),
                        ("residual_calibration", "eval")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ContractError(f"honest split contamination: {left}/{right} overlap: {sorted(overlap)[:5]}")
    for name in ("residual_fit", "residual_validation"):
        outside = sets[name] - sets["residual_calibration"]
        if outside:
            raise ContractError(f"{name} contains IDs outside residual_calibration: {sorted(outside)[:5]}")
    if sets["residual_fit"] & sets["residual_validation"]:
        raise ContractError("residual fit/validation IDs overlap")
    if (sets["residual_fit"] | sets["residual_validation"]) != sets["residual_calibration"]:
        raise ContractError("residual fit/validation do not partition the residual_calibration block")
    if not isinstance(upstream_base_fit_ids, Mapping) or not upstream_base_fit_ids:
        raise ContractError("actual upstream base training-ID records are required")

    upstream: dict[str, Any] = {}
    for member, values in upstream_base_fit_ids.items():
        if not isinstance(member, str) or not member.strip():
            raise ContractError("upstream base member names must be non-empty strings")
        member_ids = _as_strings(values, f"upstream_base_fit_ids[{member!r}]")
        member_set = set(member_ids.tolist())
        for forbidden in ("residual_calibration", "eval"):
            contamination = member_set & sets[forbidden]
            if contamination:
                raise ContractError(
                    f"upstream base {member!r} includes {forbidden} IDs: {sorted(contamination)[:5]}"
                )
        outside = member_set - sets["base_fit"]
        if outside:
            raise ContractError(f"upstream base {member!r} includes IDs outside base_fit: {sorted(outside)[:5]}")
        upstream[member] = {"n": len(member_ids), "id_set_sha256": sha256_array(np.sort(member_ids))}
    return {
        "status": "PASS",
        "check_scope": "declared_training_dependency_ids",
        "limitation": "Checks declared ID ancestry; does not inspect fitted weights or prove metadata completeness.",
        "blocks": {name: {"n": len(values), "id_set_sha256": sha256_array(np.sort(values))}
                   for name, values in blocks.items()},
        "upstream_base_fits": upstream,
    }


class PredictionRegistry:
    """Collect predictions and reject missing, duplicate or misaligned rows."""

    def __init__(self, expected_ids: Mapping[str, Sequence[str]], grid: Sequence[float], arms: Sequence[str] | None = None, horizon: float = 1095.0):
        if not expected_ids:
            raise ContractError("prediction registry needs at least one cohort")
        self.expected_ids = {str(k): _as_strings(v, f"{k}:expected_ids") for k, v in expected_ids.items()}
        self.grid = np.asarray(grid, dtype=float)
        if self.grid.ndim != 1 or len(self.grid) < 2 or not np.isfinite(self.grid).all() or np.any(np.diff(self.grid) <= 0) or not np.isclose(self.grid[0], 0):
            raise ContractError("prediction grid must be finite, strictly increasing and start at zero")
        if not np.isfinite(float(horizon)) or float(horizon) <= 0 or float(horizon) > float(self.grid[-1]):
            raise ContractError("prediction horizon must be positive and within the prediction grid")
        self.horizon = float(horizon)
        self.arms = None if arms is None else tuple(str(a) for a in arms)
        if self.arms is not None and len(set(self.arms)) != len(self.arms):
            raise ContractError("prediction arms contain duplicates")
        self._predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def add(self, cohort: str, arm: str, ids: Sequence[str], survival: np.ndarray) -> None:
        cohort, arm = str(cohort), str(arm)
        if cohort not in self.expected_ids:
            raise ContractError(f"unregistered prediction cohort {cohort!r}")
        if self.arms is not None and arm not in self.arms:
            raise ContractError(f"unregistered prediction arm {arm!r}")
        key = (cohort, arm)
        if key in self._predictions:
            raise ContractError(f"duplicate prediction artifact for {cohort}/{arm}")
        pred_ids = validate_prediction_ids(self.expected_ids[cohort], ids)
        curves = validate_curves(pred_ids, survival, self.grid)
        self._predictions[key] = (pred_ids.copy(), curves.copy())

    def add_npz(self, cohort: str, arm: str, path: str | Path) -> None:
        ids, curves, grid = load_prediction(path)
        if not np.array_equal(grid, self.grid):
            raise ContractError(f"{path}: prediction grid differs from registry grid")
        self.add(cohort, arm, ids, curves)

    def missing(self) -> list[dict[str, str]]:
        if self.arms is None:
            return []
        return [{"cohort": c, "arm": a} for c in self.expected_ids for a in self.arms if (c, a) not in self._predictions]

    def require_complete(self) -> None:
        missing = self.missing()
        if missing:
            raise ContractError(f"incomplete predictions: {missing}")

    @property
    def predictions(self) -> Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        return dict(self._predictions)

    def save_npz(self, path: str | Path) -> None:
        """Save a complete registry archive, including cohorts of different sizes.

        Each cohort/arm pair is stored under ``ids_<i>`` and ``survival_<i>``;
        this avoids object arrays and the incorrect assumption that external
        cohorts have equal sample counts.
        """
        self.require_complete()
        cohorts = list(self.expected_ids)
        arms = list(self.arms or sorted({a for _, a in self._predictions}))
        if not arms:
            raise ContractError("cannot save an empty prediction registry")
        payload: dict[str, Any] = {
            "cohorts": np.asarray(cohorts, dtype=str),
            "arms": np.asarray(arms, dtype=str),
            "grid": self.grid.astype(np.float32),
        }
        keys: list[str] = []
        for i, cohort in enumerate(cohorts):
            for j, arm in enumerate(arms):
                key = f"{i}_{j}"
                ids, curves = self._predictions[(cohort, arm)]
                payload[f"ids_{key}"] = ids.astype(str)
                payload[f"survival_{key}"] = curves.astype(np.float32)
                keys.append(key)
        payload["pair_keys"] = np.asarray(keys, dtype=str)
        atomic_npz(path, **payload)


@dataclass(frozen=True)
class DevelopmentData:
    """Pooled development data passed to a model callback.

    This object deliberately contains no external labels.  A callback receives
    only pooled development outcomes and an expression-only test object, which
    makes it difficult for a runner to accidentally use an external endpoint
    during fitting or model selection.
    """

    ids: np.ndarray
    genes: np.ndarray
    x: np.ndarray
    p: np.ndarray
    time: np.ndarray
    event: np.ndarray
    cohort: np.ndarray
    endpoint: str

    @property
    def n(self) -> int:
        return int(len(self.ids))


@dataclass(frozen=True)
class FoldContext:
    """One immutable split context supplied to a model callback."""

    name: str
    train_indices: np.ndarray
    test_indices: np.ndarray | None
    # Indices are local to the training view supplied to the callback.
    inner_folds: tuple[tuple[np.ndarray, np.ndarray], ...] = ()


FitPredict = Callable[[FoldContext, DevelopmentData, ExpressionData], np.ndarray]


def _subset_expression(data: ExpressionData, indices: Sequence[int], cohort: str | None = None) -> ExpressionData:
    """Create a validated expression-only view without copying the gene axis."""
    ix = np.asarray(indices, dtype=int)
    _check_partition(ix, data.n, "expression subset")
    return ExpressionData(
        cohort=str(data.cohort if cohort is None else cohort),
        ids=data.ids[ix].copy(), genes=data.genes.copy(),
        x=data.x[ix].copy(), p=data.p[ix].copy(), source_path=data.source_path,
    )


class UnifiedExperimentRunner:
    """Reusable orchestration for fixed GeneNarrator survival experiments.

    The runner owns cohort registration, common axes, nested split-plan
    persistence, prediction validation and metric/manifest output.  A model is
    supplied as a small ``fit_predict`` callback; the callback is never given
    external labels.  Its signature is::

        fit_predict(context, development, test_expression) -> survival_curves

    For an outer context, ``test_expression`` is the held-out development
    block.  For the ``full`` context it is one external expression cohort.
    The callback may cache a fitted model for repeated full-context calls.
    Every returned array must have shape ``(len(test_expression.ids), len(grid))``
    and satisfy the shared monotone survival-curve contract.

    This class is intentionally model-agnostic: neural, Cox, teacher-residual
    and future network models all use the same protocol and artifact layout.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        protocol: UnifiedProtocol,
        development: Sequence[CohortBundle],
        external: Mapping[str, ExpressionData],
        grid: Sequence[float],
        arms: Sequence[str],
        outcome_loader: Callable[[], Mapping[str, Outcomes]] | None = None,
        outcome_sources: Mapping[str, str | Path] | None = None,
    ) -> None:
        if not development:
            raise ContractError("at least one development cohort is required")
        self.output_dir = Path(output_dir)
        self.protocol = protocol
        self.grid = np.asarray(grid, dtype=float)
        self.arms = tuple(str(a) for a in arms)
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ContractError("runner arms must be non-empty and unique")
        if self.protocol.endpoint != "OS" and self.protocol.endpoint not in {"DSS", "DFS", "PFS"}:
            raise ContractError(f"unsupported protocol endpoint {self.protocol.endpoint}")
        if tuple(b.expression.cohort for b in development) != tuple(protocol.development_cohorts):
            raise ContractError(
                "development cohort order must match protocol.development_cohorts"
            )
        assert_endpoint([b.outcomes for b in development], protocol.endpoint)
        all_expr = [b.expression for b in development] + list(external.values())
        self.axis_audit = audit_expression_axes(all_expr, require_same_pathways=True)
        ids, genes, x, p, time, event, cohort, endpoint = combine_expression(
            [b.expression for b in development], [b.outcomes for b in development]
        )
        self.data = DevelopmentData(ids, genes, x, p, time, event, cohort, endpoint)
        self.external = {str(k): v for k, v in external.items()}
        if len(self.external) != len(external):
            raise ContractError("external cohort names contain duplicates")
        self._plan: dict[str, Any] | None = None
        self._prepared = False
        self._development_bundles = tuple(development)
        self._outcome_loader = outcome_loader
        self._outcome_sources = {str(k): str(v) for k, v in (outcome_sources or {}).items()}

    @classmethod
    def from_standard_registry(
        cls, root: str | Path, *, output_dir: str | Path,
        protocol: UnifiedProtocol, grid: Sequence[float], arms: Sequence[str],
        external_names: Sequence[str] | None = None,
    ) -> "UnifiedExperimentRunner":
        """Construct a runner from the canonical ``experiment_base`` registry."""
        root = Path(root)
        development = load_standard_cohorts(root, list(protocol.development_cohorts))
        names = list(external_names or ())
        external = load_standard_expressions(root, names) if names else {}
        specs = standard_cohort_specs(root)
        outcome_sources = {
            name: str(specs[name].label_path or specs[name].expression_path)
            for name in names
        }
        return cls(
            output_dir=output_dir, protocol=protocol,
            development=[development[n] for n in protocol.development_cohorts],
            external=external, grid=grid, arms=arms,
            outcome_loader=lambda: load_standard_outcomes(root, names),
            outcome_sources=outcome_sources,
        )

    @property
    def plan(self) -> Mapping[str, Any]:
        if self._plan is None:
            raise ContractError("runner.prepare() must be called first")
        return self._plan

    @property
    def contexts(self) -> tuple[FoldContext, ...]:
        folds = self.plan["folds"]
        result = []
        for i, row in enumerate(folds):
            tr = np.asarray(row["train"], dtype=int)
            tr_pos = {int(v): j for j, v in enumerate(tr)}
            inner = tuple((
                np.asarray([tr_pos[int(v)] for v in r["train"]], dtype=int),
                np.asarray([tr_pos[int(v)] for v in r["valid"]], dtype=int),
            ) for r in row["inner"])
            result.append(FoldContext(
                name=f"outer{int(row.get('fold', i))}",
                train_indices=tr, test_indices=np.asarray(row["test"], dtype=int),
                inner_folds=inner,
            ))
        return tuple(result)

    def prepare(self) -> "UnifiedExperimentRunner":
        """Validate and persist the immutable protocol, axes and split plan."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with lock_protocol(self.output_dir, self.protocol):
            config_hash = hashlib.sha256(canonical_json(self.config).encode("utf-8")).hexdigest()
            # IDs and axes alone are insufficient to identify an experiment:
            # a caller can silently replace expression values or development
            # outcomes while retaining every patient name.  Fingerprint every
            # in-memory array (and every declared source file) before allowing
            # an existing run directory to be reused.
            input_fingerprint = self._input_fingerprint()
            existing_inputs = self.output_dir / "inputs.json"
            existing_plan_path = self.output_dir / "split_plan.json"
            existing_inputs_obj: Mapping[str, Any] | None = None
            if existing_inputs.exists():
                try:
                    existing = json.loads(existing_inputs.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ContractError(f"cannot read existing runner inputs: {existing_inputs}") from exc
                if existing.get("runner_config_sha256") != config_hash:
                    raise ContractError("runner configuration changed; use a new output directory")
                existing_inputs_obj = existing
                old_fingerprint = existing.get("input_fingerprint")
                if old_fingerprint is None:
                    raise ContractError(
                        "existing inputs lack full data fingerprints; use a new output directory"
                    )
                if canonical_json(old_fingerprint) != canonical_json(input_fingerprint):
                    raise ContractError(
                        "input arrays or source files changed; use a new output directory"
                    )
            self._plan = build_split_plan(
                self.data.cohort, self.data.event, self.protocol, self.data.ids
            )
            if existing_plan_path.exists():
                try:
                    previous_plan = json.loads(existing_plan_path.read_text(encoding="utf-8"))
                    validate_split_plan(previous_plan, protocol=self.protocol,
                                        cohort=self.data.cohort, event=self.data.event,
                                        ids=self.data.ids)
                except Exception as exc:
                    raise ContractError(f"existing split plan is invalid: {existing_plan_path}") from exc
                if canonical_json(previous_plan) != canonical_json(self._plan):
                    raise ContractError("split plan changed; use a new output directory")
            if existing_inputs_obj is not None:
                old_dev = existing_inputs_obj.get("development", {})
                if old_dev.get("ids_sha256") != sha256_array(self.data.ids) or old_dev.get("gene_sha256") != sha256_array(self.data.genes):
                    raise ContractError("development IDs or gene axis changed; use a new output directory")
                old_ext = existing_inputs_obj.get("external_cohorts", {})
                for name, expr in self.external.items():
                    if old_ext.get(name, {}).get("ids_sha256") != sha256_array(expr.ids):
                        raise ContractError(f"external IDs changed for {name}; use a new output directory")
            save_split_plan(self.output_dir / "split_plan.json", self._plan)
            source_paths = {
                f"{b.expression.cohort}:expression": str(b.expression.source_path)
                for b in self._development_bundles
            }
            source_paths.update({f"{b.outcomes.cohort}:outcomes": str(b.outcomes.source_path)
                                 for b in self._development_bundles})
            source_paths.update({f"{c}:expression": str(v.source_path) for c, v in self.external.items()})
            source_paths.update({f"{c}:outcomes": path for c, path in self._outcome_sources.items()})
            # DevelopmentData is pooled; recover source paths from protocol
            # cohorts by reading the canonical registry only when available.
            source_hashes: dict[str, str] = {}
            for path_text in source_paths.values():
                path = Path(path_text)
                if path.exists() and path.is_file():
                    source_hashes[path_text] = sha256_file(path)
            atomic_json(self.output_dir / "inputs.json", {
                "runner_version": RUNNER_VERSION,
                "runner_config": self.config,
                "runner_config_sha256": config_hash,
                "protocol_sha256": self.protocol.sha256,
                "development": {
                    "cohorts": list(self.protocol.development_cohorts),
                    "n": self.data.n, "events": int(self.data.event.sum()),
                    "endpoint": self.data.endpoint,
                    "ids_sha256": sha256_array(self.data.ids),
                    "gene_sha256": sha256_array(self.data.genes),
                },
                "external_cohorts": {
                    name: {"n": d.n, "ids_sha256": sha256_array(d.ids),
                           "source_path": str(d.source_path)}
                    for name, d in self.external.items()
                },
                "axis_audit": self.axis_audit,
                "source_sha256": source_hashes,
                "input_fingerprint": input_fingerprint,
                # ``extra`` is part of the locked protocol.  Keeping the
                # declared hashes visible here makes provenance audits easy:
                # model callbacks should put code/checkpoint/database hashes
                # in protocol.extra["declared_hashes"].
                "declared_hashes": dict(self.protocol.extra.get("declared_hashes", {}))
                if isinstance(self.protocol.extra, Mapping) else {},
            })
        self._prepared = True
        return self

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise ContractError("runner.prepare() must be called before fitting")

    @property
    def config(self) -> dict[str, Any]:
        """Stable runner-level configuration that complements protocol hash."""
        return {
            "runner_version": RUNNER_VERSION,
            "protocol_sha256": self.protocol.sha256,
            "arms": list(self.arms),
            "external_cohorts": list(self.external),
            "grid": self.grid.tolist(),
        }

    def _input_fingerprint(self) -> dict[str, Any]:
        """Return a complete, deterministic fingerprint of run inputs.

        This is intentionally based on loaded arrays rather than only file
        paths.  It catches an overwritten NPZ, a label edit, and an accidental
        patient reorder even when the source path is unchanged.
        """
        def expr_fp(expr: ExpressionData) -> dict[str, Any]:
            return {
                "ids": sha256_array(expr.ids), "genes": sha256_array(expr.genes),
                "x": sha256_array(expr.x), "p": sha256_array(expr.p),
                "source_path": str(expr.source_path),
            }
        development = {
            "ids": sha256_array(self.data.ids), "genes": sha256_array(self.data.genes),
            "x": sha256_array(self.data.x), "p": sha256_array(self.data.p),
            "time": sha256_array(self.data.time),
            "event": sha256_array(self.data.event.astype(np.uint8)),
            "cohort": sha256_array(self.data.cohort), "endpoint": self.data.endpoint,
        }
        source_paths: dict[str, str] = {
            f"{b.expression.cohort}:expression": str(b.expression.source_path)
            for b in self._development_bundles
        }
        source_paths.update({f"{b.outcomes.cohort}:outcomes": str(b.outcomes.source_path)
                             for b in self._development_bundles})
        source_paths.update({f"{c}:expression": str(v.source_path) for c, v in self.external.items()})
        source_paths.update({f"{c}:outcomes": path for c, path in self._outcome_sources.items()})
        source_files = {}
        for key, path_text in sorted(source_paths.items()):
            path = Path(path_text)
            source_files[key] = {
                "path": path_text,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            }
        return {
            "development": development,
            "external": {name: expr_fp(expr) for name, expr in sorted(self.external.items())},
            "source_files": source_files,
        }

    def _save_checked(self, path: Path, ids: Sequence[str], curves: np.ndarray) -> None:
        if path.exists():
            raise ContractError(f"refusing to overwrite existing artifact: {path}")
        save_prediction(path, ids, curves, self.grid)

    def run_arm(self, arm: str, fit_predict: FitPredict) -> list[Path]:
        """Fit one arm across fixed outer folds and the full external transfer."""
        self._require_prepared()
        arm = str(arm)
        if arm not in self.arms:
            raise ContractError(f"unregistered arm {arm!r}")
        arm_dir = self.output_dir / arm
        artifacts: list[Path] = []
        oof = np.full((self.data.n, len(self.grid)), np.nan, dtype=float)
        pooled_expr = ExpressionData(
            "development_pooled", self.data.ids, self.data.genes,
            self.data.x, self.data.p, "pooled",
        )
        with lock_protocol(self.output_dir, self.protocol):
            complete_path = arm_dir / "complete.json"
            if complete_path.exists():
                return self._verify_completed_arm(arm)
            for context in self.contexts:
                assert context.test_indices is not None
                test = _subset_expression(pooled_expr, context.test_indices, cohort=context.name)
                train_idx = context.train_indices
                train_data = DevelopmentData(
                    ids=self.data.ids[train_idx].copy(), genes=self.data.genes.copy(),
                    x=self.data.x[train_idx].copy(), p=self.data.p[train_idx].copy(),
                    time=self.data.time[train_idx].copy(), event=self.data.event[train_idx].copy(),
                    cohort=self.data.cohort[train_idx].copy(), endpoint=self.data.endpoint,
                )
                curves = np.asarray(fit_predict(context, train_data, test), dtype=float)
                if curves.shape != (test.n, len(self.grid)):
                    raise ContractError(
                        f"{arm}/{context.name}: expected {(test.n, len(self.grid))}, got {curves.shape}"
                    )
                path = arm_dir / f"{context.name}_predictions.npz"
                self._save_checked(path, test.ids, curves)
                artifacts.append(path)
                oof[context.test_indices] = curves
            if not np.isfinite(oof).all():
                raise ContractError(f"{arm}: outer OOF predictions are incomplete")
            pooled = arm_dir / "source_oof.npz"
            self._save_checked(pooled, self.data.ids, oof)
            artifacts.append(pooled)
            full_idx = np.arange(self.data.n, dtype=int)
            full_inner = tuple(inner_folds(full_idx, self.data.cohort, self.data.event,
                                           self.protocol.inner_splits, self.protocol.inner_seed))
            full = FoldContext("full", full_idx, None, full_inner)
            full_data = DevelopmentData(
                ids=self.data.ids.copy(), genes=self.data.genes.copy(),
                x=self.data.x.copy(), p=self.data.p.copy(), time=self.data.time.copy(),
                event=self.data.event.copy(), cohort=self.data.cohort.copy(), endpoint=self.data.endpoint,
            )
            for name, expr in self.external.items():
                curves = np.asarray(fit_predict(full, full_data, expr), dtype=float)
                if curves.shape != (expr.n, len(self.grid)):
                    raise ContractError(
                        f"{arm}/{name}: expected {(expr.n, len(self.grid))}, got {curves.shape}"
                    )
                path = arm_dir / f"{name}_predictions.npz"
                self._save_checked(path, expr.ids, curves)
                artifacts.append(path)
            atomic_json(arm_dir / "complete.json", {
                "runner_version": RUNNER_VERSION, "arm": arm,
                "protocol_sha256": self.protocol.sha256,
                "runner_config_sha256": hashlib.sha256(canonical_json(self.config).encode("utf-8")).hexdigest(),
                "inputs_sha256": sha256_file(self.output_dir / "inputs.json"),
                "artifacts": [artifact_record(p) for p in artifacts],
            })
            artifacts.append(arm_dir / "complete.json")
        return artifacts

    def _verify_completed_arm(self, arm: str) -> list[Path]:
        """Verify a completed arm before returning it as a resumable result."""
        arm_dir = self.output_dir / arm
        complete_path = arm_dir / "complete.json"
        try:
            obj = json.loads(complete_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ContractError(f"cannot read completed arm record: {complete_path}") from exc
        expected_config = hashlib.sha256(canonical_json(self.config).encode("utf-8")).hexdigest()
        if obj.get("protocol_sha256") != self.protocol.sha256 or obj.get("runner_config_sha256") != expected_config:
            raise ContractError(f"completed arm {arm!r} belongs to a different protocol/config")
        if obj.get("inputs_sha256") != sha256_file(self.output_dir / "inputs.json"):
            raise ContractError(f"completed arm {arm!r} was produced from different inputs")
        records = obj.get("artifacts")
        if not isinstance(records, list) or not records:
            raise ContractError(f"completed arm {arm!r} has no artifact records; use a new output directory")
        expected_names = {f"outer{i}_predictions.npz" for i in range(len(self.contexts))}
        expected_names.add("source_oof.npz")
        expected_names.update(f"{name}_predictions.npz" for name in self.external)
        seen: set[str] = set()
        result: list[Path] = []
        for record in records:
            if not isinstance(record, Mapping) or "path" not in record or "sha256" not in record:
                raise ContractError(f"completed arm {arm!r} has an invalid artifact record")
            path = Path(str(record["path"]))
            if path.parent.resolve() != arm_dir.resolve() or path.name not in expected_names:
                raise ContractError(f"completed arm {arm!r} contains an unexpected artifact path: {path}")
            if path.name in seen or not path.exists() or sha256_file(path) != record["sha256"]:
                raise ContractError(f"completed arm {arm!r} artifact is missing, duplicated, or changed: {path}")
            ids, curves, grid = load_prediction(path)
            if not np.array_equal(grid, self.grid):
                raise ContractError(f"completed arm {arm!r} has a grid mismatch: {path}")
            if path.name == "source_oof.npz":
                validate_prediction_ids(self.data.ids, ids)
            elif path.name.startswith("outer"):
                k = int(path.name[len("outer"):].split("_", 1)[0])
                if k >= len(self.contexts) or not np.array_equal(ids, self.data.ids[self.contexts[k].test_indices]):
                    raise ContractError(f"completed arm {arm!r} has misaligned outer predictions: {path}")
            else:
                name = path.name[:-len("_predictions.npz")]
                validate_prediction_ids(self.external[name].ids, ids)
            seen.add(path.name)
            result.append(path)
        if seen != expected_names:
            raise ContractError(f"completed arm {arm!r} is incomplete: missing {sorted(expected_names - seen)}")
        result.append(complete_path)
        return result

    def score(self, *, include_source_oof: bool = True) -> pd.DataFrame:
        """Load labels only now, score all arms, and write metrics + manifest."""
        self._require_prepared()
        for arm in self.arms:
            if not (self.output_dir / arm / "complete.json").exists():
                raise ContractError(f"arm {arm!r} has not completed")
        # This is intentionally the first place this runner reads external
        # labels; predictions must already be frozen and complete.
        outcomes = self._outcome_loader() if self._outcome_loader is not None else {}
        if set(outcomes) != set(self.external):
            raise ContractError(
                "outcome loader must return exactly the registered external cohorts"
            )
        assert_endpoint(list(outcomes.values()), self.protocol.endpoint)
        rows = pd.DataFrame()
        if self.external:
            registry = PredictionRegistry(
                {name: expr.ids for name, expr in self.external.items()},
                self.grid, arms=self.arms, horizon=self.protocol.horizon,
            )
            for arm in self.arms:
                for name in self.external:
                    registry.add_npz(name, arm, self.output_dir / arm / f"{name}_predictions.npz")
            rows = score_registry(registry, outcomes, self.protocol.endpoint).copy()
        if include_source_oof:
            for arm in self.arms:
                ids, curves, _ = load_prediction(self.output_dir / arm / "source_oof.npz")
                if not np.array_equal(ids, self.data.ids):
                    raise ContractError(f"{arm}: source OOF IDs changed")
                scored = score_survival_curves(
                    self.data.time, self.data.event, curves, self.grid,
                    horizon=self.protocol.horizon,
                )
                rows = pd.concat([rows, pd.DataFrame([{
                    "cohort": "source_OOF", "arm": arm,
                    "endpoint": self.protocol.endpoint, **scored,
                }])], ignore_index=True)
        if rows.empty:
            raise ContractError("no scored predictions")
        metrics_path = self.output_dir / "metrics.csv"
        if metrics_path.exists():
            raise ContractError(f"refusing to overwrite existing metrics: {metrics_path}")
        rows.to_csv(metrics_path, index=False)
        artifacts = [self.output_dir / "protocol.lock.json", self.output_dir / "split_plan.json",
                     self.output_dir / "inputs.json", metrics_path]
        artifacts.extend(self.output_dir.glob("*/complete.json"))
        # Include pooled OOF and per-cohort prediction archives in the top
        # level manifest (the arm completion records contain the same hashes,
        # but keeping the files here makes a quick audit possible).
        artifacts.extend(self.output_dir.glob("*/*.npz"))
        write_run_manifest(self.output_dir / "manifest.json", self.protocol.as_dict(), artifacts)
        return rows


def _curve_at(curves: np.ndarray, grid: np.ndarray, horizon: float) -> np.ndarray:
    if horizon < grid[0] or horizon > grid[-1]:
        raise ContractError(f"horizon {horizon} is outside prediction grid [{grid[0]}, {grid[-1]}]")
    # ``np.interp`` only accepts one-dimensional ordinates.  Use one shared
    # pair of bracketing grid points so interpolation remains vectorized over
    # patients and preserves the curve shape.
    j = int(np.searchsorted(grid, float(horizon), side="left"))
    if j == 0:
        return curves[:, 0].copy()
    if j == len(grid):
        return curves[:, -1].copy()
    if grid[j] == float(horizon):
        return curves[:, j].copy()
    left, right = j - 1, j
    w = (float(horizon) - grid[left]) / (grid[right] - grid[left])
    return curves[:, left] * (1.0 - w) + curves[:, right] * w


def rmst_risk(curves: np.ndarray, grid: Sequence[float], horizon: float) -> np.ndarray:
    """Return ``1 - RMST(horizon)/horizon`` with no post-horizon leakage.

    If the requested horizon falls between grid points, the survival value at
    that horizon is linearly interpolated and appended before integration.  This
    is the historical GeneNarrator 3-year risk convention when ``horizon=1095``;
    unlike integrating the entire prediction grid, it remains correct when a
    runner stores a grid extending beyond the scoring horizon.
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or len(g) < 2 or not np.isfinite(g).all() or np.any(np.diff(g) <= 0) or not np.isclose(g[0], 0):
        raise ContractError("risk grid must be finite, strictly increasing and start at zero")
    if not np.isfinite(float(horizon)) or float(horizon) <= 0 or float(horizon) > float(g[-1]):
        raise ContractError("risk horizon must be positive and within the prediction grid")
    raw_curves = np.asarray(curves)
    if raw_curves.ndim != 2:
        raise ContractError("survival curves must be a two-dimensional patient-by-grid array")
    c = validate_curves(np.arange(raw_curves.shape[0]).astype(str), raw_curves, g)
    keep = g < float(horizon)
    # Include zero and every original point strictly before the horizon, then
    # append the interpolated endpoint exactly once.
    gh = np.r_[g[keep], float(horizon)]
    ch = np.column_stack([c[:, keep], _curve_at(c, g, float(horizon))])
    integrate = getattr(np, "trapezoid", None) or np.trapz
    risk = 1.0 - integrate(ch, gh, axis=1) / float(horizon)
    if not np.isfinite(risk).all():
        raise ContractError("RMST risk is non-finite")
    return risk


def score_survival_curves(
    time: Sequence[float],
    event: Sequence[bool],
    curves: np.ndarray,
    grid: Sequence[float],
    horizon: float = 1095.0,
    reference_time: Sequence[float] | None = None,
    reference_event: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Score one cohort with Uno, Harrell and Brier under one fixed convention."""
    from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw
    from sksurv.util import Surv

    t = np.asarray(time, dtype=float)
    e_raw = np.asarray(event)
    if t.ndim != 1 or e_raw.ndim != 1 or len(t) != len(e_raw) or len(t) == 0 or not np.isfinite(t).all() or np.any(t <= 0):
        raise ContractError("time and event must be aligned, positive and finite")
    if e_raw.dtype.kind not in "biuf" or not np.isfinite(e_raw).all() or not np.isin(e_raw, [0, 1]).all():
        raise ContractError("event must contain only numeric 0/1 or boolean labels")
    e = e_raw.astype(bool)
    g = np.asarray(grid, dtype=float)
    c = validate_curves(np.arange(len(t)).astype(str), curves, g)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ContractError("horizon must be positive and finite")
    risk = rmst_risk(c, g, float(horizon))
    if not np.isfinite(risk).all():
        raise ContractError("risk derived from survival curves is non-finite")
    y = Surv.from_arrays(e, t)
    ref_t = t if reference_time is None else np.asarray(reference_time, dtype=float)
    ref_e = e if reference_event is None else np.asarray(reference_event).astype(bool)
    if len(ref_t) != len(ref_e) or len(ref_t) == 0 or not np.isfinite(ref_t).all() or np.any(ref_t <= 0):
        raise ContractError("reference outcomes are invalid")
    y_ref = Surv.from_arrays(ref_e, ref_t)
    result: dict[str, Any] = {
        "uno_c_3y": np.nan,
        "harrell_c": np.nan,
        "brier_3y": np.nan,
        "n": int(len(t)),
        "events": int(e.sum()),
        "events_by_horizon": int(np.sum(e & (t <= float(horizon)))),
        "horizon": float(horizon),
        "metric_convention": METRIC_CONVENTION,
        "metric_errors": {},
    }
    try:
        result["uno_c_3y"] = float(concordance_index_ipcw(y_ref, y, risk, tau=float(horizon))[0])
    except Exception as exc:  # retain diagnostic instead of silently dropping a metric
        result["metric_errors"]["uno_c_3y"] = f"{type(exc).__name__}: {exc}"
    try:
        result["harrell_c"] = float(concordance_index_censored(e, t, risk)[0])
    except Exception as exc:
        result["metric_errors"]["harrell_c"] = f"{type(exc).__name__}: {exc}"
    try:
        at_horizon = _curve_at(c, g, float(horizon))[:, None]
        result["brier_3y"] = float(brier_score(y_ref, y, at_horizon, np.asarray([horizon]))[1][0])
    except Exception as exc:
        result["metric_errors"]["brier_3y"] = f"{type(exc).__name__}: {exc}"
    result["metric_status"] = "PASS" if not result["metric_errors"] else "PARTIAL"
    return result


def score_registry(
    registry: PredictionRegistry,
    outcomes: Mapping[str, Outcomes],
    endpoint: str,
    reference_outcomes: Mapping[str, Outcomes] | None = None,
) -> pd.DataFrame:
    """Score every registered prediction after endpoint and ID checks."""
    endpoint = str(endpoint).upper()
    if not outcomes:
        raise ContractError("outcomes are required for scoring")
    registry.require_complete()
    rows: list[dict[str, Any]] = []
    for cohort, outcome in outcomes.items():
        if outcome.endpoint != endpoint:
            raise ContractError(f"{cohort}: endpoint {outcome.endpoint} != requested {endpoint}")
        if cohort not in registry.expected_ids:
            raise ContractError(f"outcomes contain unregistered cohort {cohort!r}")
        for (pred_cohort, arm), (ids, curves) in registry.predictions.items():
            if pred_cohort != cohort:
                continue
            validate_prediction_ids(outcome.ids, ids)
            # ID equality is deliberately set-based in the shared contract;
            # align the curve rows to the outcome order before scoring so a
            # differently ordered prediction artifact cannot swap patients.
            pred_pos = {value: i for i, value in enumerate(ids)}
            order = np.asarray([pred_pos[value] for value in outcome.ids], dtype=int)
            aligned_curves = curves[order]
            ref = None if reference_outcomes is None else reference_outcomes.get(cohort)
            if ref is not None:
                if ref.endpoint != endpoint:
                    raise ContractError(f"{cohort}: reference endpoint {ref.endpoint} != requested {endpoint}")
            scored = score_survival_curves(
                outcome.time, outcome.event, aligned_curves, registry.grid,
                horizon=float(registry.horizon),
                reference_time=None if ref is None else ref.time,
                reference_event=None if ref is None else ref.event,
            )
            rows.append({"cohort": cohort, "arm": arm, "endpoint": endpoint, **scored})
    if not rows:
        raise ContractError("no predictions matched supplied outcomes")
    return pd.DataFrame(rows)


def self_check() -> dict[str, Any]:
    """Run a tiny in-memory audit; usable as ``python unified_experiment_base.py``."""
    protocol = UnifiedProtocol("self-check", outer_splits=3, inner_splits=2, horizon=2.0)
    cohort = np.repeat(["a", "b"], 12)
    event = np.tile([False, True], 12)
    ids = np.asarray([f"p{i}" for i in range(len(cohort))])
    plan = build_split_plan(cohort, event, protocol, ids)
    grid = np.asarray([0.0, 1.0, 2.0])
    expected = {"toy": ids}
    registry = PredictionRegistry(expected, grid, arms=["BASE", "LLM"], horizon=2.0)
    curves = np.column_stack([np.ones(len(ids)), np.full(len(ids), .8), np.full(len(ids), .6)])
    registry.add("toy", "BASE", ids, curves)
    registry.add("toy", "LLM", ids, curves)
    registry.require_complete()
    score = score_survival_curves(np.arange(1, 25, dtype=float), event, curves, grid, horizon=2.0)
    return {"status": "PASS", "protocol_sha256": protocol.sha256, "outer_folds": len(plan["folds"]), "score_keys": sorted(score)}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2, ensure_ascii=False))
