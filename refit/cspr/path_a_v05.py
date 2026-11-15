"""Prospective Track A primitives; no file access or model execution on import.

These are secondary analyses. Historical matrices and kernels remain unchanged.
Target responses never enter input calibration or source-only CV selection.
"""
from dataclasses import replace
import hashlib
import numpy as np
from scipy.stats import beta as beta_distribution
from .path_a_assay import SourceScale, hierarchical_weights, row_weights, weighted_ridge_design
from .path_a_portability import source_coordinate
from .path_a_v03_experiment import units_and_membership

DF_GRID = tuple(np.geomspace(.01, .95, 15))


def target_fixed_units(samples, axes):
    """Fit each preparation's own S coordinate, independent of every source row.

    Call once per target population and reuse the returned object in all rows.
    Insufficient 20-PC rank is a failure, never an invitation to lower dimension.
    """
    parts, ledger = [], []
    for sample in sorted(samples, key=lambda s: s.key):
        coordinate = source_coordinate((sample,), axes)
        units, members = units_and_membership((sample,), axes, coordinate)
        parts.append(units)
        ledger.extend(members)
    if not parts:
        raise ValueError('No preparations')
    fields = ('expression', 'state', 'dominant', 'study', 'sample', 'identities')
    return replace(parts[0], **{k: np.concatenate([getattr(p, k) for p in parts]) for k in fields},
                   planned_samples=tuple(k for p in parts for k in p.planned_samples),
                   remainders={k: v for p in parts for k, v in p.remainders.items()}), ledger


def endpoint_digest(units, y):
    """Bind exact target identities, outcomes and weights, including dtype/shape."""
    h = hashlib.sha256()
    for array in (units.identities, units.study, units.sample, y, units.weights):
        array = np.ascontiguousarray(array)
        h.update(str((array.dtype.str, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def scores(y, prediction, weights):
    """Pooled and family-level signed alignment and centred R²; constants explicit."""
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 2 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Aligned finite unit-by-family arrays required')
    w = row_weights(weights, len(y))
    yc, pc = y-w@y, p-w@p
    s, v, c = w@(yc*yc), w@(pc*pc), w@(yc*pc)
    s[np.all(y == y[0], axis=0)] = 0
    v[np.all(p == p[0], axis=0)] = 0
    valid = s > np.finfo(float).tiny
    paired = valid & (v > np.finfo(float).tiny)
    family_r2 = np.full(y.shape[1], np.nan)
    family_rho = family_r2.copy()
    family_r2[valid] = (2*c[valid]-v[valid])/s[valid]
    family_rho[paired] = c[paired]/np.sqrt(s[paired]*v[paired])
    total_s, total_v, total_c = s.sum(), v.sum(), c.sum()
    rho = float(total_c/np.sqrt(total_s*total_v)) if total_s > 0 and total_v > 0 else None
    clean = lambda a: [float(x) if np.isfinite(x) else None for x in a]
    return dict(rho=rho, amplitude=float(np.sqrt(total_v/total_s)) if total_s > 0 else None,
        R2_centered=float((2*total_c-total_v)/total_s) if total_s > 0 else None,
        oracle_R2=max(0., rho)**2 if rho is not None else None,
        family_R2=clean(family_r2), family_rho=clean(family_rho),
        family_mean_R2=float(family_r2[valid].mean()) if valid.any() else None,
        constant_outcome_families=int((~valid).sum()))


def input_calibrated_predict(model, target_training_x, target_x, weights):
    """PI-approved target-X-only location/scale adaptation; no target Y argument."""
    scale = SourceScale.fit(target_training_x, weights)
    z = scale.transform(target_x)
    z[:, scale.constant | model.constant] = 0
    return z @ model.coefficient + model.target_mean


def source_cv_path(x, y, studies, samples, identities, fractions=DF_GRID):
    """Five deterministic unit folds within source; all scaling refit per fold.

    This is conditional unit-level CV, not independent study confirmation. Outer
    target studies are absent from this API. Lowest df wins exact loss ties.
    """
    x, y = np.asarray(x), np.asarray(y)
    studies, samples, identities = map(np.asarray, (studies, samples, identities))
    if len(set(identities)) != len(x) or len(x) < 10 or len(y) != len(x):
        raise ValueError('Unique aligned source units, at least ten, required')
    fractions = np.asarray(fractions)
    if np.any(fractions <= 0) or np.any(fractions >= 1) or np.any(np.diff(fractions) <= 0):
        raise ValueError('Increasing df fractions strictly between zero and one required')
    order = sorted(range(len(x)), key=lambda i: (hashlib.sha256(str(identities[i]).encode()).hexdigest(), str(identities[i])))
    folds = np.empty(len(x), int)
    folds[order] = np.arange(len(x)) % 5
    losses = np.zeros((5, len(fractions)))
    for fold in range(5):
        train, test = folds != fold, folds == fold
        w = hierarchical_weights(studies[train], samples[train])
        wt = hierarchical_weights(studies[test], samples[test])
        design = weighted_ridge_design(x[train], w)
        for j, fraction in enumerate(fractions):
            p = design.fit(y[train], design.rank*fraction).predict(x[test])
            losses[fold, j] = np.sum(wt[:, None]*(y[test]-p)**2)
    selected = int(np.argmin(losses.mean(axis=0)))
    return dict(selected_fraction=float(fractions[selected]), losses=losses.tolist(), folds=folds.tolist())


def hierarchical_intervals(y, p, preparation, *, replicates=1000, seed=20261106):
    """Conditional target uncertainty; fixed predictions, preparation then unit.

    With a single preparation these are unit-only intervals, explicitly labelled.
    Model-fitting and study-selection uncertainty are not covered.
    """
    y, p, preparation = np.asarray(y), np.asarray(p), np.asarray(preparation)
    groups = np.unique(preparation)
    if not len(groups) or replicates < 2:
        raise ValueError('Nonempty preparations and at least two replicates required')
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ('rho', 'amplitude', 'R2_centered')}
    for _ in range(replicates):
        indices, weights = [], []
        for group in rng.choice(groups, len(groups), replace=True):
            ix = np.flatnonzero(preparation == group)
            indices.extend(rng.choice(ix, len(ix), replace=True))
            weights.extend([1/(len(groups)*len(ix))]*len(ix))
        score = scores(y[indices], p[indices], weights)
        for k in draws:
            draws[k].append(score[k])
    return dict(scope='unit_only' if len(groups) == 1 else 'preparation_then_unit', replicates=replicates,
        intervals={k: dict(valid=sum(x is not None for x in v),
            ci95=np.quantile([x for x in v if x is not None], [.025, .975]).tolist()
            if all(x is not None for x in v) else None) for k, v in draws.items()})


def permutation_p(observed, null):
    null = np.asarray(null, float)
    if not np.isfinite(observed) or null.ndim != 1 or len(null) < 19 or not np.isfinite(null).all():
        raise ValueError('Finite statistic and at least 19 null draws required')
    return float((1 + (null >= observed).sum())/(1+len(null)))


def power_summary(pvalues):
    """Exact binomial uncertainty; 10 seeds cannot establish precise 80% power."""
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Valid p-values required')
    n, k = len(p), int((p <= .05).sum())
    lo = 0. if k == 0 else float(beta_distribution.ppf(.025, k, n-k+1))
    hi = 1. if k == n else float(beta_distribution.ppf(.975, k+1, n-k))
    return dict(seeds=n, detected=k, power=k/n, ci95=[lo, hi])


def g1(mde_rho, *, generator_validated, null_calibrated):
    """Conditional assay sensitivity, never absence of conserved biology."""
    if not generator_validated or not null_calibrated:
        return 'INCONCLUSIVE_CALIBRATION'
    if mde_rho is None or not np.isfinite(mde_rho):
        return 'INCONCLUSIVE_MDE_NOT_IDENTIFIED'
    if mde_rho < 0:
        raise ValueError('MDE correlation must be nonnegative')
    return 'POWERED_CONDITIONAL_ASSAY' if mde_rho <= .15 else 'UNDERPOWERED' if mde_rho > .25 else 'INCONCLUSIVE_INTERVAL'


def ridge_path(source_x, source_y, source_weights, target_x, fractions=DF_GRID):
    """Fit every preregistered fraction using one source-only spectral design.

    The caller must supply admitted populations and disjoint target units.
    This returns predictions/audits, not a target-selected best model.
    """
    fractions = np.asarray(fractions, float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or np.any((fractions <= 0) | (fractions >= 1)) or np.any(np.diff(fractions) <= 0):
        raise ValueError('Increasing finite fractions within (0,1) required')
    design = weighted_ridge_design(source_x, source_weights)
    predictions, audits = [], []
    for fraction in fractions:
        model = design.fit(source_y, design.rank*fraction)
        predictions.append(model.predict(target_x))
        audits.append(dict(fraction=float(fraction), **model.audit))
    return np.stack(predictions), audits
