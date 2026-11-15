"""Exploratory diagnostics under the Nov2 amendment; frozen kernels unchanged."""
from dataclasses import replace
import hashlib

import numpy as np

from .path_a_assay import FamilyCounts, hierarchical_weights, row_weights
from .path_a_portability import Sample


def decompose(y, prediction, weights):
    """Exact weighted squared-error decomposition; centring is diagnostic only."""
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 2 or prediction.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Aligned finite two-dimensional outcomes/predictions required")
    w = row_weights(weights, len(y))
    mu_y, mu_p = w @ y, w @ prediction
    yc, pc = y - mu_y, prediction - mu_p
    s = float(np.sum(w[:, None] * yc**2))
    ec = float(np.sum(w[:, None] * (yc - pc)**2))
    bp = float(np.sum((mu_y - mu_p)**2))
    sse = float(np.sum(w[:, None] * (y - prediction)**2))
    closure = abs(sse - ec - bp)
    if closure > 1e-10 * max(1., sse, ec + bp):
        raise ArithmeticError("Squared-error decomposition did not close")
    constant = np.all(y == y[0]) or s <= np.finfo(float).tiny
    return dict(S=s, Ec=ec, BP=bp, SSE=sse, closure_absolute_error=closure,
                R2_raw=None if constant else 1 - sse/s,
                R2_centered_oracle=None if constant else 1 - ec/s,
                BP_over_S=None if constant else bp/s,
                mean_error_share=None if sse == 0 else bp/sse)


def cell_halves(samples):
    """Partition before PCA/grouping. Membership depends only on cell identity."""
    halves = [[], []]
    proof = []
    for sample in samples:
        order = sorted(range(len(sample.barcodes)), key=lambda i: (
            hashlib.sha256(sample.barcodes[i].encode()).hexdigest(), sample.barcodes[i]))
        sets = []
        for fold in (0, 1):
            indices = np.asarray(sorted(order[fold::2]))
            if len(indices) < 20:
                raise ValueError("Every preparation/half requires a complete twenty-cell unit")
            barcodes = tuple(sample.barcodes[i] for i in indices)
            sets.append(set(barcodes))
            counts = sample.counts
            halves[fold].append(replace(sample, barcodes=barcodes, counts=FamilyCounts(
                counts.linear[indices], counts.denominator[indices], counts.denominator_detected[indices])))
        if sets[0] & sets[1] or sets[0] | sets[1] != set(sample.barcodes):
            raise ValueError("Cell partition overlap or loss")
        proof.append(dict(study=sample.study, sample=sample.sample,
                          half_cells=list(map(len, sets)), intersection=0,
                          membership_sha256=[hashlib.sha256("\n".join(sorted(x)).encode()).hexdigest()
                                             for x in sets]))
    return tuple(map(tuple, halves)), proof


def concatenate_values(parts):
    """Combine held-out crossfit outputs; each fold uses its own train-only null."""
    result = []
    for arm in range(4):
        keys = set(parts[0][arm])
        if any(set(part[arm]) != keys for part in parts):
            raise ValueError("Crossfit prediction panels differ")
        result.append({name: np.concatenate([p[arm][name] for p in parts], axis=0 if arm < 2 else 1)
                       for name in sorted(keys)})
    return tuple(result)


def combined_units(parts):
    from .path_a_portability import Units
    # Identities repeat between halves in the frozen grouping kernel: disambiguate.
    return Units(np.concatenate([p.expression for p in parts]),
                 np.concatenate([p.state for p in parts]), np.concatenate([p.dominant for p in parts]),
                 np.concatenate([p.study for p in parts]), np.concatenate([p.sample for p in parts]),
                 np.concatenate([[f"fold{f}/{x}" for x in p.identities] for f, p in enumerate(parts)]),
                 parts[0].planned_samples,
                 {key: sum(p.remainders[key] for p in parts) for key in parts[0].remainders})


def fixed_module_indices(axes):
    names = sorted({name for groups in axes.modules.values() for name in groups})
    if len(names) != 6:
        raise ValueError("Exactly six frozen effector annotations required")
    return {name: [i for i, family in enumerate(axes.y) if name in axes.modules[family]] for name in names}


def metrics(y, prediction, weights, modules):
    result = dict(full=decompose(y, prediction, weights))
    result["modules"] = {name: decompose(y[:, ix].mean(axis=1, keepdims=True),
        prediction[:, ix].mean(axis=1, keepdims=True), weights) for name, ix in modules.items()}
    return result


def prediction_report(y, weights, values, modules):
    from .path_a_portability import REFERENCES
    result = {}
    for arm, base, draws in (("raw", values[0], values[2]), ("train_mean_calibrated", values[1], values[3])):
        models = {name: metrics(y, p, weights, modules) for name, p in base.items()}
        draw_metrics = {name: [metrics(y, p, weights, modules) for p in ps] for name, ps in draws.items()}
        scores = {}
        for population in ("full", *modules):
            pick = (lambda d: d["full"]) if population == "full" else (lambda d: d["modules"][population])
            scores[population] = {}
            for metric in ("R2_raw", "R2_centered_oracle"):
                points = {name: pick(d)[metric] for name, d in models.items()}
                for name, ds in draw_metrics.items():
                    numbers = [pick(d)[metric] for d in ds]
                    points[name] = (None if any(v is None for v in numbers)
                                    else float(np.quantile(numbers, .95, method="higher")))
                complete = set(points) == {"TF", *REFERENCES} and all(v is not None for v in points.values())
                strongest = max(REFERENCES, key=lambda n: points[n]) if complete else None
                scores[population][metric] = dict(scores=points, strongest_reference=strongest,
                    TF_minus_strongest=None if strongest is None else points["TF"] - points[strongest])
        result[arm] = dict(model_error_components=models, conditional_draw_error_components=draw_metrics,
                           comparisons=scores)
    return result


def per_sample_scores(units, y, values):
    from .path_a_portability import score_predictions
    result = {}
    for study, sample in units.planned_samples:
        mask = (units.study == study) & (units.sample == sample)
        if not mask.any():
            result[f"{study}/{sample}"] = dict(status="NO_COMPLETE_UNIT")
            continue
        selected = tuple({k: v[mask] if i < 2 else v[:, mask] for k, v in item.items()}
                         for i, item in enumerate(values))
        result[f"{study}/{sample}"] = score_predictions(y[mask], units.subset(mask).weights, *selected)
    return result
