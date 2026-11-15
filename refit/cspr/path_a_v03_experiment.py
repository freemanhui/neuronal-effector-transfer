"""Replay and readout helpers for the frozen-assay v0.3 capacity contrast."""
from __future__ import annotations

import numpy as np

from .path_a_assay import build_sample_units, hierarchical_weights
from .path_a_course_correction import decompose
from .path_a_portability import Units, check_samples


def qualified_cells(samples):
    keys = [(s.study, s.sample, b) for s in samples for b in s.barcodes]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate sample-qualified cell identity')
    return set(keys)


def require_disjoint_cells(training, evaluation):
    left, right = qualified_cells(training), qualified_cells(evaluation)
    if left & right:
        raise ValueError('Training/evaluation cells overlap')
    return dict(training_cells=len(left), evaluation_cells=len(right), overlap=0)


def units_and_membership(samples, axes, coordinate):
    """Use the frozen aggregation kernel and retain its previously unsaved ledger."""
    samples = check_samples(samples, axes)
    state_indices = [axes.ids.index(i) for i in axes.s]
    expression, state, dominant, studies, names, identities, membership = [], [], [], [], [], [], []
    remainders = {}
    for sample in samples:
        part = build_sample_units(sample.counts, state_indices, coordinate, sample.barcodes)
        count = len(part['expression'])
        expression.append(part['expression'])
        state.append(part['state_features'])
        dominant.append(part['dominant'])
        studies.extend([sample.study]*count)
        names.extend([sample.sample]*count)
        unit_ids = [f'{sample.study}/{sample.sample}/{i:08d}' for i in range(count)]
        identities.extend(unit_ids)
        remainders[sample.key] = int(np.sum(part['assignment'] < 0))
        for barcode, assignment in zip(sample.barcodes, part['assignment']):
            membership.append((sample.study, sample.sample, barcode,
                               '' if assignment < 0 else unit_ids[int(assignment)]))
    units = Units(np.vstack(expression), np.vstack(state), np.concatenate(dominant),
        np.asarray(studies), np.asarray(names), np.asarray(identities),
        tuple(s.key for s in samples), remainders)
    if len({r[:3] for r in membership}) != len(membership):
        raise ValueError('Duplicate membership identity')
    return units, membership


def require_equal(actual, expected, *, label, tolerance=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f'{label}: replay shape mismatch')
    if actual.dtype.kind in 'biufc' and expected.dtype.kind in 'biufc':
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f'{label}: nonfinite replay value')
        error = float(np.max(np.abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
        if error > tolerance:
            raise ValueError(f'{label}: replay error {error} exceeds {tolerance}')
        return error
    if not np.array_equal(actual, expected):
        raise ValueError(f'{label}: replay identity mismatch')
    return 0.


def compare_units(actual, expected):
    errors = {field: require_equal(getattr(actual, field), getattr(expected, field), label=field)
              for field in ('expression', 'state', 'dominant', 'study', 'sample', 'identities')}
    if actual.planned_samples != expected.planned_samples or actual.remainders != expected.remainders:
        raise ValueError('Source unit preparation/remainder replay mismatch')
    errors['weights'] = require_equal(actual.weights, expected.weights, label='source weights')
    return errors


def metrics_by_draw(y, predictions, weights):
    predictions = np.asarray(predictions, float)
    if predictions.ndim != 3 or predictions.shape[1:] != np.asarray(y).shape:
        raise ValueError('Draw, unit, response axes required')
    return [decompose(y, p, weights) for p in predictions]


def summarize_new_controls(y, weights, tf, full_df, matched, calibration_shift):
    """Summarize actual predictions, taking quantiles after pooled-fold scoring."""
    result = {}
    for arm, shift in [('raw', np.zeros_like(calibration_shift)),
                       ('train_mean_calibrated', calibration_shift)]:
        point = {name: decompose(y, value+shift, weights)
                 for name, value in [('TF', tf), ('Nfull_df', full_df)]}
        draws = metrics_by_draw(y, matched+shift, weights)
        comparisons = {}
        for metric in ('R2_raw', 'R2_centered_oracle'):
            values = [d[metric] for d in draws]
            q95 = (None if any(v is None for v in values)
                   else float(np.quantile(values, .95, method='higher')))
            tf_value, full_value = point['TF'][metric], point['Nfull_df'][metric]
            comparisons[metric] = dict(TF=tf_value, Nfull_df=full_value,
                N1_active_q95=q95, N1_active_median=None if q95 is None else float(np.median(values)),
                TF_minus_N1_active_q95=None if q95 is None or tf_value is None else tf_value-q95,
                TF_minus_Nfull_df=None if tf_value is None or full_value is None else tf_value-full_value)
        result[arm] = dict(models=point, N1_active_draws=draws, comparisons=comparisons)
    return result


def per_preparation_new_controls(y, studies, samples, tf, full_df, matched, shift):
    result = {}
    for study, sample in sorted(set(zip(studies, samples))):
        mask = (studies == study) & (samples == sample)
        weights = hierarchical_weights(studies[mask], samples[mask])
        result[f'{study}/{sample}'] = dict(units=int(mask.sum()), scores=summarize_new_controls(
            y[mask], weights, tf[mask], full_df[mask], matched[:, mask], shift[mask]))
    return result
