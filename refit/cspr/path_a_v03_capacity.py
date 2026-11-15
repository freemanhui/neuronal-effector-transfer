"""Source-only capacity-contrast primitives; no outcomes or execution on import."""
from __future__ import annotations

import numpy as np

from .path_a_assay import SourceScale, row_weights
from .path_a_portability import descriptor_bins


def match_background_panels(tf, background, weights, tf_ids, background_ids, *, draws=128):
    """Exact source descriptor strata, shared cuts, fixed seeds, no replacement.

    A shortage returns a HOLD with no draws. It must not trigger a silent change
    to the population, matching strata, panel size or candidate family universe.
    """
    tf, background = np.asarray(tf, float), np.asarray(background, float)
    tf_ids, background_ids = tuple(tf_ids), tuple(background_ids)
    if (tf.ndim != 2 or background.ndim != 2 or len(tf) != len(background)
            or tf.shape[1] != len(tf_ids) or background.shape[1] != len(background_ids)
            or not tf_ids or not background_ids or draws < 1):
        raise ValueError('Aligned nonempty source panels and positive draw count required')
    if (len(set(tf_ids)) != len(tf_ids) or len(set(background_ids)) != len(background_ids)
            or set(tf_ids) & set(background_ids)):
        raise ValueError('Unique, disjoint TF/background family identities required')
    weights = row_weights(weights, len(tf))
    bins, cuts = descriptor_bins(np.column_stack((tf, background)), weights)
    tf_bins, bg_bins = bins[:len(tf_ids)], bins[len(tf_ids):]
    keys = sorted(set(map(tuple, tf_bins)))
    strata = []
    for key in keys:
        requested = int(np.all(tf_bins == key, axis=1).sum())
        indices = np.flatnonzero(np.all(bg_bins == key, axis=1))
        indices = np.asarray(sorted(indices, key=lambda i: background_ids[i]), dtype=int)
        strata.append(dict(key=list(map(int, key)), requested=requested,
                           available=len(indices), indices=indices))
    audit = dict(rule='shared_source_X_plus_U_zero_and_positive_median_bins',
        zero_threshold=1e-12, source_only=True,
        cuts=[{k:v for k,v in cut.items() if k != 'values'} for cut in cuts],
        strata=[{k:v for k,v in s.items() if k != 'indices'} for s in strata],
        shortages=[{k:v for k,v in s.items() if k != 'indices'} for s in strata
                   if s['available'] < s['requested']])
    if audit['shortages']:
        return None, dict(audit, status='HOLD_INSUFFICIENT_EXACT_STRATUM_CAPACITY')
    panels = []
    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([20261102, 11, draw]))
        panel = np.concatenate([rng.choice(s['indices'], size=s['requested'], replace=False)
                                for s in strata])
        panel = np.asarray(sorted(panel, key=lambda i: background_ids[i]), dtype=int)
        if len(set(panel)) != len(tf_ids):
            raise AssertionError('Background panel cardinality/replacement failure')
        panels.append(panel)
    panels = np.stack(panels)
    return panels, dict(audit, status='MATCHED_SOURCE_DESCRIPTORS', draws=draws,
        unique_panels=len({tuple(p) for p in panels}), families_per_panel=len(tf_ids))


def match_ridge_trace(eigenvalues, target_df):
    """Match slope trace only; an intercept is common to all compared models.

    Equal trace does not imply equal shrinkage spectrum or prediction variance.
    """
    values = np.asarray(eigenvalues, float)
    if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0)
            or not 0 < target_df < len(values)):
        raise ValueError('Positive spectrum with rank strictly above target df required')
    low, high = 0., float(values.sum()/target_df)
    for _ in range(100):
        regularization = (low+high)/2
        trace = np.sum(values/(values+regularization))
        if trace > target_df:
            low = regularization
        else:
            high = regularization
    regularization = (low+high)/2
    shrinkage = values/(values+regularization)
    achieved = float(shrinkage.sum())
    if abs(achieved-target_df) >= 1e-9:
        raise ArithmeticError('Absolute df matching tolerance not met')
    return dict(regularization=regularization, achieved_df=achieved,
                sum_squared_shrinkage=float(np.square(shrinkage).sum()))


def match_active_background_panels(tf, background, weights, tf_ids, background_ids, *, draws=128):
    """Explicit active-family amendment; preserve nominal matching as a separate test."""
    tf, background = np.asarray(tf, float), np.asarray(background, float)
    tf_ids, background_ids = tuple(tf_ids), tuple(background_ids)
    if (tf.ndim != 2 or background.ndim != 2 or tf.shape[1] != len(tf_ids)
            or background.shape[1] != len(background_ids) or len(tf) != len(background)):
        raise ValueError('Aligned source panels required')
    if (len(set(tf_ids)) != len(tf_ids) or len(set(background_ids)) != len(background_ids)
            or set(tf_ids) & set(background_ids)):
        raise ValueError('Unique, disjoint original TF/background axes required')
    tf_active = np.flatnonzero(~SourceScale.fit(tf, weights).constant)
    bg_active = np.flatnonzero(~SourceScale.fit(background, weights).constant)
    panels, audit = match_background_panels(tf[:, tf_active], background[:, bg_active], weights,
        [tf_ids[i] for i in tf_active], [background_ids[i] for i in bg_active], draws=draws)
    audit.update(amendment='active_family_count_not_nominal_family_count',
        constant_rule='source_weighted_standard_deviation_le_1e-12',
        nominal_TF_families=len(tf_ids), active_TF_families=len(tf_active),
        nominal_U_families=len(background_ids), active_U_families=len(bg_active),
        active_TF_indices=tf_active.tolist(),
        omitted_constant_TF_families=[x for i,x in enumerate(tf_ids) if i not in set(tf_active)])
    return (None if panels is None else bg_active[panels]), audit
