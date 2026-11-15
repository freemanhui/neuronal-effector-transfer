"""WP1 execution mechanics: source-only selection and auditable readouts.

Target endpoint construction remains in the frozen v0.5 helper. No real data
access occurs on import. Matched panels are reselected inside each CV fold.
"""
import hashlib
from decimal import Decimal, localcontext
import numpy as np
from .path_a_assay import hierarchical_weights, weighted_ridge_design
from .path_a_v03_capacity import match_background_panels, match_active_background_panels
from .path_a_v05 import DF_GRID, scores

MODELS = ('TF', 'Nfull', 'Nfull_df') + tuple(f'N1_active_{i:03d}' for i in range(128))
FRACTIONS = tuple(DF_GRID) + (.35,)


def folds_for(identities):
    ids = list(map(str, identities))
    if len(ids) < 10 or len(set(ids)) != len(ids):
        raise ValueError('At least ten unique source unit identities required')
    order = sorted(range(len(ids)), key=lambda i: (hashlib.sha256(ids[i].encode()).hexdigest(), ids[i]))
    folds = np.empty(len(ids), int)
    folds[order] = np.arange(len(ids)) % 5
    return folds


def matching_plan(tf, background, study, sample, identities, tf_ids, background_ids):
    """No Y input. Full-source and five strictly training-only panel collections."""
    folds = folds_for(identities)
    plans, audits = [], []
    for fold in (-1, 0, 1, 2, 3, 4):
        train = np.ones(len(tf), bool) if fold == -1 else folds != fold
        w = hierarchical_weights(study[train], sample[train])
        _, nominal = match_background_panels(tf[train], background[train], w, tf_ids, background_ids)
        selected, active = match_active_background_panels(tf[train], background[train], w, tf_ids, background_ids)
        if selected is None:
            raise ValueError(f'Matched-panel shortage in training fold {fold}: {active}')
        plans.append(selected)
        audits.append(dict(fold=fold, training_units=int(train.sum()),
                           training_ids_sha256=hashlib.sha256('\n'.join(map(str, identities[train])).encode()).hexdigest(),
                           nominal=nominal, active=active))
    # Active family counts can legitimately differ across training folds.
    # Preserve each fold's exact panel width instead of padding/substituting.
    return folds, plans, audits


def features(data, name, panel_indices, group=0):
    if name == 'TF':
        return data['TF']
    if name in ('Nfull', 'Nfull_df'):
        return data['Nfull']
    if name not in MODELS:
        raise ValueError('Unknown registered model')
    return data['Nfull'][:, panel_indices[group][int(name.rsplit('_', 1)[1])]]


def select_source(data, name, panel_indices, folds):
    """Five-fold loss path with fold-specific family sets and feature scaling."""
    losses = np.empty((5, len(DF_GRID)))
    ranks = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        x = features(data, name, panel_indices, fold+1)
        w = hierarchical_weights(data['study'][train], data['sample'][train])
        wt = hierarchical_weights(data['study'][valid], data['sample'][valid])
        design = weighted_ridge_design(x[train], w)
        rank = (weighted_ridge_design(data['TF'][train], w).rank
                if name == 'Nfull_df' else design.rank)
        ranks.append(dict(model_rank=design.rank, reference_rank=rank))
        for j, fraction in enumerate(DF_GRID):
            model = design.fit(data['y'][train], rank*fraction)
            losses[fold, j] = np.sum(wt[:, None]*(data['y'][valid]-model.predict(x[valid]))**2)
    chosen = int(np.argmin(losses.mean(axis=0)))
    return dict(selected_index=chosen, selected_fraction=float(DF_GRID[chosen]),
                losses=losses.tolist(), ranks=ranks)


def checked_scores(y, prediction, weights):
    """Cross-check the covariance formula with independently formed residual SSE."""
    out = scores(y, prediction, weights)
    w = np.asarray(weights, float) / np.sum(weights)
    yc, pc = y-w@y, prediction-w@prediction
    s = np.sum(w[:, None]*yc**2, axis=0)
    ec = np.sum(w[:, None]*(yc-pc)**2, axis=0)
    valid = np.array([v is not None for v in out['family_R2']])
    errors = np.zeros(y.shape[1]+1)
    errors[:-1][valid] = np.abs(np.array(out['family_R2'], dtype=float)[valid] - (1-ec[valid]/s[valid]))
    if out['R2_centered'] is not None:
        errors[-1] = abs(out['R2_centered']-(1-ec.sum()/s.sum()))
    out['float64_check_max_error'] = float(errors.max())
    indices = np.flatnonzero(errors[:-1] > 1e-12).tolist()
    pooled = bool(errors[-1] > 1e-12)
    out['high_precision_family_indices'] = indices
    out['high_precision_pooled'] = pooled
    out['R2_raw'] = float(1-np.sum(w[:, None]*(y-prediction)**2)/s.sum()) if s.sum() > 0 else None
    if indices or pooled:
        # Float64 absolute-error checks can fail for large negative R2 even
        # though the two formulas are equivalent. Recompute the affected
        # statistics, not the models, in 50-digit arithmetic. Never widen tol.
        with localcontext() as ctx:
            ctx.prec = 50
            dw = [Decimal.from_float(float(v)) for v in w]
            total = sum(dw); dw = [v/total for v in dw]
            totals = [Decimal(0)]*5
            for j in (range(y.shape[1]) if pooled else indices):
                dy = [Decimal.from_float(float(v)) for v in y[:, j]]
                dp = [Decimal.from_float(float(v)) for v in prediction[:, j]]
                my, mp = sum(a*b for a,b in zip(dw,dy)), sum(a*b for a,b in zip(dw,dp))
                cy, cp = [v-my for v in dy], [v-mp for v in dp]
                sy = sum(a*b*b for a,b in zip(dw,cy))
                sp = sum(a*b*b for a,b in zip(dw,cp))
                c = sum(a*b*d for a,b,d in zip(dw,cy,cp))
                residual = sum(a*(b-d)**2 for a,b,d in zip(dw,cy,cp))
                raw = sum(a*(b-d)**2 for a,b,d in zip(dw,dy,dp))
                totals = [a+b for a,b in zip(totals,(sy,sp,c,residual,raw))]
                if j in indices:
                    out['family_R2'][j] = float((2*c-sp)/sy)
                    errors[j] = float(abs((2*c-sp)/sy-(1-residual/sy)))
            if pooled:
                sy, sp, c, residual, raw = totals
                rho = c/(sy*sp).sqrt() if sp else None
                out.update(R2_centered=float((2*c-sp)/sy), R2_raw=float(1-raw/sy),
                    rho=float(rho) if rho is not None else None, amplitude=float((sp/sy).sqrt()),
                    oracle_R2=float(max(Decimal(0),rho)**2) if rho is not None else None)
                errors[-1] = float(abs((2*c-sp)/sy-(1-residual/sy)))
        out['family_mean_R2'] = float(np.mean([v for v in out['family_R2'] if v is not None])) if valid.any() else None
    if errors.max() > 1e-12:
        raise ArithmeticError('Independent residual-SSE score check exceeds 1e-12')
    out['score_check_max_error'] = float(errors.max())
    return out


def require_column_digest(actual, expected):
    if actual != expected:
        raise ValueError('Target endpoint changed between source rows')
