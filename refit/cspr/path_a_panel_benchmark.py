"""Registered linear-panel diagnostic, distinct from failed matched-gene pilot."""
from dataclasses import dataclass
import numpy as np
from .path_a_pilot import collection_weights, finite_matrix, pooled_r2, common_support

PANELS = ("TF", "Nbackground_all3132", "Nstate")


@dataclass
class SpectralDesign:
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    dual: bool
    rank: int

    @classmethod
    def prepare(cls, x, labels):
        x = finite_matrix(x)
        if len(x) != len(labels):
            raise ValueError("Training labels misaligned")
        w = collection_weights(labels); mean = w @ x
        sd = np.sqrt(w @ ((x-mean)**2)); constant = sd <= 1e-12
        sd[constant] = 1.
        z = (x-mean)/sd; z[:, constant] = 0.
        a = np.sqrt(w)[:, None]*z
        dual = x.shape[1] > len(x)
        gram = a @ a.T if dual else a.T @ a
        values, vectors = np.linalg.eigh(gram)
        largest = max(float(values[-1]), 0.)
        cutoff = np.finfo(float).eps * max(x.shape) * largest
        keep = values > cutoff
        if not keep.any():
            raise ValueError("Zero numerical input rank; no silent model omission")
        return cls(mean, sd, constant, z, w, values[keep], vectors[:, keep], dual, int(keep.sum()))

    def fit(self, y, target_df):
        y = finite_matrix(y)
        if len(y) != len(self.z) or not 0 < target_df < self.rank:
            raise ValueError("Invalid target trace or response axis")
        low, high = 0., float(self.eigenvalues.sum()/target_df)
        for _ in range(100):
            lam = (low+high)/2
            trace = np.sum(self.eigenvalues/(self.eigenvalues+lam))
            if trace > target_df: low = lam
            else: high = lam
        lam = (low+high)/2
        shrink = self.eigenvalues/(self.eigenvalues+lam)
        if abs(float(shrink.sum())-target_df) >= 1e-8:
            raise ValueError("Ridge trace equation did not converge")
        mean_y = self.weights @ y
        if self.dual:
            a = np.sqrt(self.weights)[:, None]*self.z
            rhs = self.eigenvectors.T @ (np.sqrt(self.weights)[:, None]*(y-mean_y))
            coefficient = a.T @ (self.eigenvectors @ (rhs/(self.eigenvalues+lam)[:, None]))
        else:
            rhs = self.eigenvectors.T @ (self.z.T @ (self.weights[:, None]*(y-mean_y)))
            coefficient = self.eigenvectors @ (rhs/(self.eigenvalues+lam)[:, None])
        coefficient[self.constant] = 0.
        return PanelModel(self.mean, self.scale, self.constant, mean_y, coefficient,
                          dict(rank=self.rank, target_df=float(target_df), achieved_df=float(shrink.sum()),
                               sum_squared_shrinkage=float(np.square(shrink).sum()), regularization=float(lam),
                               columns=len(self.mean), constant_columns=np.flatnonzero(self.constant).tolist()))


@dataclass
class PanelModel:
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray
    target_mean: np.ndarray
    coefficient: np.ndarray
    audit: dict

    def predict(self, x):
        x = finite_matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Input panel axis changed")
        z = (x-self.mean)/self.scale; z[:, self.constant] = 0.
        return z @ self.coefficient + self.target_mean


def panel_scores(y, predictions):
    if set(predictions) != {"N0", *PANELS}:
        raise ValueError("Every declared panel/control is required")
    scores = {k:pooled_r2(y, v) for k,v in predictions.items()}
    if any(v is None for v in scores.values()):
        return dict(scores=scores, delta_panel=None, reason="all_outputs_constant")
    strongest = max(("N0", "Nbackground_all3132", "Nstate"), key=lambda k:scores[k])
    return dict(scores=scores, delta_panel=scores["TF"]-scores[strongest], strongest_control=strongest, reason=None)


def nested_panel_fold(builder, train, test, fractions):
    if len(train) != 2 or len(set(train)) != 2 or test in train:
        raise ValueError("Two training collections and one held-out collection required")
    fractions = tuple(sorted(fractions))
    if len(set(fractions)) != len(fractions) or not all(0 < f < 1 for f in fractions):
        raise ValueError("Invalid complexity fraction grid")
    tuning = {name:{f:[] for f in fractions} for name in PANELS}; audit = []; fits = 0
    for validation in train:
        inner = [s for s in train if s != validation]
        x,y,labels,vx,vy = builder(inner, validation)
        if set(x) != set(PANELS) or set(vx) != set(PANELS) or set(labels) != set(inner):
            raise ValueError("Invalid inner builder contract")
        designs = {k:SpectralDesign.prepare(v, labels) for k,v in x.items()}
        common_rank = min(d.rank for d in designs.values())
        for name in PANELS:
            for fraction in fractions:
                model = designs[name].fit(y, fraction*common_rank); fits += 1
                score = pooled_r2(vy, model.predict(vx[name]))
                if score is None:
                    raise ValueError("Undefined inner target variation")
                tuning[name][fraction].append(score)
                audit.append(dict(stage="inner", train=inner, evaluation=validation, model=name, fraction=fraction,
                                  shared_rank=common_rank, **model.audit))
    chosen = {name:max(fractions, key=lambda f:(np.mean(tuning[name][f]), -f)) for name in PANELS}
    x,y,labels,tx,ty = builder(train, test)
    if set(x) != set(PANELS) or set(tx) != set(PANELS) or set(labels) != set(train):
        raise ValueError("Invalid outer builder contract")
    designs = {k:SpectralDesign.prepare(v, labels) for k,v in x.items()}
    common_rank = min(d.rank for d in designs.values())
    models = {k:d.fit(y, chosen[k]*common_rank) for k,d in designs.items()}; fits += len(models)
    predictions = {k:m.predict(tx[k]) for k,m in models.items()}
    predictions["N0"] = np.broadcast_to(collection_weights(labels) @ y, ty.shape).copy()
    for name,m in models.items():
        audit.append(dict(stage="outer", train=list(train), evaluation=test, model=name, fraction=chosen[name],
                          shared_rank=common_rank, **m.audit))
    return dict(**panel_scores(ty, predictions), fractions=chosen, inner_scores=tuning, models=models,
                predictions=predictions, complexity_audit=audit, ridge_fits=fits)


def diagnostic_route(deltas, support):
    advantage = all(d is not None and d > 0 for d in deltas) and float(np.mean(deltas)) >= .02
    supported = all(s >= .8 for s in support)
    return dict(panel_advantage=advantage, broad_support_pass=supported,
                broad_panel_screen=advantage and supported, original_matched_control_pass=False,
                TF_specificity=False, causal_regulation=False, A1=False, A2=False, C1=False)


class PanelBuilder:
    """Reuse immutable count/group code without reinterpreting failed matches."""
    def __init__(self, source):
        self.source = source
        self.calls = []

    def build(self, arm, train, evaluation):
        train = tuple(sorted(train))
        if evaluation in train:
            raise ValueError("Held-out collection entered training")
        measured = [self.source.measured(arm, train, s) for s in train]
        test = self.source.measured(arm, train, evaluation)
        def features(m):
            return dict(TF=m["x"], Nbackground_all3132=m["candidates"], Nstate=m["geometry"]["nstate"])
        feature_sets = [features(m) for m in measured]
        x = {k:np.vstack([f[k] for f in feature_sets]) for k in PANELS}
        y = np.vstack([m["y"] for m in measured])
        labels = np.concatenate([np.repeat(s, len(m["y"])) for s,m in zip(train, measured)])
        self.calls.append(dict(arm=arm, train=list(train), evaluation=evaluation))
        return x,y,labels,features(test),test["y"]

    def support(self, arm, train, evaluation):
        geometry = [self.source.geometry_for(arm, train, s) for s in train]
        states = np.concatenate([g["dominant"] for g in geometry])
        labels = np.concatenate([np.repeat(s, len(g["dominant"])) for s,g in zip(train, geometry)])
        test = self.source.geometry_for(arm, train, evaluation)
        return common_support(states, labels, test["dominant"])


def subset_scores(y, predictions, support):
    result = {}
    support = np.asarray(support, dtype=bool)
    for name, mask in [("supported", support), ("unsupported", ~support)]:
        result[name] = dict(units=int(mask.sum()), result=panel_scores(y[mask], {k:v[mask] for k,v in predictions.items()}) if mask.sum() >= 20 else None)
    return result
