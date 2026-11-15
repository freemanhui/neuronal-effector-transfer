"""Prospective P1-AD mechanics; biological exposure/admission is a runner gate."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse

from .path_a_assay import (
    FamilyCounts, SourceState, StateNeighbours, build_sample_units,
    hierarchical_weights, weighted_pooled_r2, weighted_ridge_design,
)
from .path_a_pilot import finite_matrix

RIDGE_FRACTIONS = (.05, .15, .35, .65, .90)
NEIGHBOUR_FRACTIONS = (.01, .03, .10, .30, 1.)
CONTROL_CODES = {"N2a":21,"N2b":22,"N3a":31,"N3b":32}
REFERENCES = ("N0_target","N0_source","Nfull","Nstate","Nstate_nn","N2a","N2b","N3a","N3b")


def pooled_score(y,prediction,weights):
    score = weighted_pooled_r2(y,prediction,weights)
    # Exact constant rows have mathematical SST=0 even if a weighted sum rounds.
    return None if np.all(np.asarray(y)==np.asarray(y)[0]) else score


@dataclass(frozen=True)
class Axes:
    ids: tuple
    x: tuple
    y: tuple
    u: tuple
    s: tuple
    size_pairs: dict
    modules: dict

    def __post_init__(self):
        if len(self.ids)!=len(set(self.ids)):
            raise ValueError("Duplicate family identity")
        x,y,u,s = map(set,(self.x,self.y,self.u,self.s))
        if not x or not y or not s or not s<=u or x&y or x&u or y&u or x|y|u!=set(self.ids):
            raise ValueError("Full disjoint X/Y/U and S subset required")
        if set(self.size_pairs)!=x|y or set(self.modules)!=y:
            raise ValueError("Exact size/module metadata required")
        if any(len(v)!=2 or any(type(n) is not int or n<1 for n in v) for v in self.size_pairs.values()):
            raise ValueError("Positive exact species size pairs required")
        if any(tuple(sorted(set(v)))!=tuple(v) for v in self.modules.values()):
            raise ValueError("Canonical complete module sets required")


@dataclass(frozen=True)
class Sample:
    species: str
    study: str
    sample: str
    barcodes: tuple
    counts: FamilyCounts

    @property
    def key(self):
        return (self.study,self.sample)


@dataclass(frozen=True)
class Units:
    expression: np.ndarray
    state: np.ndarray
    dominant: np.ndarray
    study: np.ndarray
    sample: np.ndarray
    identities: np.ndarray
    planned_samples: tuple
    remainders: dict

    @property
    def weights(self):
        return hierarchical_weights(self.study,self.sample)

    def subset(self,mask):
        mask = np.asarray(mask,bool)
        return Units(self.expression[mask],self.state[mask],self.dominant[mask],self.study[mask],
            self.sample[mask],self.identities[mask],self.planned_samples,self.remainders)


def check_samples(samples,axes):
    samples = tuple(samples)
    if not samples or len({s.species for s in samples})!=1 or not all(s.species and s.study and s.sample for s in samples):
        raise ValueError("One named species and nonempty samples required")
    if len({s.key for s in samples})!=len(samples):
        raise ValueError("Repeated biological sample")
    for sample in samples:
        shape = sample.counts.linear.shape
        if shape!=(len(sample.barcodes),len(axes.ids)) or len(set(sample.barcodes))!=shape[0] or shape[0]==0:
            raise ValueError("Exact aligned unique cell/family axis required")
        if len(sample.counts.denominator)!=shape[0] or np.any(np.asarray(sample.counts.denominator)<=0):
            raise ValueError("Positive normalized D population required")
    return tuple(sorted(samples,key=lambda s:s.key))


def _state_cells(samples,axes):
    index = [axes.ids.index(i) for i in axes.s]
    x = np.vstack([np.log1p(s.counts.linear[:,index].toarray()) for s in samples])
    study = np.concatenate([[s.study]*len(s.barcodes) for s in samples])
    sample = np.concatenate([[s.sample]*len(s.barcodes) for s in samples])
    return x,hierarchical_weights(study,sample)


def source_coordinate(samples,axes):
    samples = check_samples(samples,axes)
    return SourceState.fit(*_state_cells(samples,axes))


def make_units(samples,axes,coordinate):
    samples = check_samples(samples,axes)
    indices = [axes.ids.index(i) for i in axes.s]
    expr,features,dominant,studies,names,identities = [],[],[],[],[],[]
    remainders = {}
    for sample in samples:
        part = build_sample_units(sample.counts,indices,coordinate,sample.barcodes)
        n = len(part["expression"])
        expr.append(part["expression"]);features.append(part["state_features"]);dominant.append(part["dominant"])
        studies += [sample.study]*n;names += [sample.sample]*n
        identities += [f"{sample.study}/{sample.sample}/{i:08d}" for i in range(n)]
        remainders[sample.key] = int(np.sum(part["assignment"]<0))
    return Units(np.vstack(expr),np.vstack(features),np.concatenate(dominant),np.asarray(studies),
        np.asarray(names),np.asarray(identities),tuple(s.key for s in samples),remainders)


def sample_unit_counts(units):
    return {key:int(np.sum((units.study==key[0])&(units.sample==key[1]))) for key in units.planned_samples}


def require_units(units,*,heldout=False):
    counts = sample_unit_counts(units)
    if not counts or min(counts.values())<1:
        raise ValueError("Every planned sample requires one complete unit")
    if heldout:
        for study in set(units.study):
            if np.sum(units.study==study)<20:
                raise ValueError("Held-out study requires twenty complete units")


def panels(units,axes):
    indices = {key:[axes.ids.index(i) for i in ids] for key,ids in [("TF",axes.x),("Nfull",axes.u)]}
    return {**{key:units.expression[:,idx] for key,idx in indices.items()},"Nstate":units.state}


def responses(units,axes):
    return units.expression[:,[axes.ids.index(i) for i in axes.y]]


def descriptor_bins(values,weights):
    x = finite_matrix(values)
    if np.any(x<0):
        raise ValueError("Nonnegative family expression required for descriptors")
    w = np.asarray(weights,float)
    mean = w@x
    descriptors = (w@(x>0),mean,w@((x-mean)**2))
    bins,cuts = [],[]
    for name,d in zip(("detection","mean_log","variance"),descriptors):
        positive = d>1e-12
        cut = float(np.median(d[positive])) if positive.any() else None
        label = np.zeros(len(d),int)
        if cut is not None:
            label[positive] = 1+(d[positive]>cut)
        bins.append(label)
        cuts.append(dict(descriptor=name,positive_median=cut,zero_threshold=1e-12,values=d.tolist()))
    return np.stack(bins,axis=1),cuts


@dataclass(frozen=True)
class Correspondence:
    name: str
    mappings: np.ndarray
    audit: dict


def correspondence(name,ids,size_pairs,modules,values,weights):
    if name not in CONTROL_CODES or len(ids)!=len(set(ids)):
        raise ValueError("Declared correspondence and unique family identities required")
    conditioned = name.endswith("b")
    output = name.startswith("N3")
    bins,cuts = descriptor_bins(values,weights)
    groups = {}
    for i,family in enumerate(ids):
        key = tuple(size_pairs[family])
        if output:
            key += (tuple(modules[family]),)
        if conditioned:
            key += tuple(map(int,bins[i]))
        groups.setdefault(key,[]).append(i)
    ordered = [(key,np.asarray(sorted(indices,key=lambda i:ids[i]))) for key,indices in sorted(groups.items())]
    movable = sorted(i for _,indices in ordered if len(indices)>1 for i in indices)
    if not movable:
        raise ValueError(f"Required {name} has no movable stratum")
    identity = np.arange(len(ids))
    mappings,attempts = [],[]
    for draw in range(128):
        rng = np.random.default_rng(np.random.SeedSequence([20261030,CONTROL_CODES[name],draw]))
        for attempt in range(1,257):
            mapping = identity.copy()
            for _,indices in ordered:
                mapping[indices] = rng.permutation(indices)
            if not np.array_equal(mapping,identity):
                mappings.append(mapping);attempts.append(attempt);break
        else:
            raise ValueError(f"Required {name} exhausted nonidentity attempts")
    mappings = np.asarray(mappings)
    audit = dict(descriptor_cuts=cuts if conditioned else None,
        strata=[dict(key=list(key),families=[ids[i] for i in indices]) for key,indices in ordered],
        singleton_families=[ids[indices[0]] for _,indices in ordered if len(indices)==1],
        movable_family_fraction=len(movable)/len(ids),
        moved_families_per_draw=np.sum(mappings!=identity,axis=1).tolist(),attempts=attempts,
        unique_mappings=len({row.tobytes() for row in mappings}),draws=128,source_only=True)
    return Correspondence(name,mappings,audit)


@dataclass(frozen=True)
class Direction:
    axes: Axes
    source_species: str
    source_studies: frozenset
    coordinate: SourceState
    units: Units
    models: dict
    chosen: dict
    tuning: dict
    source_mean: np.ndarray
    correspondences: dict
    execution_audit: dict
    target_species: str = ""
    target_training_studies: frozenset = frozenset()
    target_mean: object = None


def fit_direction(development,validation,axes):
    development,validation = check_samples(development,axes),check_samples(validation,axes)
    if development[0].species!=validation[0].species or {s.study for s in development}&{s.study for s in validation}:
        raise ValueError("Independent same-species source validation studies required")
    training_state = source_coordinate(development,axes)
    train = make_units(development,axes,training_state)
    val = make_units(validation,axes,training_state)
    require_units(train);require_units(val,heldout=True)
    y_train,y_val = responses(train,axes),responses(val,axes)
    train_x,val_x = panels(train,axes),panels(val,axes)
    chosen,tuning = {},{}
    for name in ("TF","Nfull","Nstate"):
        design = weighted_ridge_design(train_x[name],train.weights)
        candidates = []
        for fraction in RIDGE_FRACTIONS:
            model = design.fit(y_train,fraction*design.rank)
            score = pooled_score(y_val,model.predict(val_x[name]),val.weights)
            if score is None:
                raise ValueError("Undefined source-validation SST; tuning HOLD")
            candidates.append(dict(fraction=fraction,score=score,rank=design.rank,audit=model.audit))
        chosen[name] = max(candidates,key=lambda r:(r["score"],-r["fraction"]))["fraction"]
        tuning[name] = candidates
    neighbour = StateNeighbours.fit(train.state,y_train,train.weights,train.identities)
    candidates = []
    cache = {}
    for fraction in NEIGHBOUR_FRACTIONS:
        k = max(1,int(np.ceil(fraction*len(train.expression))))
        reused = k in cache
        if not reused:
            cache[k] = neighbour.predict(val.state,fraction)
        score = pooled_score(y_val,cache[k],val.weights)
        if score is None:
            raise ValueError("Undefined neighbour source-validation score")
        candidates.append(dict(fraction=fraction,k=k,score=score,duplicate_k_prediction_reused=reused))
    chosen["Nstate_nn"] = max(candidates,key=lambda r:(r["score"],r["fraction"]))["fraction"]
    tuning["Nstate_nn"] = candidates
    coordinate = source_coordinate(development+validation,axes)
    source = make_units(development+validation,axes,coordinate)
    require_units(source)
    y = responses(source,axes)
    x = panels(source,axes)
    models = {}
    for name in ("TF","Nfull","Nstate"):
        design = weighted_ridge_design(x[name],source.weights)
        models[name] = design.fit(y,chosen[name]*design.rank)
    models["Nstate_nn"] = StateNeighbours.fit(source.state,y,source.weights,source.identities)
    controls = {name:correspondence(name,axes.x if name.startswith("N2") else axes.y,
        axes.size_pairs,axes.modules,x["TF"] if name.startswith("N2") else y,source.weights) for name in CONTROL_CODES}
    audit = dict(tuning_ridge_fits=15,final_ridge_fits=3,source_PCA_fits=2,source_KMeans_fits=2,
        tuning_studies=sorted({s.study for s in development}),validation_studies=sorted({s.study for s in validation}),
        final_studies=sorted({s.study for s in development+validation}),
        upper_bound_ridge_choices={name:chosen[name]==RIDGE_FRACTIONS[-1] for name in ("TF","Nfull","Nstate")},
        tuning_unit_identities=train.identities.tolist(),validation_unit_identities=val.identities.tolist(),
        final_unit_identities=source.identities.tolist())
    return Direction(axes,development[0].species,frozenset(s.study for s in development+validation),coordinate,
        source,models,chosen,tuning,source.weights@y,controls,audit)


def attach_target_training(direction,samples):
    from dataclasses import replace
    samples = check_samples(samples,direction.axes)
    studies = frozenset(s.study for s in samples)
    if samples[0].species==direction.source_species or studies&direction.source_studies:
        raise ValueError("Independent opposite-species target training required")
    units = make_units(samples,direction.axes,direction.coordinate)
    require_units(units)
    return replace(direction,target_species=samples[0].species,target_training_studies=studies,
        target_mean=units.weights@responses(units,direction.axes))


def predictions(direction,units):
    if direction.target_mean is None:
        raise ValueError("Independent target-training mean missing")
    x = panels(units,direction.axes)
    raw = {name:model.predict(x[name]) for name,model in direction.models.items() if name!="Nstate_nn"}
    raw["Nstate_nn"] = direction.models["Nstate_nn"].predict(units.state,direction.chosen["Nstate_nn"])
    shape = raw["TF"].shape
    raw["N0_source"] = np.broadcast_to(direction.source_mean,shape).copy()
    raw["N0_target"] = np.broadcast_to(direction.target_mean,shape).copy()
    calibrated = {name:direction.target_mean+value-direction.source_mean for name,value in raw.items()
                  if name not in ("N0_source","N0_target")}
    calibrated.update({name:raw[name].copy() for name in ("N0_source","N0_target")})
    controls_raw,controls_calibrated = {},{}
    for name,control in direction.correspondences.items():
        if name.startswith("N2"):
            draws = np.asarray([direction.models["TF"].predict(x["TF"][:,p]) for p in control.mappings])
            shifted = direction.target_mean+draws-direction.source_mean
        else:
            draws = np.asarray([raw["TF"][:,p] for p in control.mappings])
            residual = raw["TF"]-direction.source_mean
            shifted = np.asarray([direction.target_mean+residual[:,p] for p in control.mappings])
        controls_raw[name] = draws
        controls_calibrated[name] = shifted
    return raw,calibrated,controls_raw,controls_calibrated


def support(source,target):
    counts = {key:np.bincount(source.dominant[(source.study==key[0])&(source.sample==key[1])],minlength=8)
        for key in source.planned_samples}
    allowed = np.flatnonzero(np.all(np.stack(list(counts.values()))>=5,axis=0))
    mask = np.isin(target.dominant,allowed)
    return mask,dict(source_state_frequencies={"/".join(k):v.tolist() for k,v in counts.items()},
        supported_states=allowed.tolist(),supported_units=int(mask.sum()),full_units=len(mask),
        supported_fraction=float(mask.mean()) if len(mask) else None,
        warning_below_80_percent=(float(mask.mean())<.8) if len(mask) else True)


def score_predictions(y,weights,raw,calibrated,controls_raw,controls_calibrated):
    result = {}
    for label,models,controls in [("raw",raw,controls_raw),("mean_calibrated",calibrated,controls_calibrated)]:
        scores = {name:pooled_score(y,prediction,weights) for name,prediction in models.items()}
        draws = {}
        for name,predictions_ in controls.items():
            values = [pooled_score(y,p,weights) for p in predictions_]
            draws[name] = values
            scores[name] = None if any(v is None for v in values) else float(np.quantile(values,.95,method="higher"))
        complete = set(scores)=={"TF",*REFERENCES} and all(scores[n] is not None for n in ("TF",*REFERENCES))
        strongest = max(REFERENCES,key=lambda n:scores[n]) if complete else None
        result[label] = dict(scores=scores,control_draw_scores=draws,strongest_reference=strongest,
            delta=scores["TF"]-scores[strongest] if complete else None,
            status="DESCRIPTIVE_COHORT_POINT_ONLY" if complete else "HOLD_UNDEFINED_REQUIRED_SCORE")
    return result


def evaluate_direction(direction,samples):
    samples = check_samples(samples,direction.axes)
    if len({s.study for s in samples})!=1 or samples[0].species!=direction.target_species:
        raise ValueError("One declared opposite-species held-out study required")
    if {s.study for s in samples}&(direction.source_studies|direction.target_training_studies):
        raise ValueError("Study leakage into held-out scoring")
    units = make_units(samples,direction.axes,direction.coordinate)
    support_mask,support_audit = support(direction.units,units)
    audit = dict(sample_units={"/".join(k):v for k,v in sample_unit_counts(units).items()},
        remainders={"/".join(k):v for k,v in units.remainders.items()},support=support_audit)
    try:
        require_units(units,heldout=True)
    except ValueError as error:
        return dict(status="HOLD_UNIT_CAPACITY",reason=str(error),audit=audit)
    values = predictions(direction,units)
    y = responses(units,direction.axes)
    full = score_predictions(y,units.weights,*values)
    per_sample = {}
    for sample in samples:
        mask = (units.study==sample.study)&(units.sample==sample.sample)
        subset = units.subset(mask)
        selected = [({k:v[mask] for k,v in item.items()} if i<2 else {k:v[:,mask] for k,v in item.items()})
            for i,item in enumerate(values)]
        per_sample["/".join(sample.key)] = score_predictions(y[mask],subset.weights,*selected)
    supported = dict(status="UNDEFINED_FEWER_THAN_20_SUPPORTED_UNITS",units=int(support_mask.sum()))
    if support_mask.sum()>=20:
        subset = units.subset(support_mask)
        selected = [({k:v[support_mask] for k,v in item.items()} if i<2 else {k:v[:,support_mask] for k,v in item.items()})
            for i,item in enumerate(values)]
        supported = dict(status="SECONDARY_SUPPORTED_POPULATION",units=int(support_mask.sum()),
            empty_planned_samples=["/".join(k) for k,v in sample_unit_counts(subset).items() if v==0],
            scores=score_predictions(y[support_mask],subset.weights,*selected))
    status = "DESCRIPTIVE_COHORT_POINT_ONLY" if all(arm["delta"] is not None for arm in full.values()) else "HOLD_UNDEFINED_REQUIRED_SCORE"
    return dict(status=status,full=full,per_sample=per_sample,supported=supported,audit=audit)
