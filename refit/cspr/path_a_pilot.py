"""Fold-local numerical kernels for the registered development source pilot.

This module has no file/network access and no biological execution entry point.
The caller must rebuild grouping and control matching inside every fold.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .path_a_measurement import aggregate_families


CONTROL_NAMES = ("N0", "N1_1", "N1_2", "N1_3", "N1_4", "N1_5", "Nstate")
MODEL_NAMES = ("TF", "N1_1", "N1_2", "N1_3", "N1_4", "N1_5", "Nstate")


def collection_weights(collections):
    labels = np.asarray(collections, dtype=str)
    if labels.ndim != 1 or not len(labels) or np.any(labels == ""):
        raise ValueError("Resolved nonempty collection vector required")
    names, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return 1.0 / (len(names) * counts[inverse])


def finite_matrix(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Nonempty finite matrix required")
    return result


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    coefficient: np.ndarray
    regularization: float

    def predict(self, x):
        x = finite_matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Changed input feature axis")
        return ((x-self.mean)/self.scale) @ self.coefficient + self.target_mean


def fit_ridge(x, y, collections, regularization):
    x, y = finite_matrix(x), finite_matrix(y)
    if len(x) != len(y) or len(collections) != len(x) or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Invalid ridge dimensions or regularization")
    weights = collection_weights(collections)
    mean = weights @ x
    sd = np.sqrt(weights @ ((x-mean)**2))
    sd[sd <= 1e-12] = 1.0
    z = (x-mean)/sd
    target_mean = weights @ y
    coefficients = np.linalg.solve(z.T @ (weights[:,None]*z) + regularization*np.eye(x.shape[1]),
                                   z.T @ (weights[:,None]*(y-target_mean)))
    return RidgeModel(mean, sd, target_mean, coefficients, float(regularization))


def pooled_r2(y, predicted):
    y, predicted = finite_matrix(y), finite_matrix(predicted)
    if y.shape != predicted.shape or len(y)<2:
        raise ValueError("Scoring requires aligned arrays and at least two units")
    denominator = float(np.square(y-y.mean(axis=0)).sum())
    if denominator == 0:
        return None
    return 1-float(np.square(y-predicted).sum())/denominator


def source_scores(y, predictions):
    required = {"TF", *CONTROL_NAMES}
    if set(predictions) != required:
        raise ValueError("All prespecified source controls are required")
    scores = {name:pooled_r2(y,pred) for name,pred in predictions.items()}
    if any(value is None for value in scores.values()):
        return dict(scores=scores,delta_dev=None,reason="all_outputs_constant")
    strongest = max(CONTROL_NAMES,key=lambda name:scores[name])
    return dict(scores=scores,delta_dev=scores["TF"]-scores[strongest],
                strongest_control=strongest,reason=None)


def nested_collection_fold(builder, training_samples, test_sample, lambdas):
    """Builder(train IDs, evaluation ID) must fit all transformations on train.

    Its return is (train feature dict, train Y, train collection IDs,
    evaluation feature dict, evaluation Y). Outer evaluation is requested only
    after tuning on the two inner splits. Biological grouping/matching belongs
    in that builder; fixed synthetic matrices alone do not validate it.
    """
    if len(training_samples)!=2 or len(set(training_samples))!=2 or test_sample in training_samples:
        raise ValueError("Two distinct training collections and one excluded test required")
    lambdas = sorted(set(float(x) for x in lambdas))
    if not lambdas or min(lambdas)<=0:
        raise ValueError("Positive registered lambda grid required")
    tuning = {name:{lam:[] for lam in lambdas} for name in MODEL_NAMES}
    for validation in training_samples:
        train = [x for x in training_samples if x!=validation]
        x,y,groups,vx,vy = builder(train,validation)
        if set(x)!=set(MODEL_NAMES) or set(vx)!=set(MODEL_NAMES) or set(groups)!=set(train):
            raise ValueError("Builder supplied wrong models or training collection IDs")
        for name in MODEL_NAMES:
            for lam in lambdas:
                score=pooled_r2(vy,fit_ridge(x[name],y,groups,lam).predict(vx[name]))
                if score is None:
                    raise ValueError("Undefined inner target variation; no silent fold deletion")
                tuning[name][lam].append(score)
    chosen={name:max(lambdas,key=lambda lam:(float(np.mean(tuning[name][lam])),lam)) for name in MODEL_NAMES}
    x,y,groups,tx,ty = builder(training_samples,test_sample)
    if set(x)!=set(MODEL_NAMES) or set(tx)!=set(MODEL_NAMES) or set(groups)!=set(training_samples):
        raise ValueError("Builder supplied wrong models or outer training IDs")
    models={name:fit_ridge(x[name],y,groups,chosen[name]) for name in MODEL_NAMES}
    predictions={name:model.predict(tx[name]) for name,model in models.items()}
    predictions["N0"] = np.broadcast_to(collection_weights(groups)@y,ty.shape).copy()
    return dict(lambdas=chosen,inner_scores=tuning,models=models,predictions=predictions,
                **source_scores(ty,predictions))


def measured_families(counts, gene_ids, families, denominator_genes, assignment, size, *, integer_required):
    raw=sparse.csr_matrix(counts,dtype=float,copy=True)
    raw.sum_duplicates();raw.eliminate_zeros()
    if not np.isfinite(raw.data).all() or (raw.data<0).any():
        raise ValueError("Finite nonnegative measurements required")
    if integer_required and not np.equal(raw.data,np.floor(raw.data)).all():
        raise ValueError("Fractional corrected expression is not original integer counts")
    index={g:i for i,g in enumerate(gene_ids)}
    if len(index)!=len(gene_ids) or raw.shape[1]!=len(gene_ids):
        raise ValueError("Unique aligned gene axis required")
    denominator_genes=list(denominator_genes)
    if not denominator_genes or len(set(denominator_genes))!=len(denominator_genes) or not set(denominator_genes)<=set(index):
        raise ValueError("Invalid normalization axis")
    totals=np.asarray(raw[:,[index[g] for g in denominator_genes]].sum(axis=1)).ravel()
    if np.any(totals<=0):
        raise ValueError("Zero denominator; do not silently discard cells")
    rr=[];cc=[];seen=set()
    for column,members in enumerate(families):
        members=list(members)
        if not members or len(members)!=len(set(members)) or seen&set(members) or not set(members)<=set(index):
            raise ValueError("Incomplete, duplicated or overlapping family; no zero fill")
        rr.extend(index[g] for g in members);cc.extend([column]*len(members));seen.update(members)
    sums=(raw@sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(gene_ids),len(families)))).toarray()
    linear=sums*(10000/totals[:,None])
    _,logged=aggregate_families(linear,assignment,size)
    return linear,logged


def balanced_fit_rows(collections, barcodes, cap=5000):
    names=np.asarray(collections,dtype=str);barcodes=np.asarray(barcodes,dtype=str)
    if len(names)!=len(barcodes) or len(set(zip(names,barcodes)))!=len(names):
        raise ValueError("Unique sample-qualified barcodes required")
    samples=np.unique(names)
    count=min(cap,min(np.count_nonzero(names==s) for s in samples))
    rows=[]
    for sample in samples:
        indices=np.flatnonzero(names==sample)
        rows.extend(sorted(indices,key=lambda i:hashlib.sha256(f"{sample}:{barcodes[i]}|group-fit-v1".encode()).digest())[:count])
    return np.asarray(rows,dtype=int)


@dataclass(frozen=True)
class GroupTransform:
    pool: tuple
    selected: tuple
    pca: PCA
    balanced_rows: np.ndarray

    def transform(self, counts, gene_ids):
        normalized=_background_log(counts,gene_ids,self.pool)
        index={g:i for i,g in enumerate(self.pool)}
        return self.pca.transform(normalized[:,[index[g] for g in self.selected]].toarray())


def _background_log(counts,gene_ids,pool):
    if len(gene_ids)!=len(set(gene_ids)) or counts.shape[1]!=len(gene_ids):
        raise ValueError("Unique gene axis required")
    index={g:i for i,g in enumerate(gene_ids)}
    if not set(pool)<=set(index):
        raise ValueError("Missing fixed grouping genes")
    values=sparse.csr_matrix(counts[:,[index[g] for g in pool]],dtype=float)
    if not np.isfinite(values.data).all() or (values.data<0).any():
        raise ValueError("Invalid background values")
    totals=np.asarray(values.sum(axis=1)).ravel()
    if np.any(totals<=0):
        raise ValueError("Zero grouping denominator")
    values=sparse.diags(10000/totals)@values
    values.data=np.log1p(values.data)
    return values.tocsr()


def fit_group_transform(counts,gene_ids,pool,collections,barcodes,*,max_genes=1000,components=20,min_cells=5,cap=5000):
    pool=tuple(sorted(pool));collections=np.asarray(collections,dtype=str)
    if len(pool)!=len(set(pool)) or len(collections)!=counts.shape[0]:
        raise ValueError("Invalid grouping pool or collection axis")
    values=_background_log(counts,gene_ids,pool)
    eligible=np.ones(len(pool),dtype=bool)
    for sample in np.unique(collections):
        eligible &= np.asarray((values[collections==sample]>0).sum(axis=0)).ravel()>=min_cells
    balanced=balanced_fit_rows(collections,barcodes,cap)
    sub=values[balanced]
    mean=np.asarray(sub.mean(axis=0)).ravel()
    variance=np.asarray(sub.power(2).mean(axis=0)).ravel()-mean**2
    candidates=np.flatnonzero(eligible & (variance>0))
    selected=candidates[np.lexsort((np.asarray(pool)[candidates],-variance[candidates]))[:max_genes]]
    if len(selected)<components or len(balanced)<=components:
        raise ValueError("Insufficient eligible grouping features/cells")
    model=PCA(n_components=components,svd_solver="randomized",random_state=0,iterated_power=4)
    model.fit(sub[:,selected].toarray())
    return GroupTransform(pool,tuple(np.asarray(pool)[selected]),model,balanced)


def training_descriptors(features,collections):
    x=finite_matrix(features)
    if (x<0).any():
        raise ValueError("Family descriptors require nonnegative log expression")
    w=collection_weights(collections);mean=w@x
    return w@(x>0),mean,w@((x-mean)**2)


def match_nonTF_sets(x,candidates,collections,target_sizes,candidate_sizes,candidate_ids,*,sets=5):
    x=finite_matrix(x);candidates=finite_matrix(candidates)
    if len(x)!=len(candidates) or x.shape[1]!=len(target_sizes) or candidates.shape[1]!=len(candidate_sizes) or len(candidate_ids)!=len(set(candidate_ids)):
        raise ValueError("Invalid matching axes")
    target_sizes=np.asarray(target_sizes);candidate_sizes=np.asarray(candidate_sizes)
    if (target_sizes<1).any() or (candidate_sizes<1).any():
        raise ValueError("Family sizes must be positive")
    tx=training_descriptors(x,collections);cc=training_descriptors(candidates,collections)
    dd=np.abs(tx[0][:,None]-cc[0][None,:])
    means=np.log2((cc[1][None,:]+1e-12)/(tx[1][:,None]+1e-12))
    variances=np.log2((cc[2][None,:]+1e-12)/(tx[2][:,None]+1e-12))
    valid=(target_sizes[:,None]==candidate_sizes[None,:]) & (dd<=.15) & (np.abs(means)<=2) & (np.abs(variances)<=2)
    valid &= (tx[2][:,None]<=1e-12)==(cc[2][None,:]<=1e-12)
    cost=(dd/.15)**2+(means/2)**2+(variances/2)**2
    cost[~valid]=np.inf
    available=sorted(range(len(candidate_ids)),key=lambda i:candidate_ids[i]);result=[]
    for _ in range(sets):
        if len(available)<x.shape[1]:
            raise ValueError("Insufficient disjoint matched controls")
        try:
            rows,cols=linear_sum_assignment(cost[:,available])
        except ValueError as exc:
            raise ValueError("Incomplete matched controls under fixed calipers") from exc
        if len(rows)!=x.shape[1] or not np.isfinite(cost[rows,np.asarray(available)[cols]]).all():
            raise ValueError("Incomplete matched controls under fixed calipers")
        assignment=np.empty(x.shape[1],dtype=int);assignment[rows]=np.asarray(available)[cols]
        result.append(assignment);used=set(assignment);available=[i for i in available if i not in used]
    return result


def fit_state_model(training_embedding,balanced_rows,k=8):
    points=finite_matrix(training_embedding)
    return KMeans(n_clusters=k,random_state=0,n_init=10).fit(points[balanced_rows])


def unit_state_features(embedding,states,background_mass,background_detected_fraction,assignment,size,*,k=8):
    embedding=finite_matrix(embedding);states=np.asarray(states);assignment=np.asarray(assignment)
    if len(states)!=len(embedding) or np.any(states<0) or np.any(states>=k):
        raise ValueError("Invalid cell state assignments")
    ids,counts=np.unique(assignment[assignment>=0],return_counts=True)
    if not np.array_equal(ids,np.arange(len(ids))) or np.any(counts!=size):
        raise ValueError("Full units required for state features")
    log_depth=np.log1p(background_mass);rows=[];dominant=[]
    for unit in ids:
        keep=assignment==unit;proportions=np.bincount(states[keep],minlength=k)/size
        rows.append(np.r_[embedding[keep].mean(axis=0),proportions,log_depth[keep].mean(),log_depth[keep].std(),np.mean(np.asarray(background_detected_fraction)[keep])])
        dominant.append(int(np.argmax(proportions)))
    return np.asarray(rows),np.asarray(dominant)


def common_support(train_states,train_collections,test_states,min_units=5):
    train_states=np.asarray(train_states);labels=np.asarray(train_collections);test_states=np.asarray(test_states)
    if len(train_states)!=len(labels):
        raise ValueError("Misaligned state/collection axes")
    return np.asarray([all(np.count_nonzero((labels==sample)&(train_states==state))>=min_units for sample in np.unique(labels)) for state in test_states])


def shuffle_within_state(x,collections,states,seed):
    x=finite_matrix(x);collections=np.asarray(collections,dtype=str);states=np.asarray(states)
    if len(x)!=len(collections) or len(x)!=len(states):
        raise ValueError("Misaligned permutation strata")
    rng=np.random.default_rng(seed);order=np.arange(len(x))
    for sample in sorted(set(collections)):
        for state in sorted(set(states[collections==sample])):
            rows=np.flatnonzero((collections==sample)&(states==state))
            order[rows]=rng.permutation(rows)
    return x[order],order,float(np.mean(order!=np.arange(len(order))))
