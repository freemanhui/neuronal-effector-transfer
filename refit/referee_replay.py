"""Independently verify the saved 20-nucleus tables and replay the TF-count shuffles.

From the root of the extracted evidence ZIP, run:
    python refit/referee_replay.py
    python refit/referee_replay.py --shuffle

The optional numerical replay uses the deposited source-fitting environment in
refit/cspr. It refits TF, matched-background and depth models on the nine original
source components. Models and input scales are fixed during target permutations.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SOURCES = ("StAnge", "Iida", "Russ")
TARGETS = ("StAnge", "Gao", "Iida", "Russ")
ARMS = ("raw", "X_calibrated")
KINDS = ("recipient_depth", "donor_depth")
METRICS = ("rho", "amplitude", "R2")
MATRIX = "manifests/path-a/v059-referee-scores-2026-11-15"
B = 128
SEED = 20261115


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def near(observed, expected, tolerance=1e-11):
    error = abs(float(observed) - float(expected))
    if not math.isfinite(error) or error > tolerance:
        raise ValueError(f"Expected {expected}, got {observed}")
    return error


def keyed(table, fields):
    result = {tuple(row[k] for k in fields): row for row in table}
    if len(result) != len(table):
        raise ValueError(f"Duplicate key {fields}")
    return result


def simple_check(root):
    folder = root / MATRIX
    transfer = keyed(rows(folder / "transfer.tsv"), ("source", "target", "arm"))
    preps = keyed(rows(folder / "preparations.tsv"), ("source", "target", "preparation"))
    summary = keyed(rows(folder / "shuffle-summary.tsv"), ("source", "target", "arm", "shuffle"))
    grouped_draws = defaultdict(list)
    draw_rows = rows(folder / "shuffle-draws.tsv")
    for row in draw_rows:
        grouped_draws[tuple(row[k] for k in ("source", "target", "arm", "shuffle"))].append(row)
    n = 0
    for table, prefixes in ((transfer, ("", "depth_", "background_")),
                            (preps, ("", "standardized_", "background_", "depth_"))):
        for row in table.values():
            for prefix in prefixes:
                rho, a, r2 = [float(row[prefix + m]) for m in METRICS]
                near(r2, 2*rho*a - a*a)
                n += 1
    for key, r in transfer.items():
        if r["source"] == r["target"]:
            for m in METRICS:
                near(r[m], transfer[key[:2] + ("raw",)][m])
        if sum(int(q["groups"]) for (s,t,_),q in preps.items() if (s,t) == key[:2]) != int(r["groups"]):
            raise ValueError("Preparation group counts do not partition the target")
    for key, r in summary.items():
        draws = grouped_draws[key]
        if {int(d["draw"]) for d in draws} != set(range(B)) or len(draws) != B:
            raise ValueError(f"Draw IDs incomplete: {key}")
        for m in METRICS:
            values = [float(d[m]) for d in draws]
            bounds = statistics.quantiles(values, n=40, method="inclusive")
            near(bounds[0], r[m+"_q025"])
            near(statistics.median(values), r[m+"_median"])
            near(bounds[-1], r[m+"_q975"])
            near(r[m+"_observed"], transfer[key[:3]][m])
            near(sum(v >= float(r[m+"_observed"]) for v in values), r[m+"_draws_ge_observed"])
    if (len(transfer), len(preps), len(summary), len(draw_rows), len(grouped_draws)) != (24,33,24,3072,24):
        raise ValueError("Incomplete fixed-score release")
    return {"transfer_rows":24, "preparation_rows":33, "score_identities":n,
            "shuffle_summaries_recalculated":24, "shuffle_draws":3072}


def numerical_replay(root, code, groups, population, member_file):
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(code))
    from cspr.path_a_assay import SourceScale, hierarchical_weights, weighted_ridge_design
    from cspr.path_a_v05 import DF_GRID
    from cspr.path_a_v05_wp1 import select_source

    def load(path, keys):
        with np.load(path, allow_pickle=False) as f:
            return {k:f[k] for k in keys}

    def metrics(y, p, weights):
        # Direct weighted Euclidean norms and inner product, independent of the producer.
        w = weights / np.sum(weights)
        y = (y - np.average(y, axis=0, weights=w))*np.sqrt(w[:,None])
        p = (p - np.average(p, axis=0, weights=w))*np.sqrt(w[:,None])
        normy, normp = np.linalg.norm(y), np.linalg.norm(p)
        return [np.vdot(y,p)/(normy*normp), normp/normy,
                1. - (np.linalg.norm(y-p)/normy)**2]

    def score_pair(source, target):
        return ((source in ("StAnge","Gao")) == (target in ("StAnge","Gao")))

    def target_draw_weights(names):
        rng = np.random.default_rng(20261106)
        labels = np.unique(names)
        groups = {s:np.flatnonzero(names == s) for s in labels}
        weights = np.zeros((1000,len(names)))
        for w in weights:
            for s in rng.choice(labels,len(labels),replace=True):
                ix = groups[s]
                positions = rng.choice(ix,len(ix),replace=True)
                np.add.at(w,positions,1/(len(labels)*len(ix)))
        return weights/weights.sum(axis=1,keepdims=True)

    def conditional_bounds(y, p, w):
        # Using fixed references makes the bootstrap second-moment differences stable.
        y = y - np.mean(y,axis=0)
        p = p - np.mean(p,axis=0)
        ybar, pbar = w@y, w@p
        sy = w@np.sum(y*y,axis=1) - np.sum(ybar*ybar,axis=1)
        sp = w@np.sum(p*p,axis=1) - np.sum(pbar*pbar,axis=1)
        cov = w@np.sum(y*p,axis=1) - np.sum(ybar*pbar,axis=1)
        if np.any(sy <= 0) or np.any(sp <= 0):
            raise ValueError("Undefined target-conditional metric")
        draws = np.stack([cov/np.sqrt(sy*sp), np.sqrt(sp/sy), (2*cov-sp)/sy],axis=-1)
        return np.quantile(draws,[.025,.975],axis=0)

    use = ("TF", "Nfull", "y", "weights", "study", "sample", "identities")
    data = {}
    source_fits = {}
    backgrounds = {}
    with threadpool_limits(limits=1):
        for target in TARGETS:
            for partition in (("full", "half0", "half1") if target in SOURCES else ("half0", "half1")):
                c = f"{target}-{partition}"
                g = data[c] = load(groups / (c+".npz"), use)
                if target in SOURCES:
                    folds = load(groups / (c+"-matching.npz"), ("folds",))["folds"]
                    cv = select_source(g,"TF",[],folds)
                    fit = weighted_ridge_design(g["TF"],g["weights"])
                    df = fit.rank*DF_GRID[cv["selected_index"]]
                    source_fits[c] = fit.fit(g["y"],df)
                    backgrounds[c] = weighted_ridge_design(g["Nfull"],g["weights"]).fit(g["y"],df)
        members = defaultdict(list)
        with gzip.open(member_file, "rt") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["unit"]:
                    members[r["component"],r["sample"],r["unit"]].append(r["barcode"])

        population_arrays = {}
        for sample in sorted({s for g in data.values() for s in g["sample"]}):
            f = load(population / (sample+"-cells.npz"), ("barcodes","denominator","detected"))
            linear = sparse.load_npz(population / (sample+"-families.npz")).tocsr()[:,:136].toarray()
            depth = f["denominator"]
            if len(depth) != len(linear) or np.any(depth <= 0):
                raise ValueError("Invalid population depth")
            raw = np.rint(linear * (depth[:,None]/10000.))
            if not np.array_equal(raw*(10000./depth[:,None]), linear) or np.min(raw) < 0:
                raise ValueError("The normalized population does not invert to integer TF counts")
            population_arrays[sample] = ({b:i for i,b in enumerate(f["barcodes"])},raw,depth,f["detected"])

        # Depth control fitted independently by weighted least squares with an intercept.
        depth_x, depth_models = {}, {}
        for c,g in data.items():
            covariates = []
            for sample,unit in zip(g["sample"],g["identities"]):
                lookup,_,d,f = population_arrays[sample]
                ix = [lookup[b] for b in members[c,sample,unit]]
                if len(ix) != 20:
                    raise ValueError(f"Incomplete depth data for {c}/{unit}")
                covariates.append((1.,np.log(d[ix]).mean(),f[ix].mean()))
            z = depth_x[c] = np.asarray(covariates)
            if c in source_fits:
                sw = np.sqrt(g["weights"])[:,None]
                depth_models[c] = np.linalg.lstsq(sw*z,sw*g["y"],rcond=None)[0]

        tests = keyed(rows(root / MATRIX / "shuffle-draws.tsv"), ("source","target","arm","shuffle","draw"))
        maps = keyed(rows(root / MATRIX / "shuffle-maps.tsv"), ("target","draw"))
        transfers = keyed(rows(root / MATRIX / "transfer.tsv"), ("source","target","arm"))
        preparations = keyed(rows(root / MATRIX / "preparations.tsv"), ("source","target","preparation"))
        n_points = n_preparations = n_null = n_bounds = 0
        max_point = max_preparation = max_null = max_interval = 0.
        for target_id,target in enumerate(TARGETS):
            print(f"Refitted prediction and count-level permutation checks for {target}",flush=True)
            halves = [data[f"{target}-half{i}"] for i in (0,1)]
            y = np.concatenate([g["y"] for g in halves])
            names = np.concatenate([g["sample"] for g in halves])
            weights = hierarchical_weights(np.concatenate([g["study"] for g in halves]),names)
            bootstrap_weights = target_draw_weights(names)
            frame = []
            for h,g in enumerate(halves):
                c = f"{target}-half{h}"
                coord = []
                for sample, unit in zip(g["sample"],g["identities"]):
                    b = sorted(members[c,sample,unit])
                    if len(b) != 20:
                        raise ValueError(f"Expected 20 original member barcodes in {unit}")
                    index,counts,ds,_ = population_arrays[sample]
                    ids = [index[barcode] for barcode in b]
                    coord.append((counts[ids],ds[ids]))
                raw = np.stack([z[0] for z in coord])
                depth = np.stack([z[1] for z in coord])
                def transform(raw, depth):
                    return np.log1p(np.mean(raw*(10000./depth[:,:,None]),axis=1))
                np.testing.assert_allclose(transform(raw,depth),g["TF"],rtol=0,atol=1e-12)
                other = halves[1-h]
                scale = SourceScale.fit(other["TF"],other["weights"])
                bg_scale = SourceScale.fit(other["Nfull"],other["weights"])
                frame.append((raw,depth,scale,bg_scale))
            def predict(source,arm,matrix):
                ps = []
                for h,g in enumerate(halves):
                    c = f"{source}-half{1-h}" if source == target else source+"-full"
                    m = source_fits[c]
                    if arm == "raw":
                        p = m.predict(matrix[h])
                    else:
                        z = frame[h][2].transform(matrix[h])
                        z[:,m.constant] = 0
                        p = z @ m.coefficient + m.target_mean
                    ps.append(p)
                return np.concatenate(ps)
            for source in SOURCES:
                cs = [f"{source}-half{1-h}" if source == target else source+"-full" for h in (0,1)]
                depth_p = np.concatenate([depth_x[f"{target}-half{h}"]@depth_models[c]
                                          for h,c in enumerate(cs)])
                for arm in ARMS:
                    bg_halves = []
                    for h,(c,g) in enumerate(zip(cs,halves)):
                        fit = backgrounds[c]
                        if arm == "raw":
                            bp = fit.predict(g["Nfull"])
                        else:
                            z = frame[h][3].transform(g["Nfull"])
                            z[:,fit.constant] = 0
                            bp = z@fit.coefficient + fit.target_mean
                        bg_halves.append(bp)
                    procedures = [("",predict(source,arm,[g["TF"] for g in halves])),
                                  ("background_",np.concatenate(bg_halves))]
                    if arm == "raw":
                        procedures.append(("depth_",depth_p))
                    for prefix,p in procedures:
                        r = transfers[source,target,arm]
                        for value,m in zip(metrics(y,p,weights),METRICS):
                            max_point = max(max_point,near(value,r[prefix+m],1e-8))
                        n_points += 1
                        if prefix != "depth_":
                            bounds = conditional_bounds(y,p,bootstrap_weights)
                            for side,values in zip(("_lo","_hi"),bounds):
                                for value,m in zip(values,METRICS):
                                    max_interval = max(max_interval,near(value,r[prefix+m+side],1e-8))
                                    n_bounds += 1
                        if arm == "X_calibrated" and prefix:
                            continue
                        prep_prefix = prefix if arm == "raw" else "standardized_"
                        for sample in np.unique(names):
                            ix = names == sample
                            r = preparations[source,target,sample]
                            for value,m in zip(metrics(y[ix],p[ix],weights[ix]),METRICS):
                                max_preparation = max(max_preparation,near(value,r[prep_prefix+m],1e-8))
                            n_preparations += 1
            for d in range(B):
                mappings, kept, paired = [], [], []
                for h,g in enumerate(halves):
                    raw,depth,_,_ = frame[h]
                    mapping = np.arange(len(g["TF"]))
                    state = np.random.SeedSequence([SEED,target_id,h,d])
                    rng = np.random.Generator(np.random.PCG64(state))
                    for s in sorted(set(g["sample"])):
                        ix = np.nonzero(g["sample"] == s)[0]
                        mapping[ix] = rng.permutation(ix)
                    mappings.append(mapping.astype("<i8",copy=False).tobytes())
                    kept.append(transform(raw[mapping],depth))
                    paired.append(transform(raw[mapping],depth[mapping]))
                if hashlib.sha256(b"".join(mappings)).hexdigest() != maps[target,str(d)]["sha256"]:
                    raise ValueError("Permutation map hashes differ")
                for source in [s for s in SOURCES if score_pair(s,target)]:
                    for arm in ARMS:
                        for condition, x in zip(KINDS,(kept,paired)):
                            r = tests[source,target,arm,condition,str(d)]
                            for value,m in zip(metrics(y,predict(source,arm,x),weights),METRICS):
                                max_null = max(max_null,near(value,r[m],1e-8))
                            n_null += 1
    return dict(source_model_refits={"TF":len(source_fits),"matched_background":len(backgrounds),
                                    "depth_detection":len(depth_models)},
                refitted_transfer_arm_checks=n_points,refitted_conditional_interval_bounds=n_bounds,
                refitted_preparation_arm_checks=n_preparations, refitted_shuffle_draw_checks=n_null,
                shuffle_map_hashes=4*B,max_transfer_metric_error=max_point,
                max_preparation_metric_error=max_preparation,max_shuffle_metric_error=max_null,
                max_conditional_bound_error=max_interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Extracted ZIP root")
    p.add_argument("--shuffle", action="store_true", help="Reconstruct original TF counts and refit nine source models")
    p.add_argument("--receipt",type=Path,help="Optional verification JSON")
    p.add_argument("--code", type=Path,help="Override the directory containing cspr")
    p.add_argument("--groups", type=Path,help="Override refit/groups")
    p.add_argument("--populations", type=Path,help="Override refit/populations")
    p.add_argument("--members", type=Path,help="Override refit/groups/membership.tsv.gz")
    args = p.parse_args()
    root = args.root.resolve()
    result = {"saved_tables": simple_check(root)}
    if args.shuffle:
        g = args.groups or root/"refit/groups"
        result["independent_refit"] = numerical_replay(root,args.code or root/"refit",g,
            args.populations or root/"refit/populations",args.members or g/"membership.tsv.gz")
    result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    msg = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    print(msg)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(msg+"\n")


if __name__ == "__main__":
    main()
