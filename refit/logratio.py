"""Refit the supplementary S1 log-ratio pilot from the released gated nuclei.

From the extracted archive root, run 'python refit/logratio.py' for the observed
scores, their original-measurement baselines and paired conditional intervals.
Add --nulls to refit all 128 draw pairs in the historical log-ratio pilot.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

DATE = "2026-11-07"
ARMS = ("raw", "X_calibrated")
METRICS = ("rho", "amplitude", "R2_centered", "R2_raw")
NULLS = ("kept_reference", "shuffled_reference")
TOL = 1e-12


def read_table(path, fields):
    with path.open(newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    by_key = {tuple(r[k] for k in fields):r for r in data}
    if len(by_key) != len(data):
        raise ValueError(f"Duplicate rows in {path}")
    return by_key


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_table(path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    diff = float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))))
    if not np.isfinite(diff) or diff > TOL:
        raise ValueError(f"Difference {diff} exceeds {TOL}")
    return diff


def metrics(y, p, w):
    w = w / w.sum()
    yc, pc = y-w@y, p-w@p
    s = np.sum(w[:,None]*yc**2)
    v = np.sum(w[:,None]*pc**2)
    c = np.sum(w[:,None]*pc*yc)
    return np.array([c/np.sqrt(s*v), np.sqrt(v/s), (2*c-v)/s, 1-np.sum(w[:,None]*(y-p)**2)/s])


def bootstrap_weights(names, repetitions, seed):
    groups = np.unique(names)
    indices = {s:np.flatnonzero(names == s) for s in groups}
    result = np.zeros((repetitions,len(names)))
    rng = np.random.default_rng(seed)
    for row in result:
        for s in rng.choice(groups,len(groups),replace=True):
            ix = indices[s]
            np.add.at(row,rng.choice(ix,len(ix),replace=True),1/(len(ix)*len(groups)))
    return result/result.sum(axis=1,keepdims=True)


def bootstrap_metrics(y, p, w):
    yc, pc = y-y.mean(axis=0), p-p.mean(axis=0)
    ym, pm = w@yc, w@pc
    s = w@np.sum(yc*yc,axis=1)-np.sum(ym*ym,axis=1)
    v = w@np.sum(pc*pc,axis=1)-np.sum(pm*pm,axis=1)
    c = w@np.sum(pc*yc,axis=1)-np.sum(ym*pm,axis=1)
    return np.column_stack((c/np.sqrt(s*v),np.sqrt(v/s),(2*c-v)/s))


def permutation_maps(data, component, cfg):
    n = len(data["sample"])
    maps = np.empty((cfg["null_draws"],2,n),dtype=np.int32)
    for prep in np.unique(data["sample"]):
        ix = np.flatnonzero(data["sample"] == prep)
        ix = ix[np.argsort(data["identities"][ix])]
        for d in range(cfg["null_draws"]):
            for i, label in enumerate(("numerator","reference")):
                key = f'wp1-xonly-v051:{cfg["seed"]}:{d}:{component}:{prep}:{label}'.encode()
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8],"big")
                maps[d,i,ix] = np.random.default_rng(seed).permutation(ix)
    return maps


def measure(data, cfg, axes, layout, groups, pop):
    columns = {family:i for i,family in enumerate(layout["axes"]["ids"])}
    nx = len(axes["X"])
    if (len(columns) != len(layout["axes"]["ids"]) or axes["X"] != layout["axes"]["x"]
            or axes["Y"] != layout["axes"]["y"] or nx != cfg["X_families"]
            or len(axes["R"]) != cfg["R_families"] or set(axes["R"]) & set(axes["X"]+axes["Y"])):
        raise ValueError("Unregistered expression axes")
    ix = [columns[family] for family in axes["X"]+axes["R"]]
    by_unit = defaultdict(list)
    with gzip.open(groups/"membership.tsv.gz","rt") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            if row["component"] in data and row["unit"]:
                by_unit[row["component"],row["sample"],row["unit"]].append(row["barcode"])
    total_eligible = 0
    max_orig = 0.
    for g in data.values():
        g["L"] = np.full(g["TF"].shape,np.nan)
        g["G"] = np.full(len(g["TF"]),np.nan)
    for library in layout["samples"]:
        sample = library["sample"]
        cells, matrix = (pop/(sample+"-"+s+".npz") for s in ("cells","families"))
        for path in (cells,matrix):
            if sha(path) != layout["files_sha256"][path.name]:
                raise ValueError(f"Changed cell-level input: {path.name}")
        with np.load(cells,allow_pickle=False) as f:
            denom = f["denominator"]
            index = {b:i for i,b in enumerate(f["barcodes"])}
        if len(denom) != len(index) or np.any(denom <= 0):
            raise ValueError("Duplicate barcodes or missing depth")
        total_eligible += len(index)
        linear = sparse.load_npz(matrix).tocsr()[:,ix].tocsr()
        raw = linear.copy()
        scale = np.repeat(10000./denom,np.diff(raw.indptr))
        integer = np.rint(raw.data/scale)
        if (not np.array_equal(integer*scale, raw.data) or np.any(integer < 0)
                or np.max(abs(raw.data/scale-integer),initial=0.) > .25):
            raise ValueError("Could not invert the original normalized counts")
        raw.data = integer.astype(np.int64)
        if not np.array_equal(np.asarray(raw[:,nx:].sum(axis=1)).ravel(),denom):
            raise ValueError("The registered reference counts do not sum to the original depth")
        for component,g in data.items():
            ids = np.flatnonzero(g["sample"] == sample)
            if not ids.size:
                continue
            members = [by_unit[component,sample,u] for u in g["identities"][ids]]
            if any(len(b) != cfg["unit_size"] for b in members):
                raise ValueError(f"Incomplete group in {component}/{sample}")
            cell_ix = np.array([[index[b] for b in unit] for unit in members])
            if len(set(cell_ix.ravel())) != cell_ix.size:
                raise ValueError(f"Duplicate cell in {component}/{sample}")
            # Group sum of raw nuclei on each family. Integral sparse summation
            # makes the original per-group count means independent of member order.
            n, size = cell_ix.shape
            group_by_cell = sparse.csr_matrix((np.ones(cell_ix.size,dtype=np.int64),
                (np.repeat(np.arange(n),size),cell_ix.ravel())),shape=(n,len(index)))
            means = np.ascontiguousarray((group_by_cell@raw).toarray()/size)
            shift = cfg["raw_pseudocount"]
            g["L"][ids] = np.log(means[:,:nx]+shift)
            g["G"][ids] = np.log(means[:,nx:]+shift).mean(axis=1)
            wp1 = np.log1p((group_by_cell@linear[:,:nx]).toarray()/size)
            max_orig = max(max_orig,close(wp1,g["TF"][ids]))
    if any(not np.isfinite(g[k]).all() for g in data.values() for k in ("L","G")):
        raise ValueError("Missing reconstructed component rows")
    return dict(eligible_nuclei=total_eligible,groups_by_component={c:len(g["G"]) for c,g in data.items()},
                max_original_TF_measurement_error=max_orig)


def run(args):
    root = args.root.resolve()
    refit, man = root/"refit", root/"manifests/path-a"
    code = args.code or refit
    sys.path.insert(0,str(code.resolve()))
    from cspr.path_a_assay import weighted_ridge_design, hierarchical_weights
    from cspr.path_a_v05 import input_calibrated_predict
    groups, pop = args.groups or refit/"groups", args.populations or refit/"populations"
    cfg_path = man/f"v051-xonly-registration-{DATE}.json"
    axes_path = man/f"v051-composition-axes-{DATE}.json"
    cfg = json.loads(cfg_path.read_text())
    if sha(axes_path) != cfg["authority_sha256"][axes_path.name]:
        raise ValueError("Original log-ratio axis manifest changed")
    axes = json.loads(axes_path.read_text())
    layout = json.loads((args.layout or pop/"layout.json").read_text())
    observed = read_table(man/f"v051-xonly-observed-{DATE}.tsv",("source","target","arm","model"))
    ints = read_table(man/f"v051-xonly-intervals-{DATE}.tsv",("source","target","arm","model","metric"))
    nulls = (read_table(man/f"v051-xonly-null-draws-{DATE}.tsv",("source","target","arm","kind","draw"))
             if args.nulls else {})
    current = [s+"-full" for s in cfg["targets"]]
    current.extend(f"{s}-half{h}" for s in cfg["preparations"] for h in (0,1))
    data = {}
    for c in current:
        with np.load(groups/(c+".npz"),allow_pickle=False) as f:
            data[c] = {k:f[k] for k in ("TF","weights","study","sample","y","identities")}
    receipt = measure(data,cfg,axes,layout,groups,pop)
    if args.nulls:
        for c,g in data.items():
            g["maps"] = permutation_maps(g,c,cfg)
    max_obs = max_interval = max_null = 0.
    record_rows, interval_rows, null_rows = [], [], []
    fit_count = null_count = interval_count = 0
    for source,targets in cfg["targets"].items():
        train = data[source+"-full"]
        df = cfg["absolute_df"][source]
        print(f"Refitting S1 models at {df:.8f} degrees of freedom for {source}",flush=True)
        fit = lambda x: weighted_ridge_design(x,train["weights"]).fit(train["y"],df)
        predict = lambda model,halves,scales,arm: np.concatenate([
            model.predict(halves[h]) if arm=="raw" else
            input_calibrated_predict(model,halves[1-h],halves[h],scales[1-h])
            for h in (0,1)])
        baseline_model = fit(train["TF"])
        transformed_model = fit(train["L"]-train["G"][:,None])
        fit_count += 1
        contexts = {}
        for target in targets:
            g = [data[f"{target}-half{h}"] for h in (0,1)]
            y = np.concatenate([b["y"] for b in g])
            names = np.concatenate([b["sample"] for b in g])
            w = hierarchical_weights(np.concatenate([b["study"] for b in g]),names)
            hw = [b["weights"] for b in g]
            bws = bootstrap_weights(names,cfg["bootstrap_draws"],cfg["bootstrap_seed"])
            contexts[target] = (g,y,w,hw)
            for arm in ARMS:
                base = predict(baseline_model,[b["TF"] for b in g],hw,arm)
                new = predict(transformed_model,[b["L"]-b["G"][:,None] for b in g],hw,arm)
                be,ne = metrics(y,base,w),metrics(y,new,w)
                bd,nd = bootstrap_metrics(y,base,bws),bootstrap_metrics(y,new,bws)
                for label,values in (("WP1",be),("X_only",ne),("X_only_minus_WP1",ne-be)):
                    key=(source,target,arm,label)
                    ref = observed[key]
                    max_obs = max(max_obs,close(values,[ref[k] for k in METRICS]))
                    record_rows.append(dict(source=source,target=target,arm=arm,model=label,
                                            **dict(zip(METRICS,values))))
                for label,draws in (("X_only",nd),("X_only_minus_WP1",nd-bd)):
                    bounds=np.quantile(draws,[.025,.975],axis=0)
                    for j,metric in enumerate(METRICS[:3]):
                        ref=ints[source,target,arm,label,metric]
                        max_interval=max(max_interval,close(bounds[:,j],[ref["ci025"],ref["ci975"]]))
                        interval_rows.append(dict(source=source,target=target,arm=arm,model=label,
                                                  metric=metric,ci025=bounds[0,j],ci975=bounds[1,j]))
                        interval_count += 1
        if args.nulls:
            for d in range(cfg["null_draws"]):
                for i,kind in enumerate(NULLS):
                    def x(b):
                        numerator, reference = b["maps"][d]
                        return b["L"][numerator]-(b["G"][reference] if i else b["G"])[:,None]
                    model = fit(x(train))
                    fit_count += 1
                    for t,(g,y,w,hw) in contexts.items():
                        for arm in ARMS:
                            result=metrics(y,predict(model,[x(b) for b in g],hw,arm),w)
                            ref=nulls[source,t,arm,kind,str(d)]
                            max_null=max(max_null,close(result,[ref[k] for k in METRICS]))
                            null_rows.append(dict(source=source,target=t,arm=arm,kind=kind,draw=d,
                                                  **dict(zip(METRICS,result))))
                            null_count += 1
    if len(record_rows)!=24 or interval_count!=48 or (args.nulls and (fit_count!=514 or null_count!=2048)):
        raise ValueError("Incomplete S1 replay")
    receipt.update(status="PASS_S1_LOGRATIO_FROM_GATED_NUCLEI",logratio_source_fits=fit_count,
                   original_measurement_baseline_fits=2,observed_score_rows=len(record_rows),
                   conditional_intervals=interval_count,max_observed_metric_error=max_obs,
                   max_conditional_bound_error=max_interval,null_draw_score_rows=null_count,
                   max_null_metric_error=max_null if args.nulls else None,
                   script_sha256=sha(Path(__file__)),registration_sha256=sha(cfg_path),axis_sha256=sha(axes_path))
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    write_table(out/"observed.tsv",record_rows)
    write_table(out/"intervals.tsv",interval_rows)
    if null_rows:
        write_table(out/"null-draws.tsv",null_rows)
    result = json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)
    (out/"receipt.json").write_text(result+"\n")
    print(result)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path.cwd(),help="Extracted reader archive root")
    p.add_argument("--output",type=Path,default=Path("reconstructed/logratio"))
    p.add_argument("--nulls",action="store_true",help="Rebuild and verify both historical shuffled-reference controls")
    p.add_argument("--code",type=Path,help="Override the directory containing the deposited cspr kernels")
    p.add_argument("--groups",type=Path,help="Override refit/groups")
    p.add_argument("--populations",type=Path,help="Override refit/populations")
    p.add_argument("--layout",type=Path,help="Override refit/populations/layout.json")
    with threadpool_limits(limits=1):
        run(p.parse_args())
