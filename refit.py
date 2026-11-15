"""Fit the original source ridge models from the supplied grouped expression matrices."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "refit"))
from cspr.path_a_assay import hierarchical_weights, weighted_ridge_design
from cspr.path_a_v05 import DF_GRID, input_calibrated_predict, scores
from cspr.path_a_v05_wp1 import MODELS, features, select_source

STUDIES = ("StAnge", "Gao", "Iida", "Russ")
SOURCES = ("StAnge", "Iida", "Russ")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def run(models, output):
    folder = ROOT / "refit/groups"
    groups = {p.stem: load(p) for p in folder.glob("*.npz") if not p.stem.endswith("-matching")}
    reference_path = ROOT / "manifests/path-a/v059-referee-scores-2026-11-15/transfer.tsv"
    with reference_path.open() as stream:
        reference = {(r["source"], r["target"], r["arm"]): r for r in csv.DictReader(stream, delimiter="\t")}
    with (ROOT / "manifests/path-a/v05-wp1-selected-matrix-2026-11-06.tsv").open() as stream:
        selected = {(r["source"], r["target"], r["model"], r["arm"]): r for r in csv.DictReader(stream, delimiter="\t")}
    panel_names = {name for name in models if name.startswith("N1_active_")}
    panel_reference = {}
    if panel_names:
        with (ROOT / "manifests/path-a/v051-wp1-intervals-2026-11-07.tsv").open() as stream:
            for r in csv.DictReader(stream, delimiter="\t"):
                if r["model"] in panel_names:
                    panel_reference[r["source"], r["target"], r["model"], r["arm"], r["metric"]] = r["point"]
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ("source", "target", "model", "arm", "rho", "amplitude", "R2_centered")
    records, max_error = [], 0.
    for source in SOURCES:
        for name in models:
            predictions = {}
            for component in ("full", "half0", "half1"):
                label = source + "-" + component
                data, match = groups[label], load(folder / (label + "-matching.npz"))
                panels = [match[f"panels{i}"] for i in range(6)]
                choice = "TF" if name == "matched_background" else name
                family = "Nfull_df" if name == "matched_background" else name
                idx = select_source(data, choice, panels, match["folds"])["selected_index"]
                design = weighted_ridge_design(features(data, family, panels), data["weights"])
                rank = weighted_ridge_design(data["TF"], data["weights"]).rank if family == "Nfull_df" else design.rank
                model = design.fit(data["y"], rank * DF_GRID[idx])
                destinations = ([(t, h) for t in STUDIES if t != source for h in (0, 1)]
                                if component == "full" else [(source, 1 - int(component[-1]))])
                for target, h in destinations:
                    tx = features(groups[f"{target}-half{h}"], family, panels)
                    other = groups[f"{target}-half{1-h}"]
                    predictions[target, h, "raw"] = model.predict(tx)
                    predictions[target, h, "X_calibrated"] = input_calibrated_predict(
                        model, features(other, family, panels), tx, other["weights"])
            for target in STUDIES:
                halves = [groups[f"{target}-half{h}"] for h in (0, 1)]
                y = np.concatenate([h["y"] for h in halves])
                weights = hierarchical_weights(np.concatenate([h["study"] for h in halves]),
                                               np.concatenate([h["sample"] for h in halves]))
                for arm in ("raw", "X_calibrated"):
                    prediction = np.concatenate([predictions[target, h, arm] for h in (0, 1)])
                    result = scores(y, prediction, weights)
                    row = dict(source=source, target=target, model=name, arm=arm)
                    for key in keys[-3:]:
                        row[key] = result[key]
                        if name in ("TF", "matched_background"):
                            archive_key = ("background_" if name == "matched_background" else "") + ("R2" if key == "R2_centered" else key)
                            expected = reference[source, target, arm][archive_key]
                        elif name in panel_names:
                            expected = panel_reference[source, target, name, arm, key]
                        else:
                            expected = selected[source, target, name, arm][key]
                        error = abs(float(expected) - float(row[key]))
                        max_error = max(max_error, error)
                        if not np.isfinite(error) or error > 1e-8:
                            raise ValueError((source, target, name, arm, key, expected, row[key]))
                    records.append(row)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} freshly fitted source-target-arm scores written to {output}. Maximum archived-score difference {max_error:.3g}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=("matched_background", *MODELS), default=["TF", "matched_background"],
                   help="Nfull_df tunes background separately on TF-rank grid. N1_active_000 through N1_active_127 refit the individual panels.")
    p.add_argument("--output", type=Path, default=ROOT / "reconstructed/refitted.tsv")
    p.add_argument("--all-panels", action="store_true",
                   help="Additionally refit all 128 count-matched panels and check each score against its archived estimate.")
    args = p.parse_args()
    if args.all_panels:
        args.models = list(dict.fromkeys([*args.models, *(m for m in MODELS if m.startswith("N1_active_"))]))
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.models, args.output)
