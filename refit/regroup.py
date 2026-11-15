#!/usr/bin/env python3
"""Rebuild the eleven 20-nucleus source and half matrices from the deposited populations."""

import argparse
from contextlib import ExitStack
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from cspr.path_a_assay import FamilyCounts
from cspr.path_a_course_correction import cell_halves
from cspr.path_a_portability import Axes, Sample, panels, responses
from cspr.path_a_v05 import target_fixed_units


HERE = Path(__file__).resolve().parent
STUDIES = ("StAnge", "Gao", "Iida", "Russ")
SOURCES = ("StAnge", "Iida", "Russ")
LEDGER_HEADER = ("component", "study", "sample", "barcode", "unit")


def load_populations(population_dir, studies=None):
    """Load hash-checked, gated nuclear family measurements on their original axes."""
    folder = Path(population_dir)
    layout = json.loads((folder / "layout.json").read_text())
    if (layout["schema_version"], layout["group_size"]) != (1, 20):
        raise ValueError("Expected the 20-nucleus release layout")
    spec = layout["axes"]
    axes = Axes(*(tuple(spec[key]) for key in ("ids", "x", "y", "u", "s")),
                {k: tuple(v) for k, v in spec["size_pairs"].items()},
                {k: tuple(v) for k, v in spec["modules"].items()})
    selected = set(STUDIES if studies is None else studies)
    if not selected or not selected <= set(STUDIES):
        raise ValueError(f"Choose one or more studies from {STUDIES}")
    samples = {name: [] for name in STUDIES if name in selected}
    for library in layout["samples"]:
        if library["name"] not in selected:
            continue
        prep = library["sample"]
        for suffix in ("families", "cells"):
            name = f"{prep}-{suffix}.npz"
            with (folder / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != layout["files_sha256"][name]:
                raise ValueError(f"Archived population input changed: {name}")
        linear = sparse.load_npz(folder / f"{prep}-families.npz")
        with np.load(folder / f"{prep}-cells.npz", allow_pickle=False) as cells:
            counts = FamilyCounts(linear, cells["denominator"], cells["detected"])
            samples[library["name"]].append(Sample(library["species"], library["study"], prep,
                                                   tuple(cells["barcodes"]), counts))
    return axes, samples


def components(samples):
    for name in STUDIES:
        halves, _ = cell_halves(samples[name])
        for fold, half in enumerate(halves):
            yield f"{name}-half{fold}", half
    for name in SOURCES:
        yield f"{name}-full", samples[name]


def check_arrays(path, generated):
    maximum = 0.0
    with np.load(path, allow_pickle=False) as frozen:
        if set(frozen.files) != set(generated):
            raise ValueError(f"Unexpected grouped fields in {path.name}")
        for key, current in generated.items():
            saved = frozen[key]
            if saved.shape != current.shape:
                raise ValueError(f"Different shape for {path.name}/{key}")
            if saved.dtype.kind in "fc":
                error = float(np.max(np.abs(saved - current)))
                if not np.isfinite(error) or error > 1e-12:
                    raise ValueError(f"Numerical mismatch for {path.name}/{key}: {error}")
                maximum = max(error, maximum)
            elif not np.array_equal(saved, current):
                raise ValueError(f"Different identifiers for {path.name}/{key}")
    return maximum


def rebuild(population_dir, groups_dir, output=None):
    axes, samples = load_populations(population_dir)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    count = members = 0
    maximum = 0.0
    with ExitStack() as files:
        original = files.enter_context(gzip.open(groups_dir / "membership.tsv.gz", "rt", newline=""))
        ledger = csv.reader(original, delimiter="\t")
        regenerated = None
        if output is not None:
            out = files.enter_context(gzip.open(output / "membership.tsv.gz", "wt", newline=""))
            regenerated = csv.writer(out, delimiter="\t", lineterminator="\n")
            regenerated.writerow(LEDGER_HEADER)
        if tuple(next(ledger)) != LEDGER_HEADER:
            raise ValueError("Unexpected membership header")
        for component, population in components(samples):
            units, assignments = target_fixed_units(population, axes)
            inputs = panels(units, axes)
            data = dict(TF=inputs["TF"], Nfull=inputs["Nfull"], y=responses(units, axes),
                        weights=units.weights, study=units.study, sample=units.sample,
                        identities=units.identities)
            maximum = max(maximum, check_arrays(groups_dir / (component + ".npz"), data))
            for row in assignments:
                if next(ledger, None) != [component, *row]:
                    raise ValueError(f"Original membership differs at {component}/{row[1]}/{row[2]}")
            if regenerated is not None:
                regenerated.writerows((component, *row) for row in assignments)
            members += len(assignments)
            count += 1
            if output is not None:
                np.savez_compressed(output / (component + ".npz"), **data)
            print(f"{component}: {len(units.identities)} groups, {len(assignments)} nuclear assignments",
                  flush=True)
        if next(ledger, None) is not None:
            raise ValueError("Unexpected trailing membership records")
    print(f"Verified {count} matrices and {members} membership records. "
          f"Maximum absolute numerical difference {maximum:.3g}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=Path, default=HERE / "populations")
    parser.add_argument("--groups", type=Path, default=HERE / "groups")
    parser.add_argument("--output", type=Path,
                        help="Write verified matrices and regenerated memberships to a new directory")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        rebuild(args.populations, args.groups, args.output)


if __name__ == "__main__":
    main()
