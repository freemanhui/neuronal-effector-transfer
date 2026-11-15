#!/usr/bin/env python3
"""Check the resolution tables from aggregate inputs using the Python standard library."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_table(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def keyed(rows, fields):
    index = {tuple(row[field] for field in fields): row for row in rows}
    assert len(rows) == len(index), 'duplicate row key'
    return index


def close(first, second):
    assert math.isclose(float(first), float(second), abs_tol=1e-12, rel_tol=1e-12), (first, second)


def verify(directory):
    receipt = json.loads((directory/'verification.json').read_text())
    for name, expected in receipt['output_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name

    config = ('variant', 'size', 'anchor')
    source_target = (*config, 'source', 'target')
    scores = keyed(read_table(directory/'scores.tsv'), source_target)
    moments = keyed(read_table(directory/'centered-moments.tsv'), source_target)
    fits = keyed(read_table(directory/'source-fits.tsv'), (*config, 'source', 'component'))
    assert scores.keys() == moments.keys() and len(scores) == 54
    target_variances = {}
    missing = []
    for key, row in scores.items():
        variant, size, order, source, target = key
        aggregate = moments[key]
        observed = float(aggregate['observed_ss'])
        target_key = (variant, size, order, target)
        close(observed, target_variances.setdefault(target_key, observed))
        components = ('half0', 'half1') if source == target else ('full',)
        available = all(int(fits[(variant,size,order,source,part)]['groups']) >= 10 for part in components)
        if not available:
            assert row['status'] == 'source_half_below_10_groups'
            assert all(row[k] == 'NA' for k in ('rho', 'amplitude', 'R2_centered'))
            assert all(aggregate[k] == 'NA' for k in ('predicted_ss', 'cross_product', 'error_ss'))
            missing.append(key)
            continue
        assert row['status'] == 'estimated'
        predicted, cross, error = (float(aggregate[k]) for k in ('predicted_ss', 'cross_product', 'error_ss'))
        rho = cross / math.sqrt(observed*predicted)
        amplitude = math.sqrt(predicted/observed)
        close(row['rho'], rho)
        close(row['amplitude'], amplitude)
        close(row['R2_centered'], 1-error/observed)
        close(row['R2_centered'], 2*rho*amplitude-amplitude**2)
    assert set(missing) == {('common_support','200','lexical',study,study) for study in ('Iida','Russ')}

    support = read_table(directory/'common-support.tsv')
    total, retained = Counter(), Counter()
    for row in support:
        eligible, retained_n = map(int, (row['eligible'], row['common_cells']))
        assert retained_n == 200*(eligible//200)
        assert eligible-retained_n == int(row['removed'])
        total[row['study']] += eligible
        retained[row['study']] += retained_n
    assert sum(total.values()) == 21935 and sum(retained.values()) == 20000

    half_key = (*config, 'target')
    agreements = keyed(read_table(directory/'half-agreement.tsv'), half_key)
    splits = defaultdict(dict)
    for row in read_table(directory/'half-mean-squares.tsv'):
        key = tuple(row[x] for x in half_key)
        between, within = float(row['between_ms']), float(row['within_ms'])
        index = int(row['split'])
        assert index not in splits[key]
        splits[key][index] = (between-within)/(between+within)
    assert agreements.keys() == splits.keys() and len(splits) == 18
    for key, row in agreements.items():
        assert set(splits[key]) == set(range(32))
        close(row['icc_split0'], splits[key][0])
        close(row['icc_min_32'], min(splits[key].values()))
        close(row['icc_max_32'], max(splits[key].values()))
        variant, size, order, target = key
        if variant == 'common_support':
            assert retained[target] == int(size)*int(row['groups'])
        assert all(int(scores[(variant,size,order,source,target)]['target_groups']) == int(row['groups'])
                   for source in total)
    return dict(score_rows=len(scores), estimates=len(scores)-len(missing), missing=len(missing),
                target_iccs=len(splits), split_iccs=sum(map(len,splits.values())),
                eligible=sum(total.values()), retained=sum(retained.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing sensitivity TSVs and verification.json')
    parser.add_argument('--compare-to', type=Path, help='Also require byte-identical TSVs to this issued edition')
    options = parser.parse_args()
    result = verify(options.directory)
    if options.compare_to:
        verify(options.compare_to)
        reference = json.loads((options.compare_to/'verification.json').read_text())['output_sha256']
        reproduced = json.loads((options.directory/'verification.json').read_text())['output_sha256']
        assert reference == reproduced, sorted(k for k in set(reference)|set(reproduced)
                                               if reference.get(k) != reproduced.get(k))
        result['byte_identical_tables'] = len(reference)
    print(json.dumps(result, sort_keys=True))
