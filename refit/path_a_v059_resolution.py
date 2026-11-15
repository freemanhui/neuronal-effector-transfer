#!/usr/bin/env python3
"""Replay the size-20 WP1 TF fits and run the prespecified reviewer resolution diagnostic.

Use the issued Python/OpenBLAS environment and one BLAS thread. For the
standalone package, pass --reader-root with the extracted archive directory.
"""
import argparse
import csv
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader-root', type=Path, help='Root of the extracted v0.5.9 refit package')
    parser.add_argument('--output', type=Path, help='Destination for the seven aggregate TSVs and receipt')
    return parser.parse_args()


ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    options = arguments()
    module_dir = options.reader_root.resolve()/'refit' if options.reader_root else ROOT/'analysis'
    sys.path.insert(0, str(module_dir))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from cspr.path_a_assay import FamilyCounts, hierarchical_weights, neighbour_groups, unit_expressions, weighted_ridge_design
from cspr.path_a_course_correction import cell_halves
from cspr.path_a_portability import source_coordinate
from cspr.path_a_v05_wp1 import checked_scores, folds_for, select_source

MANIFESTS = Path('manifests/path-a')
EXPERIMENT = MANIFESTS / 'v059-resolution-2026-11-15'
STUDIES = ('StAnge', 'Iida', 'Russ')
SIZES = (20, 50, 100, 200)
HASH_SEED = '20261115'
TRAINING_COMPONENTS = ('full', 'half0', 'half1')
METRICS = ('rho', 'amplitude', 'R2_centered')


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t', fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def full_sha(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def keys(namespace, sample):
    prefix = f'path-a-v059/{namespace}/{HASH_SEED}/{sample.sample}/'.encode()
    return np.asarray([hashlib.sha256(prefix + b.encode()).hexdigest() for b in sample.barcodes])


def subset(sample, mask):
    idx = np.flatnonzero(mask)
    counts = sample.counts
    return replace(sample, barcodes=tuple(sample.barcodes[i] for i in idx),
                   counts=FamilyCounts(counts.linear[idx], counts.denominator[idx], counts.denominator_detected[idx]))


def populations(samples):
    original, common, support = {}, {}, []
    for study in STUDIES:
        halves, _ = cell_halves(samples[study])
        original[study] = dict(full=samples[study], half0=halves[0], half1=halves[1])
        kept_by_sample = defaultdict(set)
        common[study] = {'full': [], 'half0': [], 'half1': []}
        for which, half in enumerate(halves):
            for sample in half:
                n = len(sample.barcodes)
                keep = n // 200 * 200
                if keep == 0:
                    raise ValueError(f'No common block in {sample.sample} half {which}')
                priority = keys('support', sample)
                ids = np.lexsort((np.asarray(sample.barcodes), priority))[:keep]
                mask = np.zeros(n, bool)
                mask[ids] = True
                retained = subset(sample, mask)
                assert not (set(retained.barcodes) & kept_by_sample[sample.sample])
                kept_by_sample[sample.sample].update(retained.barcodes)
                common[study][f'half{which}'].append(retained)
                support.append(dict(study=study, preparation=sample.sample, outer_half=which,
                                    eligible=n, common_cells=keep, removed=n-keep,
                                    eligible_barcodes_sha256=full_sha(sorted(sample.barcodes)),
                                    common_barcodes_sha256=full_sha(sorted(retained.barcodes))))
        for sample in samples[study]:
            ids = kept_by_sample[sample.sample]
            joined = subset(sample, np.asarray([b in ids for b in sample.barcodes]))
            assert ids == set(joined.barcodes)
            common[study]['full'].append(joined)
    return original, common, support


def state_spaces(population, axes):
    columns = [axes.ids.index(s) for s in axes.s]
    spaces = {}
    for study in STUDIES:
        for part, cohort in population[study].items():
            preparations = []
            for sample in sorted(cohort, key=lambda s: s.key):
                coordinate = source_coordinate((sample,), axes)
                state = np.log1p(sample.counts.linear[:, columns].toarray())
                points, _ = coordinate.transform(state)
                preparations.append((sample, points))
            spaces[study,part] = preparations
    return spaces


def seeded_anchor_groups(points, sample, size):
    """Only the order of choosing anchors changes. Lexical neighbour ties remain."""
    barcodes = np.asarray(sample.barcodes)
    lexical = np.argsort(barcodes, kind='stable')
    order = np.lexsort((barcodes, keys('anchor', sample)))
    unused = np.ones(len(barcodes), bool)
    assignment = np.full(len(barcodes), -1)
    group = 0
    for anchor in order:
        if not unused[anchor]:
            continue
        if group == len(barcodes) // size:
            break
        unused[anchor] = False
        candidates = lexical[unused[lexical]]
        distances = np.sum((points[candidates] - points[anchor])**2, axis=1)
        chosen = candidates[np.argsort(distances, kind='stable')[:size-1]]
        assignment[anchor] = assignment[chosen] = group
        unused[chosen] = False
        group += 1
    return assignment


def build_components(spaces, axes, size, order, original_ledger=None):
    xy = [axes.ids.index(family) for family in (*axes.x, *axes.y)]
    result, group_lists, evidence = {}, defaultdict(list), []
    for (study, part), preparations in spaces.items():
        components = []
        for sample, points in preparations:
            if order == 'lexical':
                membership = neighbour_groups(points, sample.barcodes, size=size)
            else:
                membership = seeded_anchor_groups(points, sample, size)
            if original_ledger is None and np.any(membership < 0):
                raise ValueError('Incomplete common-support group')
            expr = unit_expressions(sample.counts.linear[:, xy], membership, size=size)
            ids = [f'{sample.study}/{sample.sample}/{i:08d}' for i in range(len(expr))]
            names = [ids[i] if i >= 0 else '' for i in membership]
            if original_ledger is not None:
                old = original_ledger[study+'-'+part, sample.sample]
                assert len(old)==len(names) and all(old[b]==n for b,n in zip(sample.barcodes,names))
            if part != 'full':
                group_lists[study].append((sample, membership))
            components.append(dict(TF=expr[:, :len(axes.x)], y=expr[:, len(axes.x):],
                study=np.repeat(sample.study,len(expr)), sample=np.repeat(sample.sample,len(expr)),
                identities=np.asarray(ids)))
            evidence.append(dict(component=study+'-'+part, preparation=sample.sample,
                                 groups=len(ids), included_cells=int((membership>=0).sum()),
                                 memberships_sha256=full_sha(f'{b}\t{n}' for b,n in sorted(zip(sample.barcodes,names)))))
        result[study,part] = data = {k:np.concatenate([p[k] for p in components]) for k in components[0]}
        data['weights'] = hierarchical_weights(data['study'], data['sample'])
    return result, group_lists, evidence


def fit_and_score(data, variant, size, order):
    models, training = {}, []
    for key, source in data.items():
        study, component = key
        n = len(source['TF'])
        row = dict(variant=variant, size=size, anchor=order, source=study, component=component, groups=n)
        if n < 10:
            models[key] = None
            training.append(dict(**row, selected_index='NA', rank_fraction='NA', effective_df='NA',
                                 status='below_10_groups_required_by_five_fold_selector'))
            continue
        folds = folds_for(source['identities'])
        cv = select_source(source, 'TF', None, folds)
        design = weighted_ridge_design(source['TF'], source['weights'])
        model = design.fit(source['y'], design.rank * cv['selected_fraction'])
        training.append(dict(**row, selected_index=cv['selected_index'], rank_fraction=cv['selected_fraction'],
                             effective_df=model.audit['achieved_df'], status='estimated'))
        models[key] = model
    scored, moments = [], []
    for target in STUDIES:
        halves = [data[target, f'half{i}'] for i in (0,1)]
        y = np.concatenate([h['y'] for h in halves])
        wt = hierarchical_weights(np.concatenate([h['study'] for h in halves]), np.concatenate([h['sample'] for h in halves]))
        y_centered = y - wt @ y
        observed_ss = np.sum(wt[:, None] * y_centered**2)
        for source in STUDIES:
            parts = [f'half{1-i}' if source == target else 'full' for i in (0,1)]
            fitted = [models[source,part] for part in parts]
            comparison = 'within_study' if source==target else ('same_species' if 'StAnge' not in (source,target) else 'cross_species')
            row = dict(variant=variant, size=size, anchor=order, source=source, target=target,
                       comparison=comparison, target_groups=len(y))
            if any(fit is None for fit in fitted):
                scored.append(dict(**row, **{k:'NA' for k in METRICS}, status='source_half_below_10_groups'))
                moments.append(dict(**row, observed_ss=observed_ss, predicted_ss='NA',
                                    cross_product='NA', error_ss='NA'))
                continue
            p = np.concatenate([model.predict(half['TF']) for model,half in zip(fitted,halves)])
            s = checked_scores(y,p,wt)
            p_centered = p - wt @ p
            error_ss = np.sum(wt[:, None] * (y_centered-p_centered)**2)
            assert abs((1-error_ss/observed_ss)-s['R2_centered']) < 1e-12
            assert abs((2*s['rho']*s['amplitude']-s['amplitude']**2)-s['R2_centered']) < 1e-12
            moments.append(dict(**row, observed_ss=observed_ss,
                                predicted_ss=np.sum(wt[:, None] * p_centered**2),
                                cross_product=np.sum(wt[:, None] * y_centered * p_centered),
                                error_ss=error_ss))
            scored.append(dict(**row, **{k:s[k] for k in METRICS}, status='estimated'))
    return scored, training, moments


def half_agreement(group_lists, axes, size, variant, order):
    """Original two-half ANOVA, generalized to even group sizes. Equal preparation contributions."""
    ycols = [axes.ids.index(y) for y in axes.y]
    out, mean_squares = [], []
    for study in STUDIES:
        grouped = defaultdict(list)
        for sample, membership in group_lists[study]:
            valid = np.flatnonzero(membership>=0)
            indices = valid[np.argsort(membership[valid], kind='stable')].reshape(-1,size)
            assert np.all(membership[indices] == np.arange(len(indices))[:,None])
            values = sample.counts.linear[:,ycols].toarray()
            grouped[sample.sample].append((values, indices, sample.barcodes))
        results = []
        for split in range(32):
            between, within = 0.0, 0.0
            for prep, parts in grouped.items():
                first, second = [], []
                for values, indices, barcodes in parts:
                    ns = f'path-a-v052/{split}/{prep}/'.encode()
                    priority = np.array([hashlib.sha256(ns+b.encode()).digest() for b in barcodes],dtype='V32')
                    # Use lexical byte order of SHA-256; preserve barcode order in the vanishingly rare ties.
                    ranked = np.argsort(priority[indices], axis=1, kind='stable')
                    aligned = np.take_along_axis(indices, ranked, axis=1)
                    a = np.log1p(values[aligned[:, :size//2]].mean(axis=1))
                    b = np.log1p(values[aligned[:, size//2:]].mean(axis=1))
                    first.append(a)
                    second.append(b)
                a, b = np.vstack(first), np.vstack(second)
                assert len(a) >= 2
                between += 2*np.var((a+b)/2, axis=0, ddof=1).sum()
                within += np.mean((a-b)**2/2, axis=0).sum()
            results.append((between-within)/(between+within))
            mean_squares.append(dict(variant=variant, size=size, anchor=order, target=study, split=split,
                                     between_ms=between/len(grouped), within_ms=within/len(grouped)))
        out.append(dict(variant=variant,size=size,anchor=order,target=study,preparations=len(grouped),
                        groups=sum(len(g) for p in grouped.values() for _,g,_ in p),half_nuclei=size//2,
                        icc_split0=results[0],icc_min_32=min(results),icc_max_32=max(results)))
    return out, mean_squares


def read_membership(prepared_dir):
    ledger=defaultdict(dict)
    with gzip.open(prepared_dir/'membership.tsv.gz','rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            ledger[r['component'],r['sample']][r['barcode']] = r['unit']
    return ledger


def load_inputs(root, reader):
    if reader:
        from regroup import load_populations
        refit = root/'refit'
        axes, samples = load_populations(refit/'populations', studies=STUDIES)
        prepared_dir = next((d for d in (refit/'groups',refit/'grouped') if (d/'membership.tsv.gz').is_file()), None)
        if prepared_dir is None:
            raise FileNotFoundError('The refit archive requires historical group matrices and membership.tsv.gz in refit/groups/.')
        return axes, samples, prepared_dir, refit/'cspr', [refit/'regroup.py',refit/'populations/layout.json']

    from path_a_v05_calibration_preflight import load_exposed
    loader, recipe, config, staged = load_exposed(root)
    axes, _, _ = loader.axes_from(config)
    subset_recipe = dict(recipe, libraries=[r for r in recipe['libraries'] if r['name'] in STUDIES])
    samples = loader.approved_samples(subset_recipe)
    return (axes, samples, root/'work/path-a/v05-wp1-prepared-2026-11-06', root/'analysis/cspr',
            [staged/'matrix-inputs.json', staged/'populations/seal.json',
             root/'analysis/scripts/path_a_v05_calibration_preflight.py'])


def run(output, reader_root=None):
    root = reader_root or ROOT
    manifests = root/MANIFESTS
    print('Loading verified exposed inputs',flush=True)
    axes, samples, prepared_dir, module_dir, upstream = load_inputs(root, reader_root is not None)
    reference_axes = json.loads((prepared_dir/'axes.json').read_text())
    assert tuple(reference_axes['TF'])==axes.x and tuple(reference_axes['Y'])==axes.y
    assert len(axes.x)==136 and len(axes.y)==115
    original, common, support=populations(samples)
    issued=manifests/'v05-wp1-selected-matrix-2026-11-06.tsv'
    published={(r['source'],r['target']):r for r in table(issued)
               if r['model']=='TF' and r['arm']=='raw' and r['point']=='source_cv'}
    icc_path=manifests/'v052-stanford-review-2026-11-11/target-repeatability.tsv'
    published_icc={r['target']:r for r in table(icc_path) if r['family']=='POOLED'}
    receipt=manifests/'v05-wp1-preparation-2026-11-06.json'
    prepared=json.loads(receipt.read_text())
    assert sha(prepared_dir/'axes.json') == prepared['files_sha256']['axes.json']
    assert sha(prepared_dir/'membership.tsv.gz') == prepared['files_sha256']['membership.tsv.gz']
    original_ledger=read_membership(prepared_dir)
    all_scores, all_training, all_agreement, all_memberships = [], [], [], []
    all_moments, all_ms = [], []
    maxima={'source_target_group_expression':0.,'metrics':0.,'icc_reference_and_range':0.}
    for variant, population, configurations in (
            ('original_support',original,[(20,'lexical')]),
            ('common_support',common,[(k,'lexical') for k in SIZES]+[(20,'seeded_anchor')])):
        print('Calculating preparation-local PCA',variant,flush=True)
        spaces=state_spaces(population,axes)
        for size,order in configurations:
            print('Grouping',variant,size,order,flush=True)
            originals=variant=='original_support'
            data, groups, membership = build_components(spaces,axes,size,order,original_ledger if originals else None)
            if originals:
                for (study, part), values in data.items():
                    tag=f'{study}-{part}.npz'
                    assert sha(prepared_dir/tag) == prepared['files_sha256'][tag]
                    with np.load(prepared_dir/tag,allow_pickle=False) as old:
                        for k in ('TF','y','weights','identities','sample','study'):
                            if k in ('TF','y','weights'):
                                error=float(np.max(np.abs(old[k]-values[k])))
                                maxima['source_target_group_expression']=max(error,maxima['source_target_group_expression'])
                                assert error<=1e-12,(tag,k,error)
                            else:
                                np.testing.assert_array_equal(old[k],values[k])
            print('Source-only CV and disjoint-half agreement',variant,size,order,flush=True)
            scores, training, moments = fit_and_score(data,variant,size,order)
            agreements, mean_squares = half_agreement(groups,axes,size,variant,order)
            if originals:
                for r in scores:
                    old=published[r['source'],r['target']]
                    for name in METRICS:
                        err=abs(float(old[name])-r[name])
                        maxima['metrics']=max(err,maxima['metrics'])
                        assert err <= 1e-12,(r['source'],r['target'],name,err)
                for r in agreements:
                    for new,old in [('icc_split0','icc_split0'),('icc_min_32','split_min'),('icc_max_32','split_max')]:
                        error=abs(r[new]-float(published_icc[r['target']][old]))
                        maxima['icc_reference_and_range']=max(error,maxima['icc_reference_and_range'])
                        assert error<=1e-12,(r['target'],new,error)
                print('Historical control passed',maxima,flush=True)
            all_scores.extend(scores);all_training.extend(training);all_agreement.extend(agreements)
            all_moments.extend(moments)
            all_ms.extend(mean_squares)
            all_memberships.extend(dict(variant=variant,size=size,anchor=order,**r) for r in membership)
            print('Completed',variant,size,order,flush=True)
    output.mkdir(exist_ok=True,parents=True)
    tables = [('scores.tsv', all_scores), ('half-agreement.tsv', all_agreement), ('source-fits.tsv', all_training),
              ('common-support.tsv', support), ('membership-digests.tsv', all_memberships),
              ('centered-moments.tsv', all_moments), ('half-mean-squares.tsv', all_ms)]
    for name, rows in tables:
        write_table(output/name,rows)
    inputs=[root/EXPERIMENT/'design.md', issued, icc_path, receipt, *upstream,
            *[module_dir/f'path_a_{m}.py' for m in ('assay','portability','course_correction','v05','v05_wp1')]]
    verified=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),
        scope='exploratory common-support resolution and anchor-order checks on three already-exposed studies',
        source_sha256={str(p.relative_to(root)):sha(p) for p in inputs},
        analysis_script_sha256=sha(Path(__file__).resolve()),
        output_sha256={name:sha(output/name) for name,_ in tables},
        nine_original_scores_and_three_split_iccs_recovered=True,max_abs_control_errors=maxima,
        backend=[{k:r[k] for k in ('internal_api','version','num_threads')} for r in threadpool_info()],
        score_rows=len(all_scores),available_rows=sum(r['status']=='estimated' for r in all_scores),
        membership_rule='All eligible original halves for the issued control; deterministic common subset separately in each preparation/half for all sensitivity arms.')
    (output/'verification.json').write_text(json.dumps(verified,indent=2)+'\n')


if __name__=='__main__':
    reader_root = options.reader_root.resolve() if options.reader_root else None
    default_output = reader_root/'reconstructed/resolution' if reader_root else ROOT/EXPERIMENT
    with threadpool_limits(limits=1):
        # On any BLAS, the original memberships, expressions and points must reproduce first.
        run((options.output or default_output).resolve(),reader_root)
