"""Development measurement helpers; no regression, labels or transfer scores."""
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from .path_a_gate import non_evaluated_columns


def validate_counts(counts):
    values = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    values.sum_duplicates()
    values.eliminate_zeros()
    if (not np.isfinite(values.data).all() or np.any(values.data < 0)
            or not np.equal(values.data, np.floor(values.data)).all()):
        raise ValueError("Expected finite nonnegative integer counts")
    return values


def diagnostic_panels(tissue_genes):
    """Disjoint neuronal halves and tissue-exclusive non-neuronal markers."""
    neuron = sorted(set(tissue_genes["Neuron"]))
    other = {k: set(v) - set(neuron) for k, v in tissue_genes.items() if k != "Neuron"}
    uses = Counter(g for genes in other.values() for g in genes)
    exclusive = {k: sorted(g for g in genes if uses[g] == 1) for k, genes in other.items()}
    return neuron[::2], neuron[1::2], exclusive


def grouping_embedding(counts, gene_ids, excluded_ids, *, min_cells=5, max_genes=1000,
                       components=20, seed=0):
    """Fit a per-collection descriptive embedding with no evaluated/marker inputs."""
    if counts.shape[1] != len(gene_ids):
        raise ValueError("Count columns and gene axis differ")
    keep = non_evaluated_columns(gene_ids, excluded_ids)
    values = sparse.csr_matrix(counts[:, keep], dtype=np.float64)
    genes = np.asarray(gene_ids)[keep]
    totals = np.asarray(values.sum(axis=1)).ravel()
    if len(totals) < 2 or np.any(totals <= 0):
        raise ValueError("Grouping needs at least two cells with nonzero allowed counts")
    detected = np.asarray((values > 0).sum(axis=0)).ravel()
    values = sparse.diags(10000.0 / totals) @ values
    values.data = np.log1p(values.data)
    means = np.asarray(values.mean(axis=0)).ravel()
    variance = np.asarray(values.power(2).mean(axis=0)).ravel() - means ** 2
    candidates = np.flatnonzero((detected >= min_cells) & (variance > 0))
    order = np.lexsort((genes[candidates], -variance[candidates]))
    selected = candidates[order[:max_genes]]
    if not len(selected):
        raise ValueError("No variable genes available for the registered grouping")
    n_components = min(components, len(selected), values.shape[0] - 1)
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed,
                iterated_power=4)
    embedding = model.fit_transform(values[:, selected].toarray())
    return embedding, genes[selected].tolist(), model.explained_variance_ratio_


def full_groups(embedding, barcodes, size, *, method="neighbors", seed=0):
    """Return one full-unit ID per cell, or -1 for an explicitly discarded remainder."""
    points = np.asarray(embedding)
    names = np.asarray(barcodes, dtype=str)
    if size < 2 or points.ndim != 2 or len(names) != len(points):
        raise ValueError("Invalid grouping dimensions or unit size")
    if len(names) != len(set(names)) or not np.isfinite(points).all():
        raise ValueError("Grouping requires unique barcodes and finite coordinates")
    if method not in {"neighbors", "random"}:
        raise ValueError("Unknown grouping method")
    # Canonical barcode order makes membership invariant to input row permutation.
    order = np.argsort(names, kind="stable")
    points = points[order]
    anchors = np.random.default_rng(seed).permutation(len(points))
    assigned = np.full(len(points), -1, dtype=int)
    full = len(points) // size
    if method == "random":
        assigned[anchors[:full * size]] = np.repeat(np.arange(full), size)
    else:
        unused = np.ones(len(points), dtype=bool)
        group = 0
        for anchor in anchors:
            if group == full:
                break
            if not unused[anchor]:
                continue
            candidates = np.flatnonzero(unused)
            candidates = candidates[candidates != anchor]
            distances = np.sum((points[candidates] - points[anchor]) ** 2, axis=1)
            nearest = candidates[np.lexsort((candidates, distances))[:size - 1]]
            members = np.concatenate(([anchor], nearest))
            assigned[members] = group
            unused[members] = False
            group += 1
    result = np.empty(len(points), dtype=int)
    result[order] = assigned
    return result


def normalized_families(counts, gene_ids, families, *, target_sum=10000):
    """Sum within family, then normalize each cell against the FULL raw library."""
    raw = validate_counts(counts)
    if raw.shape[1] != len(gene_ids) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Family counts need a unique matching gene axis")
    totals = np.asarray(raw.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("A measured cell has zero total counts")
    index = {g: i for i, g in enumerate(gene_ids)}
    rows, cols, used = [], [], set()
    for j, members in enumerate(families):
        if not members or len(members) != len(set(members)) or used & set(members):
            raise ValueError("Family membership must be nonempty and globally unique")
        if not set(members) <= set(index):
            raise ValueError("Family member missing from the count axis")
        used.update(members)
        rows.extend(index[g] for g in members)
        cols.extend([j] * len(members))
    mapping = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                shape=(len(gene_ids), len(families)))
    summed = (raw @ mapping).toarray()
    return summed, summed * (target_sum / totals[:, None])


def aggregate_families(linear, assignment, size):
    """Average normalized cells, then log1p; remainders never enter an estimate."""
    values = np.asarray(linear, dtype=float)
    groups = np.asarray(assignment)
    if values.ndim != 2 or len(groups) != len(values) or groups.ndim != 1:
        raise ValueError("Group assignment must match the cell matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative normalized values")
    if not np.issubdtype(groups.dtype, np.integer) or np.any(groups < -1):
        raise ValueError("Group IDs must be integers with -1 for remainders")
    ids, sizes = np.unique(groups[groups >= 0], return_counts=True)
    if not np.array_equal(ids, np.arange(len(ids))) or np.any(sizes != size):
        raise ValueError("Every unit must be full, nonoverlapping and consecutively numbered")
    means = np.stack([values[groups == g].mean(axis=0) for g in ids]) if len(ids) else np.empty((0, values.shape[1]))
    return means, np.log1p(means)


def upper_mad_flags(values, reference):
    """Descriptive upper tail using four normalized MADs, not a fitted QC cutoff."""
    log_values = np.log1p(np.asarray(values))
    reference = np.asarray(reference, dtype=bool)
    if not reference.any():
        return np.zeros(len(values), dtype=bool), None
    median = np.median(log_values[reference])
    mad = np.median(np.abs(log_values[reference] - median))
    if mad == 0:
        return np.zeros(len(values), dtype=bool), None
    threshold = median + 4 * 1.4826 * mad
    return log_values > threshold, float(threshold)
