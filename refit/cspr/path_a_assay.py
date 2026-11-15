"""P1-AD mechanics. No study discovery, target tuning or execution on import."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from .path_a_panel_benchmark import SpectralDesign
from .path_a_pilot import finite_matrix, unit_state_features


def hierarchical_weights(studies, samples):
    """Equal study, then sample within study, then row within sample."""
    studies, samples = np.asarray(studies, str), np.asarray(samples, str)
    if studies.ndim != 1 or studies.shape != samples.shape or not len(studies):
        raise ValueError("Aligned nonempty study/sample axes required")
    if np.any(studies == "") or np.any(samples == ""):
        raise ValueError("Missing study/sample identity")
    weights = np.empty(len(studies))
    for study in np.unique(studies):
        selected = studies == study
        names = np.unique(samples[selected])
        for sample in names:
            keep = selected & (samples == sample)
            weights[keep] = 1 / (len(np.unique(studies)) * len(names) * keep.sum())
    return weights


def row_weights(weights, n):
    weights = np.asarray(weights, float)
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("Positive normalized row weights required")
    return weights / weights.sum()


def integer_counts(counts):
    # Validate stored entries before summing duplicates: cancellation/rounding is forbidden.
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values != np.floor(values))):
        raise ValueError("Finite nonnegative integer stored counts required")
    counts = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    return counts


@dataclass
class FamilyCounts:
    linear: sparse.csr_matrix
    denominator: np.ndarray
    denominator_detected: np.ndarray


def normalize_families(counts, gene_ids, families, denominator_columns):
    """Complete disjoint families; D normalization precedes unit averaging/log."""
    counts = integer_counts(counts)
    genes = list(gene_ids)
    if len(genes) != counts.shape[1] or len(set(genes)) != len(genes):
        raise ValueError("Unique matching measured gene axis required")
    index = {gene: i for i, gene in enumerate(genes)}
    rows, columns, seen = [], [], set()
    for column, members in enumerate(families):
        members = list(members)
        if not members or len(set(members)) != len(members) or seen.intersection(members):
            raise ValueError("Nonempty globally disjoint family membership required")
        if not set(members) <= index.keys():
            raise ValueError("Missing measured family member; no zero fill")
        seen.update(members)
        rows.extend(index[gene] for gene in members)
        columns.extend([column] * len(members))
    d = np.asarray(denominator_columns, int)
    if (d.ndim != 1 or not len(d) or len(set(d)) != len(d)
            or np.any(d < 0) or np.any(d >= len(families))):
        raise ValueError("Unique measured denominator family indices required")
    mapping = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                                shape=(len(genes), len(families)))
    family = (counts @ mapping).tocsr()
    mass = np.asarray(family[:, d].sum(axis=1)).ravel()
    if np.any(mass <= 0):
        raise ValueError("Eligible cell has zero D mass; do not silently remove it")
    detected = np.asarray((family[:, d] > 0).sum(axis=1)).ravel() / len(d)
    return FamilyCounts((sparse.diags(10000 / mass) @ family).tocsr(), mass, detected)


def mouse_population(counts, gene_ids, xy_ids, marker_panels):
    counts = integer_counts(counts)
    genes = list(gene_ids)
    if len(set(genes)) != len(genes) or len(genes) != counts.shape[1]:
        raise ValueError("Unique matching measured gene axis required")
    index = {gene: i for i, gene in enumerate(genes)}
    if len(marker_panels) != 2:
        raise ValueError("Both registered marker panels required")
    supports = []
    for panel in marker_panels:
        if (len(set(panel)) != len(panel) or len(panel) < 2
                or not set(panel) <= index.keys() or set(panel) & set(xy_ids)):
            raise ValueError("Complete distinct non-X/Y markers required")
        supports.append(np.asarray((counts[:, [index[g] for g in panel]] >= 1)
                                   .sum(axis=1)).ravel() >= 2)
    allowed = [i for i, gene in enumerate(genes) if gene not in set(xy_ids)]
    detected = np.asarray((counts[:, allowed] > 0).sum(axis=1)).ravel()
    return supports[0] & supports[1] & (detected > 200), detected, supports


@dataclass
class SourceScale:
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray

    @classmethod
    def fit(cls, x, weights):
        x = finite_matrix(x)
        w = row_weights(weights, len(x))
        mean = w @ x
        sd = np.sqrt(w @ ((x - mean) ** 2))
        constant = sd <= 1e-12
        sd[constant] = 1
        return cls(mean, sd, constant)

    def transform(self, x):
        x = finite_matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Source feature axis changed")
        z = (x - self.mean) / self.scale
        z[:, self.constant] = 0
        return z


def weighted_ridge_design(x, weights):
    """Reuse the pinned spectral solver with explicit hierarchical weights."""
    x = finite_matrix(x)
    w = row_weights(weights, len(x))
    scale = SourceScale.fit(x, w)
    z = scale.transform(x)
    a = np.sqrt(w)[:, None] * z
    dual = x.shape[1] > len(x)
    gram = a @ a.T if dual else a.T @ a
    values, vectors = np.linalg.eigh(gram)
    keep = values > np.finfo(float).eps * max(x.shape) * max(float(values[-1]), 0)
    if not keep.any():
        raise ValueError("Zero numerical input rank; required model cannot be omitted")
    return SpectralDesign(scale.mean, scale.scale, scale.constant, z, w,
                          values[keep], vectors[:, keep], dual, int(keep.sum()))


@dataclass
class SourceState:
    scale: SourceScale
    components: np.ndarray
    eigenvalues: np.ndarray
    centroids: np.ndarray
    numerical_rank: int

    @classmethod
    def fit(cls, state_expression, weights, *, components=20, states=8):
        x = finite_matrix(state_expression)
        w = row_weights(weights, len(x))
        scale = SourceScale.fit(x, w)
        z = scale.transform(x)
        covariance = z.T @ (w[:, None] * z)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(-values, kind="stable")
        values, vectors = values[order], vectors[:, order]
        rank = int(np.sum(values > np.finfo(float).eps * max(x.shape)
                          * max(float(values[0]), 0)))
        if rank < components or len(x) < states:
            raise ValueError("Insufficient source PCA rank/cells; fixed dimension cannot shrink")
        loadings = vectors[:, :components].T.copy()
        for loading in loadings:
            if loading[np.argmax(np.abs(loading))] < 0:
                loading *= -1
        points = z @ loadings.T
        model = KMeans(n_clusters=states, random_state=0, n_init=10).fit(
            points, sample_weight=w)
        if len(np.unique(model.labels_)) != states:
            raise ValueError("Source state fit did not produce all fixed states")
        centers = np.asarray(sorted(map(tuple, model.cluster_centers_)))
        return cls(scale, loadings, values[:components], centers, rank)

    def transform(self, state_expression):
        points = self.scale.transform(state_expression) @ self.components.T
        distances = np.sum((points[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        return points, np.argmin(distances, axis=1)


def neighbour_groups(points, barcodes, *, size=20):
    """First remaining barcode plus nearest remaining cells; remainder is -1."""
    points = finite_matrix(points)
    barcodes = np.asarray(barcodes, str)
    if (len(barcodes) != len(points) or len(set(barcodes)) != len(barcodes)
            or not isinstance(size, int) or size < 2):
        raise ValueError("Unique aligned barcodes and full unit size required")
    order = np.argsort(barcodes, kind="stable")
    sorted_points = points[order]
    assignment = np.full(len(points), -1, dtype=int)
    unused = np.ones(len(points), bool)
    for unit in range(len(points) // size):
        candidates = np.flatnonzero(unused)
        anchor, others = candidates[0], candidates[1:]
        distance = np.sum((sorted_points[others] - sorted_points[anchor]) ** 2, axis=1)
        nearest = others[np.lexsort((others, distance))[:size - 1]]
        members = np.r_[anchor, nearest]
        assignment[members] = unit
        unused[members] = False
    result = np.empty(len(points), int)
    result[order] = assignment
    return result


def unit_expressions(linear, assignment, *, size=20):
    assignment = np.asarray(assignment, int)
    if assignment.shape != (linear.shape[0],) or np.any(assignment < -1):
        raise ValueError("Invalid cell-to-unit axis")
    ids, counts = np.unique(assignment[assignment >= 0], return_counts=True)
    if not len(ids) or not np.array_equal(ids, np.arange(len(ids))) or np.any(counts != size):
        raise ValueError("Contiguous complete units required")
    cells = np.flatnonzero(assignment >= 0)
    averaging = sparse.csr_matrix((np.full(len(cells), 1 / size),
                                  (assignment[cells], cells)), shape=(len(ids), len(assignment)))
    mean = averaging @ sparse.csr_matrix(linear)
    return np.log1p(mean.toarray())


def build_sample_units(family_counts, state_columns, coordinate, barcodes, *, size=20):
    state = np.log1p(family_counts.linear[:, state_columns].toarray())
    points, labels = coordinate.transform(state)
    assignment = neighbour_groups(points, barcodes, size=size)
    expression = unit_expressions(family_counts.linear, assignment, size=size)
    features, dominant = unit_state_features(
        points, labels, family_counts.denominator, family_counts.denominator_detected,
        assignment, size, k=len(coordinate.centroids))
    return dict(expression=expression, state_features=features, dominant=dominant,
                assignment=assignment, cell_states=labels)


@dataclass
class StateNeighbours:
    scale: SourceScale
    points: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    identities: np.ndarray

    @classmethod
    def fit(cls, x, y, weights, identities):
        x, y = finite_matrix(x), finite_matrix(y)
        names = np.asarray(identities, str)
        if len(x) != len(y) or names.shape != (len(x),) or len(set(names)) != len(names):
            raise ValueError("Aligned unique source unit identities required")
        w = row_weights(weights, len(x))
        scale = SourceScale.fit(x, w)
        return cls(scale, scale.transform(x), y.copy(), w, names)

    def predict(self, x, fraction):
        if not 0 < fraction <= 1:
            raise ValueError("Neighbour fraction must be in (0,1]")
        points = self.scale.transform(x)
        k = max(1, int(np.ceil(fraction * len(self.points))))
        if k == len(self.points):
            return np.broadcast_to(self.weights @ self.y, (len(points), self.y.shape[1])).copy()
        predictions = []
        for point in points:
            distances = np.sum((self.points - point) ** 2, axis=1)
            nearest = np.lexsort((self.identities, distances))[:k]
            w = self.weights[nearest] / self.weights[nearest].sum()
            predictions.append(w @ self.y[nearest])
        return np.asarray(predictions)


def weighted_pooled_r2(y, prediction, weights):
    y, prediction = finite_matrix(y), finite_matrix(prediction)
    if y.shape != prediction.shape:
        raise ValueError("Prediction/output axes differ")
    w = row_weights(weights, len(y))
    total = np.sum(w[:, None] * (y - w @ y) ** 2)
    if total <= 0:
        return None
    return float(1 - np.sum(w[:, None] * (y - prediction) ** 2) / total)
