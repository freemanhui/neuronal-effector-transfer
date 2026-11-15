"""Input boundary for a gate independent of evaluated gene expression."""
import numpy as np


def non_evaluated_columns(gene_ids, evaluated_ids):
    """Exclude evaluated genes before any normalization, selection or control scoring."""
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError("Gene axis must be unique")
    return np.array([g not in evaluated_ids for g in gene_ids], dtype=bool)
