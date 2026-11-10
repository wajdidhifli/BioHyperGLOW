"""Tests for data loading, negative pair sampling, and CV splitting."""
import numpy as np
import pytest

from biohyperglow.data import (
    generate_synthetic_dataset,
    load_edges_csv,
    negative_pairs,
    train_val_test_split,
)


def test_load_edges_and_negative_sampling(synthetic_data):
    edges_path, _ = synthetic_data
    edges, diseases, genes, sha = load_edges_csv(edges_path)

    assert len(edges) == 20
    assert len(diseases) <= 8
    assert len(genes) <= 12
    assert len(sha) == 64

    # Negative sampling
    rng = np.random.default_rng(42)
    pos_set = set(map(tuple, edges.tolist()))
    negs = negative_pairs(len(diseases), len(genes), pos_set, 10, rng)

    assert len(negs) == 10
    # Negatives must not overlap with positives
    neg_set = set(map(tuple, negs.tolist()))
    assert len(pos_set & neg_set) == 0


def test_train_val_test_split_integrity(synthetic_data):
    edges_path, _ = synthetic_data
    edges, diseases, genes, _ = load_edges_csv(edges_path)

    folds = train_val_test_split(edges, len(diseases), len(genes), seed=42, folds=3, val_frac=0.1)
    assert len(folds) == 3

    all_pos = set(map(tuple, edges.tolist()))
    for fold in folds:
        tr = set(map(tuple, fold["train_pos"].tolist()))
        va = set(map(tuple, fold["val_pos"].tolist()))
        te = set(map(tuple, fold["test_pos"].tolist()))

        # Disjoint partitions
        assert len(tr & va) == 0
        assert len(tr & te) == 0
        assert len(va & te) == 0
        assert (tr | va | te) == all_pos

        vn = set(map(tuple, fold["val_neg"].tolist()))
        tn = set(map(tuple, fold["test_neg"].tolist()))
        assert len((vn | tn) & all_pos) == 0
        assert len(vn & tn) == 0
