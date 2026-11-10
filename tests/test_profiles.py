"""Tests for structural target profile computation T0."""
import numpy as np
import pytest

from biohyperglow.profiles import compute_structural_targets, normalize_rows


def test_normalize_rows():
    mat = np.array([[2.0, 3.0], [0.0, 0.0]])
    norm = normalize_rows(mat)
    assert norm[0, 0] == pytest.approx(0.4)
    assert norm[0, 1] == pytest.approx(0.6)
    # Zero row falls back to uniform
    assert norm[1, 0] == pytest.approx(0.5)
    assert norm[1, 1] == pytest.approx(0.5)


def test_compute_structural_targets():
    # 3 diseases, 4 genes, 2 hypernodes
    n_d = 3
    n_g = 4
    train_pos = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
        [2, 3],
    ])
    b0 = np.array([
        [0.8, 0.2],
        [0.5, 0.5],
        [0.1, 0.9],
    ])

    t0, audit = compute_structural_targets(train_pos, n_d, n_g, b0, tau=2.0, penalty=30.0)

    assert t0.shape == (n_d + n_g, 2)
    # Check row sums equal 1.0
    np.testing.assert_allclose(t0.sum(axis=1), 1.0, atol=1e-5)
    assert (t0 >= 0.0).all()
    assert audit["n_diseases"] == 3
    assert audit["n_genes"] == 4
    assert audit["connected_components"] >= 1
