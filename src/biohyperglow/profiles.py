"""Graph structural target profiles T0 from the positive training graph.

Computes fixed group-level structural targets by propagating shortest-path affinities
from disease reference groups to all graph nodes, using the fictive component bridge.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    """Row-normalize a nonnegative matrix to the simplex, falling back to uniform if row sum is zero."""
    arr = np.asarray(scores, dtype=float)
    mass = arr.sum(axis=1, keepdims=True)
    uniform = np.full_like(arr, 1.0 / arr.shape[1])
    return np.divide(arr, mass, out=uniform, where=mass > 0.0).astype(np.float32)


def compute_structural_targets(
    train_pos: np.ndarray,
    n_diseases: int,
    n_genes: int,
    b0: np.ndarray,
    tau: float = 2.0,
    penalty: float = 30.0,
    block_size: int = 64,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute fixed structural target profiles T0 for all graph nodes.

    Parameters
    ----------
    train_pos : np.ndarray of shape (N_train, 2)
        Training positive edges as pairs of (disease_index, gene_index).
    n_diseases : int
        Number of diseases n_D.
    n_genes : int
        Number of genes n_G.
    b0 : np.ndarray of shape (n_diseases, H)
        Prior disease memberships B0.
    tau : float, default=2.0
        Distance temperature parameter for exponential decay exp(-dist / tau).
    penalty : float, default=30.0
        Cross-component penalty rho added to distances between disconnected components.
    block_size : int, default=64
        Block size for shortest path queries to manage memory.
    progress : callable, optional
        Callback receiving (current_disease_index, total_diseases).

    Returns
    -------
    t0 : np.ndarray of shape (n_diseases + n_genes, H)
        Fixed row-stochastic structural targets for diseases (rows 0..n_D-1)
        and genes (rows n_D..n_D+n_G-1).
    audit : dict
        Diagnostics including component counts and runtime.
    """
    n = n_diseases + n_genes
    train_arr = np.asarray(train_pos, dtype=np.int64)
    if len(train_arr) > 0:
        d_idx = train_arr[:, 0]
        g_idx = n_diseases + train_arr[:, 1]
        row = np.r_[d_idx, g_idx]
        col = np.r_[g_idx, d_idx]
        data = np.ones(len(row), dtype=float)
        adj = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    else:
        adj = coo_matrix((n, n), dtype=float).tocsr()

    nc, comp = connected_components(adj, directed=False)
    degree = np.diff(adj.indptr)

    # Select highest-degree, lowest-ID hub per component
    order = np.lexsort((np.arange(n), -degree, comp))
    ordered_comp = comp[order]
    hubs = order[np.r_[True, ordered_comp[1:] != ordered_comp[:-1]]]

    # Augment graph with a bridge node connected to each component hub
    if len(train_arr) > 0:
        aug_row = np.r_[row, hubs, np.full(nc, n)]
        aug_col = np.r_[col, np.full(nc, n), hubs]
    else:
        aug_row = np.r_[hubs, np.full(nc, n)]
        aug_col = np.r_[np.full(nc, n), hubs]

    aug_data = np.ones(len(aug_row), dtype=float)
    augmented = coo_matrix((aug_data, (aug_row, aug_col)), shape=(n + 1, n + 1)).tocsr()

    # Column-normalize disease prior B0 to prevent large clusters from dominating target mass
    b0_float = np.asarray(b0, dtype=float)
    col_sums = b0_float.sum(axis=0, keepdims=True)
    c0 = np.divide(b0_float, col_sums, out=np.full_like(b0_float, 1.0 / b0_float.shape[0]), where=col_sums > 0)

    h_dim = b0.shape[1]
    pooled = np.zeros((n, h_dim), dtype=np.float64)

    start_time = time.perf_counter()
    for begin in range(0, n_diseases, block_size):
        end = min(n_diseases, begin + block_size)
        indices = np.arange(begin, end)
        # Compute shortest paths from disease block to all graph nodes
        dist = shortest_path(augmented, method="D", directed=False, unweighted=True, indices=indices)[:, :n]
        # Cross-component penalty for nodes in different original components
        diff_comp = comp[begin:end, None] != comp[None, :]
        dist += diff_comp * (penalty + 1.0)

        affinity = np.exp(-dist.T / tau)
        pooled += affinity @ c0[begin:end]
        if progress:
            progress(end, n_diseases)

    t0 = normalize_rows(pooled)
    elapsed = time.perf_counter() - start_time

    audit = {
        "n_nodes": n,
        "n_diseases": n_diseases,
        "n_genes": n_genes,
        "connected_components": int(nc),
        "isolated_diseases": int((degree[:n_diseases] == 0).sum()),
        "isolated_genes": int((degree[n_diseases:] == 0).sum()),
        "tau": tau,
        "bridge_penalty": penalty,
        "preprocessing_seconds": elapsed,
        "h": h_dim,
    }

    return t0, audit
