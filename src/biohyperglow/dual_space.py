"""Dual-space representation, intensity mixture, and normalized concatenation.

Implements the adaptive degree- and mapping-dependent gate:
  alpha_d = m_d * (kappa / (kappa + k_d))
where m_d is 1 for accepted ontology mappings and 0 otherwise, and k_d is the positive
training degree of disease d. Reference configuration uses kappa = 10.0.

Also provides the exact intensity mixture and normalized concatenation check (Proposition 1).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def compute_gate(
    mapped: Union[np.ndarray, list],
    degree: Union[np.ndarray, list],
    kappa: float = 10.0,
) -> np.ndarray:
    """Compute the mapping- and degree-dependent gate weights alpha_d.

    Parameters
    ----------
    mapped : array-like of bool or int
        Indicator of accepted ontology mapping for each disease (1 if mapped, 0 otherwise).
    degree : array-like of int
        Number of positive training associations for each disease.
    kappa : float, default=10.0
        Degree scale parameter at which a mapped disease assigns equal weight (0.5)
        to the semantic and structural branches.

    Returns
    -------
    alpha : np.ndarray of shape (n_diseases,)
        Branch gate weight in [0, 1]. For disease d:
        - If mapped is 0: alpha_d = 0.0 (relies entirely on association-only branch).
        - If mapped is 1: alpha_d = kappa / (kappa + degree_d).
    """
    mapped_arr = np.asarray(mapped, dtype=float)
    deg_arr = np.asarray(degree, dtype=float)
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if np.any(deg_arr < 0):
        raise ValueError("Training degrees cannot be negative")

    alpha = mapped_arr * (kappa / (kappa + deg_arr))
    return np.clip(alpha, 0.0, 1.0)


def mix_intensities(
    semantic_log_rate: np.ndarray,
    free_log_rate: np.ndarray,
    weight: Union[float, np.ndarray],
) -> np.ndarray:
    """Mix association intensities in log space: log(weight * exp(s) + (1 - weight) * exp(f)).

    This operates on association intensities (Poisson rates), preserving activity scales,
    rather than averaging post-link Bernoulli probabilities.

    Parameters
    ----------
    semantic_log_rate : np.ndarray
        Semantic branch log-intensity log(lambda^S).
    free_log_rate : np.ndarray
        Free association-only branch log-intensity log(lambda^F).
    weight : float or np.ndarray
        Gate weights alpha in [0, 1], broadcastable to input shapes.

    Returns
    -------
    combined_log_rate : np.ndarray
        Fused log intensity log(lambda_{dg}).
    """
    s = np.asarray(semantic_log_rate, dtype=float)
    f = np.asarray(free_log_rate, dtype=float)
    w = np.broadcast_to(np.asarray(weight, dtype=float), s.shape)

    if np.any((w < 0.0) | (w > 1.0)):
        raise ValueError("Gate weights must be in [0, 1]")

    log_a = np.full(w.shape, -np.inf)
    log_b = np.full(w.shape, -np.inf)
    np.log(w, out=log_a, where=w > 0.0)
    np.log1p(-w, out=log_b, where=w < 1.0)

    score = np.logaddexp(log_a + s, log_b + f)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite value encountered during intensity mixing")

    # Verify edge cases exactly
    if np.any(w == 0.0):
        score[w == 0.0] = f[w == 0.0]
    if np.any(w == 1.0):
        score[w == 1.0] = s[w == 1.0]

    return score


def branch_scores(
    branch: Dict[str, Any],
    d: Union[int, np.ndarray, list],
    g: Optional[Union[int, np.ndarray, list]] = None,
) -> np.ndarray:
    """Compute log association intensities from stored branch embeddings.

    Parameters
    ----------
    branch : dict
        Branch dictionary containing:
        - 'b': disease memberships of shape (n_diseases, H)
        - 'q': gene memberships of shape (n_genes, H)
        - 'a': gene log activities s_g of shape (n_genes,)
        - 'c': centered disease log activities t_d of shape (n_diseases,)
    d : int or array-like
        Disease indices.
    g : int or array-like, optional
        Gene indices. If None, evaluates all genes for the specified diseases,
        returning an array of shape (len(d), n_genes).

    Returns
    -------
    log_rates : np.ndarray
    """
    b = branch["b"]
    q = branch["q"]
    a = branch["a"]
    c = branch["c"]
    ds = np.asarray(d, dtype=int)

    if g is None:
        overlap = b[ds] @ q.T
        exposure = np.clip(c[ds, None] + a[None, :], -20.0, 20.0)
    else:
        gs = np.asarray(g, dtype=int)
        overlap = (b[ds] * q[gs]).sum(axis=-1)
        exposure = np.clip(c[ds] + a[gs], -20.0, 20.0)

    return np.log(np.maximum(overlap, np.finfo(float).tiny)) + exposure


def normalized_concatenation_check(
    semantic: Dict[str, Any],
    free: Dict[str, Any],
    pairs: np.ndarray,
    alpha: np.ndarray,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Verify Proposition 1: Normalized concatenation reproduces the intensity mixture.

    Concatenates gene memberships [Q^S; Q^F] and alpha-weighted disease memberships
    [alpha * B^S; (1 - alpha) * B^F], normalizes them to the 2H-coordinate simplex,
    and shows that the scaled path sum reconstructs the fused association intensity.

    Parameters
    ----------
    semantic : dict
        Semantic branch dictionary with 'b', 'q', 'a', 'c'.
    free : dict
        Free branch dictionary with 'b', 'q', 'a', 'c'.
    pairs : np.ndarray of shape (N, 2)
        Pairs (disease_index, gene_index).
    alpha : np.ndarray of shape (n_diseases,)
        Gate weights for all diseases.
    atol : float, default=1e-12
        Maximum allowed absolute difference between path sum and mixture intensity.

    Returns
    -------
    audit : dict
        Verification diagnostics including error bounds and simplex row sums.
    """
    pairs_arr = np.asarray(pairs, dtype=int)
    d = pairs_arr[:, 0]
    g = pairs_arr[:, 1]
    weight = np.asarray(alpha)[d]

    # Concatenate representations
    q_concat = np.concatenate([semantic["q"][g], free["q"][g]], axis=1)
    b_concat = np.concatenate(
        [weight[:, None] * semantic["b"][d], (1.0 - weight[:, None]) * free["b"][d]],
        axis=1,
    )

    zq = q_concat.sum(axis=1)
    zb = b_concat.sum(axis=1)

    # Simplex projection by L1 row normalization
    q_tilde = q_concat / zq[:, None]
    b_tilde = b_concat / zb[:, None]

    np.testing.assert_allclose(q_tilde.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b_tilde.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    assert (q_tilde >= 0.0).all() and (b_tilde >= 0.0).all()

    h = semantic["q"].shape[1]
    es = np.exp(np.clip(semantic["c"][d] + semantic["a"][g], -20.0, 20.0))
    ef = np.exp(np.clip(free["c"][d] + free["a"][g], -20.0, 20.0))
    scale = zq * zb

    paths = np.concatenate(
        [
            scale[:, None] * es[:, None] * q_tilde[:, :h] * b_tilde[:, :h],
            scale[:, None] * ef[:, None] * q_tilde[:, h:] * b_tilde[:, h:],
        ],
        axis=1,
    )
    assert np.isfinite(paths).all() and (paths >= 0.0).all()

    reconstructed = np.log(paths.sum(axis=1))
    s_score = branch_scores(semantic, d, g)
    f_score = branch_scores(free, d, g)
    expected = mix_intensities(s_score, f_score, weight)

    error = float(np.max(np.abs(reconstructed - expected)))
    np.testing.assert_allclose(reconstructed, expected, atol=atol, rtol=0)

    return {
        "pairs_checked": len(pairs_arr),
        "total_coordinates": 2 * h,
        "max_log_intensity_error": error,
        "max_gene_simplex_error": float(np.abs(q_tilde.sum(axis=1) - 1.0).max()),
        "max_disease_simplex_error": float(np.abs(b_tilde.sum(axis=1) - 1.0).max()),
        "nonnegative_paths": True,
        "proposition_1_verified": True,
    }
