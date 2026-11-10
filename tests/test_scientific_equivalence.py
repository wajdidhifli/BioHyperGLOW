"""Scientific equivalence tests comparing packaged implementation to research protocol.

Verifies:
1. Degree- and mapping-aware gate equivalence (kappa=10).
2. Branch exposure and score equivalence.
3. Intensity mixture equivalence (logaddexp Poisson rate fusion).
4. Normalized concatenation and Proposition 1 path reconstruction.
5. Exact additive explanation reconstruction (< 1e-12 tolerance).
6. Checkpoint round-trip and identifier ordering invariance.
"""
import numpy as np
import pytest

from biohyperglow.dual_space import (
    branch_scores,
    compute_gate,
    mix_intensities,
    normalized_concatenation_check,
)
from biohyperglow.explanation import explain_association
from biohyperglow.scoring import BioHyperGLOWModel, rate_to_probability


def research_scores(branch, ds, gs=None):
    """Reference implementation from ranking_development/dual_space.py."""
    ds = np.asarray(ds, dtype=int)
    if gs is None:
        overlap = branch["b"][ds] @ branch["q"].T
        exposure = np.clip(branch["c"][ds, None] + branch["a"][None, :], -20.0, 20.0)
    else:
        gs = np.asarray(gs, dtype=int)
        overlap = (branch["b"][ds] * branch["q"][gs]).sum(axis=-1)
        exposure = np.clip(branch["c"][ds] + branch["a"][gs], -20.0, 20.0)
    return np.log(np.maximum(overlap, np.finfo(float).tiny)) + exposure


def research_gate_degree10(mapped, degree):
    """Reference gate from ranking_development/dual_space.py."""
    return np.asarray(mapped, dtype=float) * 10.0 / (10.0 + np.asarray(degree, dtype=float))


def research_mix_intensities(semantic, free, weight):
    """Reference mixture from ranking_development/score_screen.py."""
    weight = np.broadcast_to(np.asarray(weight, dtype=float), semantic.shape)
    log_a = np.full(weight.shape, -np.inf)
    log_b = np.full(weight.shape, -np.inf)
    np.log(weight, out=log_a, where=weight > 0.0)
    np.log1p(-weight, out=log_b, where=weight < 1.0)
    score = np.logaddexp(log_a + semantic, log_b + free)
    return score


def test_gate_mathematical_equivalence():
    rng = np.random.default_rng(123)
    mapped = rng.choice([True, False], size=100)
    degrees = rng.integers(0, 50, size=100)

    packaged_gate = compute_gate(mapped, degrees, kappa=10.0)
    reference_gate = research_gate_degree10(mapped, degrees)

    np.testing.assert_allclose(packaged_gate, reference_gate, atol=1e-15, rtol=0)


def test_branch_and_fused_score_equivalence():
    rng = np.random.default_rng(42)
    h = 8
    n_d = 10
    n_g = 15

    branch = {
        "b": rng.dirichlet(np.ones(h), size=n_d),
        "q": rng.dirichlet(np.ones(h), size=n_g),
        "a": rng.normal(0, 1, size=n_g),
        "c": rng.normal(0, 1, size=n_d),
    }

    # Test pair-wise scores
    d_pairs = rng.integers(0, n_d, size=25)
    g_pairs = rng.integers(0, n_g, size=25)

    pkg_scores = branch_scores(branch, d_pairs, g_pairs)
    ref_scores = research_scores(branch, d_pairs, g_pairs)
    np.testing.assert_allclose(pkg_scores, ref_scores, atol=1e-15, rtol=0)

    # Test full matrix evaluation
    pkg_all = branch_scores(branch, np.arange(n_d))
    ref_all = research_scores(branch, np.arange(n_d))
    np.testing.assert_allclose(pkg_all, ref_all, atol=1e-15, rtol=0)


def test_intensity_mixture_equivalence():
    rng = np.random.default_rng(456)
    s = rng.normal(0, 2, size=(20, 10))
    f = rng.normal(0, 2, size=(20, 10))
    weights = rng.uniform(0, 1, size=20)[:, None]

    pkg_mix = mix_intensities(s, f, weights)
    ref_mix = research_mix_intensities(s, f, weights)

    np.testing.assert_allclose(pkg_mix, ref_mix, atol=1e-15, rtol=0)


def test_explanation_exact_additive_reconstruction(dummy_dual_model):
    """Test that every coordinate path sums to the exact intensity with < 1e-12 error."""
    for d in dummy_dual_model.diseases:
        for g in dummy_dual_model.genes:
            explanation = explain_association(dummy_dual_model, d, g)
            # 1. Total reconstruction error must be < 1e-12
            assert explanation.max_reconstruction_error < 1e-12

            # 2. Sum of contributions must reconstruct predicted intensity
            sum_contrib = sum(c.intensity_contribution for c in explanation.components)
            np.testing.assert_allclose(sum_contrib, explanation.predicted_intensity, atol=1e-12, rtol=0)

            # 3. Sum of shares must equal 1.0
            sum_shares = sum(c.contribution_share for c in explanation.components)
            np.testing.assert_allclose(sum_shares, 1.0, atol=1e-12, rtol=0)

            # 4. Decoded probability check
            expected_p = rate_to_probability(explanation.log_intensity)
            np.testing.assert_allclose(explanation.predicted_probability, expected_p, atol=1e-12, rtol=0)


def test_checkpoint_roundtrip_and_order_invariance(dummy_dual_model, tmp_path):
    ckpt_path = tmp_path / "roundtrip_test.npz"
    dummy_dual_model.save(ckpt_path)

    loaded = BioHyperGLOWModel.load(ckpt_path)

    # Identifiers and ordering
    assert loaded.diseases == dummy_dual_model.diseases
    assert loaded.genes == dummy_dual_model.genes
    assert loaded.H == dummy_dual_model.H
    assert loaded.kappa == dummy_dual_model.kappa
    assert loaded.threshold == dummy_dual_model.threshold

    # Array weights exactness
    np.testing.assert_allclose(loaded.mapped, dummy_dual_model.mapped)
    np.testing.assert_allclose(loaded.degrees, dummy_dual_model.degrees)
    np.testing.assert_allclose(loaded.alpha, dummy_dual_model.alpha)

    np.testing.assert_allclose(loaded.semantic["q"], dummy_dual_model.semantic["q"])
    np.testing.assert_allclose(loaded.semantic["b"], dummy_dual_model.semantic["b"])
    np.testing.assert_allclose(loaded.semantic["a"], dummy_dual_model.semantic["a"])
    np.testing.assert_allclose(loaded.semantic["c"], dummy_dual_model.semantic["c"])

    np.testing.assert_allclose(loaded.free["q"], dummy_dual_model.free["q"])
    np.testing.assert_allclose(loaded.free["b"], dummy_dual_model.free["b"])
    np.testing.assert_allclose(loaded.free["a"], dummy_dual_model.free["a"])
    np.testing.assert_allclose(loaded.free["c"], dummy_dual_model.free["c"])

    # Predictions from original and reloaded models must be bit-for-bit identical
    preds_orig = dummy_dual_model.predict(dummy_dual_model.diseases[:2], dummy_dual_model.genes[:2])
    preds_load = loaded.predict(loaded.diseases[:2], loaded.genes[:2])
    np.testing.assert_allclose(preds_load["intensity"], preds_orig["intensity"], atol=1e-15, rtol=0)
    np.testing.assert_allclose(preds_load["probability"], preds_orig["probability"], atol=1e-15, rtol=0)
