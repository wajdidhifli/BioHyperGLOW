"""Tests for the dual-space intensity mixture, gate, and Proposition 1."""
import numpy as np
import pytest

from biohyperglow.dual_space import compute_gate, mix_intensities, normalized_concatenation_check


def test_compute_gate_behavior():
    # Test degree gate with kappa = 10.0
    mapped = [True, True, True, False]
    degree = [0, 10, 30, 0]

    alpha = compute_gate(mapped, degree, kappa=10.0)
    # k=0 mapped -> 10 / (10 + 0) = 1.0
    assert pytest.approx(alpha[0]) == 1.0
    # k=10 mapped -> 10 / (10 + 10) = 0.5
    assert pytest.approx(alpha[1]) == 0.5
    # k=30 mapped -> 10 / (10 + 30) = 0.25
    assert pytest.approx(alpha[2]) == 0.25
    # unmapped -> 0.0
    assert pytest.approx(alpha[3]) == 0.0


def test_mix_intensities_logaddexp():
    s = np.array([0.0, -1.0, 2.0])
    f = np.array([1.0, 0.5, -0.5])

    # Weight 1.0 should equal s
    np.testing.assert_allclose(mix_intensities(s, f, 1.0), s)
    # Weight 0.0 should equal f
    np.testing.assert_allclose(mix_intensities(s, f, 0.0), f)

    # 50/50 mixture
    expected = np.log(0.5 * np.exp(s) + 0.5 * np.exp(f))
    mixed = mix_intensities(s, f, 0.5)
    np.testing.assert_allclose(mixed, expected, atol=1e-12)


def test_proposition1_normalized_concatenation_reconstruction():
    """Reproduces and verifies Proposition 1 across varying activities and gate weights."""
    rng = np.random.default_rng(42)

    def make_branch():
        return {
            "b": rng.dirichlet(np.ones(5), size=4),
            "q": rng.dirichlet(np.ones(5), size=6),
            "a": np.array([-45.0, -3.0, 0.0, 1.0, 8.0, 45.0]),
            "c": np.array([30.0, -30.0, 0.0, 1.0]),
        }

    semantic, free = make_branch(), make_branch()
    free["a"] = free["a"][::-1].copy()
    free["c"] = np.roll(free["c"], 1)

    pairs = np.array([(d, g) for d in range(4) for g in range(6)])
    for weights in (np.full(4, 0.5), np.array([0.0, 1.0, 0.25, 0.9])):
        audit = normalized_concatenation_check(semantic, free, pairs, weights, atol=1e-12)
        assert audit["pairs_checked"] == 24
        assert audit["total_coordinates"] == 10
        assert audit["max_log_intensity_error"] <= 1e-12
        assert audit["proposition_1_verified"]
