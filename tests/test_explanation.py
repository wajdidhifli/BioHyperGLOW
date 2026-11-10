"""Tests for intrinsic explanation, share decomposition, and Proposition 1 verification."""
import numpy as np
import pytest


def test_explanation_shares_and_reconstruction(dummy_dual_model):
    res = dummy_dual_model.explain("C0000001", "101")

    # 1. Intensity decomposition
    assert res.reconstructed_intensity == pytest.approx(res.predicted_intensity, abs=1e-10)
    assert res.max_reconstruction_error < 1e-10

    # 2. Probability check
    expected_prob = 1.0 - np.exp(-res.predicted_intensity)
    assert res.predicted_probability == pytest.approx(expected_prob, abs=1e-10)

    # 3. Component shares must sum to 1.0
    shares = [c.contribution_share for c in res.components]
    assert sum(shares) == pytest.approx(1.0, abs=1e-10)

    # 4. Branch shares must match branch intensities
    assert (res.semantic_intensity + res.structural_intensity) == pytest.approx(res.predicted_intensity, abs=1e-10)
    assert (res.semantic_share + res.structural_share) == pytest.approx(1.0, abs=1e-10)

    # 5. Component counts: 2 * H = 10 coordinates
    assert len(res.components) == 10
    sem_comps = [c for c in res.components if c.branch == "semantic"]
    str_comps = [c for c in res.components if c.branch == "structural"]
    assert len(sem_comps) == 5
    assert len(str_comps) == 5

    # 6. Check that structural coordinates are not presented as biological mechanisms
    for sc in str_comps:
        assert not sc.biological_interpretation_valid
        assert "latent" in sc.label.lower() or "structural" in sc.label.lower()

    # 7. Summary format check
    summary_text = res.summary(top_k=3)
    assert "BioHyperGLOW Explanation" in summary_text
    assert "Notice: Semantic hypernodes" in summary_text
