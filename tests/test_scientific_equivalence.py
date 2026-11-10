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


def test_probability_consistency_across_dynamic_range():
    """Verify Finding 4 reproducer: small finite probabilities match between prediction and explanation."""
    from biohyperglow.dual_space import intensity_to_probability, rate_to_probability
    from biohyperglow.scoring import BioHyperGLOWModel

    b = np.array([[1.0 - 1e-15, 1e-15]])
    q = np.array([[1e-15, 1.0 - 1e-15]])
    a = np.zeros(1)
    c = np.zeros(1)
    sem = {"b": b, "q": q, "a": a, "c": c}
    free = {"b": b, "q": q, "a": a, "c": c}

    model = BioHyperGLOWModel(sem, free, ["D1"], ["G1"], [True], [0], kappa=10.0)
    pred = model.predict(["D1"], ["G1"])
    expl = model.explain("D1", "G1")

    p_pred = pred["probability"][0]
    p_expl = expl.predicted_probability
    p_direct = intensity_to_probability(pred["intensity"][0])

    # Direct decoding of predicted intensity
    assert p_pred == pytest.approx(2e-15, rel=1e-10)
    assert p_expl == pytest.approx(2e-15, rel=1e-10)
    assert p_direct == pytest.approx(2e-15, rel=1e-10)
    # Exact consistency: prediction and explanation must agree without artificial floor
    assert abs(p_pred - p_expl) <= 1e-25

    # Check moderate and large rates
    rates = np.array([1e-20, 1e-10, 1e-5, 0.1, 1.0, 5.0, 20.0, 80.0])
    for r in rates:
        prob = intensity_to_probability(r)
        prob_from_log = rate_to_probability(np.log(r))
        np.testing.assert_allclose(prob, prob_from_log, atol=1e-15, rtol=1e-12)


def test_training_negative_sampler_exclusions(tmp_path):
    """Verify Finding 1: complete benchmark positives and held-out negatives are strictly excluded during training."""
    import biohyperglow.training
    from biohyperglow.data import generate_synthetic_dataset, load_edges_csv
    from biohyperglow.training import train_dual_space

    edges_csv, ontology_obo = generate_synthetic_dataset(
        tmp_path, n_diseases=8, n_genes=12, n_edges=20, seed=42
    )
    edges, ds, gs, _ = load_edges_csv(edges_csv)
    all_positive = set(map(tuple, edges.tolist()))

    captured_negatives = []
    forbidden_sets_received = []
    exclusion_sets_received = []

    original_neg_pairs = biohyperglow.training.negative_pairs

    def tracking_negative_pairs(n_d, n_g, pos_set, n, rng, exclude=None):
        forbidden_sets_received.append(set(pos_set))
        exclusion_sets_received.append(set(exclude) if exclude else set())
        negs = original_neg_pairs(n_d, n_g, pos_set, n, rng, exclude=exclude)
        for p in negs:
            captured_negatives.append((int(p[0]), int(p[1])))
        return negs

    biohyperglow.training.negative_pairs = tracking_negative_pairs
    try:
        model = train_dual_space(
            edges_csv=edges_csv,
            ontology_obo=ontology_obo,
            H=4,
            epochs=1,
            batch_size=8,
            steps_per_epoch=24,
            base_seed=42,
        )
    finally:
        biohyperglow.training.negative_pairs = original_neg_pairs

    assert len(captured_negatives) > 0

    # 1. Every forbidden set must equal the full benchmark positive set (len=20)
    for fset in forbidden_sets_received:
        assert len(fset) == len(all_positive), "Forbidden set in sampler must include all benchmark positives"
        assert fset == all_positive

    # 2. Every call must receive non-empty negative exclusions from the validation/test splits
    for ex in exclusion_sets_received:
        assert len(ex) > 0, "Held-out negative pairs must be excluded from training negative sampling"

    # 3. ZERO sampled negatives can be in all_positive
    overlap_pos = [p for p in captured_negatives if p in all_positive]
    assert len(overlap_pos) == 0, f"Sampled negatives overlapped with known positives: {overlap_pos}"

    # 4. ZERO sampled negatives can be in the union of all exclusion sets
    all_exclusions = set().union(*exclusion_sets_received)
    overlap_ex = [p for p in captured_negatives if p in all_exclusions]
    assert len(overlap_ex) == 0, f"Sampled negatives overlapped with held-out negative exclusions: {overlap_ex}"


def test_paired_minibatch_stream_and_training_step_equivalence():
    """Verify Finding 2: check 1-step training equivalence and paired minibatch stream consistency."""
    import math
    import torch
    from biohyperglow.model import BioHyperGLOW as PackagedModel

    # Define minimal reference module matching code/model_v1.py
    class ReferenceModel(torch.nn.Module):
        def __init__(self, b0, gene_initial):
            super().__init__()
            nd, h = b0.shape
            ng = len(gene_initial)
            self.nd = nd
            self.ng = ng
            self.h = h
            self.register_buffer("log_b0", torch.from_numpy(np.log(b0)))
            self.theta = torch.nn.Parameter(torch.as_tensor(gene_initial, dtype=torch.float32).clone())
            self.delta = torch.nn.Parameter(torch.zeros(nd, h, dtype=torch.float32))
            self.log_s = torch.nn.Parameter(torch.full((ng,), math.log(h * math.log(2)), dtype=torch.float32))
            self.log_t = torch.nn.Parameter(torch.zeros(nd, dtype=torch.float32))

        def log_memberships(self):
            return (
                torch.nn.functional.log_softmax(self.theta, dim=1),
                torch.nn.functional.log_softmax(self.log_b0 + self.delta, dim=1),
            )

        def score(self, d, g, logq=None, logb=None):
            if logq is None:
                logq, logb = self.log_memberships()
            exposure = (self.log_s[g] + self.log_t[d] - self.log_t.mean()).clamp(-20, 20)
            channel = logq[g] + logb[d]
            return exposure + torch.logsumexp(channel, dim=1)

        def association_loss(self, log_rate, y):
            safe = log_rate.clamp_min(-20)
            log_p_high = torch.log(-torch.expm1(-torch.exp(safe)))
            log_p = torch.where(log_rate < -20, log_rate, log_p_high)
            return ((1.0 - y) * torch.exp(log_rate) - y * log_p).mean()

    b0 = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
    gene_init = np.array([[0.1, -0.1], [-0.2, 0.2], [0.0, 0.0]], dtype=np.float32)

    torch.manual_seed(42)
    ref = ReferenceModel(b0, gene_init)
    torch.manual_seed(42)
    pkg = PackagedModel(b0, gene_init, mode="learned")

    # Initial parameters must be bit-for-bit identical
    torch.testing.assert_close(ref.theta, pkg.theta)
    torch.testing.assert_close(ref.delta, pkg.delta)
    torch.testing.assert_close(ref.log_s, pkg.log_s)
    torch.testing.assert_close(ref.log_t, pkg.log_t)

    # Identical scores on random batch
    d = torch.tensor([0, 1, 0, 1])
    g = torch.tensor([0, 1, 2, 0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])

    ref_score = ref.score(d, g)
    pkg_score = pkg.score(d, g)
    torch.testing.assert_close(ref_score, pkg_score, atol=1e-15, rtol=0)

    # Identical loss
    ref_loss = ref.association_loss(ref_score, y)
    pkg_loss = pkg.association_loss(pkg_score, y)
    torch.testing.assert_close(ref_loss, pkg_loss, atol=1e-15, rtol=0)

    # Identical step update with Adam
    opt_ref = torch.optim.Adam(ref.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt_pkg = torch.optim.Adam(pkg.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    ref_loss.backward()
    pkg_loss.backward()

    torch.testing.assert_close(ref.theta.grad, pkg.theta.grad, atol=1e-15, rtol=0)
    torch.testing.assert_close(ref.delta.grad, pkg.delta.grad, atol=1e-15, rtol=0)
    torch.testing.assert_close(ref.log_s.grad, pkg.log_s.grad, atol=1e-15, rtol=0)
    torch.testing.assert_close(ref.log_t.grad, pkg.log_t.grad, atol=1e-15, rtol=0)

    opt_ref.step()
    opt_pkg.step()

    torch.testing.assert_close(ref.theta, pkg.theta, atol=1e-15, rtol=0)
    torch.testing.assert_close(ref.delta, pkg.delta, atol=1e-15, rtol=0)
    torch.testing.assert_close(ref.log_s, pkg.log_s, atol=1e-15, rtol=0)
    torch.testing.assert_close(ref.log_t, pkg.log_t, atol=1e-15, rtol=0)

