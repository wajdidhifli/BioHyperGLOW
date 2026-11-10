"""Tests for the single-branch BioHyperGLOW PyTorch module."""
import math
import numpy as np
import pytest
import torch

from biohyperglow.model import BioHyperGLOW


def test_model_initialization():
    b0 = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    gene_init = np.array([[0.1, -0.1], [0.0, 0.0], [-0.2, 0.2]], dtype=np.float32)

    model = BioHyperGLOW(b0, gene_init, mode="learned")
    assert model.nd == 2
    assert model.ng == 3
    assert model.h == 2
    assert model.theta.requires_grad
    assert model.delta.requires_grad

    # Test invalid b0
    with pytest.raises(ValueError, match="strictly positive"):
        BioHyperGLOW(np.array([[0.0, 1.0]]), np.array([[0.1, 0.2]]))

    with pytest.raises(ValueError, match="row-stochastic"):
        BioHyperGLOW(np.array([[0.5, 0.2]]), np.array([[0.1, 0.2]]))


def test_log_memberships_simplex_constraint():
    b0 = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
    gene_init = np.zeros((3, 2), dtype=np.float32)
    model = BioHyperGLOW(b0, gene_init, mode="learned")

    log_q, log_b = model.log_memberships()
    q = torch.exp(log_q)
    b = torch.exp(log_b)

    # Exponentiated softmax row sums must be 1.0
    torch.testing.assert_close(q.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(b.sum(dim=1), torch.ones(2))


def test_score_and_components_reconstruction():
    b0 = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
    gene_init = np.array([[0.2, -0.2], [-0.1, 0.1]], dtype=np.float32)
    model = BioHyperGLOW(b0, gene_init, mode="learned")

    d = torch.tensor([0, 1], dtype=torch.long)
    g = torch.tensor([1, 0], dtype=torch.long)

    log_rate, phi = model.score(d, g, components=True)
    rate = torch.exp(log_rate)

    # Sum of path components must equal total rate
    torch.testing.assert_close(phi.sum(dim=1), rate, atol=1e-6, rtol=1e-6)
    assert (phi >= 0).all()


def test_association_loss_gradient_flow():
    b0 = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
    gene_init = np.zeros((2, 2), dtype=np.float32)
    model = BioHyperGLOW(b0, gene_init, mode="learned")

    d = torch.tensor([0, 1])
    g = torch.tensor([0, 1])
    y = torch.tensor([1.0, 0.0])

    log_rate = model.score(d, g)
    loss = model.association_loss(log_rate, y)

    assert torch.isfinite(loss)
    loss.backward()

    assert model.theta.grad is not None
    assert model.delta.grad is not None
    assert model.log_s.grad is not None
    assert model.log_t.grad is not None


def test_frozen_and_activity_mode_parameter_freezing():
    b0 = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
    gene_init = np.zeros((3, 2), dtype=np.float32)

    # In 'frozen' mode: disease memberships are frozen at B0, but gene memberships are learned
    frozen_model = BioHyperGLOW(b0, gene_init, mode="frozen")
    assert frozen_model.theta.requires_grad is True, "gene logits theta must remain trainable in frozen mode"
    assert frozen_model.delta.requires_grad is False, "disease delta must be frozen in frozen mode"

    # In 'activity' mode: memberships are frozen, only log_s and log_t are learned
    activity_model = BioHyperGLOW(b0, gene_init, mode="activity")
    assert activity_model.theta.requires_grad is False
    assert activity_model.delta.requires_grad is False
    assert activity_model.log_s.requires_grad is True
    assert activity_model.log_t.requires_grad is True

    # In 'learned' mode: both theta and delta are learned
    learned_model = BioHyperGLOW(b0, gene_init, mode="learned")
    assert learned_model.theta.requires_grad is True
    assert learned_model.delta.requires_grad is True

    # In 'free' mode: both theta and delta are learned
    free_model = BioHyperGLOW(b0, gene_init, mode="free")
    assert free_model.theta.requires_grad is True
    assert free_model.delta.requires_grad is True

    # Optimization step test in frozen mode: theta updates, delta stays zero
    opt = torch.optim.SGD([p for p in frozen_model.parameters() if p.requires_grad], lr=0.1)
    d = torch.tensor([0, 1])
    g = torch.tensor([0, 1])
    y = torch.tensor([1.0, 0.0])
    loss = frozen_model.association_loss(frozen_model.score(d, g), y)
    loss.backward()
    opt.step()

    # Theta should have changed
    assert not torch.allclose(frozen_model.theta, torch.zeros((3, 2), dtype=torch.float32))
    # Delta should remain strictly zero
    assert torch.all(frozen_model.delta == 0.0)
    # Disease memberships remain identical to B0
    _, log_b = frozen_model.log_memberships()
    torch.testing.assert_close(torch.exp(log_b), torch.from_numpy(b0))

