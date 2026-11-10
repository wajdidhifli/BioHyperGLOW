"""Tests for synthetic training workflow, early stopping, and checkpoint selection."""
import numpy as np
import pytest

from biohyperglow.training import EarlyStopping, train_dual_space


def test_early_stopping():
    stopping = EarlyStopping(patience=3, min_delta=1e-3)
    assert not stopping.stopped

    stopping.update(0.500)
    assert stopping.bad_checks == 0

    # Below delta -> bad check 1
    stopping.update(0.5005)
    assert stopping.bad_checks == 1

    # Improvement -> resets bad checks
    stopping.update(0.510)
    assert stopping.bad_checks == 0

    # 3 consecutive bad checks
    stopping.update(0.510)
    assert stopping.bad_checks == 1
    stopping.update(0.505)
    assert stopping.bad_checks == 2
    stopped = stopping.update(0.500)
    assert stopped
    assert stopping.stopped


def test_train_dual_space_short_workflow(synthetic_data, tmp_path):
    edges_path, ontology_path = synthetic_data
    model_path = tmp_path / "model.npz"

    # Fast 5-epoch test
    model = train_dual_space(
        edges_csv=edges_path,
        ontology_obo=ontology_path,
        H=4,
        kappa=10.0,
        epochs=5,
        eval_every=1,
        patience=5,
        batch_size=32,
        device="cpu",
        seed=42,
    )

    assert model.H == 4
    assert len(model.diseases) > 0
    assert len(model.genes) > 0
    assert model.threshold is not None

    # Test saving and re-loading
    model.save(model_path)
    assert model_path.exists()

    from biohyperglow.scoring import BioHyperGLOWModel
    loaded = BioHyperGLOWModel.load(model_path)
    assert loaded.H == model.H
    assert loaded.diseases == model.diseases
    assert loaded.genes == model.genes
    assert loaded.threshold == pytest.approx(model.threshold)
