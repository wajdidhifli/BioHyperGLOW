"""Regression coverage for the final release review, without benchmark training."""
import json

import numpy as np
import pytest

import biohyperglow.training as training
from biohyperglow.data import load_edges_csv, train_val_test_split
from biohyperglow.scoring import BioHyperGLOWModel


@pytest.mark.parametrize("as_file", [False, True])
@pytest.mark.parametrize("field", ["sha256", "n_d", "n_g", "folds"])
def test_reject_mismatched_saved_split_identity(synthetic_data, tmp_path, as_file, field):
    edges_path, _ = synthetic_data
    edges, diseases, genes, digest = load_edges_csv(edges_path)
    folds = train_val_test_split(edges, len(diseases), len(genes), folds=3)
    metadata = {"sha256": digest, "n_d": len(diseases), "n_g": len(genes), "folds": 3}
    metadata[field] = "0" * 64 if field == "sha256" else metadata[field] + 100
    saved = {"meta": metadata, "splits": [{k: v.tolist() for k, v in f.items()} for f in folds]}
    if as_file:
        path = tmp_path / "mismatched.json"
        path.write_text(json.dumps(saved))
        saved = path
    with pytest.raises(ValueError, match=f"metadata mismatch for {field}"):
        training.train_dual_space(edges_path, splits=saved, H=4, epochs=1)


@pytest.mark.parametrize("fold", [0, 6])
def test_generated_split_fold_bounds(synthetic_data, fold):
    edges_path, _ = synthetic_data
    with pytest.raises(ValueError, match="Requested fold="):
        training.train_dual_space(edges_path, fold=fold, H=4, epochs=1)


def test_paper_preset_uses_ten_folds_without_running_training(synthetic_data, monkeypatch):
    edges_path, _ = synthetic_data
    edges, diseases, genes, _ = load_edges_csv(edges_path)
    expected = train_val_test_split(edges, len(diseases), len(genes), folds=10)[0]
    calls = []

    def capture_trainer(**kwargs):
        calls.append(kwargs)
        branch = {
            "b": kwargs["b0"].astype(float),
            "q": np.full((kwargs["n_genes"], kwargs["h_dim"]), 1 / kwargs["h_dim"]),
            "a": np.zeros(kwargs["n_genes"]),
            "c": np.zeros(kwargs["n_diseases"]),
        }
        return branch, None, {}

    monkeypatch.setattr(training, "train_single_branch", capture_trainer)
    model = training.train_dual_space(edges_path, preset="paper")
    assert len(calls) == 2
    for call in calls:
        np.testing.assert_array_equal(call["train_pos"], expected["train_pos"])
        assert call["epochs"] == 5000
        assert call["patience"] == 50
        assert call["seed"] == 42
    assert model.metadata["folds"] == 10
    assert model.metadata["eval_every"] == 5
    assert len(model.metadata["split_sha256"]) == 64


def test_checkpoint_resave_preserves_current_authoritative_fields(dummy_dual_model, tmp_path):
    original = tmp_path / "original.npz"
    revised = tmp_path / "revised.npz"
    dummy_dual_model.metadata["custom_note"] = "retained"
    dummy_dual_model.save(original)
    loaded = BioHyperGLOWModel.load(original)
    loaded.threshold = -2.75
    loaded.semantic["group_labels"][0] = "Updated description"
    loaded.save(revised)
    restored = BioHyperGLOWModel.load(revised)
    assert restored.threshold == -2.75
    assert restored.semantic["group_labels"][0] == "Updated description"
    assert restored.metadata["custom_note"] == "retained"
    disease, gene = restored.diseases[0], restored.genes[0]
    before = loaded.predict([disease], [gene])
    after = restored.predict([disease], [gene])
    np.testing.assert_array_equal(before["binary_prediction"], after["binary_prediction"])
    np.testing.assert_array_equal(before["intensity"], after["intensity"])
