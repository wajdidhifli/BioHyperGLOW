"""Shared pytest fixtures for BioHyperGLOW test suite."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from biohyperglow.data import generate_synthetic_dataset
from biohyperglow.scoring import BioHyperGLOWModel


@pytest.fixture
def synthetic_data(tmp_path):
    """Generate synthetic edges and ontology for testing."""
    edges_path, ontology_path = generate_synthetic_dataset(
        tmp_path, n_diseases=8, n_genes=12, n_edges=20, seed=42
    )
    return edges_path, ontology_path


@pytest.fixture
def dummy_dual_model():
    """Create a small deterministic BioHyperGLOWModel for unit tests."""
    rng = np.random.default_rng(42)
    n_d = 4
    n_g = 6
    h = 5

    diseases = [f"C000000{i+1}" for i in range(n_d)]
    genes = [f"10{i+1}" for i in range(n_g)]

    b_s = rng.dirichlet(np.ones(h), size=n_d)
    q_s = rng.dirichlet(np.ones(h), size=n_g)
    a_s = np.array([-1.0, 0.0, 0.5, -0.2, 0.8, -0.5])
    c_s = np.array([0.2, -0.3, 0.1, 0.0])

    b_f = rng.dirichlet(np.ones(h), size=n_d)
    q_f = rng.dirichlet(np.ones(h), size=n_g)
    a_f = np.array([0.5, -0.2, 0.0, 0.3, -0.8, 0.2])
    c_f = np.array([-0.1, 0.2, 0.0, -0.1])

    mapped = np.array([True, True, True, False])
    degrees = np.array([0, 5, 10, 2])

    semantic = {
        "b": b_s,
        "q": q_s,
        "a": a_s,
        "c": c_s,
        "group_ids": [f"CLUSTER_{i+1}" for i in range(h - 1)] + ["UNASSIGNED"],
        "group_labels": [f"Cluster {i+1} concept" for i in range(h - 1)] + ["Unassigned fallback"],
    }
    free = {
        "b": b_f,
        "q": q_f,
        "a": a_f,
        "c": c_f,
    }

    return BioHyperGLOWModel(
        semantic=semantic,
        free=free,
        diseases=diseases,
        genes=genes,
        mapped=mapped,
        degrees=degrees,
        kappa=10.0,
        threshold=-1.5,
    )
