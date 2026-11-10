"""BioHyperGLOW: Adaptive Dual-Space Embeddings for Interpretable Disease-Gene Association Prediction."""
from __future__ import annotations

from biohyperglow.data import (
    check_split,
    generate_synthetic_dataset,
    load_edges_csv,
    load_splits_json,
    negative_pairs,
    train_val_test_split,
)
from biohyperglow.dual_space import compute_gate, mix_intensities, normalized_concatenation_check
from biohyperglow.explanation import ExplanationResult, explain_association
from biohyperglow.metrics import pair_metrics, ranking_metrics, select_threshold
from biohyperglow.model import BioHyperGLOW
from biohyperglow.scoring import BioHyperGLOWModel, intensity_to_probability, rate_to_probability
from biohyperglow.training import train_dual_space, train_single_branch

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BioHyperGLOW",
    "BioHyperGLOWModel",
    "check_split",
    "compute_gate",
    "explain_association",
    "ExplanationResult",
    "generate_synthetic_dataset",
    "intensity_to_probability",
    "load_edges_csv",
    "load_splits_json",
    "mix_intensities",
    "negative_pairs",
    "normalized_concatenation_check",
    "pair_metrics",
    "ranking_metrics",
    "rate_to_probability",
    "select_threshold",
    "train_dual_space",
    "train_single_branch",
    "train_val_test_split",
]
