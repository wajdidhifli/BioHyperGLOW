"""BioHyperGLOW: Adaptive Dual-Space Embeddings for Interpretable Disease-Gene Association Prediction."""
from __future__ import annotations

from biohyperglow.dual_space import compute_gate, mix_intensities, normalized_concatenation_check
from biohyperglow.explanation import ExplanationResult, explain_association
from biohyperglow.metrics import pair_metrics, ranking_metrics, select_threshold
from biohyperglow.model import BioHyperGLOW
from biohyperglow.scoring import BioHyperGLOWModel, rate_to_probability
from biohyperglow.training import train_dual_space, train_single_branch

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BioHyperGLOW",
    "BioHyperGLOWModel",
    "compute_gate",
    "explain_association",
    "ExplanationResult",
    "mix_intensities",
    "normalized_concatenation_check",
    "pair_metrics",
    "ranking_metrics",
    "rate_to_probability",
    "select_threshold",
    "train_dual_space",
    "train_single_branch",
]
