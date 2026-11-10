"""Evaluation metrics and validation-based classification threshold selection.

Includes pair-level classification metrics (AP, AUROC, AUPRC, Accuracy, F1, MCC),
per-disease ranking metrics (Recall@K, NDCG@K, Macro AP), and exact validation threshold selection.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Select the score cutoff that maximizes binary F1 on validation data.

    When multiple thresholds achieve the same maximum F1, the largest exact score
    resolves the tie.

    Parameters
    ----------
    y : np.ndarray of shape (N,)
        Binary ground-truth labels (0 or 1). Must contain both classes.
    score : np.ndarray of shape (N,)
        Predicted scores (e.g., log-intensity or probability).

    Returns
    -------
    best_threshold : float
        Optimal cutoff maximizing validation F1.
    """
    yy = np.asarray(y, dtype=int)
    ss = np.asarray(score, dtype=float)

    if len(np.unique(yy)) != 2:
        raise ValueError("Threshold selection requires at least one positive and one negative sample")

    order = np.argsort(-ss, kind="stable")
    sorted_s, sorted_y = ss[order], yy[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s) != 0) + 1, len(sorted_s)]
    tp = np.cumsum(sorted_y)[ends - 1]
    f1 = 2.0 * tp / (ends + yy.sum())

    # Descending score order makes the first maximum the largest tied threshold.
    best_idx = int(np.flatnonzero(f1 == f1.max())[0])
    eta = float(sorted_s[ends[best_idx] - 1])

    # Cross-check against precision_recall_curve
    prec, rec, _ = precision_recall_curve(yy, ss)
    indep_f1 = 2.0 * prec * rec / np.maximum(prec + rec, 1e-300)
    np.testing.assert_allclose(f1[best_idx], indep_f1.max(), atol=1e-7, rtol=0)

    return eta


def pair_metrics(
    labels: Union[np.ndarray, Sequence[int]],
    log_intensity: Union[np.ndarray, Sequence[float]],
    threshold: Optional[float] = None,
    probability_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute pair-level discrimination and classification metrics.

    Parameters
    ----------
    labels : array-like of shape (N,)
        Binary ground truth labels (0 or 1).
    log_intensity : array-like of shape (N,)
        Predicted log association intensities.
    threshold : float, optional
        Classification cutoff on the log-intensity scale. If None, derived from
        `probability_threshold` via eta = log(-log(1 - p)).
    probability_threshold : float, default=0.5
        Default post-link probability threshold if explicit cutoff is not supplied.

    Returns
    -------
    metrics : dict
        Dictionary of AP, AUROC, AUPRC_trapezoid, Accuracy, Precision, Recall, F1, MCC,
        and confusion matrix counts.
    """
    y = np.asarray(labels, dtype=int)
    score = np.asarray(log_intensity, dtype=float)

    if y.ndim != 1 or y.shape != score.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching 1D binary labels and score arrays")
    if not np.isfinite(score).all():
        raise ValueError("Predicted scores must be finite")

    if threshold is None:
        if not (0.0 < probability_threshold < 1.0):
            raise ValueError(f"probability_threshold must lie in (0, 1), got {probability_threshold}")
        threshold = math.log(-math.log1p(-probability_threshold))

    prediction = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    result: Dict[str, Any] = {
        "pairs": len(y),
        "positives": int(y.sum()),
        "threshold": float(threshold),
        "Accuracy": float(accuracy_score(y, prediction)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Recall": float(recall_score(y, prediction, zero_division=0)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, prediction)) if len(np.unique(np.r_[y, prediction])) > 1 else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    if len(np.unique(y)) == 2:
        precision_curve, recall_curve, _ = precision_recall_curve(y, score)
        result.update({
            "AP": float(average_precision_score(y, score)),
            "AUPRC_trapezoid": float(auc(recall_curve, precision_curve)),
            "AUROC": float(roc_auc_score(y, score)),
        })
    else:
        result.update({"AP": None, "AUPRC_trapezoid": None, "AUROC": None})

    return result


def ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    cutoffs: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> Dict[str, Any]:
    """Compute per-disease candidate ranking metrics (Recall@K, NDCG@K, AP).

    Ties are resolved using expected gain within the tied score block.

    Parameters
    ----------
    labels : np.ndarray of shape (N,)
        Binary indicator of held-out positive candidate genes.
    scores : np.ndarray of shape (N,)
        Predicted association scores for candidates.
    cutoffs : sequence of int, default=(1, 5, 10, 20, 50, 100)
        Rank cutoffs K for Recall@K and NDCG@K.

    Returns
    -------
    result : dict
    """
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)

    if y.ndim != 1 or y.shape != score.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Expected matching 1D binary labels and finite scores")
    if len(y) == 0 or y.sum() == 0:
        raise ValueError("Ranking evaluation requires at least one positive candidate")

    order = np.argsort(-score, kind="stable")
    sorted_y, sorted_score = y[order], score[order]

    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    ends = np.r_[starts[1:], len(y)]
    sizes = ends - starts

    positives = np.add.reduceat(sorted_y, starts)
    cumulative = np.cumsum(positives)
    ap = float(np.sum(positives / y.sum() * cumulative / ends))

    expected_gain = np.repeat(positives / sizes, sizes)
    hits = np.cumsum(expected_gain)
    dcg = np.cumsum(expected_gain / np.log2(np.arange(len(y)) + 2))
    ideal = np.cumsum(1.0 / np.log2(np.arange(int(y.sum())) + 2))

    res: Dict[str, Any] = {
        "AP": ap,
        "candidates": len(y),
        "positives": int(y.sum()),
    }
    for k in cutoffs:
        if k < 1:
            continue
        n = min(k, len(y))
        res[f"Recall@{k}"] = float(hits[n - 1] / y.sum())
        res[f"NDCG@{k}"] = float(dcg[n - 1] / ideal[min(n, int(y.sum())) - 1])

    return res
