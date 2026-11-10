"""Training workflows with validation-based early stopping and checkpoint selection."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from biohyperglow.data import load_edges_csv, negative_pairs, train_val_test_split
from biohyperglow.dual_space import compute_gate, mix_intensities
from biohyperglow.metrics import pair_metrics, select_threshold
from biohyperglow.model import BioHyperGLOW
from biohyperglow.ontology import build_ontology_prior
from biohyperglow.profiles import compute_structural_targets
from biohyperglow.scoring import BioHyperGLOWModel


class EarlyStopping:
    """Monitor validation AP and stop when improvement ceases for a set patience."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-5) -> None:
        if patience < 1 or min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError(f"Invalid early stopping parameters: patience={patience}, min_delta={min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.reference: Optional[float] = None
        self.bad_checks = 0

    @property
    def stopped(self) -> bool:
        return self.bad_checks >= self.patience

    def update(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite validation metric encountered: {value}")
        if self.stopped:
            raise ValueError("Early stopping monitor has already triggered")

        if self.reference is None or value > self.reference + self.min_delta:
            self.reference = value
            self.bad_checks = 0
        else:
            self.bad_checks += 1

        return self.stopped


def _evaluate_validation(
    model: BioHyperGLOW,
    val_pairs: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """Compute validation log-intensities and Average Precision (AP)."""
    model.eval()
    with torch.no_grad():
        lq, lb = model.log_memberships()
        scores = []
        batch_size = 4096
        for offset in range(0, len(val_pairs), batch_size):
            batch = val_pairs[offset : offset + batch_size]
            d = torch.as_tensor(batch[:, 0], device=device)
            g = torch.as_tensor(batch[:, 1], device=device)
            eta = model.score(d, g, lq, lb)
            scores.append(eta.cpu().numpy())

        score_arr = np.concatenate(scores) if scores else np.empty(0, dtype=float)
        val_metric = pair_metrics(val_labels, score_arr)
        ap = val_metric["AP"] if val_metric["AP"] is not None else 0.0
        return ap, score_arr


def train_single_branch(
    mode: str,
    h_dim: int,
    b0: np.ndarray,
    initial_theta: np.ndarray,
    train_pos: np.ndarray,
    val_pairs: np.ndarray,
    val_labels: np.ndarray,
    n_diseases: int,
    n_genes: int,
    target: Optional[np.ndarray] = None,
    assigned_mask: Optional[np.ndarray] = None,
    epochs: int = 100,
    steps_per_epoch: int = 24,
    batch_size: int = 1024,
    eval_every: int = 5,
    patience: int = 20,
    lr: float = 0.003,
    lambda_profile: float = 0.1,
    lambda_ontology: float = 0.1,
    scale_l2: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], BioHyperGLOW, Dict[str, Any]]:
    """Train a single BioHyperGLOW branch (semantic or free) with validation early stopping."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    model = BioHyperGLOW(b0, initial_theta, mode=mode).to(dev)
    if mode == "free":
        # Free branch initializes random perturbation on disease residual delta
        with torch.no_grad():
            model.delta.copy_(0.1 * torch.randn_like(model.delta))

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    stopping = EarlyStopping(patience=patience, min_delta=1e-5)
    rng = np.random.default_rng(seed)

    target_t = torch.from_numpy(target).to(dev) if target is not None else None
    mask_t = torch.as_tensor(assigned_mask, device=dev) if assigned_mask is not None else None

    all_train_pos = set(map(tuple, train_pos.tolist()))
    best_ap = -1.0
    best_epoch = 0
    best_state = None

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for _ in range(steps_per_epoch):
            # Sample positive training edges
            n_pos = min(batch_size, len(train_pos))
            pos_batch = train_pos[rng.integers(0, len(train_pos), size=n_pos)]
            neg_batch = negative_pairs(n_diseases, n_genes, all_train_pos, n_pos, rng)

            pairs = np.concatenate([pos_batch, neg_batch])
            y = torch.cat([
                torch.ones(len(pos_batch), device=dev),
                torch.zeros(len(neg_batch), device=dev),
            ])

            d = torch.as_tensor(pairs[:, 0], device=dev)
            g = torch.as_tensor(pairs[:, 1], device=dev)

            logq, logb = model.log_memberships()
            log_rate = model.score(d, g, logq, logb)
            loss_assoc = model.association_loss(log_rate, y)

            nodes = torch.as_tensor(rng.integers(0, n_diseases + n_genes, size=batch_size), device=dev)
            diseases_sub = torch.as_tensor(rng.integers(0, n_diseases, size=batch_size), device=dev)

            loss_prof, loss_ont, loss_scale = model.regularizers(
                target_t, nodes, diseases_sub, logq, logb, assigned_mask=mask_t
            )

            loss = loss_assoc + scale_l2 * loss_scale
            if mode != "free" and target_t is not None:
                loss = loss + lambda_profile * loss_prof + lambda_ontology * loss_ont

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            val_ap, val_scores = _evaluate_validation(model, val_pairs, val_labels, dev)
            if val_ap > best_ap:
                best_ap = val_ap
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            history.append({
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "val_ap": float(val_ap),
            })
            if progress:
                progress(f"[{mode}] Epoch {epoch}/{epochs}: loss={np.mean(epoch_losses):.4f}, val_AP={val_ap:.4f} (best={best_ap:.4f} at epoch {best_epoch})")

            stopping.update(val_ap)
            if stopping.stopped:
                if progress:
                    progress(f"[{mode}] Early stopping triggered at epoch {epoch}")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract final embeddings at best checkpoint
    model.eval()
    with torch.no_grad():
        lq, lb = model.log_memberships()
        q = lq.exp().cpu().numpy()
        b = lb.exp().cpu().numpy()
        log_s = model.log_s.cpu().numpy()
        log_t = (model.log_t - model.log_t.mean()).cpu().numpy()

    branch_dict = {
        "q": q,
        "b": b,
        "a": log_s,
        "c": log_t,
        "best_epoch": best_epoch,
        "best_val_ap": best_ap,
    }

    train_info = {
        "best_epoch": best_epoch,
        "best_val_ap": best_ap,
        "history": history,
    }

    return branch_dict, model, train_info


def train_dual_space(
    edges_csv: Union[str, Path],
    ontology_obo: Optional[Union[str, Path]] = None,
    H: int = 512,
    kappa: float = 10.0,
    epochs: int = 200,
    eval_every: int = 5,
    patience: int = 30,
    batch_size: int = 1024,
    lr: float = 0.003,
    lambda_profile: float = 0.1,
    lambda_ontology: float = 0.1,
    device: str = "cpu",
    seed: int = 42,
    val_frac: float = 0.1,
    progress: Optional[Callable[[str], None]] = None,
) -> BioHyperGLOWModel:
    """Train complete BioHyperGLOW dual-space model from association network and ontology.

    Parameters
    ----------
    edges_csv : str or Path
        Path to CSV file with 'disease' and 'gene' columns.
    ontology_obo : str or Path, optional
        Path to MONDO ontology .obo file.
    H : int, default=512
        Coordinate dimension for each branch.
    kappa : float, default=10.0
        Degree-gate scale parameter.
    epochs : int, default=200
    eval_every : int, default=5
    patience : int, default=30
    batch_size : int, default=1024
    lr : float, default=0.003
    lambda_profile : float, default=0.1
    lambda_ontology : float, default=0.1
    device : str, default='cpu'
    seed : int, default=42
    val_frac : float, default=0.1
    progress : callable, optional

    Returns
    -------
    model : BioHyperGLOWModel
        Fitted dual-space model ready for prediction and explanation.
    """
    edges, diseases, genes, _ = load_edges_csv(edges_csv)
    n_d = len(diseases)
    n_g = len(genes)

    if progress:
        progress(f"Loaded {len(edges)} associations across {n_d} diseases and {n_g} genes")

    # Generate leakage-safe train/val split
    splits = train_val_test_split(edges, n_d, n_g, seed=seed, folds=5, val_frac=val_frac)
    fold0 = splits[0]
    train_pos = fold0["train_pos"]
    val_pos = fold0["val_pos"]
    val_neg = fold0["val_neg"]
    val_pairs = np.concatenate([val_pos, val_neg])
    val_labels = np.r_[np.ones(len(val_pos), dtype=int), np.zeros(len(val_neg), dtype=int)]

    # Compute training degree for each disease
    degrees = np.bincount(train_pos[:, 0], minlength=n_d)

    # 1. Prepare Semantic Branch Inputs
    group_ids: List[str] = []
    group_labels: List[str] = []
    mapped = np.zeros(n_d, dtype=bool)

    if ontology_obo is not None:
        if progress:
            progress(f"Building ontology concept clusters (H={H})...")
        b0_sem, group_ids, group_labels, report, records = build_ontology_prior(
            diseases, ontology_obo, H=H, progress=progress
        )
        mapped = np.array([r["status"] == "mapped" for r in records], dtype=bool)
        if progress:
            progress(f"Mapped {report['unique_mapping']}/{n_d} diseases ({report['biological_coverage']*100:.1f}%)")

        if progress:
            progress("Computing fixed graph structural targets T0...")
        t0, _ = compute_structural_targets(train_pos, n_d, n_g, b0_sem, progress=None)
        initial_theta_sem = np.log(np.maximum(t0[n_d:], 1e-30))
    else:
        # Fallback if no ontology provided: uniform prior and random initial theta
        b0_sem = np.full((n_d, H), 1.0 / H, dtype=np.float32)
        initial_theta_sem = np.zeros((n_g, H), dtype=np.float32)
        t0 = None

    # 2. Train Semantic Branch
    if progress:
        progress("Starting Semantic Branch optimization...")
    sem_branch, _, sem_info = train_single_branch(
        mode="learned" if ontology_obo is not None else "free",
        h_dim=H,
        b0=b0_sem,
        initial_theta=initial_theta_sem,
        train_pos=train_pos,
        val_pairs=val_pairs,
        val_labels=val_labels,
        n_diseases=n_d,
        n_genes=n_g,
        target=t0,
        assigned_mask=mapped,
        epochs=epochs,
        eval_every=eval_every,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        lambda_profile=lambda_profile,
        lambda_ontology=lambda_ontology,
        device=device,
        seed=seed,
        progress=progress,
    )
    sem_branch["group_ids"] = group_ids
    sem_branch["group_labels"] = group_labels

    # 3. Train Free (Association-Only) Branch
    if progress:
        progress("Starting Free (Association-Only) Branch optimization...")
    b0_free = np.full((n_d, H), 1.0 / H, dtype=np.float32)
    rng_free = np.random.default_rng(seed + 99991)
    initial_theta_free = rng_free.normal(0.0, 0.1, (n_g, H)).astype(np.float32)

    free_branch, _, free_info = train_single_branch(
        mode="free",
        h_dim=H,
        b0=b0_free,
        initial_theta=initial_theta_free,
        train_pos=train_pos,
        val_pairs=val_pairs,
        val_labels=val_labels,
        n_diseases=n_d,
        n_genes=n_g,
        target=None,
        epochs=epochs,
        eval_every=eval_every,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed + 1,
        progress=progress,
    )

    # 4. Construct Dual-Space Gate & Select Validation Threshold
    gate_alpha = compute_gate(mapped, degrees, kappa=kappa)

    # Compute validation predictions from both restored branches
    d_val = val_pairs[:, 0]
    g_val = val_pairs[:, 1]
    from biohyperglow.dual_space import branch_scores

    ss_val = branch_scores(sem_branch, d_val, g_val)
    sf_val = branch_scores(free_branch, d_val, g_val)
    weights_val = gate_alpha[d_val]
    fused_val = mix_intensities(ss_val, sf_val, weights_val)

    # Validation-based F1 threshold selection
    best_threshold = select_threshold(val_labels, fused_val)
    val_metrics = pair_metrics(val_labels, fused_val, threshold=best_threshold)

    if progress:
        progress(
            f"Dual-space validation AP: {val_metrics['AP']:.4f}, AUROC: {val_metrics['AUROC']:.4f}, "
            f"F1: {val_metrics['F1']:.4f} (threshold={best_threshold:.4f})"
        )

    model = BioHyperGLOWModel(
        semantic=sem_branch,
        free=free_branch,
        diseases=diseases,
        genes=genes,
        mapped=mapped,
        degrees=degrees,
        kappa=kappa,
        threshold=best_threshold,
    )

    return model
