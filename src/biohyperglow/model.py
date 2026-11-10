"""BioHyperGLOW single-branch neural module.

Implements the mixed-membership disease-gene representation, activity multipliers,
and numerically stable Bernoulli-Poisson association loss.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BioHyperGLOW(nn.Module):
    """Single-branch mixed-membership model with activity scaling.

    Each entity (gene and disease) is mapped to a probability simplex over H hypernodes:
      Q_g = softmax(theta_g) in Delta^{H-1}
      B_d = softmax(log(B0_d) + delta_d) in Delta^{H-1}

    The predicted association rate (intensity) for pair (d, g) is:
      lambda_{dg} = exp(clip(s_g + t_d - mean(t), -20, 20)) * sum_{h=1}^H B_{dh} Q_{gh}

    Under the Bernoulli-Poisson link, the marginal association probability is:
      p_{dg} = 1 - exp(-lambda_{dg}) = -expm1(-lambda_{dg})

    Parameters
    ----------
    b0 : array-like of shape (n_diseases, H)
        Row-normalized prior disease memberships, strictly positive.
    gene_initial : array-like of shape (n_genes, H)
        Initial gene logits (e.g., from log structural target profiles or random normal).
    mode : {'learned', 'free', 'frozen', 'activity'}, default='learned'
        - 'learned': gene logits and disease residuals are both optimized.
        - 'free': association-only branch with uniform reference prior.
        - 'frozen': disease memberships held fixed at b0 (delta frozen at 0).
        - 'activity': hypernode memberships frozen; only activities are optimized.
    """

    def __init__(
        self,
        b0: Union[np.ndarray, torch.Tensor],
        gene_initial: Union[np.ndarray, torch.Tensor],
        mode: str = "learned",
    ) -> None:
        super().__init__()
        valid_modes = {"learned", "free", "frozen", "activity"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {sorted(valid_modes)}")

        b0_arr = np.asarray(b0, dtype=np.float32)
        if np.any(b0_arr <= 0):
            raise ValueError("Prior memberships B0 must be strictly positive")
        if not np.allclose(b0_arr.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Prior memberships B0 must be row-stochastic (sum to 1)")

        nd, h = b0_arr.shape
        gene_arr = np.asarray(gene_initial, dtype=np.float32)
        ng = len(gene_arr)
        if gene_arr.shape != (ng, h):
            raise ValueError(f"Invalid gene initialization shape {gene_arr.shape}; expected ({ng}, {h})")

        self.nd = nd
        self.ng = ng
        self.h = h
        self.mode = mode

        # Register prior log memberships as a persistent buffer
        self.register_buffer("log_b0", torch.from_numpy(np.log(b0_arr)))

        # Gene membership logits theta (trainable in all modes except 'activity')
        requires_grad_theta = mode != "activity"
        self.theta = nn.Parameter(
            torch.as_tensor(gene_arr, dtype=torch.float32).clone(),
            requires_grad=requires_grad_theta,
        )

        # Disease membership residual delta (trainable only in 'learned' and 'free' modes; frozen at 0 in 'frozen' and 'activity')
        requires_grad_delta = mode in ("learned", "free")
        self.delta = nn.Parameter(
            torch.zeros(nd, h, dtype=torch.float32),
            requires_grad=requires_grad_delta,
        )

        # Gene log-activity s_g, initialized to log(h * ln(2))
        self.log_s = nn.Parameter(torch.full((ng,), math.log(h * math.log(2)), dtype=torch.float32))

        # Disease log-activity t_d, initialized to 0
        self.log_t = nn.Parameter(torch.zeros(nd, dtype=torch.float32))

    def log_memberships(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log memberships for genes (log Q) and diseases (log B).

        Returns
        -------
        log_q : torch.Tensor of shape (n_genes, H)
        log_b : torch.Tensor of shape (n_diseases, H)
        """
        log_q = F.log_softmax(self.theta, dim=1)
        log_b = F.log_softmax(self.log_b0 + self.delta, dim=1)
        return log_q, log_b

    def score(
        self,
        d: torch.Tensor,
        g: torch.Tensor,
        logq: Optional[torch.Tensor] = None,
        logb: Optional[torch.Tensor] = None,
        components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute log association intensity for disease indices d and gene indices g.

        Parameters
        ----------
        d : torch.Tensor of shape (N,)
            Disease indices.
        g : torch.Tensor of shape (N,)
            Gene indices.
        logq : torch.Tensor, optional
            Precomputed gene log memberships.
        logb : torch.Tensor, optional
            Precomputed disease log memberships.
        components : bool, default=False
            If True, also return the additive path contributions phi of shape (N, H).

        Returns
        -------
        log_rate : torch.Tensor of shape (N,)
            Predicted log association intensity log(lambda_{dg}).
        phi : torch.Tensor of shape (N, H), optional
            Per-hypernode path intensity contributions, such that sum_h phi_{dgh} = exp(log_rate).
        """
        if logq is None or logb is None:
            logq, logb = self.log_memberships()

        # Mean-centering disease log-scales fixes the global multiplicative gauge.
        exposure = (self.log_s[g] + self.log_t[d] - self.log_t.mean()).clamp(-20, 20)

        if self.mode == "activity":
            log_u = torch.full_like(exposure, -math.log(self.h))
            parts = None
        else:
            channel = logq[g] + logb[d]
            log_u = torch.logsumexp(channel, dim=1)
            parts = F.softmax(channel, dim=1)

        log_rate = exposure + log_u

        if components:
            rate = torch.exp(log_rate)
            if parts is not None:
                phi = rate[:, None] * parts
            else:
                phi = rate[:, None] * (torch.ones((len(d), self.h), device=d.device, dtype=rate.dtype) / self.h)
            return log_rate, phi

        return log_rate

    @staticmethod
    def association_loss(log_rate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Numerically stable Bernoulli-Poisson negative log-likelihood.

        Loss = (1 - y) * exp(log_rate) - y * log(1 - exp(-exp(log_rate)))

        Parameters
        ----------
        log_rate : torch.Tensor of shape (N,)
            Predicted log association intensity.
        y : torch.Tensor of shape (N,)
            Binary ground truth labels (1 for positive association, 0 for nonpositive).
        """
        safe = log_rate.clamp_min(-20)
        log_p_high = torch.log(-torch.expm1(-torch.exp(safe)))
        log_p = torch.where(log_rate < -20, log_rate, log_p_high)
        return ((1.0 - y) * torch.exp(log_rate) - y * log_p).mean()

    def regularizers(
        self,
        target: Optional[torch.Tensor],
        nodes: torch.Tensor,
        diseases: torch.Tensor,
        logq: torch.Tensor,
        logb: torch.Tensor,
        assigned_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute structural profile KL, ontology prior retention, and scale L2 regularizers.

        Parameters
        ----------
        target : torch.Tensor or None
            Structural target profiles T0 of shape (n_diseases + n_genes, H).
        nodes : torch.Tensor
            Minibatch of sampled node indices (disease and gene indices combined).
        diseases : torch.Tensor
            Minibatch of sampled disease indices.
        logq : torch.Tensor
            Gene log memberships of shape (n_genes, H).
        logb : torch.Tensor
            Disease log memberships of shape (n_diseases, H).
        assigned_mask : torch.Tensor, optional
            Boolean mask indicating diseases with accepted ontology mappings.

        Returns
        -------
        profile_loss : torch.Tensor
        ontology_loss : torch.Tensor
        scale_loss : torch.Tensor
        """
        device = logq.device
        zero = torch.zeros((), device=device, dtype=logq.dtype)

        if target is not None:
            all_log = torch.cat([logb, logq], dim=0)
            profile_loss = F.kl_div(all_log[nodes], target[nodes], reduction="batchmean")
        else:
            profile_loss = zero

        if self.mode == "learned" and target is not None:
            prior = self.log_b0[diseases]
            current_lb = logb[diseases]
            per_row = (prior.exp() * (prior - current_lb)).sum(dim=1)
            if assigned_mask is not None:
                per_row = per_row * assigned_mask[diseases].to(dtype=per_row.dtype)
            ontology_loss = per_row.mean()
        else:
            ontology_loss = zero

        scale_loss = (
            self.log_s.square().mean() + (self.log_t - self.log_t.mean()).square().mean()
        ) / 2.0

        return profile_loss, ontology_loss, scale_loss
