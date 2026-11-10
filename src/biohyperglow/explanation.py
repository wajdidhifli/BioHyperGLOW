"""Intrinsic explanation of predicted disease-gene association strength.

Decomposes the predicted association intensity lambda_{dg} into exact, nonnegative
contributions from 2H hypernodes (H semantic coordinates and H association-only structural coordinates):

  lambda_{dg} = sum_{h=1}^H phi^S_{dgh} + sum_{h=1}^H phi^F_{dgh}

where:
  phi^S_{dgh} = alpha_d * e^S_{dg} * B^S_{dh} * Q^S_{gh}
  phi^F_{dgh} = (1 - alpha_d) * e^F_{dg} * B^F_{dh} * Q^F_{gh}

Under Proposition 1, this exact sum is identical to evaluating normalized concatenated embeddings.

NOTE: Free structural coordinates represent latent association patterns learned directly from
network topology and must NOT be presented or interpreted as established biological mechanisms.
Semantic coordinates reflect concept clusters derived from the disease ontology (MONDO).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class CoordinateContribution:
    """Contribution of an individual hypernode coordinate to association strength."""

    coordinate_index: int
    branch: str  # 'semantic' or 'structural'
    branch_coordinate: int
    label: str
    intensity_contribution: float
    contribution_share: float  # Fraction of total intensity
    gene_membership: float  # Q_{gh}
    disease_membership: float  # B_{dh}
    exposure: float  # exp(clip(s_g + t_d, -20, 20))
    biological_interpretation_valid: bool


@dataclass
class ExplanationResult:
    """Complete additive decomposition of predicted link strength for a disease-gene pair."""

    disease: str
    gene: str
    predicted_intensity: float  # lambda_{dg}
    predicted_probability: float  # p_{dg} = 1 - exp(-lambda_{dg})
    log_intensity: float
    binary_prediction: Optional[bool]
    classification_threshold: Optional[float]
    gate_alpha: float  # alpha_d
    disease_training_degree: int
    is_ontology_mapped: bool
    semantic_intensity: float
    structural_intensity: float
    semantic_share: float
    structural_share: float
    reconstructed_intensity: float
    max_reconstruction_error: float
    components: List[CoordinateContribution]

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation result to a standard dictionary."""
        return asdict(self)

    def summary(self, top_k: int = 5) -> str:
        """Return a human-readable text summary of the explanation."""
        lines = [
            f"BioHyperGLOW Explanation: Disease {self.disease} <-> Gene {self.gene}",
            "=" * 60,
            f"Predicted intensity (Poisson rate lambda): {self.predicted_intensity:.6f}",
            f"Decoded probability (Bernoulli p):        {self.predicted_probability:.6f}",
            f"Log intensity:                           {self.log_intensity:.6f}",
        ]
        if self.classification_threshold is not None:
            pred_str = "POSITIVE" if self.binary_prediction else "NEGATIVE"
            lines.append(f"Classification (threshold {self.classification_threshold:.4f}): {pred_str}")

        lines.extend([
            f"Disease training degree:                 {self.disease_training_degree}",
            f"Ontology mapping status:                 {'Mapped' if self.is_ontology_mapped else 'Unmapped'}",
            f"Gate weight alpha_d:                     {self.gate_alpha:.4f} (Semantic {self.semantic_share * 100:.1f}%, Structural {self.structural_share * 100:.1f}%)",
            f"Exact reconstruction error:              {self.max_reconstruction_error:.2e}",
            "",
            f"Top {top_k} Contributing Hypernodes (out of {len(self.components)}):",
            "-" * 60,
        ])

        for comp in self.components[:top_k]:
            tag = "[Semantic]" if comp.branch == "semantic" else "[Latent Structural]"
            lines.append(
                f"  {tag:<19} #{comp.coordinate_index:03d} {comp.label[:36]:<36} | "
                f"Contrib: {comp.intensity_contribution:8.5f} ({comp.contribution_share * 100:5.1f}%)"
            )

        lines.extend([
            "-" * 60,
            "Notice: Semantic hypernodes reflect MONDO-derived concept clusters.",
            "Latent structural hypernodes represent unconstrained topological patterns,",
            "not established biological mechanisms.",
        ])
        return "\n".join(lines)


def explain_association(
    model: Any,
    disease: str,
    gene: str,
    top_k: Optional[int] = None,
) -> ExplanationResult:
    """Compute the exact additive hypernode path breakdown for a disease-gene pair.

    Parameters
    ----------
    model : BioHyperGLOWModel
        Fitted or loaded BioHyperGLOW dual-space model.
    disease : str
        Disease identifier (e.g. UMLS CUI 'C0006142').
    gene : str
        Gene identifier (e.g. Entrez ID '101' or symbol).
    top_k : int, optional
        If specified, retains only the top_k components in the returned list.
        All components are used to verify exact reconstruction.

    Returns
    -------
    result : ExplanationResult
    """
    if disease not in model.disease_index:
        raise KeyError(f"Disease identifier '{disease}' not in model vocabulary")
    if gene not in model.gene_index:
        raise KeyError(f"Gene identifier '{gene}' not in model vocabulary")

    d = model.disease_index[disease]
    g = model.gene_index[gene]

    alpha = float(model.alpha[d])
    weight_s = alpha
    weight_f = 1.0 - alpha

    # Semantic branch values
    q_s = model.semantic["q"][g]
    b_s = model.semantic["b"][d]
    a_s = float(model.semantic["a"][g])
    c_s = float(model.semantic["c"][d])
    exp_s = float(np.exp(np.clip(a_s + c_s, -20.0, 20.0)))

    # Free branch values
    q_f = model.free["q"][g]
    b_f = model.free["b"][d]
    a_f = float(model.free["a"][g])
    c_f = float(model.free["c"][d])
    exp_f = float(np.exp(np.clip(a_f + c_f, -20.0, 20.0)))

    h = model.H
    # Exact branch path contributions
    paths_s = weight_s * exp_s * b_s * q_s  # shape (H,)
    paths_f = weight_f * exp_f * b_f * q_f  # shape (H,)

    intensity_s = float(paths_s.sum())
    intensity_f = float(paths_f.sum())
    total_intensity = intensity_s + intensity_f

    # Guard against zero intensity for division
    safe_tot = total_intensity if total_intensity > 0 else 1.0
    decoded_prob = float(-np.expm1(-total_intensity))
    log_int = float(np.log(np.maximum(total_intensity, np.finfo(float).tiny)))

    # Cross-verify against normalized concatenation (Proposition 1)
    u_g = np.concatenate([q_s, q_f])
    v_d = np.concatenate([weight_s * b_s, weight_f * b_f])
    z_g = u_g.sum()
    z_d = v_d.sum()
    q_tilde = u_g / z_g
    b_tilde = v_d / z_d
    scale = z_g * z_d

    prop1_paths = np.concatenate([
        scale * exp_s * q_tilde[:h] * b_tilde[:h],
        scale * exp_f * q_tilde[h:] * b_tilde[h:],
    ])
    reconstructed_int = float(prop1_paths.sum())
    reconstruction_err = float(abs(reconstructed_int - total_intensity))
    if reconstruction_err > 1e-10:
        raise RuntimeError(f"Explanation reconstruction error {reconstruction_err:.2e} exceeds tolerance")

    # Build coordinate components
    components: List[CoordinateContribution] = []
    semantic_labels = model.semantic.get("group_labels", [f"MONDO Cluster {k+1}" for k in range(h - 1)] + ["Unassigned fallback"])

    # Semantic coordinates 1..H
    for i in range(h):
        lbl = semantic_labels[i] if i < len(semantic_labels) else f"Semantic coordinate {i + 1}"
        phi = float(paths_s[i])
        components.append(CoordinateContribution(
            coordinate_index=i + 1,
            branch="semantic",
            branch_coordinate=i + 1,
            label=lbl,
            intensity_contribution=phi,
            contribution_share=phi / safe_tot,
            gene_membership=float(q_s[i]),
            disease_membership=float(b_s[i]),
            exposure=exp_s,
            biological_interpretation_valid=True,
        ))

    # Structural coordinates H+1..2H
    for i in range(h):
        phi = float(paths_f[i])
        components.append(CoordinateContribution(
            coordinate_index=h + i + 1,
            branch="structural",
            branch_coordinate=i + 1,
            label=f"Latent structural group {i + 1:03d} (topological)",
            intensity_contribution=phi,
            contribution_share=phi / safe_tot,
            gene_membership=float(q_f[i]),
            disease_membership=float(b_f[i]),
            exposure=exp_f,
            biological_interpretation_valid=False,
        ))

    # Sort components by descending intensity contribution
    components.sort(key=lambda c: -c.intensity_contribution)

    binary_pred = None
    if model.threshold is not None:
        binary_pred = log_int >= model.threshold

    result_components = components[:top_k] if top_k is not None else components

    return ExplanationResult(
        disease=disease,
        gene=gene,
        predicted_intensity=total_intensity,
        predicted_probability=decoded_prob,
        log_intensity=log_int,
        binary_prediction=binary_pred,
        classification_threshold=model.threshold,
        gate_alpha=alpha,
        disease_training_degree=int(model.degrees[d]),
        is_ontology_mapped=bool(model.mapped[d]),
        semantic_intensity=intensity_s,
        structural_intensity=intensity_f,
        semantic_share=intensity_s / safe_tot,
        structural_share=intensity_f / safe_tot,
        reconstructed_intensity=reconstructed_int,
        max_reconstruction_error=reconstruction_err,
        components=result_components,
    )
