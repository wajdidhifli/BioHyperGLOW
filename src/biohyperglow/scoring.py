"""Scoring, candidate prioritization, and model checkpoint persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from biohyperglow.dual_space import (
    branch_scores,
    compute_gate,
    intensity_to_probability,
    mix_intensities,
    rate_to_probability,
)


class BioHyperGLOWModel:
    """Loaded or fitted dual-space BioHyperGLOW model for prediction, ranking, and explanation.

    Combines the semantic branch S and association-only structural branch F using the
    adaptive degree- and mapping-aware gate:
      alpha_d = m_d * (kappa / (kappa + degree_d))

    Parameters
    ----------
    semantic : dict
        Semantic branch weights with 'b', 'q', 'a', 'c', and optional 'group_ids', 'group_labels'.
    free : dict
        Free branch weights with 'b', 'q', 'a', 'c'.
    diseases : sequence of str
        List of disease identifiers.
    genes : sequence of str
        List of gene identifiers.
    mapped : sequence of bool or int
        Binary indicator of accepted ontology mapping for each disease.
    degrees : sequence of int
        Positive training degree for each disease.
    kappa : float, default=10.0
        Reference degree scale parameter.
    threshold : float, optional
        Classification cutoff on log-intensity scale (selected on validation split).
    """

    def __init__(
        self,
        semantic: Dict[str, Any],
        free: Dict[str, Any],
        diseases: Sequence[str],
        genes: Sequence[str],
        mapped: Sequence[Union[bool, int]],
        degrees: Sequence[int],
        kappa: float = 10.0,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.diseases = [str(d) for d in diseases]
        self.genes = [str(g) for g in genes]
        self.disease_index = {d: i for i, d in enumerate(self.diseases)}
        self.gene_index = {g: i for i, g in enumerate(self.genes)}

        self.mapped = np.asarray(mapped, dtype=bool)
        self.degrees = np.asarray(degrees, dtype=int)
        self.kappa = float(kappa)
        self.threshold = float(threshold) if threshold is not None else None
        self.metadata = dict(metadata or {})

        self.alpha = compute_gate(self.mapped, self.degrees, self.kappa)

        # Store branches
        self.semantic = {
            "b": np.asarray(semantic["b"], dtype=float),
            "q": np.asarray(semantic["q"], dtype=float),
            "a": np.asarray(semantic["a"], dtype=float),
            "c": np.asarray(semantic["c"], dtype=float),
            "group_ids": list(semantic.get("group_ids", [])),
            "group_labels": list(semantic.get("group_labels", [])),
        }
        self.free = {
            "b": np.asarray(free["b"], dtype=float),
            "q": np.asarray(free["q"], dtype=float),
            "a": np.asarray(free["a"], dtype=float),
            "c": np.asarray(free["c"], dtype=float),
        }

        self.H = self.semantic["q"].shape[1]
        if self.free["q"].shape[1] != self.H:
            raise ValueError(
                f"Dimension mismatch between semantic (H={self.H}) and free (H={self.free['q'].shape[1]})"
            )

    def _resolve_indices(
        self,
        diseases: Union[str, Sequence[str]],
        genes: Union[str, Sequence[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert disease and gene string identifiers to integer indices."""
        if isinstance(diseases, str):
            d_list = [diseases]
        else:
            d_list = list(diseases)

        if isinstance(genes, str):
            g_list = [genes]
        else:
            g_list = list(genes)

        if len(d_list) != len(g_list):
            raise ValueError(f"Lengths of disease ({len(d_list)}) and gene ({len(g_list)}) inputs must match")

        d_indices = np.array([self.disease_index[d] for d in d_list], dtype=int)
        g_indices = np.array([self.gene_index[g] for g in g_list], dtype=int)
        return d_indices, g_indices

    def predict_log_intensity(
        self,
        diseases: Union[str, Sequence[str]],
        genes: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """Predict fused log association intensity log(lambda_{dg}) for pairs."""
        d, g = self._resolve_indices(diseases, genes)
        ss = branch_scores(self.semantic, d, g)
        sf = branch_scores(self.free, d, g)
        weights = self.alpha[d]
        return mix_intensities(ss, sf, weights)

    def predict_intensity(
        self,
        diseases: Union[str, Sequence[str]],
        genes: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """Predict fused association intensity lambda_{dg} (Poisson rate) for pairs."""
        log_rates = self.predict_log_intensity(diseases, genes)
        return np.exp(log_rates)

    def predict_probability(
        self,
        diseases: Union[str, Sequence[str]],
        genes: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """Predict decoded association probability p_{dg} = 1 - exp(-lambda_{dg}) for pairs."""
        log_rates = self.predict_log_intensity(diseases, genes)
        return rate_to_probability(log_rates)

    def predict(
        self,
        diseases: Union[str, Sequence[str]],
        genes: Union[str, Sequence[str]],
    ) -> Dict[str, Any]:
        """Comprehensive prediction returning log-intensity, intensity, probability, and classification.

        Returns
        -------
        dict containing:
        - 'disease': list of str
        - 'gene': list of str
        - 'log_intensity': np.ndarray
        - 'intensity': np.ndarray
        - 'probability': np.ndarray
        - 'binary_prediction': np.ndarray of bool (if threshold is set)
        - 'semantic_intensity': np.ndarray
        - 'structural_intensity': np.ndarray
        - 'gate_alpha': np.ndarray
        """
        d, g = self._resolve_indices(diseases, genes)
        ss = branch_scores(self.semantic, d, g)
        sf = branch_scores(self.free, d, g)
        weights = self.alpha[d]
        fused_log = mix_intensities(ss, sf, weights)
        fused_rate = np.exp(fused_log)
        prob = rate_to_probability(fused_log)

        d_str = [self.diseases[idx] for idx in d]
        g_str = [self.genes[idx] for idx in g]

        sem_int = weights * np.exp(ss)
        str_int = (1.0 - weights) * np.exp(sf)

        result: Dict[str, Any] = {
            "disease": d_str,
            "gene": g_str,
            "log_intensity": fused_log,
            "intensity": fused_rate,
            "probability": prob,
            "semantic_intensity": sem_int,
            "structural_intensity": str_int,
            "gate_alpha": weights,
        }

        if self.threshold is not None:
            result["binary_prediction"] = fused_log >= self.threshold

        return result

    def rank_genes(
        self,
        disease: str,
        candidate_genes: Optional[Sequence[str]] = None,
        exclude_genes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank candidate genes for a specified disease.

        Parameters
        ----------
        disease : str
            Query disease identifier.
        candidate_genes : sequence of str, optional
            Subset of candidate genes to score. Defaults to all model genes.
        exclude_genes : sequence of str, optional
            Genes to exclude from ranking (e.g. known training positives).

        Returns
        -------
        ranked_genes : list of dict
            Rank-ordered candidates with 'rank', 'gene', 'log_intensity', 'intensity', 'probability'.
        """
        if disease not in self.disease_index:
            raise KeyError(f"Disease '{disease}' not found in model vocabulary")

        d = self.disease_index[disease]
        weight = float(self.alpha[d])

        ss = branch_scores(self.semantic, [d])[0]  # shape (ng,)
        sf = branch_scores(self.free, [d])[0]  # shape (ng,)
        scores = mix_intensities(ss, sf, weight)

        excluded_set = set(exclude_genes or [])
        if candidate_genes is not None:
            eligible_indices = [
                self.gene_index[g]
                for g in candidate_genes
                if g in self.gene_index and g not in excluded_set
            ]
        else:
            eligible_indices = [
                i for i, g in enumerate(self.genes) if g not in excluded_set
            ]

        # Sort eligible genes by descending score, breaking ties by gene identifier
        eligible_indices.sort(key=lambda idx: (-scores[idx], self.genes[idx]))

        results = []
        for rank, g_idx in enumerate(eligible_indices, 1):
            sc = float(scores[g_idx])
            results.append({
                "rank": rank,
                "disease": disease,
                "gene": self.genes[g_idx],
                "log_intensity": sc,
                "intensity": float(np.exp(sc)),
                "probability": float(rate_to_probability(sc)),
            })

        return results

    def explain(
        self,
        disease: str,
        gene: str,
        top_k: Optional[int] = None,
    ):
        """Explain the predicted link strength for a disease-gene pair."""
        from biohyperglow.explanation import explain_association
        return explain_association(self, disease, gene, top_k=top_k)

    def save(self, path: Union[str, Path]) -> None:
        """Save complete model checkpoint to an .npz archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            **self.metadata,
            "version": "0.1.0",
            "H": self.H,
            "kappa": self.kappa,
            "threshold": self.threshold,
            "group_ids": self.semantic["group_ids"],
            "group_labels": self.semantic["group_labels"],
        }

        np.savez_compressed(
            p,
            metadata=json.dumps(meta),
            diseases=np.array(self.diseases),
            genes=np.array(self.genes),
            mapped=self.mapped,
            degrees=self.degrees,
            alpha=self.alpha,
            semantic_b=self.semantic["b"],
            semantic_q=self.semantic["q"],
            semantic_a=self.semantic["a"],
            semantic_c=self.semantic["c"],
            free_b=self.free["b"],
            free_q=self.free["q"],
            free_a=self.free["a"],
            free_c=self.free["c"],
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> BioHyperGLOWModel:
        """Load a saved BioHyperGLOWModel checkpoint."""
        p = Path(path)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            diseases = z["diseases"].tolist()
            genes = z["genes"].tolist()
            mapped = z["mapped"]
            degrees = z["degrees"]
            kappa = float(meta["kappa"])
            threshold = meta.get("threshold")

            semantic = {
                "b": z["semantic_b"],
                "q": z["semantic_q"],
                "a": z["semantic_a"],
                "c": z["semantic_c"],
                "group_ids": meta.get("group_ids", []),
                "group_labels": meta.get("group_labels", []),
            }
            free = {
                "b": z["free_b"],
                "q": z["free_q"],
                "a": z["free_a"],
                "c": z["free_c"],
            }

            return cls(
                semantic=semantic,
                free=free,
                diseases=diseases,
                genes=genes,
                mapped=mapped,
                degrees=degrees,
                kappa=kappa,
                threshold=threshold,
                metadata=meta,
            )

    @classmethod
    def from_embeddings(
        cls,
        semantic_path: Union[str, Path],
        free_path: Union[str, Path],
        mapped: Sequence[Union[bool, int]],
        degrees: Sequence[int],
        kappa: float = 10.0,
        threshold: Optional[float] = None,
    ) -> BioHyperGLOWModel:
        """Construct BioHyperGLOWModel from separate semantic and free embeddings.npz files.

        Compatible with the saved research workspace checkpoints.
        """
        with np.load(semantic_path, allow_pickle=False) as zs:
            semantic = {
                "b": zs["disease_memberships"].astype(float),
                "q": zs["gene_memberships"].astype(float),
                "a": zs["gene_log_activity"].astype(float),
                "c": zs["centered_disease_log_activity"].astype(float),
                "group_ids": zs["group_ids"].tolist() if "group_ids" in zs else [],
                "group_labels": zs["group_labels"].tolist() if "group_labels" in zs else [],
            }
            diseases = zs["diseases"].tolist()
            genes = zs["genes"].tolist()

        with np.load(free_path, allow_pickle=False) as zf:
            free = {
                "b": zf["disease_memberships"].astype(float),
                "q": zf["gene_memberships"].astype(float),
                "a": zf["gene_log_activity"].astype(float),
                "c": zf["centered_disease_log_activity"].astype(float),
            }
            if zf["diseases"].tolist() != diseases or zf["genes"].tolist() != genes:
                raise ValueError("Disease or gene order differs between semantic and free branches")

        return cls(
            semantic=semantic,
            free=free,
            diseases=diseases,
            genes=genes,
            mapped=mapped,
            degrees=degrees,
            kappa=kappa,
            threshold=threshold,
        )
