"""Ontology parsing, hierarchical concept clustering, and soft membership priors.

Builds ontology-guided disease membership priors B0 from an OBO ontology (such as MONDO)
and disease cross-references (such as UMLS CUIs).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.sparse import csr_matrix

ORGANIZATIONAL_ROOTS = frozenset({
    "MONDO:0000001",
    "MONDO:0700096",
    "MONDO:7770006",
    "MONDO:7770007",
    "MONDO:7770008",
})


def read_obo(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Parse an OBO ontology file for terms, ancestors, and UMLS cross-references.

    Parameters
    ----------
    path : str or Path
        Path to the .obo ontology file.

    Returns
    -------
    terms : dict
        Mapping from term ID (e.g. 'MONDO:0000001') to dictionary with:
        - 'id': str
        - 'name': str
        - 'parents': list of parent term IDs
        - 'umls': list of UMLS CUI strings (e.g. 'C0006142')
        - 'obsolete': bool
    """
    terms: Dict[str, Dict[str, Any]] = {}
    term: Optional[Dict[str, Any]] = None

    def finish():
        if term and term.get("id", "").startswith("MONDO:"):
            terms[term["id"]] = term

    with open(path, encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line == "[Term]":
                finish()
                term = {"parents": [], "umls": [], "obsolete": False}
            elif line.startswith("["):
                finish()
                term = None
            elif term is not None:
                if line.startswith("id: "):
                    term["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    term["name"] = line[6:].strip()
                elif line.startswith("is_a: MONDO:"):
                    term["parents"].append(line[6:].split()[0].strip())
                elif line.startswith("xref: UMLS:"):
                    cui = line[11:].split()[0].strip('"').strip()
                    term["umls"].append(cui)
                elif line == "is_obsolete: true":
                    term["obsolete"] = True
        finish()

    if not terms:
        raise ValueError(f"No MONDO terms parsed from {path}")
    return terms


def _build_features(
    diseases: List[str],
    terms: Dict[str, Dict[str, Any]],
) -> Tuple[List[List[str]], List[str], List[Set[str]], List[str], csr_matrix, np.ndarray]:
    """Map disease identifiers to unique MONDO terms and compute IDF feature matrix."""
    xrefs = defaultdict(set)
    for tid, term in terms.items():
        if not term["obsolete"]:
            for cui in term["umls"]:
                xrefs[cui].add(tid)

    mappings = [sorted(xrefs[str(d).removeprefix("UMLS:").strip()]) for d in diseases]
    term_ids = sorted({m[0] for m in mappings if len(m) == 1})
    if not term_ids:
        raise ValueError("No uniquely mapped active MONDO terms found for the supplied diseases")

    @lru_cache(None)
    def ancestors(tid: str) -> Set[str]:
        seen = {tid}
        stack = [tid]
        while stack:
            v = stack.pop()
            for parent in terms.get(v, {}).get("parents", []):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    sets = [(ancestors(tid) - ORGANIZATIONAL_ROOTS) | {tid} for tid in term_ids]
    vocabulary = sorted(set().union(*sets))
    index = {tid: i for i, tid in enumerate(vocabulary)}

    rows, columns = [], []
    for i, feature_set in enumerate(sets):
        for tid in sorted(feature_set):
            rows.append(i)
            columns.append(index[tid])

    matrix = csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(len(term_ids), len(vocabulary)),
    )

    # Document frequency over unique mapped MONDO concepts
    df = np.asarray(matrix.sum(axis=0)).ravel()
    weights = 1.0 + np.log((1.0 + len(term_ids)) / (1.0 + df))
    return mappings, term_ids, sets, vocabulary, matrix, weights


def _condensed_distances(
    matrix: csr_matrix,
    weights: np.ndarray,
    progress: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Compute condensed distance matrix using IDF-weighted ancestor Jaccard similarity."""
    n = matrix.shape[0]
    weighted = matrix.multiply(weights).tocsr()
    mass = np.asarray(weighted.sum(axis=1)).ravel()
    distances = np.empty(n * (n - 1) // 2, dtype=np.float64)

    block_size = 128
    for begin in range(0, n, block_size):
        end = min(n, begin + block_size)
        intersection = (weighted[begin:end] @ matrix.T).toarray()
        union = mass[begin:end, None] + mass[None, :] - intersection
        similarity = np.clip(intersection / np.maximum(union, 1e-12), 0.0, 1.0)
        for local, i in enumerate(range(begin, end)):
            start = n * i - i * (i + 1) // 2
            distances[start : start + n - i - 1] = 1.0 - similarity[local, i + 1 :]
        if progress:
            progress(f"Computed ontology distances for {end}/{n} concepts")

    return distances


def _similarity_row(distances: np.ndarray, n: int, i: int) -> np.ndarray:
    row = np.ones(n, dtype=np.float64)
    left = np.arange(i, dtype=np.int64)
    offsets = n * left - left * (left + 1) // 2 + i - left - 1
    row[:i] -= distances[offsets]
    start = n * i - i * (i + 1) // 2
    row[i + 1 :] -= distances[start : start + n - i - 1]
    return row


def build_ontology_prior(
    diseases: List[str],
    ontology_path: Union[str, Path],
    H: int,
    temperature: float = 0.035,
    epsilon: float = 0.01,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """Build soft ontology membership prior B0 with Laplace smoothing.

    Parameters
    ----------
    diseases : list of str
        List of disease identifiers in benchmark order.
    ontology_path : str or Path
        Path to the MONDO .obo ontology file.
    H : int
        Total coordinate dimension for the semantic branch (H >= 2).
        Contains H - 1 ontology clusters plus 1 unassigned/fallback coordinate.
    temperature : float, default=0.035
        Softmax temperature for assigning soft cluster affinities.
    epsilon : float, default=0.01
        Laplace smoothing parameter in (0, 1).
    progress : callable, optional
        Optional logging or status callback.

    Returns
    -------
    b0 : np.ndarray of shape (n_diseases, H)
        Row-normalized prior disease memberships, strictly positive.
    group_ids : list of str
        Cluster identifiers of length H (H-1 cluster IDs + 'UNASSIGNED').
    group_labels : list of str
        Human-readable labels of length H.
    report : dict
        Mapping coverage and clustering statistics.
    records : list of dict
        Per-disease mapping records with disease, status, and mapped MONDO terms.
    """
    diseases_str = [str(d).strip() for d in diseases]
    if len(diseases_str) != len(set(diseases_str)):
        raise ValueError("Disease identifiers must be unique")
    if not isinstance(H, int) or H < 2:
        raise ValueError(f"H must be an integer >= 2 (got {H}) to accommodate clusters and fallback")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite and positive, got {temperature}")

    terms = read_obo(ontology_path)
    mappings, term_ids, sets, vocabulary, matrix, weights = _build_features(diseases_str, terms)
    n = len(term_ids)

    k_clusters = H - 1
    if k_clusters > n:
        raise ValueError(
            f"Requested H-1={k_clusters} clusters, but only {n} unique mapped MONDO terms are available. "
            f"Reduce H to at most {n + 1}."
        )

    distances = _condensed_distances(matrix, weights, progress=progress)
    tree = linkage(distances, method="average", optimal_ordering=False) if n > 1 else np.empty((0, 4))

    raw_labels = cut_tree(tree, n_clusters=[k_clusters]).ravel() if n > 1 else np.zeros(1, dtype=int)
    groups = sorted(
        (np.flatnonzero(raw_labels == label) for label in np.unique(raw_labels)),
        key=lambda grp: term_ids[grp[0]],
    )
    labels = np.empty(n, dtype=int)
    for j, members in enumerate(groups):
        labels[members] = j

    affinity = np.zeros((n, k_clusters), dtype=float)
    counts = np.bincount(labels, minlength=k_clusters)
    for i in range(n):
        sim = _similarity_row(distances, n, i)
        affinity[i] = np.bincount(labels, weights=sim, minlength=k_clusters) / counts

    term_index = {tid: i for i, tid in enumerate(term_ids)}
    mapped_rows = np.array([i for i, m in enumerate(mappings) if len(m) == 1], dtype=int)
    concept_rows = np.array([term_index[mappings[i][0]] for i in mapped_rows], dtype=int)

    group_ids = [f"MONDO_CLUSTER_H{H}_{j + 1:03d}" for j in range(k_clusters)]
    cluster_rows, group_labels = [], []
    for j, members in enumerate(groups):
        representative = members[np.argmax(affinity[members, j])]
        tid = term_ids[representative]
        label = f"Cluster {j + 1:03d}: {terms[tid]['name']}"
        group_labels.append(label)
        size = len(members)
        coherence = (size * affinity[members, j].mean() - 1.0) / (size - 1.0) if size > 1 else 1.0
        cluster_rows.append({
            "cluster_id": group_ids[j],
            "label": label,
            "representative_mondo_id": tid,
            "unique_mondo_terms": size,
            "mean_within_cluster_jaccard": float(coherence),
        })

    # Soft memberships via temperature-scaled softmax
    logits = (affinity - affinity.max(axis=1, keepdims=True)) / temperature
    proportions = np.exp(logits)
    proportions /= proportions.sum(axis=1, keepdims=True)

    # Initialize raw memberships: 1 on fallback coordinate
    raw_m = np.zeros((len(diseases_str), H), dtype=float)
    raw_m[:, -1] = 1.0
    if len(mapped_rows) > 0:
        raw_m[mapped_rows, -1] = 0.0
        raw_m[mapped_rows, :-1] = proportions[concept_rows]

    # Apply Laplace smoothing
    b0 = ((1.0 - epsilon) * raw_m + epsilon / H).astype(np.float32)

    records = []
    for d, mapping in zip(diseases_str, mappings):
        assigned = len(mapping) == 1
        records.append({
            "disease": d,
            "status": "mapped" if assigned else ("ambiguous" if len(mapping) > 1 else "unmapped"),
            "mondo_ids": ";".join(mapping),
        })

    status_counts = Counter(r["status"] for r in records)
    report = {
        "H": H,
        "clusters": k_clusters,
        "temperature": temperature,
        "epsilon": epsilon,
        "diseases": len(diseases_str),
        "unique_mapping": status_counts.get("mapped", 0),
        "ambiguous_mapping": status_counts.get("ambiguous", 0),
        "no_mapping": status_counts.get("unmapped", 0),
        "biological_coverage": status_counts.get("mapped", 0) / len(diseases_str),
        "cluster_details": cluster_rows,
    }

    all_ids = group_ids + ["UNASSIGNED"]
    all_labels = group_labels + ["Unassigned or unmapped fallback"]
    return b0, all_ids, all_labels, report, records
