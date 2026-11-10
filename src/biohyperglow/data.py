"""Data utilities for association networks, negative pair sampling, and synthetic benchmark generation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


def load_edges_csv(
    path: Union[str, Path]
) -> Tuple[np.ndarray, List[str], List[str], str]:
    """Load disease-gene positive associations from a CSV file.

    Expects a header with at least 'disease' and 'gene' columns.

    Parameters
    ----------
    path : str or Path
        Path to the associations CSV file.

    Returns
    -------
    edges : np.ndarray of shape (N, 2), dtype=int64
        Indices into the sorted disease and gene lists.
    diseases : list of str
        Sorted unique disease identifiers.
    genes : list of str
        Sorted unique gene identifiers.
    sha256 : str
        Deterministic SHA-256 fingerprint of the normalized edge set.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"disease", "gene"}.issubset(reader.fieldnames):
            raise ValueError(f"CSV file at {path} must contain 'disease' and 'gene' columns")
        rows = {
            (r["disease"].strip(), r["gene"].strip())
            for r in reader
            if r["disease"] and r["gene"] and r["disease"].strip() and r["gene"].strip()
        }

    if not rows:
        raise ValueError(f"No valid associations found in {path}")

    diseases = sorted({d for d, _ in rows})
    genes = sorted({g for _, g in rows})
    d_to_idx = {d: i for i, d in enumerate(diseases)}
    g_to_idx = {g: i for i, g in enumerate(genes)}

    edges = np.array(sorted((d_to_idx[d], g_to_idx[g]) for d, g in rows), dtype=np.int64)
    normalized_text = "\n".join(f"{d}\t{g}" for d, g in sorted(rows)).encode("utf-8")
    sha256 = hashlib.sha256(normalized_text).hexdigest()

    return edges, diseases, genes, sha256


def negative_pairs(
    n_diseases: int,
    n_genes: int,
    positive_set: Set[Tuple[int, int]],
    n_samples: int,
    rng: np.random.Generator,
    exclude: Optional[Set[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Sample disjoint negative (unobserved) pairs not in any known positive association.

    Parameters
    ----------
    n_diseases : int
        Total number of diseases.
    n_genes : int
        Total number of genes.
    positive_set : set of (int, int)
        All known benchmark positive associations.
    n_samples : int
        Number of negative pairs to sample.
    rng : np.random.Generator
        Numpy random generator.
    exclude : set of (int, int), optional
        Additional pairs to exclude (e.g., test or validation negatives).

    Returns
    -------
    negatives : np.ndarray of shape (n_samples, 2), dtype=int64
    """
    occupied = set(positive_set)
    if exclude:
        occupied.update(exclude)

    total_possible = n_diseases * n_genes
    if n_samples > total_possible - len(occupied):
        raise ValueError(
            f"Cannot sample {n_samples} negative pairs: only {total_possible - len(occupied)} remain"
        )

    out: Set[Tuple[int, int]] = set()
    while len(out) < n_samples:
        batch_size = max(32, (n_samples - len(out)) * 2)
        d_cand = rng.integers(n_diseases, size=batch_size)
        g_cand = rng.integers(n_genes, size=batch_size)
        for d, g in zip(d_cand, g_cand):
            p = (int(d), int(g))
            if p not in occupied and p not in out:
                out.add(p)
            if len(out) == n_samples:
                break

    return np.asarray(sorted(out), dtype=np.int64).reshape(-1, 2)


def train_val_test_split(
    edges: np.ndarray,
    n_diseases: int,
    n_genes: int,
    seed: int = 42,
    folds: int = 5,
    val_frac: float = 0.1,
) -> List[Dict[str, np.ndarray]]:
    """Partition edges into leakage-safe CV folds with disjoint validation and test negatives.

    Parameters
    ----------
    edges : np.ndarray of shape (N, 2)
        All positive edges.
    n_diseases : int
    n_genes : int
    seed : int, default=42
    folds : int, default=5
    val_frac : float, default=0.1

    Returns
    -------
    splits : list of dict
        Each dict has 'train_pos', 'val_pos', 'test_pos', 'val_neg', 'test_neg'.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    all_positive = set(map(tuple, edges.tolist()))
    results = []

    for test_indices in np.array_split(perm, folds):
        remainder = np.setdiff1d(perm, test_indices, assume_unique=True)
        local_perm = rng.permutation(remainder)
        n_val = max(1, int(round(len(remainder) * val_frac)))

        val_pos = edges[local_perm[:n_val]]
        train_pos = edges[local_perm[n_val:]]
        test_pos = edges[test_indices]

        test_neg = negative_pairs(n_diseases, n_genes, all_positive, len(test_pos), rng)
        test_neg_set = set(map(tuple, test_neg.tolist()))
        val_neg = negative_pairs(n_diseases, n_genes, all_positive, len(val_pos), rng, exclude=test_neg_set)

        results.append({
            "train_pos": train_pos,
            "val_pos": val_pos,
            "test_pos": test_pos,
            "val_neg": val_neg,
            "test_neg": test_neg,
        })

    return results


def check_split(fold: Dict[str, np.ndarray], all_edges: np.ndarray) -> None:
    """Validate that a fold partition satisfies strict closed-world leakage-safe constraints.

    Checks:
    1. Positive sets (train, val, test) are pairwise disjoint and partition all positive edges.
    2. Negative sets (val, test) are disjoint from all known positives.
    3. Validation and test negatives are disjoint from each other.
    """
    all_pos = set(map(tuple, all_edges.tolist()))
    tr = set(map(tuple, fold["train_pos"].tolist()))
    va = set(map(tuple, fold["val_pos"].tolist()))
    te = (
        set(map(tuple, fold["test_pos"].tolist()))
        if "test_pos" in fold and len(fold["test_pos"])
        else set()
    )

    if te:
        if tr & va or tr & te or va & te:
            raise ValueError("Train, val, and test positive sets must be pairwise disjoint")
        if (tr | va | te) != all_pos:
            raise ValueError("Train, val, and test positive sets must partition all known positive associations")
    else:
        if tr & va:
            raise ValueError("Train and validation positive sets must be disjoint")
        if (tr | va) != all_pos:
            raise ValueError("Train and validation positive sets must partition all known positive associations")

    vn = set(map(tuple, fold["val_neg"].tolist()))
    tn = (
        set(map(tuple, fold["test_neg"].tolist()))
        if "test_neg" in fold and len(fold["test_neg"])
        else set()
    )
    if (vn | tn) & all_pos:
        raise ValueError("Validation or test negative pairs overlap with known positive associations")
    if tn and (vn & tn):
        raise ValueError("Validation and test negative sets must be disjoint")


def load_splits_json(path: Union[str, Path]) -> List[Dict[str, np.ndarray]]:
    """Load precomputed CV splits from a JSON file.

    Compatible with publication split format containing 'splits' list or raw list of fold dicts.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    raw_splits = data["splits"] if isinstance(data, dict) and "splits" in data else data
    if not isinstance(raw_splits, list):
        raise ValueError(f"Invalid split file {path}: expected a list of folds under 'splits' or root")

    converted = []
    for f in raw_splits:
        converted.append({
            k: np.asarray(v, dtype=np.int64).reshape(-1, 2)
            for k, v in f.items()
            if isinstance(v, (list, np.ndarray))
        })
    return converted


def generate_synthetic_dataset(
    output_dir: Union[str, Path],
    n_diseases: int = 12,
    n_genes: int = 20,
    n_edges: int = 40,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Generate a clean synthetic dataset and ontology for quickstart examples and testing.

    Parameters
    ----------
    output_dir : str or Path
        Directory where 'edges.csv' and 'ontology.obo' will be written.
    n_diseases : int, default=12
    n_genes : int, default=20
    n_edges : int, default=40
    seed : int, default=42

    Returns
    -------
    edges_path : Path
    ontology_path : Path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Diseases named with CUI format: C0000001, C0000002, ...
    disease_ids = [f"C{i + 1:07d}" for i in range(n_diseases)]
    # Genes named with Entrez ID format: 101, 102, ...
    gene_ids = [f"{100 + i + 1}" for i in range(n_genes)]

    # Sample unique random edges
    edge_set: Set[Tuple[str, str]] = set()
    while len(edge_set) < n_edges:
        d = rng.choice(disease_ids)
        g = rng.choice(gene_ids)
        edge_set.add((d, g))

    edges_path = out / "synthetic_edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "gene"])
        for d, g in sorted(edge_set):
            writer.writerow([d, g])

    # Generate synthetic MONDO OBO ontology with hierarchical clusters
    # 2 high-level categories under root, each with subclasses mapping to diseases
    ontology_path = out / "synthetic_ontology.obo"
    obo_lines = [
        "format-version: 1.2",
        "ontology: synthetic-mondo",
        "",
        "[Term]",
        "id: MONDO:0000001",
        "name: disease or disorder",
        "",
        "[Term]",
        "id: MONDO:1000001",
        "name: Neoplasm or neoplastic disease",
        "is_a: MONDO:0000001 ! disease or disorder",
        "",
        "[Term]",
        "id: MONDO:1000002",
        "name: Nervous system disorder",
        "is_a: MONDO:0000001 ! disease or disorder",
        "",
        "[Term]",
        "id: MONDO:1000003",
        "name: Cardiovascular disease",
        "is_a: MONDO:0000001 ! disease or disorder",
        "",
    ]

    # Map the first 9 diseases to MONDO terms (3 in each category); remaining 3 are unmapped fallbacks
    categories = ["MONDO:1000001", "MONDO:1000002", "MONDO:1000003"]
    for i in range(min(9, n_diseases)):
        cat = categories[i % len(categories)]
        term_id = f"MONDO:200000{i + 1}"
        obo_lines.extend([
            "[Term]",
            f"id: {term_id}",
            f"name: Synthetic Disease {i + 1}",
            f"is_a: {cat}",
            f'xref: UMLS:{disease_ids[i]}',
            "",
        ])

    ontology_path.write_text("\n".join(obo_lines), encoding="utf-8")
    return edges_path, ontology_path
