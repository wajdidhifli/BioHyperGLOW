"""Tests for ontology parsing, hierarchical concept clustering, and B0 priors."""
import numpy as np
import pytest

from biohyperglow.ontology import build_ontology_prior, read_obo


def test_read_obo(synthetic_data):
    _, ontology_path = synthetic_data
    terms = read_obo(ontology_path)

    assert "MONDO:0000001" in terms
    assert terms["MONDO:0000001"]["name"] == "disease or disorder"
    assert "MONDO:2000001" in terms
    assert "C0000001" in terms["MONDO:2000001"]["umls"]


def test_build_ontology_prior(synthetic_data):
    _, ontology_path = synthetic_data
    diseases = [f"C{i+1:07d}" for i in range(8)]

    # Request H=4 (3 clusters + 1 fallback)
    b0, group_ids, group_labels, report, records = build_ontology_prior(
        diseases, ontology_path, H=4, temperature=0.035, epsilon=0.01
    )

    assert b0.shape == (8, 4)
    # Row sums must be 1.0
    np.testing.assert_allclose(b0.sum(axis=1), 1.0, atol=1e-5)
    # Strictly positive (Laplace smoothing)
    assert (b0 > 0.0).all()

    assert len(group_ids) == 4
    assert group_ids[-1] == "UNASSIGNED"
    assert len(group_labels) == 4
    assert report["H"] == 4
    assert len(records) == 8
