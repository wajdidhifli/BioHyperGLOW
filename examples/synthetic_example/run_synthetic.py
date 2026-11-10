"""Self-contained synthetic example demonstrating the complete BioHyperGLOW workflow."""
from pathlib import Path

from biohyperglow import train_dual_space
from biohyperglow.scoring import BioHyperGLOWModel


def main():
    here = Path(__file__).resolve().parent
    edges_csv = here / "synthetic_edges.csv"
    ontology_obo = here / "synthetic_ontology.obo"
    model_output = here / "synthetic_model.npz"

    print("=== Training BioHyperGLOW on Synthetic Benchmark ===")
    print(f"Edges: {edges_csv}")
    print(f"Ontology: {ontology_obo}")

    # Train a compact dual-space model (H=8 coordinates per branch)
    model = train_dual_space(
        edges_csv=edges_csv,
        ontology_obo=ontology_obo,
        H=8,
        kappa=10.0,
        epochs=30,
        eval_every=5,
        patience=10,
        batch_size=64,
        device="cpu",
        seed=42,
        progress=print,
    )

    # Save checkpoint
    model.save(model_output)
    print(f"\nModel saved to {model_output}")

    # Reload checkpoint to verify round-trip
    loaded = BioHyperGLOWModel.load(model_output)
    print(f"Reloaded model with {len(loaded.diseases)} diseases and {len(loaded.genes)} genes")

    # Predict association scores
    query_disease = loaded.diseases[0]
    query_gene = loaded.genes[0]
    print(f"\n=== Predicting Association for {query_disease} <-> {query_gene} ===")
    pred = loaded.predict([query_disease], [query_gene])
    print(f"  Log-intensity: {pred['log_intensity'][0]:.4f}")
    print(f"  Intensity (lambda): {pred['intensity'][0]:.4f}")
    print(f"  Decoded Probability: {pred['probability'][0]:.4f}")
    print(f"  Gate alpha_d: {pred['gate_alpha'][0]:.4f}")

    # Prioritize candidate genes for the disease
    print(f"\n=== Top 5 Candidate Genes for {query_disease} ===")
    ranked = loaded.rank_genes(query_disease)[:5]
    for r in ranked:
        print(f"  Rank {r['rank']}: Gene {r['gene']} | Intensity: {r['intensity']:.4f} | Prob: {r['probability']:.4f}")

    # Explain the prediction via exact hypernode path decomposition (Proposition 1)
    print(f"\n=== Intrinsic Explanation for {query_disease} <-> {query_gene} ===")
    explanation = loaded.explain(query_disease, query_gene, top_k=5)
    print(explanation.summary(top_k=5))


if __name__ == "__main__":
    main()
