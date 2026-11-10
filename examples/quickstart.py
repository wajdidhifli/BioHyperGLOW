"""BioHyperGLOW quickstart script showing training, prediction, and explanation."""
import tempfile
from pathlib import Path

from biohyperglow import BioHyperGLOWModel, train_dual_space
from biohyperglow.data import generate_synthetic_dataset


def quickstart():
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp = Path(temp_dir)
        print("1. Generating synthetic association network and MONDO ontology...")
        edges_csv, ontology_obo = generate_synthetic_dataset(tmp, n_diseases=10, n_genes=15, n_edges=30)

        print("\n2. Training BioHyperGLOW dual-space model (H=8, kappa=10.0)...")
        model = train_dual_space(
            edges_csv=edges_csv,
            ontology_obo=ontology_obo,
            H=8,
            kappa=10.0,
            epochs=20,
            eval_every=5,
            patience=5,
            device="cpu",
            seed=42,
            progress=print,
        )

        model_path = tmp / "model.npz"
        model.save(model_path)
        print(f"\n3. Checkpoint saved to {model_path}")

        d = model.diseases[0]
        g = model.genes[0]
        print(f"\n4. Scoring association pair ({d}, {g})...")
        pred = model.predict([d], [g])
        print(f"   Intensity: {pred['intensity'][0]:.4f}")
        print(f"   Probability: {pred['probability'][0]:.4f}")

        print("\n5. Computing exact additive explanation (Proposition 1)...")
        explanation = model.explain(d, g, top_k=5)
        print(explanation.summary(top_k=5))


if __name__ == "__main__":
    quickstart()
