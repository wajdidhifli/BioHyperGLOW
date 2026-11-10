"""BioHyperGLOW Command-Line Interface."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

from biohyperglow import __version__
from biohyperglow.data import generate_synthetic_dataset
from biohyperglow.metrics import pair_metrics
from biohyperglow.scoring import BioHyperGLOWModel
from biohyperglow.training import train_dual_space


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biohyperglow",
        description="BioHyperGLOW: Adaptive Dual-Space Embeddings for Interpretable Disease-Gene Association Prediction",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command help")

    # 1. train
    p_train = subparsers.add_parser("train", help="Train BioHyperGLOW dual-space model")
    p_train.add_argument("--edges", required=True, help="Path to disease-gene association CSV")
    p_train.add_argument("--ontology", help="Path to MONDO ontology .obo file")
    p_train.add_argument("--out", required=True, help="Output path for trained model (.npz)")
    p_train.add_argument("--splits", help="Optional path to precomputed splits JSON")
    p_train.add_argument("--fold", type=int, default=1, help="1-based fold index (default: 1)")
    p_train.add_argument("--preset", choices=["paper", "demo"], help="Preset configuration ('paper' for publication benchmark, 'demo' for fast smoke run)")
    p_train.add_argument("--H", type=int, default=512, help="Coordinates per branch (default: 512)")
    p_train.add_argument("--kappa", type=float, default=10.0, help="Degree-gate parameter kappa (default: 10.0)")
    p_train.add_argument("--epochs", type=int, default=200, help="Maximum epochs per branch (default: 200)")
    p_train.add_argument("--steps-per-epoch", type=int, default=24, help="Minibatch steps per epoch (default: 24)")
    p_train.add_argument("--eval-every", type=int, default=5, help="Validation frequency (default: 5)")
    p_train.add_argument("--patience", type=int, default=30, help="Patience for early stopping (default: 30)")
    p_train.add_argument("--min-delta", type=float, default=1e-5, help="Minimum AP delta for early stopping (default: 1e-5)")
    p_train.add_argument("--batch-size", type=int, default=1024, help="Minibatch size (default: 1024)")
    p_train.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    p_train.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Compute device (default: cpu)")
    p_train.add_argument("--seed", type=int, default=42, help="Random base seed (default: 42)")

    # 2. predict
    p_predict = subparsers.add_parser("predict", help="Predict association scores for pairs")
    p_predict.add_argument("--model", required=True, help="Path to trained model (.npz)")
    p_predict.add_argument("--pairs", required=True, help="Path to CSV with disease,gene pairs")
    p_predict.add_argument("--out", help="Output path for predictions CSV (default: stdout)")

    # 3. explain
    p_explain = subparsers.add_parser("explain", help="Decompose predicted association strength")
    p_explain.add_argument("--model", required=True, help="Path to trained model (.npz)")
    p_explain.add_argument("--disease", required=True, help="Query disease identifier (e.g. UMLS CUI)")
    p_explain.add_argument("--gene", required=True, help="Query gene identifier")
    p_explain.add_argument("--top-k", type=int, default=10, help="Number of top hypernodes to display (default: 10)")
    p_explain.add_argument("--out", help="Optional output JSON or CSV path for explanation components")

    # 4. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model discrimination on test pairs")
    p_eval.add_argument("--model", required=True, help="Path to trained model (.npz)")
    p_eval.add_argument("--test-pairs", required=True, help="Path to CSV with disease,gene,label")
    p_eval.add_argument("--out", help="Optional output JSON summary path")

    # 5. demo
    p_demo = subparsers.add_parser("demo", help="Run self-contained synthetic benchmark end-to-end")
    p_demo.add_argument("--out-dir", default="biohyperglow_demo", help="Demo directory (default: biohyperglow_demo)")
    p_demo.add_argument("--H", type=int, default=8, help="Hypernode dimension for demo (default: 8)")
    p_demo.add_argument("--epochs", type=int, default=30, help="Training epochs for demo (default: 30)")

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    print(f"BioHyperGLOW {__version__} - Training Dual-Space Model")
    print(f"Edges: {args.edges}")
    print(f"Ontology: {args.ontology or 'None (free association-only branch)'}")
    print(f"Splits: {args.splits or 'Generated automatically'}, Fold: {args.fold}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"H={args.H}, kappa={args.kappa}, device={args.device}")

    model = train_dual_space(
        edges_csv=args.edges,
        ontology_obo=args.ontology,
        splits=args.splits,
        fold=args.fold,
        preset=args.preset,
        H=args.H,
        kappa=args.kappa,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_every=args.eval_every,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        base_seed=args.seed,
        progress=print,
    )

    model.save(args.out)
    print(f"Trained model checkpoint saved successfully to: {args.out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = BioHyperGLOWModel.load(args.model)
    with open(args.pairs, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"disease", "gene"}.issubset(reader.fieldnames):
            raise ValueError(f"Input pairs CSV must contain 'disease' and 'gene' columns")
        pairs = [(r["disease"].strip(), r["gene"].strip()) for r in reader if r["disease"] and r["gene"]]

    if not pairs:
        print("No pairs found in input file", file=sys.stderr)
        return 1

    d_list, g_list = zip(*pairs)
    preds = model.predict(d_list, g_list)

    fieldnames = [
        "disease",
        "gene",
        "log_intensity",
        "intensity",
        "probability",
        "semantic_intensity",
        "structural_intensity",
        "gate_alpha",
    ]
    if "binary_prediction" in preds:
        fieldnames.append("binary_prediction")

    rows = []
    for i in range(len(d_list)):
        row = {
            "disease": preds["disease"][i],
            "gene": preds["gene"][i],
            "log_intensity": f"{preds['log_intensity'][i]:.6f}",
            "intensity": f"{preds['intensity'][i]:.6f}",
            "probability": f"{preds['probability'][i]:.6f}",
            "semantic_intensity": f"{preds['semantic_intensity'][i]:.6f}",
            "structural_intensity": f"{preds['structural_intensity'][i]:.6f}",
            "gate_alpha": f"{preds['gate_alpha'][i]:.4f}",
        }
        if "binary_prediction" in preds:
            row["binary_prediction"] = str(bool(preds["binary_prediction"][i]))
        rows.append(row)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} predictions to {args.out}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    model = BioHyperGLOWModel.load(args.model)
    res = model.explain(args.disease, args.gene, top_k=args.top_k)
    print(res.summary(top_k=args.top_k))

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".json":
            p.write_text(json.dumps(res.to_dict(), indent=2) + "\n", encoding="utf-8")
        else:
            with open(p, "w", newline="", encoding="utf-8") as f:
                fields = [
                    "coordinate_index",
                    "branch",
                    "branch_coordinate",
                    "label",
                    "intensity_contribution",
                    "contribution_share",
                    "gene_membership",
                    "disease_membership",
                    "exposure",
                ]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for c in res.components:
                    d = {k: getattr(c, k) for k in fields}
                    writer.writerow(d)
        print(f"Explanation saved to {args.out}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = BioHyperGLOWModel.load(args.model)
    with open(args.test_pairs, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"disease", "gene", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"Test CSV must contain 'disease', 'gene', and 'label' columns")
        rows = [r for r in reader if r["disease"] and r["gene"] and r["label"]]

    d_list = [r["disease"].strip() for r in rows]
    g_list = [r["gene"].strip() for r in rows]
    labels = [int(r["label"].strip()) for r in rows]

    preds = model.predict(d_list, g_list)
    metrics = pair_metrics(labels, preds["log_intensity"], threshold=model.threshold)

    print("BioHyperGLOW Evaluation Metrics:")
    print("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.6f}")
        else:
            print(f"  {k:<18}: {v}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics saved to {args.out}")

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    print(f"=== BioHyperGLOW Demo: Generating Synthetic Benchmark in {out_dir} ===")
    edges_csv, ontology_obo = generate_synthetic_dataset(out_dir, n_diseases=12, n_genes=20, n_edges=40)
    print(f"Generated synthetic associations: {edges_csv}")
    print(f"Generated synthetic ontology:    {ontology_obo}")

    model_path = out_dir / "model_checkpoint.npz"
    print(f"\nTraining model (H={args.H}, epochs={args.epochs})...")
    model = train_dual_space(
        edges_csv=edges_csv,
        ontology_obo=ontology_obo,
        H=args.H,
        kappa=10.0,
        epochs=args.epochs,
        eval_every=5,
        patience=10,
        batch_size=128,
        device="cpu",
        seed=42,
        progress=print,
    )
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Run sample prediction
    test_d, test_g = model.diseases[0], model.genes[0]
    print(f"\nScoring sample association: {test_d} <-> {test_g}")
    pred = model.predict([test_d], [test_g])
    print(f"  Predicted intensity: {pred['intensity'][0]:.4f}")
    print(f"  Predicted probability: {pred['probability'][0]:.4f}")
    print(f"  Gate alpha: {pred['gate_alpha'][0]:.4f}")

    print("\nGenerating intrinsic explanation:")
    expl = model.explain(test_d, test_g, top_k=5)
    print(expl.summary(top_k=5))

    print("\n=== Demo completed successfully! ===")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "explain":
        return cmd_explain(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "demo":
        return cmd_demo(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
