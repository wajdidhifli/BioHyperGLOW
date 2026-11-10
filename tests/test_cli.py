"""Tests for BioHyperGLOW command-line interface."""
import csv
import json
from pathlib import Path
import pytest

from biohyperglow.cli import main


def test_cli_demo(tmp_path):
    demo_dir = tmp_path / "demo_run"
    ret = main(["demo", "--out-dir", str(demo_dir), "--H", "4", "--epochs", "5"])
    assert ret == 0
    assert (demo_dir / "synthetic_edges.csv").exists()
    assert (demo_dir / "synthetic_ontology.obo").exists()
    assert (demo_dir / "model_checkpoint.npz").exists()


def test_cli_predict_explain_evaluate(tmp_path):
    demo_dir = tmp_path / "full_cli_run"
    main(["demo", "--out-dir", str(demo_dir), "--H", "4", "--epochs", "5"])
    model_path = demo_dir / "model_checkpoint.npz"

    # 1. Test predict
    pairs_csv = tmp_path / "pairs.csv"
    with open(pairs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "gene"])
        writer.writerow(["C0000001", "101"])
        writer.writerow(["C0000002", "102"])

    preds_csv = tmp_path / "preds.csv"
    ret_pred = main(["predict", "--model", str(model_path), "--pairs", str(pairs_csv), "--out", str(preds_csv)])
    assert ret_pred == 0
    assert preds_csv.exists()

    # 2. Test explain
    explain_json = tmp_path / "explain.json"
    ret_expl = main(["explain", "--model", str(model_path), "--disease", "C0000001", "--gene", "101", "--out", str(explain_json)])
    assert ret_expl == 0
    assert explain_json.exists()
    data = json.loads(explain_json.read_text())
    assert "predicted_intensity" in data
    assert "components" in data

    # 3. Test evaluate
    test_csv = tmp_path / "test.csv"
    with open(test_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "gene", "label"])
        writer.writerow(["C0000001", "101", "1"])
        writer.writerow(["C0000002", "102", "0"])

    eval_json = tmp_path / "eval.json"
    ret_eval = main(["evaluate", "--model", str(model_path), "--test-pairs", str(test_csv), "--out", str(eval_json)])
    assert ret_eval == 0
    assert eval_json.exists()
