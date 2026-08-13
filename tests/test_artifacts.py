import csv
import json
import subprocess
import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFICATION_DIR = PROJECT_ROOT / "verification"
MODELS_DIR = PROJECT_ROOT / "models"
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
LOGS_DIR = PROJECT_ROOT / "logs"


def test_verify_script_execution():
    """Test that running python verify.py succeeds with exit code 0."""
    verify_script = PROJECT_ROOT / "verify.py"
    result = subprocess.run(
        [sys.executable, str(verify_script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"verify.py failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "ALL PASS" in result.stdout


def test_attention_output_artifact():
    """Test verification/attention_output.json existence, exact schema keys, and exact shape values."""
    artifact_path = VERIFICATION_DIR / "attention_output.json"
    assert artifact_path.exists(), f"Artifact missing: {artifact_path}"

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = {"input_shape", "output_shape", "attention_weights_shape"}
    assert set(data.keys()) == expected_keys, f"Keys {set(data.keys())} do not match expected {expected_keys}"

    assert data["input_shape"] == [1, 10, 128]
    assert data["output_shape"] == [1, 10, 128]
    assert data["attention_weights_shape"] == [1, 4, 10, 10]


def test_encodings_output_artifact():
    """Test verification/encodings_output.json existence, exact schema keys, and exact shape values."""
    artifact_path = VERIFICATION_DIR / "encodings_output.json"
    assert artifact_path.exists(), f"Artifact missing: {artifact_path}"

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = {"sinusoidal_encoding_shape", "learned_encoding_shape"}
    assert set(data.keys()) == expected_keys, f"Keys {set(data.keys())} do not match expected {expected_keys}"

    assert data["sinusoidal_encoding_shape"] == [1, 20, 128]
    assert data["learned_encoding_shape"] == [1, 20, 128]


def test_final_model_checkpoint_artifact():
    """Test models/final_model.pth existence, non-emptiness, and model_state_dict contents."""
    ckpt_path = MODELS_DIR / "final_model.pth"
    assert ckpt_path.exists(), f"Artifact missing: {ckpt_path}"
    assert ckpt_path.stat().st_size > 0, "final_model.pth is empty!"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "config" in ckpt
    assert ckpt["config"]["d_model"] == 128
    assert ckpt["config"]["num_heads"] == 4
    assert ckpt["config"]["num_layers"] == 2


def test_epoch_1_weights_snapshot_artifact():
    """Test snapshots/epoch_1_weights.pt existence, shapes, and normalization."""
    snap_path = SNAPSHOTS_DIR / "epoch_1_weights.pt"
    assert snap_path.exists(), f"Artifact missing: {snap_path}"
    assert snap_path.stat().st_size > 0, "epoch_1_weights.pt is empty!"

    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    assert "attention_weights" in snap
    assert snap["epoch"] == 1

    attentions = snap["attention_weights"]
    assert len(attentions) == 2  # 2 encoder layers

    for attn in attentions:
        assert torch.isfinite(attn).all()
        assert attn.shape[1:] == (4, 128, 128)
        sum_weights = attn.sum(dim=-1)
        expected_ones = torch.ones_like(sum_weights)
        torch.testing.assert_close(sum_weights, expected_ones, rtol=1e-4, atol=1e-4)


def test_final_epoch_weights_snapshot_artifact():
    """Test snapshots/final_epoch_weights.pt existence, shapes, and normalization."""
    snap_path = SNAPSHOTS_DIR / "final_epoch_weights.pt"
    assert snap_path.exists(), f"Artifact missing: {snap_path}"
    assert snap_path.stat().st_size > 0, "final_epoch_weights.pt is empty!"

    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    assert "attention_weights" in snap
    assert snap["epoch"] >= 2

    attentions = snap["attention_weights"]
    assert len(attentions) == 2

    for attn in attentions:
        assert torch.isfinite(attn).all()
        assert attn.shape[1:] == (4, 128, 128)
        sum_weights = attn.sum(dim=-1)
        expected_ones = torch.ones_like(sum_weights)
        torch.testing.assert_close(sum_weights, expected_ones, rtol=1e-4, atol=1e-4)


def test_training_metrics_csv_artifact():
    """Test logs/training_metrics.csv existence, exact header columns, row count, and numeric types."""
    csv_path = LOGS_DIR / "training_metrics.csv"
    assert csv_path.exists(), f"Artifact missing: {csv_path}"
    assert csv_path.stat().st_size > 0, "training_metrics.csv is empty!"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        expected_header = ["epoch", "layer", "head", "attention_entropy"]
        assert header == expected_header, f"Header {header} does not match exact schema {expected_header}"

        rows = list(reader)
        assert len(rows) > 0, "training_metrics.csv has no data rows!"

        for row in rows:
            assert len(row) == 4, f"Row {row} does not have exactly 4 columns"
            epoch = int(row[0])
            layer = int(row[1])
            head = int(row[2])
            entropy = float(row[3])

            assert epoch >= 1
            assert layer in (0, 1)
            assert head in (0, 1, 2, 3)
            assert torch.isfinite(torch.tensor(entropy))
            assert entropy >= 0.0, f"Entropy value {entropy} is negative!"
