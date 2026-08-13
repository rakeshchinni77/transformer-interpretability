import csv
import json
import sys
from pathlib import Path
import torch
from model import MultiHeadAttention, PositionalEncoding

PROJECT_ROOT = Path(__file__).resolve().parent
VERIFICATION_DIR = PROJECT_ROOT / "verification"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def generate_attention_verification():
    """Generates verification/attention_output.json using custom MultiHeadAttention module on CPU."""
    torch.manual_seed(42)

    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    x = torch.randn(1, 10, 128)

    with torch.no_grad():
        output, attention = model(x, x, x)

    assert x.shape == (1, 10, 128), f"Expected input shape (1, 10, 128), got {x.shape}"
    assert output.shape == (1, 10, 128), f"Expected output shape (1, 10, 128), got {output.shape}"
    assert attention.shape == (1, 4, 10, 10), f"Expected attention shape (1, 4, 10, 10), got {attention.shape}"

    assert torch.isfinite(output).all(), "Output contains non-finite values (NaN/Inf)"
    assert torch.isfinite(attention).all(), "Attention weights contain non-finite values (NaN/Inf)"

    sum_weights = attention.sum(dim=-1)
    expected_ones = torch.ones_like(sum_weights)
    assert torch.allclose(sum_weights, expected_ones, atol=1e-5), "Attention weights sum along key dimension does not equal 1.0"

    model_train = MultiHeadAttention(d_model=128, h=4)
    x_grad = torch.randn(1, 10, 128, requires_grad=True)
    out_grad, _ = model_train(x_grad, x_grad, x_grad)
    loss = out_grad.mean()
    loss.backward()
    assert x_grad.grad is not None, "Gradient flow failed for input tensor"
    assert torch.isfinite(x_grad.grad).all(), "Input gradient contains non-finite values"

    data = {
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "attention_weights_shape": list(attention.shape),
    }

    out_file = VERIFICATION_DIR / "attention_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {out_file.relative_to(PROJECT_ROOT)}")
    return data


def generate_encodings_verification():
    """Generates verification/encodings_output.json using custom PositionalEncoding module on CPU."""
    torch.manual_seed(42)

    sinusoidal_model = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")
    learned_model = PositionalEncoding(d_model=128, max_len=128, method="learned")

    sinusoidal_model.eval()
    learned_model.eval()

    x = torch.randn(1, 20, 128)

    with torch.no_grad():
        sinusoidal_output = sinusoidal_model(x)
        learned_output = learned_model(x)

    assert sinusoidal_output.shape == (1, 20, 128), f"Expected sinusoidal output shape (1, 20, 128), got {sinusoidal_output.shape}"
    assert learned_output.shape == (1, 20, 128), f"Expected learned output shape (1, 20, 128), got {learned_output.shape}"

    assert torch.isfinite(sinusoidal_output).all(), "Sinusoidal encoding output contains non-finite values"
    assert torch.isfinite(learned_output).all(), "Learned encoding output contains non-finite values"

    data = {
        "sinusoidal_encoding_shape": list(sinusoidal_output.shape),
        "learned_encoding_shape": list(learned_output.shape),
    }

    out_file = VERIFICATION_DIR / "encodings_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {out_file.relative_to(PROJECT_ROOT)}")
    return data


def verify_phase11_artifacts():
    """Verifies existence, non-emptiness, schemas, and numeric integrity of Phase 11 interpretability artifacts."""
    print("\nArtifact verification:")

    # 1. Verify models/final_model.pth
    final_model_path = PROJECT_ROOT / "models" / "final_model.pth"
    assert final_model_path.exists(), f"Missing file: {final_model_path}"
    assert final_model_path.stat().st_size > 0, "final_model.pth is empty!"
    print("- final_model.pth: PASS")

    # 2. Verify snapshots/epoch_1_weights.pt
    epoch1_snap_path = PROJECT_ROOT / "snapshots" / "epoch_1_weights.pt"
    assert epoch1_snap_path.exists(), f"Missing file: {epoch1_snap_path}"
    assert epoch1_snap_path.stat().st_size > 0, "epoch_1_weights.pt is empty!"
    snap1 = torch.load(epoch1_snap_path, map_location="cpu", weights_only=False)
    assert "attention_weights" in snap1
    assert len(snap1["attention_weights"]) == 2
    assert snap1["attention_weights"][0].shape[1:] == (4, 128, 128)
    print("- epoch_1_weights.pt: PASS")

    # 3. Verify snapshots/final_epoch_weights.pt
    final_snap_path = PROJECT_ROOT / "snapshots" / "final_epoch_weights.pt"
    assert final_snap_path.exists(), f"Missing file: {final_snap_path}"
    assert final_snap_path.stat().st_size > 0, "final_epoch_weights.pt is empty!"
    snap_final = torch.load(final_snap_path, map_location="cpu", weights_only=False)
    assert "attention_weights" in snap_final
    assert len(snap_final["attention_weights"]) == 2
    assert snap_final["attention_weights"][0].shape[1:] == (4, 128, 128)
    print("- final_epoch_weights.pt: PASS")

    # 4. Verify logs/training_metrics.csv
    csv_path = PROJECT_ROOT / "logs" / "training_metrics.csv"
    assert csv_path.exists(), f"Missing file: {csv_path}"
    assert csv_path.stat().st_size > 0, "training_metrics.csv is empty!"
    print("- training_metrics.csv: PASS")

    # 5. Verify CSV schema and values
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_header = ["epoch", "layer", "head", "attention_entropy"]
        assert header == expected_header, f"CSV header {header} does not match expected {expected_header}"

        rows = list(reader)
        assert len(rows) > 0, "training_metrics.csv contains no data rows!"

        for row in rows:
            assert len(row) == 4, f"Row {row} does not have exactly 4 columns"
            epoch_val = int(row[0])
            layer_val = int(row[1])
            head_val = int(row[2])
            entropy_val = float(row[3])

            assert epoch_val >= 1
            assert layer_val in (0, 1)
            assert head_val in (0, 1, 2, 3)
            assert torch.isfinite(torch.tensor(entropy_val))
            assert entropy_val >= 0.0, f"Entropy value {entropy_val} is negative!"

    print("- CSV schema: PASS")
    print("- entropy values: PASS")


def main():
    try:
        generate_attention_verification()
        generate_encodings_verification()
        verify_phase11_artifacts()
        print("\nVerification system: ALL PASS")
        sys.exit(0)
    except Exception as e:
        print(f"\nVerification FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
