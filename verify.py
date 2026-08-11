import json
from pathlib import Path
import torch
from model import MultiHeadAttention, PositionalEncoding

VERIFICATION_DIR = Path(__file__).resolve().parent / "verification"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def generate_attention_verification():
    """Generates verification/attention_output.json from custom MultiHeadAttention module."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    x = torch.randn(1, 10, 128)
    with torch.no_grad():
        output, attention = model(x, x, x)

    data = {
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "attention_weights_shape": list(attention.shape),
    }

    out_file = VERIFICATION_DIR / "attention_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {out_file.relative_to(VERIFICATION_DIR.parent)}")
    return data


def generate_encodings_verification():
    """Generates verification/encodings_output.json from custom PositionalEncoding module."""
    sinusoidal_model = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")
    learned_model = PositionalEncoding(d_model=128, max_len=128, method="learned")

    sinusoidal_model.eval()
    learned_model.eval()

    x = torch.zeros(1, 20, 128)
    with torch.no_grad():
        sinusoidal_output = sinusoidal_model(x)
        learned_output = learned_model(x)

    data = {
        "sinusoidal_encoding_shape": list(sinusoidal_output.shape),
        "learned_encoding_shape": list(learned_output.shape),
    }

    out_file = VERIFICATION_DIR / "encodings_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {out_file.relative_to(VERIFICATION_DIR.parent)}")
    return data


if __name__ == "__main__":
    generate_attention_verification()
    generate_encodings_verification()
