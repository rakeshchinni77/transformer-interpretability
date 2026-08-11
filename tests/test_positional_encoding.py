import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import pytest
from model import PositionalEncoding


def test_required_shapes():
    """Test required contract: sinusoidal and learned encodings produce (1, 20, 128)."""
    sinusoidal_model = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")
    learned_model = PositionalEncoding(d_model=128, max_len=128, method="learned")

    sinusoidal_model.eval()
    learned_model.eval()

    x = torch.zeros(1, 20, 128)

    sinusoidal_output = sinusoidal_model(x)
    learned_output = learned_model(x)

    assert sinusoidal_output.shape == (1, 20, 128)
    assert learned_output.shape == (1, 20, 128)


def test_sinusoidal_numerical_correctness():
    """Test position 0 even indices (sine) are 0.0 and odd indices (cosine) are 1.0."""
    model = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")
    pe = model.pe

    # Position 0: sin(0) = 0, cos(0) = 1
    even_indices = pe[0, 0::2]
    odd_indices = pe[0, 1::2]

    torch.testing.assert_close(even_indices, torch.zeros_like(even_indices), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(odd_indices, torch.ones_like(odd_indices), atol=1e-6, rtol=1e-6)

    # Verify position 1 is different from position 0
    assert not torch.allclose(pe[0], pe[1])


def test_sinusoidal_determinism():
    """Test that two sinusoidal modules produce identical encodings."""
    model1 = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")
    model2 = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")

    torch.testing.assert_close(model1.pe, model2.pe)


def test_sinusoidal_buffer_not_trainable():
    """Test that sinusoidal pe is registered as a non-trainable buffer."""
    model = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")

    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())

    assert "pe" in buffers
    assert "pe" not in parameters
    assert not buffers["pe"].requires_grad


def test_sinusoidal_state_dict():
    """Test that sinusoidal pe is included in model.state_dict()."""
    model = PositionalEncoding(d_model=128, max_len=128, method="sinusoidal")
    state = model.state_dict()

    assert "pe" in state
    assert state["pe"].shape == (128, 128)


def test_learned_embedding():
    """Test that learned positional encoding contains a trainable nn.Embedding."""
    model = PositionalEncoding(d_model=128, max_len=128, method="learned")

    assert isinstance(model.position_embedding, nn.Embedding)
    assert model.position_embedding.num_embeddings == 128
    assert model.position_embedding.embedding_dim == 128

    parameters = dict(model.named_parameters())
    assert "position_embedding.weight" in parameters
    assert parameters["position_embedding.weight"].requires_grad


def test_learned_position_sensitivity():
    """Test that learned embedding rows for different positions are distinct."""
    model = PositionalEncoding(d_model=128, max_len=128, method="learned")
    weights = model.position_embedding.weight

    assert not torch.allclose(weights[0], weights[1])


def test_learned_gradient_flow():
    """Test backward pass gradients propagate to learned embedding weights."""
    model = PositionalEncoding(d_model=128, max_len=128, method="learned")
    x = torch.randn(2, 20, 128, requires_grad=True)

    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 20, 128)
    assert model.position_embedding.weight.grad is not None
    assert model.position_embedding.weight.grad.shape == (128, 128)


def test_variable_sequence_lengths():
    """Test variable sequence lengths 5, 20, 64 for both methods."""
    for method in ["sinusoidal", "learned"]:
        model = PositionalEncoding(d_model=128, max_len=128, method=method)
        model.eval()

        for seq_len in [5, 20, 64]:
            x = torch.zeros(2, seq_len, 128)
            output = model(x)
            assert output.shape == (2, seq_len, 128)


def test_batch_broadcasting():
    """Test batch broadcasting for batch size 4."""
    for method in ["sinusoidal", "learned"]:
        model = PositionalEncoding(d_model=128, max_len=128, method=method)
        model.eval()

        x = torch.zeros(4, 20, 128)
        output = model(x)
        assert output.shape == (4, 20, 128)


def test_max_length_failure():
    """Test ValueError raised when sequence length exceeds max_len."""
    model_sin = PositionalEncoding(d_model=128, max_len=20, method="sinusoidal")
    model_lrn = PositionalEncoding(d_model=128, max_len=20, method="learned")

    x = torch.zeros(1, 21, 128)

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        model_sin(x)

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        model_lrn(x)


def test_invalid_method():
    """Test ValueError raised for unsupported method name."""
    with pytest.raises(ValueError, match="Unsupported positional encoding method"):
        PositionalEncoding(d_model=128, max_len=128, method="invalid")


def test_invalid_dimensions():
    """Test ValueError raised for non-positive d_model or max_len."""
    with pytest.raises(ValueError, match="d_model must be > 0"):
        PositionalEncoding(d_model=0, max_len=128, method="sinusoidal")

    with pytest.raises(ValueError, match="max_len must be > 0"):
        PositionalEncoding(d_model=128, max_len=-5, method="sinusoidal")


def test_json_verification_artifact():
    """Test that verification/encodings_output.json exists and satisfies the required schema."""
    json_path = Path(__file__).resolve().parent.parent / "verification" / "encodings_output.json"
    assert json_path.exists(), f"Verification file {json_path} does not exist."

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = {"sinusoidal_encoding_shape", "learned_encoding_shape"}
    assert set(data.keys()) == expected_keys, f"JSON keys {set(data.keys())} do not match expected {expected_keys}"

    assert data["sinusoidal_encoding_shape"] == [1, 20, 128]
    assert data["learned_encoding_shape"] == [1, 20, 128]


def test_cuda_compatibility():
    """Smoke test for GPU execution if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    for method in ["sinusoidal", "learned"]:
        model = PositionalEncoding(d_model=128, max_len=128, method=method).to("cuda")
        model.eval()

        x = torch.zeros(1, 20, 128, device="cuda")
        output = model(x)

        assert output.device.type == "cuda"
        assert output.shape == (1, 20, 128)
