import torch
import torch.nn as nn
import pytest
from model import MultiHeadAttention


def test_required_shape_contract():
    """Test required contract: [1, 10, 128] -> output [1, 10, 128], attention [1, 4, 10, 10]."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    x = torch.randn(1, 10, 128)
    output, attention = model(x, x, x)

    assert output.shape == (1, 10, 128)
    assert attention.shape == (1, 4, 10, 10)


def test_batch_dimension():
    """Test batch dimension vectorization [2, 10, 128]."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    x = torch.randn(2, 10, 128)
    output, attention = model(x, x, x)

    assert output.shape == (2, 10, 128)
    assert attention.shape == (2, 4, 10, 10)


def test_different_sequence_length():
    """Test arbitrary sequence length [2, 7, 128]."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    x = torch.randn(2, 7, 128)
    output, attention = model(x, x, x)

    assert output.shape == (2, 7, 128)
    assert attention.shape == (2, 4, 7, 7)


def test_cross_attention_shapes():
    """Test cross-attention inputs where query length != key/value length."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    query = torch.randn(2, 5, 128)
    key = torch.randn(2, 7, 128)
    value = torch.randn(2, 7, 128)

    output, attention = model(query, key, value)

    assert output.shape == (2, 5, 128)
    assert attention.shape == (2, 4, 5, 7)


def test_mask_propagation():
    """Test mask propagation through MultiHeadAttention to Phase 4 attention."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    batch_size = 1
    query_len = 3
    key_len = 7
    d_model = 128

    query = torch.randn(batch_size, query_len, d_model)
    key = torch.randn(batch_size, key_len, d_model)
    value = torch.randn(batch_size, key_len, d_model)

    # Mask key positions 5 and 6
    mask = torch.tensor([[[[1, 1, 1, 1, 1, 0, 0]]]], dtype=torch.int64)

    output, attention = model(query, key, value, mask=mask)

    assert output.shape == (batch_size, query_len, d_model)
    assert attention.shape == (batch_size, 4, query_len, key_len)

    # Masked positions 5 and 6 should receive probability ~0
    assert torch.all(attention[:, :, :, 5] < 1e-6)
    assert torch.all(attention[:, :, :, 6] < 1e-6)


def test_gradient_flow():
    """Test backward pass gradient propagation to input tensor and projection layers."""
    model = MultiHeadAttention(d_model=128, h=4)
    x = torch.randn(1, 10, 128, requires_grad=True)

    output, attention = model(x, x, x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 10, 128)

    assert model.W_q.weight.grad is not None
    assert model.W_k.weight.grad is not None
    assert model.W_v.weight.grad is not None
    assert model.W_o.weight.grad is not None

    assert model.W_q.weight.grad.shape == (128, 128)
    assert model.W_k.weight.grad.shape == (128, 128)
    assert model.W_v.weight.grad.shape == (128, 128)
    assert model.W_o.weight.grad.shape == (128, 128)


def test_linear_projection_parameters():
    """Test that W_q, W_k, W_v, W_o linear layers map 128 -> 128."""
    model = MultiHeadAttention(d_model=128, h=4)

    assert isinstance(model.W_q, nn.Linear)
    assert isinstance(model.W_k, nn.Linear)
    assert isinstance(model.W_v, nn.Linear)
    assert isinstance(model.W_o, nn.Linear)

    assert model.W_q.in_features == 128 and model.W_q.out_features == 128
    assert model.W_k.in_features == 128 and model.W_k.out_features == 128
    assert model.W_v.in_features == 128 and model.W_v.out_features == 128
    assert model.W_o.in_features == 128 and model.W_o.out_features == 128


def test_head_divisibility_validation():
    """Test ValueError raised when d_model is not divisible by num_heads."""
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=130, h=4)

    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=128, h=3)


def test_attention_weight_normalization():
    """Test that attention weights sum to 1.0 along the key dimension."""
    model = MultiHeadAttention(d_model=128, h=4)
    model.eval()

    x = torch.randn(2, 10, 128)
    _, attention = model(x, x, x)

    sum_weights = attention.sum(dim=-1)
    expected_ones = torch.ones_like(sum_weights)
    torch.testing.assert_close(sum_weights, expected_ones, rtol=1e-5, atol=1e-5)


def test_evaluation_determinism():
    """Test that model in eval mode yields identical outputs for identical inputs."""
    model = MultiHeadAttention(d_model=128, h=4, dropout=0.1)
    model.eval()

    x = torch.randn(2, 8, 128)
    out1, att1 = model(x, x, x)
    out2, att2 = model(x, x, x)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(att1, att2)


def test_cuda_compatibility():
    """Smoke test for GPU execution if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    model = MultiHeadAttention(d_model=128, h=4).to("cuda")
    model.eval()

    x = torch.randn(2, 10, 128, device="cuda")
    output, attention = model(x, x, x)

    assert output.device.type == "cuda"
    assert attention.device.type == "cuda"
    assert output.shape == (2, 10, 128)
    assert attention.shape == (2, 4, 10, 10)
