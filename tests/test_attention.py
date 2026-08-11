import math
import torch
import torch.nn.functional as F
import pytest
from model import scaled_dot_product_attention


def test_basic_dimensions():
    """Test output and attention weights shapes for standard multi-head dimensions."""
    batch_size = 2
    num_heads = 4
    seq_len = 8
    d_k = 32

    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    output, weights = scaled_dot_product_attention(query, key, value)

    assert output.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_different_value_dimension():
    """Test when value dimension d_v differs from key dimension d_k."""
    batch_size = 2
    num_heads = 4
    seq_len = 6
    d_k = 32
    d_v = 64

    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_v)

    output, weights = scaled_dot_product_attention(query, key, value)

    assert output.shape == (batch_size, num_heads, seq_len, d_v)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_attention_weights_normalization():
    """Test that attention weights sum to 1.0 along the key dimension."""
    query = torch.randn(3, 2, 5, 16)
    key = torch.randn(3, 2, 5, 16)
    value = torch.randn(3, 2, 5, 16)

    _, weights = scaled_dot_product_attention(query, key, value)

    sum_weights = weights.sum(dim=-1)
    expected_ones = torch.ones_like(sum_weights)
    torch.testing.assert_close(sum_weights, expected_ones, rtol=1e-5, atol=1e-5)


def test_scaling_correctness():
    """Test numerical scaling by sqrt(d_k) before softmax."""
    d_k = 4
    query = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])  # shape (1, 1, 1, 4)
    key = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])  # shape (1, 1, 2, 4)
    value = torch.tensor([[[[10.0], [20.0]]]])  # shape (1, 1, 2, 1)

    # Q * K^T = [[1.0, 0.0]]
    # Raw scores scaled by sqrt(4)=2.0 => [[0.5, 0.0]]
    # Softmax([0.5, 0.0]) = [exp(0.5) / (exp(0.5) + 1), 1 / (exp(0.5) + 1)]
    expected_prob_0 = math.exp(0.5) / (math.exp(0.5) + 1.0)
    expected_prob_1 = 1.0 / (math.exp(0.5) + 1.0)
    expected_weights = torch.tensor([[[[expected_prob_0, expected_prob_1]]]])

    _, weights = scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-5)


def test_masking():
    """Test masking out key positions (0 = masked out, 1 = allowed)."""
    batch_size = 1
    heads = 1
    seq_len = 3
    d_k = 4

    query = torch.randn(batch_size, heads, seq_len, d_k)
    key = torch.randn(batch_size, heads, seq_len, d_k)
    value = torch.randn(batch_size, heads, seq_len, d_k)

    # Mask key index 1 (0 = masked out, 1 = allowed)
    mask = torch.tensor([[[[1, 0, 1]]]])

    _, weights = scaled_dot_product_attention(query, key, value, mask=mask)

    # Masked position (index 1) should have probability close to 0
    masked_weights = weights[..., 1]
    assert torch.all(masked_weights < 1e-6)

    # Sum across key dimension should still be 1.0
    sum_weights = weights.sum(dim=-1)
    expected_ones = torch.ones_like(sum_weights)
    torch.testing.assert_close(sum_weights, expected_ones, rtol=1e-5, atol=1e-5)


def test_output_correctness():
    """Test output computation against a deterministic expected value."""
    query = torch.tensor([[[[1.0, 0.0]]]])  # (1, 1, 1, 2)
    key = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])  # (1, 1, 2, 2)
    value = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])  # (1, 1, 2, 2)

    # Q * K^T = [[1.0, 1.0]]
    # Scaled by sqrt(2): [[1/sqrt(2), 1/sqrt(2)]]
    # Softmax yields equal weights: [[0.5, 0.5]]
    # Output = 0.5 * [2, 4] + 0.5 * [6, 8] = [4.0, 6.0]
    expected_output = torch.tensor([[[[4.0, 6.0]]]])

    output, weights = scaled_dot_product_attention(query, key, value)

    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)


def test_gradient_flow():
    """Test backward pass differentiability for query, key, and value tensors."""
    query = torch.randn(2, 2, 4, 8, requires_grad=True)
    key = torch.randn(2, 2, 4, 8, requires_grad=True)
    value = torch.randn(2, 2, 4, 8, requires_grad=True)

    output, weights = scaled_dot_product_attention(query, key, value)
    loss = output.sum() + weights.sum()
    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None

    assert query.grad.shape == query.shape
    assert key.grad.shape == key.shape
    assert value.grad.shape == value.shape


def test_mask_broadcasting():
    """Test broadcasting of attention mask shape (batch, 1, 1, key_length)."""
    batch_size = 2
    num_heads = 4
    seq_len = 5
    d_k = 16

    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Mask shape: (batch_size, 1, 1, seq_len)
    mask = torch.tensor([
        [[[1, 1, 1, 1, 0]]],
        [[[1, 1, 1, 0, 0]]]
    ])

    output, weights = scaled_dot_product_attention(query, key, value, mask=mask)

    assert output.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Verify batch 0 index 4 is masked
    assert torch.all(weights[0, :, :, 4] < 1e-6)

    # Verify batch 1 index 3 & 4 are masked
    assert torch.all(weights[1, :, :, 3] < 1e-6)
    assert torch.all(weights[1, :, :, 4] < 1e-6)


def test_incompatible_dimensions():
    """Test error handling for mismatched dimensions."""
    query = torch.randn(2, 4, 8, 32)
    key = torch.randn(2, 4, 8, 16)  # mismatched d_k
    value = torch.randn(2, 4, 8, 32)

    with pytest.raises(ValueError, match="must match"):
        scaled_dot_product_attention(query, key, value)
