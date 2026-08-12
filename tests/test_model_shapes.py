import torch
import torch.nn as nn
import pytest
from model import MultiHeadAttention, EncoderLayer, TransformerEncoder, TransformerClassifier


def test_required_shape_contract():
    """Test required contract: MultiHeadAttention [1, 10, 128] -> output [1, 10, 128], attention [1, 4, 10, 10]."""
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


# =====================================================================
# PHASE 7 TESTS: EncoderLayer, TransformerEncoder, TransformerClassifier
# =====================================================================

def test_encoder_layer_shapes():
    """Test single Pre-LN EncoderLayer input/output and attention shapes."""
    layer = EncoderLayer(d_model=128, num_heads=4, d_ff=256, dropout=0.1)
    layer.eval()

    x = torch.randn(2, 20, 128)
    output, attention = layer(x)

    assert output.shape == (2, 20, 128)
    assert attention.shape == (2, 4, 20, 20)


def test_encoder_layer_structure():
    """Test structural components of Pre-LN EncoderLayer."""
    layer = EncoderLayer(d_model=128, num_heads=4, d_ff=256, dropout=0.1)

    assert isinstance(layer.norm1, nn.LayerNorm)
    assert isinstance(layer.norm2, nn.LayerNorm)
    assert isinstance(layer.self_attn, MultiHeadAttention)

    # Check Feed-Forward architecture: 128 -> 256 -> 128 with ReLU
    ffn = layer.feed_forward
    assert isinstance(ffn[0], nn.Linear) and ffn[0].in_features == 128 and ffn[0].out_features == 256
    assert isinstance(ffn[1], nn.ReLU)
    assert isinstance(ffn[2], nn.Linear) and ffn[2].in_features == 256 and ffn[2].out_features == 128


def test_transformer_encoder_shapes():
    """Test TransformerEncoder hidden states and multi-layer attention list collection."""
    encoder = TransformerEncoder(d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.1)
    encoder.eval()

    x = torch.randn(2, 20, 128)
    hidden, attentions = encoder(x)

    assert hidden.shape == (2, 20, 128)
    assert isinstance(attentions, list)
    assert len(attentions) == 2
    assert attentions[0].shape == (2, 4, 20, 20)
    assert attentions[1].shape == (2, 4, 20, 20)


def test_transformer_classifier_forward_contract():
    """Test required Phase 7 contract: input_ids [2, 20] -> logits [2, 2], layer 0 [2, 4, 20, 20], layer 1 [2, 4, 20, 20]."""
    model = TransformerClassifier(
        vocab_size=30522,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        num_classes=2,
        max_len=128,
        positional_encoding="sinusoidal",
    )
    model.eval()

    input_ids = torch.randint(0, 30522, (2, 20))
    logits, attentions = model(input_ids)

    assert logits.shape == (2, 2)
    assert isinstance(attentions, list)
    assert len(attentions) == 2
    assert attentions[0].shape == (2, 4, 20, 20)
    assert attentions[1].shape == (2, 4, 20, 20)


def test_classifier_backward_pass():
    """Test full forward + CrossEntropyLoss backward pass and finite gradients across parameters."""
    model = TransformerClassifier(
        vocab_size=30522,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        num_classes=2,
    )

    input_ids = torch.randint(0, 30522, (2, 20))
    logits, attentions = model(input_ids)

    targets = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(logits, targets)

    assert torch.isfinite(loss)

    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()

    layer0_wq_grad = model.encoder.layers[0].self_attn.W_q.weight.grad
    assert layer0_wq_grad is not None
    assert torch.isfinite(layer0_wq_grad).all()

    layer0_ff_grad = model.encoder.layers[0].feed_forward[0].weight.grad
    assert layer0_ff_grad is not None
    assert torch.isfinite(layer0_ff_grad).all()

    classifier_grad = model.classifier.weight.grad
    assert classifier_grad is not None
    assert torch.isfinite(classifier_grad).all()


def test_learned_positional_encoding_classifier():
    """Test TransformerClassifier integration with learned positional encoding."""
    model = TransformerClassifier(
        vocab_size=30522,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        num_classes=2,
        positional_encoding="learned",
    )

    input_ids = torch.randint(0, 30522, (2, 20))
    logits, attentions = model(input_ids)

    assert logits.shape == (2, 2)
    assert len(attentions) == 2

    loss = nn.CrossEntropyLoss()(logits, torch.tensor([1, 0]))
    loss.backward()

    assert model.positional_encoding.position_embedding.weight.grad is not None
    assert torch.isfinite(model.positional_encoding.position_embedding.weight.grad).all()


def test_variable_sequence_lengths_classifier():
    """Test variable sequence lengths (5, 20, 64) on complete classifier."""
    model = TransformerClassifier(vocab_size=30522, d_model=128, num_heads=4, num_layers=2)
    model.eval()

    for seq_len in [5, 20, 64]:
        input_ids = torch.randint(0, 30522, (2, seq_len))
        logits, attentions = model(input_ids)

        assert logits.shape == (2, 2)
        assert len(attentions) == 2
        assert attentions[0].shape == (2, 4, seq_len, seq_len)
        assert attentions[1].shape == (2, 4, seq_len, seq_len)


def test_mask_propagation_classifier():
    """Test mask propagation in full classifier."""
    model = TransformerClassifier(vocab_size=30522, d_model=128, num_heads=4, num_layers=2)
    model.eval()

    seq_len = 10
    input_ids = torch.randint(0, 30522, (1, seq_len))

    # Mask key index 8 and 9
    mask = torch.tensor([[[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]]], dtype=torch.int64)

    logits, attentions = model(input_ids, mask=mask)

    assert logits.shape == (1, 2)
    assert torch.all(attentions[0][:, :, :, 8] < 1e-6)
    assert torch.all(attentions[0][:, :, :, 9] < 1e-6)
    assert torch.all(attentions[1][:, :, :, 8] < 1e-6)
    assert torch.all(attentions[1][:, :, :, 9] < 1e-6)


def test_all_encoder_layer_attentions_normalized():
    """Test that attention weights from every layer sum to 1.0 along key dimension."""
    model = TransformerClassifier(vocab_size=30522, d_model=128, num_heads=4, num_layers=2)
    model.eval()

    input_ids = torch.randint(0, 30522, (2, 15))
    _, attentions = model(input_ids)

    for idx, attn in enumerate(attentions):
        sum_weights = attn.sum(dim=-1)
        expected_ones = torch.ones_like(sum_weights)
        torch.testing.assert_close(sum_weights, expected_ones, rtol=1e-5, atol=1e-5)


def test_classifier_eval_determinism():
    """Test deterministic evaluation output."""
    model = TransformerClassifier(vocab_size=30522, d_model=128, dropout=0.2)
    model.eval()

    input_ids = torch.randint(0, 30522, (2, 20))
    logits1, att1 = model(input_ids)
    logits2, att2 = model(input_ids)

    torch.testing.assert_close(logits1, logits2)
    for a1, a2 in zip(att1, att2):
        torch.testing.assert_close(a1, a2)


def test_classifier_cuda_compatibility():
    """Smoke test for full classifier on GPU if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    model = TransformerClassifier(vocab_size=30522, d_model=128, num_heads=4, num_layers=2).to("cuda")
    model.eval()

    input_ids = torch.randint(0, 30522, (2, 20), device="cuda")
    logits, attentions = model(input_ids)

    assert logits.device.type == "cuda"
    assert logits.shape == (2, 2)
    assert len(attentions) == 2
    assert attentions[0].device.type == "cuda"
    assert attentions[0].shape == (2, 4, 20, 20)
