import math
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import pytest

from train import (
    set_seed,
    get_warmup_inv_sqrt_lambda,
    build_model,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    run_training,
)
from model import TransformerClassifier
from config.settings import settings


def test_scheduler_warmup_and_decay_math():
    """Test custom LR scheduler warmup linearity, exact boundary continuity, and inverse sqrt decay."""
    warmup_steps = 500
    lr_func = get_warmup_inv_sqrt_lambda(warmup_steps=warmup_steps)

    # Warmup phase: linear increase
    assert lr_func(0) == 1.0 / 500.0  # max(1, 0) / 500
    assert pytest.approx(lr_func(250)) == 0.5
    assert pytest.approx(lr_func(500)) == 1.0

    # Boundary continuity at step 500
    warmup_end = float(500) / 500.0
    decay_start = math.sqrt(500.0 / 500.0)
    assert pytest.approx(warmup_end) == decay_start == 1.0

    # Inverse square root decay phase
    # At step 2000: sqrt(500 / 2000) = sqrt(0.25) = 0.5
    assert pytest.approx(lr_func(2000)) == 0.5
    # At step 4500: sqrt(500 / 4500) = sqrt(1/9) = 1/3
    assert pytest.approx(lr_func(4500)) == 1.0 / 3.0

    # Verify monotonic decrease after warmup
    assert lr_func(500) > lr_func(1000) > lr_func(2000) > lr_func(5000)


def test_gradient_clipping():
    """Test that clip_grad_norm_ caps large parameter gradients to max_norm."""
    model = TransformerClassifier(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
    )
    # Inject large gradients into model parameters
    for p in model.parameters():
        p.grad = torch.full_like(p, fill_value=100.0)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert grad_norm.item() > 1.0  # Original norm was huge

    # Re-calculate norm after clipping
    clipped_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)).item()
    assert pytest.approx(clipped_norm, abs=1e-3) == 1.0


def test_single_training_step_parameter_update():
    """Test that a single training step (forward + backward + clip + opt.step) updates parameters."""
    set_seed(42)
    device = torch.device("cpu")
    model = build_model(vocab_size=1000, device=device)
    optimizer = build_optimizer(model, lr=0.01)
    scheduler = build_scheduler(optimizer, warmup_steps=10)

    input_ids = torch.randint(0, 1000, (2, 20))
    attention_mask = torch.ones(2, 20, dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)

    # Save snapshot of target weight tensor
    initial_weight = model.classifier.weight.clone().detach()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(input_ids, attention_mask=attention_mask)
    loss = nn.CrossEntropyLoss()(logits, labels)

    assert torch.isfinite(loss)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    updated_weight = model.classifier.weight.clone().detach()

    assert not torch.equal(initial_weight, updated_weight), "Parameters should be updated after optimizer.step()!"


def test_checkpoint_save_and_reload():
    """Test saving model state dict and config to checkpoint file and reloading for inference."""
    set_seed(42)
    device = torch.device("cpu")
    model = build_model(vocab_size=1000, device=device)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pth"
        save_checkpoint(model, tokenizer_name="google-bert/bert-base-uncased", epoch=1, filepath=ckpt_path)

        assert ckpt_path.exists()

        # Reload checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

        reconstructed_model = TransformerClassifier(
            vocab_size=checkpoint["config"]["vocab_size"],
            d_model=checkpoint["config"]["d_model"],
            num_heads=checkpoint["config"]["num_heads"],
            num_layers=checkpoint["config"]["num_layers"],
            d_ff=checkpoint["config"]["d_ff"],
            dropout=checkpoint["config"]["dropout"],
            num_classes=checkpoint["config"]["num_classes"],
            max_len=checkpoint["config"]["max_len"],
        ).to(device)

        reconstructed_model.load_state_dict(checkpoint["model_state_dict"])
        reconstructed_model.eval()

        input_ids = torch.randint(0, 1000, (2, 20))
        with torch.no_grad():
            logits, attns = reconstructed_model(input_ids)

        assert logits.shape == (2, 2)
        assert len(attns) == 2


def test_smoke_training_execution():
    """Test running run_training(smoke=True) execution end-to-end."""
    ckpt_path = run_training(smoke=True)
    assert ckpt_path.exists()
