import argparse
import math
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import settings
from data.dataset import get_imdb_dataloaders, get_tokenizer, load_imdb_raw_dataset, IMDBDataset
from model import TransformerClassifier

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = settings.seed):
    """Sets random seeds across random, numpy, PyTorch CPU, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_attention_entropy(attention_weights: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    """
    Computes Shannon Entropy H(p) = -sum(p * log(p)) over key dimension (dim=-1).

    Args:
        attention_weights (torch.Tensor): Attention weights tensor of shape (batch, heads, seq_len, seq_len).
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Mean scalar entropy for each head, shape (heads,).
    """
    p = attention_weights.clamp_min(epsilon)
    # Entropy per (batch_item, head, query_token)
    entropy_per_query = -(p * torch.log(p)).sum(dim=-1)
    # Average across batch (dim 0) and query tokens (dim 2) -> shape (heads,)
    mean_head_entropy = entropy_per_query.mean(dim=(0, 2))
    return mean_head_entropy


def get_warmup_inv_sqrt_lambda(warmup_steps: int = settings.warmup_steps):
    """
    Returns a learning rate schedule lambda function for LambdaLR:
    - Linear warmup from step 0 to warmup_steps.
    - Inverse square root decay after warmup_steps.
    Matches exact continuity at current_step == warmup_steps.
    """
    def lr_lambda(current_step: int) -> float:
        step = max(1, current_step)
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return math.sqrt(float(warmup_steps) / float(step))

    return lr_lambda


def build_model(vocab_size: int, device: torch.device):
    """Instantiates TransformerClassifier with configured hyperparameters and moves to target device."""
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=settings.d_model,
        num_heads=settings.num_heads,
        num_layers=settings.num_layers,
        d_ff=settings.d_ff,
        dropout=settings.dropout,
        num_classes=settings.num_classes,
        max_len=settings.max_len,
    )
    return model.to(device)


def build_optimizer(model: nn.Module, lr: float = settings.learning_rate):
    """Constructs AdamW optimizer for model parameters."""
    return torch.optim.AdamW(model.parameters(), lr=lr)


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int = settings.warmup_steps):
    """Constructs LambdaLR scheduler implementing linear warmup and inverse sqrt decay."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_warmup_inv_sqrt_lambda(warmup_steps)
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    max_grad_norm: float = settings.max_grad_norm,
):
    """Runs one training epoch, applying forward, backward, gradient clipping, optimizer step, and scheduler step."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    total_grad_norm = 0.0
    step_count = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()

        # Mandatory gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        total_grad_norm += grad_norm.item()
        step_count += 1

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    avg_grad_norm = total_grad_norm / max(1, step_count)
    current_lr = scheduler.get_last_lr()[0]

    return avg_loss, accuracy, avg_grad_norm, current_lr


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluates the model on evaluation split without gradient computation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits, _ = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)

    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, tokenizer_name: str, epoch: int, filepath: Path):
    """Saves model weights, config, tokenizer name, and epoch metadata into checkpoint file."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": settings.d_model,
            "num_heads": settings.num_heads,
            "num_layers": settings.num_layers,
            "d_ff": settings.d_ff,
            "dropout": settings.dropout,
            "num_classes": settings.num_classes,
            "max_len": settings.max_len,
            "vocab_size": model.token_embedding.num_embeddings,
        },
        "tokenizer_name": tokenizer_name,
        "epoch": epoch,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved cleanly to {filepath}")


def run_training(smoke: bool = False):
    """Main execution function for running training and generating Phase 11 interpretability artifacts."""
    set_seed(settings.seed)

    device_str = settings.resolved_device
    device = torch.device(device_str)

    print(f"--- TRANSFORMER TRAINING & ARTIFACT GENERATION ---")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = get_tokenizer(settings.tokenizer_name)

    if smoke:
        print("\nRunning SMOKE training mode (subset data, fast verification)...")
        raw_dataset = load_imdb_raw_dataset(settings.dataset_name)

        train_ds = IMDBDataset(raw_dataset["train"].select(range(128)), tokenizer, max_len=settings.max_len)
        test_ds = IMDBDataset(raw_dataset["test"].select(range(64)), tokenizer, max_len=settings.max_len)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
        epochs = 2
        checkpoint_path = MODELS_DIR / "smoke_model.pth"
        csv_path = LOGS_DIR / "smoke_training_metrics.csv"
        epoch_1_snap_path = SNAPSHOTS_DIR / "smoke_epoch_1_weights.pt"
        final_epoch_snap_path = SNAPSHOTS_DIR / "smoke_final_epoch_weights.pt"
    else:
        print("\nLoading full IMDB dataset and building DataLoaders...")
        train_loader, test_loader, tokenizer = get_imdb_dataloaders(
            train_batch_size=settings.train_batch_size,
            eval_batch_size=settings.eval_batch_size,
            max_len=settings.max_len,
            tokenizer_name=settings.tokenizer_name,
            dataset_name=settings.dataset_name,
        )
        epochs = settings.epochs
        checkpoint_path = MODELS_DIR / "final_model.pth"
        csv_path = LOGS_DIR / "training_metrics.csv"
        epoch_1_snap_path = SNAPSHOTS_DIR / "epoch_1_weights.pt"
        final_epoch_snap_path = SNAPSHOTS_DIR / "final_epoch_weights.pt"

    model = build_model(vocab_size=tokenizer.vocab_size, device=device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameter count: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=settings.learning_rate)

    warmup_steps = 10 if smoke else settings.warmup_steps
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps)

    # Initialize CSV metrics log file with mandatory header
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,layer,head,attention_entropy\n")

    # Select deterministic reference batch from test loader for attention snapshots and entropy metrics
    ref_batch = next(iter(test_loader))
    ref_input_ids = ref_batch["input_ids"][:8].to(device)
    ref_attention_mask = ref_batch["attention_mask"][:8].to(device)

    print("\nStarting Training & Artifact Generation...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, grad_norm, lr = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=settings.max_grad_norm,
        )

        eval_loss, eval_acc = evaluate(
            model=model,
            eval_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc * 100:.2f}% | "
            f"Grad Norm: {grad_norm:.4f} | LR: {lr:.6f}"
        )

        # Collect attention weights on reference batch under eval mode
        model.eval()
        with torch.no_grad():
            ref_logits, ref_attentions = model(ref_input_ids, attention_mask=ref_attention_mask)

        # 1. Compute and record Shannon entropy metrics into CSV
        with open(csv_path, "a", encoding="utf-8") as f:
            for layer_idx, layer_attn in enumerate(ref_attentions):
                head_entropies = compute_attention_entropy(layer_attn)
                for head_idx, entropy_val in enumerate(head_entropies.tolist()):
                    f.write(f"{epoch},{layer_idx},{head_idx},{entropy_val:.6f}\n")
            f.flush()

        # 2. Save genuine attention snapshots for epoch 1 and final epoch
        snapshot_payload = {
            "attention_weights": [attn.detach().cpu() for attn in ref_attentions],
            "input_ids": ref_input_ids.detach().cpu(),
            "attention_mask": ref_attention_mask.detach().cpu(),
            "epoch": epoch,
        }

        if epoch == 1:
            torch.save(snapshot_payload, epoch_1_snap_path)
            print(f"Snapshot saved cleanly to {epoch_1_snap_path}")

        if epoch == epochs:
            torch.save(snapshot_payload, final_epoch_snap_path)
            print(f"Snapshot saved cleanly to {final_epoch_snap_path}")

    # Save final model checkpoint
    save_checkpoint(model, settings.tokenizer_name, epochs, checkpoint_path)

    # Verify reload of saved checkpoint
    print("\nVerifying checkpoint reload...")
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        reload_model = TransformerClassifier(
            vocab_size=ckpt["config"]["vocab_size"],
            d_model=ckpt["config"]["d_model"],
            num_heads=ckpt["config"]["num_heads"],
            num_layers=ckpt["config"]["num_layers"],
            d_ff=ckpt["config"]["d_ff"],
            dropout=ckpt["config"]["dropout"],
            num_classes=ckpt["config"]["num_classes"],
            max_len=ckpt["config"]["max_len"],
        ).to(device)
        reload_model.load_state_dict(ckpt["model_state_dict"])
        reload_model.eval()

        with torch.no_grad():
            reload_logits, reload_attns = reload_model(ref_input_ids, attention_mask=ref_attention_mask)

        print(f"Reload model inference successful! Logits shape: {reload_logits.shape}")

    print("--- TRAINING & ARTIFACT GENERATION COMPLETE ---")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Transformer Encoder Training & Artifact Generation Pipeline")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke-training on a tiny subset of IMDB dataset for verification.",
    )
    args = parser.parse_args()
    run_training(smoke=args.smoke)


if __name__ == "__main__":
    main()
