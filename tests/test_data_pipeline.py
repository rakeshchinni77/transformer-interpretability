import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader
from data.dataset import (
    get_tokenizer,
    load_imdb_raw_dataset,
    tokenize_batch,
    IMDBDataset,
    get_imdb_dataloaders,
)
from model import TransformerClassifier
from config.settings import settings


def test_imdb_raw_dataset_loading():
    """Test loading Stanford IMDB raw dataset structure, splits, and labels."""
    raw_dataset = load_imdb_raw_dataset()

    assert "train" in raw_dataset
    assert "test" in raw_dataset
    assert "unsupervised" in raw_dataset

    assert len(raw_dataset["train"]) == 25000
    assert len(raw_dataset["test"]) == 25000

    sample = raw_dataset["train"][0]
    assert "text" in sample
    assert "label" in sample
    assert sample["label"] in (0, 1)


def test_unsupervised_split_exclusion():
    """Test that get_imdb_dataloaders processes ONLY train (25k) and test (25k) splits, excluding unsupervised (50k)."""
    tokenizer = get_tokenizer()
    raw_dataset = load_imdb_raw_dataset()

    # Use small subset for fast assertion
    train_ds = IMDBDataset(raw_dataset["train"].select(range(10)), tokenizer, max_len=settings.max_len)
    test_ds = IMDBDataset(raw_dataset["test"].select(range(10)), tokenizer, max_len=settings.max_len)

    assert len(train_ds) == 10
    assert len(test_ds) == 10
    assert not hasattr(train_ds, "unsupervised")


def test_tokenizer_properties():
    """Test tokenizer loading, vocabulary size, and special token IDs."""
    tokenizer = get_tokenizer()

    assert tokenizer.vocab_size == 30522
    assert tokenizer.cls_token_id is not None
    assert tokenizer.sep_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert tokenizer.unk_token_id is not None

    assert tokenizer.cls_token_id == 101
    assert tokenizer.sep_token_id == 102
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 100


def test_tokenized_single_sample_shapes():
    """Test tokenization of text into settings.max_len (128) max length tensors."""
    tokenizer = get_tokenizer()
    text = "This movie was absolutely wonderful and brilliant!"

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=settings.max_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)

    assert input_ids.shape == (settings.max_len,)
    assert attention_mask.shape == (settings.max_len,)
    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[1] != tokenizer.pad_token_id
    assert attention_mask[0] == 1


def test_imdb_dataset_batch_shapes():
    """Test IMDBDataset PyTorch wrapper batch output shapes."""
    tokenizer = get_tokenizer()
    raw_dataset = load_imdb_raw_dataset()

    small_split = raw_dataset["train"].select(range(32))
    dataset = IMDBDataset(small_split, tokenizer, max_len=settings.max_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (32, settings.max_len)
    assert batch["attention_mask"].shape == (32, settings.max_len)
    assert batch["labels"].shape == (32,)

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_real_batch_custom_model_integration():
    """Smoke test: Pass ONE REAL BATCH from DataLoader into custom TransformerClassifier with attention_mask keyword arg."""
    tokenizer = get_tokenizer()
    raw_dataset = load_imdb_raw_dataset()

    small_split = raw_dataset["train"].select(range(8))
    dataset = IMDBDataset(small_split, tokenizer, max_len=settings.max_len)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        num_classes=2,
        max_len=settings.max_len,
    )
    model.eval()

    with torch.no_grad():
        logits, attentions = model(
            input_ids,
            attention_mask=attention_mask,
        )

    batch_size = input_ids.size(0)
    assert input_ids.shape == (batch_size, 128)
    assert attention_mask.shape == (batch_size, 128)
    assert labels.shape == (batch_size,)

    assert logits.shape == (batch_size, 2)
    assert len(attentions) == 2
    assert attentions[0].shape == (batch_size, 4, 128, 128)
    assert attentions[1].shape == (batch_size, 4, 128, 128)


def test_padding_behavior_and_mask_propagation():
    """Verify that padding positions (where attention_mask == 0) receive ~0 attention weight in custom attention."""
    tokenizer = get_tokenizer()

    # Short real review text guaranteed to be < 128 tokens
    short_text = "Great movie! Highly recommended."
    encoded = tokenizer(
        short_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]        # shape [1, 128]
    attention_mask = encoded["attention_mask"] # shape [1, 128]

    # Confirm there are zero-padded positions
    pad_positions = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
    assert len(pad_positions) > 0, "Short text should produce zero-padded positions!"

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        num_classes=2,
        max_len=128,
    )
    model.eval()

    with torch.no_grad():
        logits, attentions = model(input_ids, attention_mask=attention_mask)

    pad_pos = pad_positions[0].item()
    # Check that attention to padded key position is near zero (< 1e-5) for layer 0 and layer 1
    assert torch.all(attentions[0][0, :, :, pad_pos] < 1e-5), "Layer 0 assigned attention to padded position!"
    assert torch.all(attentions[1][0, :, :, pad_pos] < 1e-5), "Layer 1 assigned attention to padded position!"


def test_attention_mask_gradient_flow():
    """Verify forward and backward pass with attention_mask keyword arg produce valid gradients."""
    tokenizer = get_tokenizer()
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 128))
    attention_mask = torch.tensor([[1] * 100 + [0] * 28, [1] * 80 + [0] * 48], dtype=torch.long)
    labels = torch.tensor([1, 0], dtype=torch.long)

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        num_classes=2,
        max_len=128,
    )
    model.train()

    logits, attentions = model(input_ids, attention_mask=attention_mask)
    loss = nn.CrossEntropyLoss()(logits, labels)

    assert torch.isfinite(loss)

    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_no_pretrained_transformer_instantiated():
    """Verify that no pretrained Transformer models (BertModel, AutoModel, etc.) are instantiated in data pipeline."""
    import data.dataset as dataset_mod
    import inspect

    source = inspect.getsource(dataset_mod)
    forbidden_classes = [
        "BertModel",
        "AutoModel",
        "BertForSequenceClassification",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    ]
    for forbidden in forbidden_classes:
        assert forbidden not in source, f"Forbidden pretrained model class '{forbidden}' referenced in data/dataset.py!"
