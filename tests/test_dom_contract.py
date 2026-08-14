import csv
import inspect
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from model import TransformerClassifier


def test_dom_contract_heatmap_container_in_app_source():
    """Verify that app.py explicitly renders div with data-testid='attention-heatmap-container'."""
    source = inspect.getsource(app)
    assert 'data-testid="attention-heatmap-container"' in source
    assert 'id="attention-heatmap-container"' in source


def test_dom_contract_entropy_container_in_app_source():
    """Verify that app.py explicitly renders div with data-testid='entropy-dashboard-container'."""
    source = inspect.getsource(app)
    assert 'data-testid="entropy-dashboard-container"' in source
    assert 'id="entropy-dashboard-container"' in source


def test_layer_selection_affects_attention_matrix():
    """Verify selecting Layer 0 vs Layer 1 extracts corresponding layer attention tensor."""
    model, tokenizer, device = app.load_model_and_tokenizer()

    text = "This movie was surprisingly good and the acting was excellent."
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        _, attentions = model(input_ids, attention_mask=attention_mask)

    assert len(attentions) == 2  # 2 encoder layers

    layer0_attn = attentions[0][0, 0].cpu().numpy()  # Layer 0, Head 0
    layer1_attn = attentions[1][0, 0].cpu().numpy()  # Layer 1, Head 0

    assert layer0_attn.shape == (128, 128)
    assert layer1_attn.shape == (128, 128)
    assert not (layer0_attn == layer1_attn).all(), "Layer 0 and Layer 1 attention matrices should be distinct"


def test_head_selection_affects_attention_matrix():
    """Verify selecting Head 0 vs Head 3 extracts corresponding head attention tensor."""
    model, tokenizer, device = app.load_model_and_tokenizer()

    text = "This movie was surprisingly good and the acting was excellent."
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        _, attentions = model(input_ids, attention_mask=attention_mask)

    head0_attn = attentions[0][0, 0].cpu().numpy()  # Layer 0, Head 0
    head3_attn = attentions[0][0, 3].cpu().numpy()  # Layer 0, Head 3

    assert head0_attn.shape == (128, 128)
    assert head3_attn.shape == (128, 128)
    assert not (head0_attn == head3_attn).all(), "Head 0 and Head 3 attention matrices should be distinct"


def test_new_review_submission_recomputes_inference():
    """Verify submitting a different review recomputes tokens, predictions, and attention weights dynamically."""
    model, tokenizer, device = app.load_model_and_tokenizer()

    review_pos = "This movie was surprisingly good and the acting was excellent."
    review_neg = "The movie was disappointing and poorly acted."

    encoded_pos = tokenizer(review_pos, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    encoded_neg = tokenizer(review_neg, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        logits_pos, attns_pos = model(encoded_pos["input_ids"].to(device), attention_mask=encoded_pos["attention_mask"].to(device))
        logits_neg, attns_neg = model(encoded_neg["input_ids"].to(device), attention_mask=encoded_neg["attention_mask"].to(device))

    probs_pos = F.softmax(logits_pos, dim=-1)[0]
    probs_neg = F.softmax(logits_neg, dim=-1)[0]

    assert probs_pos[1] > probs_pos[0], "Positive review should yield positive sentiment"
    assert probs_neg[0] > probs_neg[1], "Negative review should yield negative sentiment"

    valid_len_pos = int(encoded_pos["attention_mask"][0].sum().item())
    valid_len_neg = int(encoded_neg["attention_mask"][0].sum().item())
    assert valid_len_pos != valid_len_neg, "Valid token length should recompute per review"


def test_entropy_csv_schema_contract():
    """Verify logs/training_metrics.csv exists with exact 4 columns and 24 data rows."""
    csv_path = PROJECT_ROOT / "logs" / "training_metrics.csv"
    assert csv_path.exists(), f"Missing CSV file: {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        assert header == ["epoch", "layer", "head", "attention_entropy"]

        rows = list(reader)
        assert len(rows) == 24, f"Expected 24 data rows, got {len(rows)}"

        epoch_counts = {}
        for r in rows:
            ep = int(r[0])
            epoch_counts[ep] = epoch_counts.get(ep, 0) + 1

        assert epoch_counts == {1: 8, 2: 8, 3: 8}
