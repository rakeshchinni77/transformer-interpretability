import inspect
import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from model import TransformerClassifier


def test_app_file_exists():
    """Verify that app.py exists in the project root."""
    app_path = PROJECT_ROOT / "app.py"
    assert app_path.exists(), f"Missing app.py in {PROJECT_ROOT}"
    assert app_path.stat().st_size > 0, "app.py is empty!"


def test_app_imports_and_components():
    """Verify app.py module imports cleanly and exposes load_model_and_tokenizer."""
    assert hasattr(app, "load_model_and_tokenizer")
    assert hasattr(app, "main")


def test_app_contains_required_dom_test_ids():
    """Verify that app.py exposes data-testid='attention-heatmap-container' and data-testid='entropy-dashboard-container'."""
    app_source = inspect.getsource(app)

    assert "data-testid=\"attention-heatmap-container\"" in app_source or "attention-heatmap-container" in app_source
    assert "data-testid=\"entropy-dashboard-container\"" in app_source or "entropy-dashboard-container" in app_source


def test_app_model_and_tokenizer_loading():
    """Verify load_model_and_tokenizer returns valid TransformerClassifier and AutoTokenizer."""
    model, tokenizer, device = app.load_model_and_tokenizer()

    assert isinstance(model, TransformerClassifier)
    assert tokenizer.vocab_size == 30522
    assert isinstance(device, torch.device)


def test_app_inference_pipeline():
    """Verify end-to-end inference pipeline from text tokenization to logits and attention weights."""
    model, tokenizer, device = app.load_model_and_tokenizer()

    text = "This movie was surprisingly good and the acting was excellent."
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits, attentions = model(input_ids, attention_mask=attention_mask)

    assert logits.shape == (1, 2)
    assert isinstance(attentions, list)
    assert len(attentions) == 2

    # Check layer 0 and layer 1 shapes
    assert attentions[0].shape == (1, 4, 128, 128)
    assert attentions[1].shape == (1, 4, 128, 128)

    # Check non-padded token count
    valid_len = int(attention_mask[0].sum().item())
    assert valid_len > 0
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:valid_len].tolist())
    assert len(tokens) == valid_len
    assert tokens[0] == tokenizer.cls_token
