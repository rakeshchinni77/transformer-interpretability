# IMDB Dataset & Tokenization Documentation

## Overview
This directory handles the data preparation pipeline for the Transformer Interpretability project.

## Dataset Details
- **Identifier**: `stanfordnlp/imdb` (Stanford IMDB Movie Review Dataset)
- **Source**: Hugging Face Datasets (`datasets.load_dataset("stanfordnlp/imdb")`)
- **Splits**:
  - `train`: 25,000 labeled movie reviews
  - `test`: 25,000 labeled movie reviews
- **Labels**:
  - `0`: Negative sentiment
  - `1`: Positive sentiment

## Tokenizer Specification
- **Tokenizer**: `google-bert/bert-base-uncased` (loaded via `transformers.AutoTokenizer`)
- **Policy**: The tokenizer is used **only** for converting text into token IDs. No pretrained Transformer model (e.g. `BertModel`, `AutoModel`) is used. The custom PyTorch Transformer in `model.py` serves as the actual architecture.
- **Vocabulary Size**: 30,522 tokens
- **Special Tokens**:
  - `[CLS]` (ID: 101) - Start of sequence token
  - `[SEP]` (ID: 102) - End of sequence token
  - `[PAD]` (ID: 0)   - Zero-padding token
  - `[UNK]` (ID: 100) - Unknown token

## Tokenization & Batching Strategy
- **Max Sequence Length**: `128` tokens
- **Padding**: `"max_length"` (padded to 128 with `pad_token_id=0`)
- **Truncation**: `True` (truncated to 128 tokens)
- **PyTorch Tensors**: `input_ids` (`torch.long`), `attention_mask` (`torch.long`), `labels` (`torch.long`)
- **Batch Sizes**: Configurable via `config/settings.py` (Default train: `32`, eval: `32`)

## Repository & Cache Policy
- **Git Policy**: The IMDB dataset files and tokenizer caches are **not** committed to Git.
- **Cache Handling**: Hugging Face `datasets` handles local caching automatically in the user's cache directory.

## Verification & Usage
Run the dataset access and tokenization verification script:
```powershell
python scripts/download_data.py
```
