import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_tokenizer, load_imdb_raw_dataset, tokenize_batch
from config.settings import settings


def main():
    print(f"Loading raw dataset '{settings.dataset_name}' via Hugging Face datasets...")
    raw_dataset = load_imdb_raw_dataset(settings.dataset_name)

    print("\n--- DATASET STRUCTURE ---")
    print(raw_dataset)

    train_size = len(raw_dataset["train"])
    test_size = len(raw_dataset["test"])
    print(f"Train split size: {train_size}")
    print(f"Test split size:  {test_size}")

    sample = raw_dataset["train"][0]
    sample_text_snippet = sample["text"][:120].replace("\n", " ") + "..."
    sample_label = sample["label"]
    label_str = "Positive (1)" if sample_label == 1 else "Negative (0)"

    print(f"\nSample 0 Text:  \"{sample_text_snippet}\"")
    print(f"Sample 0 Label: {sample_label} ({label_str})")

    print(f"\nLoading tokenizer '{settings.tokenizer_name}'...")
    tokenizer = get_tokenizer(settings.tokenizer_name)

    print("\n--- TOKENIZER SPECIAL TOKENS ---")
    print(f"Vocab size:     {tokenizer.vocab_size}")
    print(f"CLS token (ID): {tokenizer.cls_token} ({tokenizer.cls_token_id})")
    print(f"SEP token (ID): {tokenizer.sep_token} ({tokenizer.sep_token_id})")
    print(f"PAD token (ID): {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print(f"UNK token (ID): {tokenizer.unk_token} ({tokenizer.unk_token_id})")

    # Test tokenizing single sample
    encoded = tokenizer(
        sample["text"],
        padding="max_length",
        truncation=True,
        max_length=settings.max_len,
        return_tensors="pt",
    )

    print("\n--- TOKENIZED SAMPLE SHAPES ---")
    print(f"input_ids shape:      {encoded['input_ids'].shape}")
    print(f"attention_mask shape: {encoded['attention_mask'].shape}")

    print("\nDataset access and tokenization pipeline verified successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
