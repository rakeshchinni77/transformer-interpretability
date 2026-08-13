import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from config.settings import settings


def get_tokenizer(tokenizer_name: str = settings.tokenizer_name):
    """
    Loads and returns the BERT tokenizer specified in configuration.
    Only the tokenizer is loaded from Hugging Face transformers.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_imdb_raw_dataset(dataset_name: str = settings.dataset_name):
    """
    Loads the Stanford IMDB dataset programmatically via Hugging Face datasets.
    Dataset is cached locally by Hugging Face datasets and not committed to Git.
    """
    return load_dataset(dataset_name)


def tokenize_batch(examples, tokenizer, max_len: int = settings.max_len):
    """
    Tokenizes a batch of text examples with padding and truncation.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )


class IMDBDataset(Dataset):
    """
    Memory-conscious PyTorch Dataset wrapper around tokenized Hugging Face IMDB dataset split.
    Avoids copying the entire dataset into separate Python structures.
    """

    def __init__(self, hf_dataset_split, tokenizer, max_len: int = settings.max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        def _encode(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )

        # Tokenize only the provided split in batched mode and format directly as PyTorch tensors
        self.dataset = hf_dataset_split.map(_encode, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["label"],
        }


def get_imdb_dataloaders(
    train_batch_size: int = settings.train_batch_size,
    eval_batch_size: int = settings.eval_batch_size,
    max_len: int = settings.max_len,
    tokenizer_name: str = settings.tokenizer_name,
    dataset_name: str = settings.dataset_name,
):
    """
    Prepares PyTorch DataLoader instances for train and test splits of the IMDB dataset.
    Excludes the 50,000 unsupervised examples completely to optimize memory and speed.

    Returns:
        tuple[DataLoader, DataLoader, AutoTokenizer]:
            - train_loader
            - test_loader
            - tokenizer instance
    """
    tokenizer = get_tokenizer(tokenizer_name)
    raw_dataset = load_imdb_raw_dataset(dataset_name)

    # Process ONLY train and test splits, explicitly excluding unsupervised split
    train_dataset = IMDBDataset(raw_dataset["train"], tokenizer, max_len=max_len)
    test_dataset = IMDBDataset(raw_dataset["test"], tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, tokenizer
