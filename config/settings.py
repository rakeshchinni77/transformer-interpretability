import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Find project root and load .env file if available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


def _get_env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return str(val) if val is not None and val.strip() != "" else default


def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable '{key}' must be an integer, got '{val}'")


def _get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable '{key}' must be a float, got '{val}'")


@dataclass
class Settings:
    app_name: str = _get_env_str("APP_NAME", "Transformer Interpretability Dashboard")
    model_path: str = _get_env_str("MODEL_PATH", "models/final_model.pth")
    tokenizer_name: str = _get_env_str("TOKENIZER_NAME", "google-bert/bert-base-uncased")
    dataset_name: str = _get_env_str("DATASET_NAME", "stanfordnlp/imdb")

    max_len: int = _get_env_int("MAX_LEN", 128)
    d_model: int = _get_env_int("D_MODEL", 128)
    num_heads: int = _get_env_int("NUM_HEADS", 4)
    num_layers: int = _get_env_int("NUM_LAYERS", 2)
    d_ff: int = _get_env_int("D_FF", 256)
    dropout: float = _get_env_float("DROPOUT", 0.1)
    num_classes: int = _get_env_int("NUM_CLASSES", 2)

    train_batch_size: int = _get_env_int("TRAIN_BATCH_SIZE", 32)
    eval_batch_size: int = _get_env_int("EVAL_BATCH_SIZE", 32)
    epochs: int = _get_env_int("EPOCHS", 3)
    learning_rate: float = _get_env_float("LEARNING_RATE", 0.0005)
    warmup_steps: int = _get_env_int("WARMUP_STEPS", 500)
    max_grad_norm: float = _get_env_float("MAX_GRAD_NORM", 1.0)
    seed: int = _get_env_int("SEED", 42)
    device: str = _get_env_str("DEVICE", "auto")

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate settings types and numeric constraints."""
        if self.max_len <= 0:
            raise ValueError(f"MAX_LEN must be > 0, got {self.max_len}")
        if self.d_model <= 0:
            raise ValueError(f"D_MODEL must be > 0, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"NUM_HEADS must be > 0, got {self.num_heads}")
        if self.num_layers <= 0:
            raise ValueError(f"NUM_LAYERS must be > 0, got {self.num_layers}")
        if self.d_ff <= 0:
            raise ValueError(f"D_FF must be > 0, got {self.d_ff}")
        if self.num_classes <= 1:
            raise ValueError(f"NUM_CLASSES must be > 1, got {self.num_classes}")
        if self.train_batch_size <= 0:
            raise ValueError(f"TRAIN_BATCH_SIZE must be > 0, got {self.train_batch_size}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"EVAL_BATCH_SIZE must be > 0, got {self.eval_batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"EPOCHS must be > 0, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"LEARNING_RATE must be > 0, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError(f"WARMUP_STEPS must be >= 0, got {self.warmup_steps}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"MAX_GRAD_NORM must be > 0, got {self.max_grad_norm}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"DROPOUT must be in range [0, 1), got {self.dropout}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"D_MODEL ({self.d_model}) must be divisible by NUM_HEADS ({self.num_heads})"
            )

    @property
    def resolved_device(self) -> str:
        """Resolve the runtime execution device ('cuda' or 'cpu')."""
        dev = self.device.lower().strip()
        if dev == "cuda":
            return "cuda"
        elif dev == "cpu":
            return "cpu"
        elif dev == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        else:
            return dev


# Canonical settings instance
settings = Settings()
