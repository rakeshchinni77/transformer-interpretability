import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import settings


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes Scaled Dot-Product Attention from scratch.

    Args:
        query (torch.Tensor): Query tensor of shape (..., query_len, d_k)
        key (torch.Tensor): Key tensor of shape (..., key_len, d_k)
        value (torch.Tensor): Value tensor of shape (..., key_len, d_v)
        mask (torch.Tensor, optional): Mask tensor where 1 indicates allowed position,
                                       and 0 indicates masked position. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - output tensor of shape (..., query_len, d_v)
            - attention weights tensor of shape (..., query_len, key_len)
    """
    if query.size(-1) != key.size(-1):
        raise ValueError(f"Query d_k ({query.size(-1)}) and Key d_k ({key.size(-1)}) must match.")
    if key.size(-2) != value.size(-2):
        raise ValueError(
            f"Key sequence length ({key.size(-2)}) and Value sequence length ({value.size(-2)}) must match."
        )

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module implemented from scratch.

    Splits embedding dimension (d_model) into `h` parallel heads of dimension `d_k = d_model // h`,
    applies linear projections W_q, W_k, W_v, computes scaled dot-product attention per head,
    concatenates head outputs, and applies output linear projection W_o.
    """

    def __init__(
        self,
        d_model: int = settings.d_model,
        h: int = settings.num_heads,
        dropout: float = settings.dropout,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if h <= 0:
            raise ValueError(f"num_heads must be > 0, got {h}")
        if d_model % h != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({h}).")

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, query_seq_len, d_model)
            key (torch.Tensor): Key tensor of shape (batch, key_seq_len, d_model)
            value (torch.Tensor): Value tensor of shape (batch, key_seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask tensor (1=allowed, 0=masked).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - output tensor of shape (batch, query_seq_len, d_model)
                - attention weights tensor of shape (batch, h, query_seq_len, key_seq_len)
        """
        if query.size(-1) != self.d_model or key.size(-1) != self.d_model or value.size(-1) != self.d_model:
            raise ValueError(
                f"Feature dimension of query ({query.size(-1)}), key ({key.size(-1)}), "
                f"and value ({value.size(-1)}) must match d_model ({self.d_model})."
            )

        batch_size = query.size(0)

        # 1) Linear projections: (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2) Split heads: (batch, seq_len, h, d_k) -> transpose to (batch, h, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 3) Apply scaled dot-product attention
        x, p_attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Apply dropout if configured and in training mode
        if self.dropout is not None:
            x = self.dropout(x)

        # 4) Concatenate heads: (batch, h, query_seq_len, d_k) -> (batch, query_seq_len, h, d_k) -> (batch, query_seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5) Output projection: (batch, query_seq_len, d_model)
        output = self.W_o(x)

        return output, p_attn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module supporting both Sinusoidal (deterministic) and Learned strategies.

    Args:
        d_model (int): Model embedding dimension (default from settings.d_model).
        max_len (int): Maximum supported sequence length (default from settings.max_len).
        method (str): Strategy name, either 'sinusoidal' or 'learned' (default 'sinusoidal').
    """

    def __init__(
        self,
        d_model: int = settings.d_model,
        max_len: int = settings.max_len,
        method: str = "sinusoidal",
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len}")

        method_clean = method.lower().strip()
        if method_clean not in ("sinusoidal", "learned"):
            raise ValueError(f"Unsupported positional encoding method '{method}'. Supported methods: 'sinusoidal', 'learned'.")

        self.d_model = d_model
        self.max_len = max_len
        self.method = method_clean

        if self.method == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # Register as buffer so it moves with the module and appears in state_dict without gradient updates
            self.register_buffer("pe", pe)
        else:
            # Learned strategy uses trainable embedding lookup
            self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass injecting positional encodings into input token embeddings.

        Args:
            x (torch.Tensor): Tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Positionally encoded tensor of shape (batch, seq_len, d_model).
        """
        if x.dim() < 2 or x.size(-1) != self.d_model:
            raise ValueError(f"Input feature dimension ({x.size(-1)}) must match d_model ({self.d_model}).")

        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum allowed length ({self.max_len}).")

        if self.method == "sinusoidal":
            return x + self.pe[:seq_len]
        else:
            positions = torch.arange(seq_len, device=x.device)
            return x + self.position_embedding(positions)


class EncoderLayer(nn.Module):
    """
    Single Pre-LN Transformer Encoder Layer implemented from scratch.

    Contains:
    - LayerNorm before multi-head attention
    - Multi-head self-attention
    - Residual connection & Dropout
    - LayerNorm before feed-forward network
    - Position-wise Feed-Forward Network (Linear -> ReLU -> Linear)
    - Residual connection & Dropout
    """

    def __init__(
        self,
        d_model: int = settings.d_model,
        num_heads: int = settings.num_heads,
        d_ff: int = settings.d_ff,
        dropout: float = settings.dropout,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model=d_model, h=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for EncoderLayer using Pre-LN architecture.

        Args:
            x (torch.Tensor): Input hidden states of shape (batch, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - output hidden states of shape (batch, seq_len, d_model)
                - attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        # Sublayer 1: Pre-LN Multi-Head Self-Attention + Residual
        norm_x1 = self.norm1(x)
        attn_out, layer_attn = self.self_attn(norm_x1, norm_x1, norm_x1, mask=mask)
        x = x + self.dropout1(attn_out)

        # Sublayer 2: Pre-LN Feed-Forward + Residual
        norm_x2 = self.norm2(x)
        ff_out = self.feed_forward(norm_x2)
        x = x + self.dropout2(ff_out)

        return x, layer_attn


class TransformerEncoder(nn.Module):
    """
    Stacked Transformer Encoder layers built from scratch.

    Maintains a ModuleList of EncoderLayer modules and collects attention weights
    from every layer during forward execution.
    """

    def __init__(
        self,
        d_model: int = settings.d_model,
        num_heads: int = settings.num_heads,
        num_layers: int = settings.num_layers,
        d_ff: int = settings.d_ff,
        dropout: float = settings.dropout,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass iterating through all encoder layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                - final hidden states of shape (batch, seq_len, d_model)
                - list of attention weight tensors from every encoder layer
        """
        attentions = []
        for layer in self.layers:
            x, layer_attn = layer(x, mask=mask)
            attentions.append(layer_attn)
        return x, attentions


class TransformerClassifier(nn.Module):
    """
    Full Transformer Text Classifier combining:
    - Token Embedding
    - Positional Encoding
    - Custom Stacked Transformer Encoder
    - First-token CLS pooling & Linear Classification Head
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = settings.d_model,
        num_heads: int = settings.num_heads,
        num_layers: int = settings.num_layers,
        d_ff: int = settings.d_ff,
        dropout: float = settings.dropout,
        num_classes: int = settings.num_classes,
        max_len: int = settings.max_len,
        positional_encoding: str = "sinusoidal",
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_len, method=positional_encoding
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for complete classification model.

        Args:
            input_ids (torch.Tensor): Integer token IDs of shape (batch, seq_len).
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                - raw classification logits of shape (batch, num_classes)
                - list of attention weight tensors from all encoder layers
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be a 2D tensor (batch, seq_len), got shape {input_ids.shape}")

        seq_len = input_ids.size(1)
        if seq_len > self.positional_encoding.max_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_len ({self.positional_encoding.max_len})"
            )

        # 1) Token Embedding + Positional Encoding: (batch, seq_len, d_model)
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        # 2) Custom Transformer Encoder: (batch, seq_len, d_model), attentions
        hidden_states, attentions = self.encoder(x, mask=mask)

        # 3) CLS Token Representation (first sequence position): (batch, d_model)
        cls_rep = hidden_states[:, 0, :]

        # 4) Linear Classifier: (batch, num_classes)
        logits = self.classifier(cls_rep)

        return logits, attentions
