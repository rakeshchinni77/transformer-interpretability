import math
import torch
import torch.nn.functional as F


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
