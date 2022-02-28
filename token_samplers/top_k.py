import math

import torch

from .token_sampler import TokenSampler, sample_from_logits


def mask_logits_top_k(logits: torch.Tensor, k: int, masked_value: float = -math.inf) -> torch.Tensor:

    rank_of_logits = torch.argsort(logits, descending=True, dim=-1).argsort(dim=-1)
    mask = torch.full_like(logits, fill_value=masked_value)
    processed_logits = torch.where(rank_of_logits < k, logits, mask)
    return processed_logits


class TopKSampling(TokenSampler):
    def __init__(self, k: int):
        if not (k > 0):
            raise ValueError(f"k ({k}) must be a positive integer.")
        self.k = k

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        processed_logits = mask_logits_top_k(logits, self.k)
        return sample_from_logits(processed_logits)
