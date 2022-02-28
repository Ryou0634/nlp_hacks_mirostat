import math

import torch

from .token_sampler import TokenSampler, sample_from_logits


def mask_logits_top_p(logits: torch.Tensor, p: float, masked_value: float = -math.inf) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    top_k_indices = (cumulative_probs < p).sum(dim=-1) + 1

    rank_of_logits = sorted_indices.argsort(dim=-1)
    mask = torch.full_like(logits, fill_value=masked_value)
    processed_logits = torch.where(rank_of_logits < top_k_indices.unsqueeze(-1), logits, mask)
    return processed_logits


class TopPSampling(TokenSampler):
    def __init__(self, p: float):

        if not (0 <= p <= 1.0):
            raise ValueError(f"p ({p}) must be between 0 and 1.")

        self.p = p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        processed_logits = mask_logits_top_p(logits, self.p)
        return sample_from_logits(processed_logits)
