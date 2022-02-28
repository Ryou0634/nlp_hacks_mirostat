import math

import torch

from .token_sampler import TokenSampler, sample_from_logits


def mask_logits_greedy(logits: torch.Tensor) -> torch.Tensor:
    max_logits, _ = logits.max(dim=-1)
    mask = torch.full_like(logits, fill_value=-math.inf)
    processed_logits = torch.where(logits == max_logits.unsqueeze(-1), logits, mask)
    return processed_logits


class GreedyDecoding(TokenSampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        processed_logits = mask_logits_greedy(logits)
        return sample_from_logits(processed_logits)
