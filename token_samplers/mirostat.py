import math
from typing import Union

import torch

from .token_sampler import TokenSampler, sample_from_logits


def estimate_zipf_coefficient(probs: torch.Tensor, truncation: int = 100) -> torch.Tensor:
    sorted_probs, _ = torch.sort(probs, descending=True)
    truncation = min(truncation, sorted_probs.size(-1) - 1)
    b = sorted_probs[..., :truncation] / sorted_probs[..., 1 : truncation + 1]
    t = torch.arange(2, truncation + 2) / torch.arange(1, truncation + 1)
    t = t.to(probs.device)
    log_b = torch.log(b)
    log_t = torch.log(t)
    return (log_b * log_t).sum(dim=-1) / (log_t**2).sum()


def compute_k(
    vocab_size: int, zipf_coefficient: torch.Tensor, max_surprise: Union[float, torch.Tensor]
) -> torch.Tensor:
    eps = zipf_coefficient - 1
    k = ((eps * (2**max_surprise)) / (1 - vocab_size ** (-eps))) ** (1 / zipf_coefficient)
    k = torch.round(k)
    return k


def mask_logits_top_ks(logits: torch.Tensor, ks: torch.Tensor, masked_value: float = -math.inf) -> torch.Tensor:

    if not ks.size() == (len(logits),):
        raise ValueError("If k is a tensor, it must be 1d vector with the batch size.")

    rank_of_logits = torch.argsort(logits, descending=True, dim=-1).argsort(dim=-1)
    mask = torch.full_like(logits, fill_value=masked_value)
    processed_logits = torch.where(rank_of_logits < ks.unsqueeze(-1), logits, mask)
    return processed_logits


class Mirostat(TokenSampler):
    """
    Mirostat decoding that controls the surprisal of generated sentences.
    adapted from https://github.com/basusourya/mirostat/blob/master/mirostat.py
    """

    def __init__(self, target_surprisal: float, learning_rate: float = 1.0):
        self.target_surprisal = target_surprisal
        self.learning_rate = learning_rate

        self._current_max_surprisal = 2 * self.target_surprisal

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:

        probs = torch.softmax(logits, dim=-1)

        # determine k
        zipf_coefficient = estimate_zipf_coefficient(probs)
        ks = (
            compute_k(
                vocab_size=probs.size(-1),
                zipf_coefficient=zipf_coefficient,
                max_surprise=self._current_max_surprisal,
            )
            + 1
        )

        # sampling from top-k
        processed_logits = mask_logits_top_ks(logits, ks)
        generated_id = sample_from_logits(processed_logits)

        # update the next max surprise
        probs_of_generated_id = probs[torch.arange(len(generated_id)), generated_id]
        surprisal = torch.log2(1 / probs_of_generated_id)
        self._current_max_surprisal -= self.learning_rate * (surprisal - self.target_surprisal)
        return generated_id

    def reset(self):
        self._current_max_surprisal = 2 * self.target_surprisal
