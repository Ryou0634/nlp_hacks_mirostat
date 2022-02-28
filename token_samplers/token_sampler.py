from abc import ABCMeta, abstractmethod
from typing import TypeVar

import torch

Context = TypeVar("Context")


def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    generated_id = torch.multinomial(probs, num_samples=1, replacement=True)
    return generated_id.squeeze(-1)


class TokenSampler(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample token ids from a batch of logits.
        """
        raise NotImplementedError()

    def reset(self):
        pass


class PureSampling(TokenSampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return sample_from_logits(logits)
