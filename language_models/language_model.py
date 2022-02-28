from abc import ABCMeta, abstractmethod
from typing import List, Tuple, TypeVar

import torch

Context = TypeVar("Context")


class LanguageModel(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare_context(self, text: str, batch_size: int) -> Context:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, context: Context) -> Tuple[torch.Tensor, Context]:
        raise NotImplementedError()

    @abstractmethod
    def update_context(self, last_generate_id: torch.Tensor, context: Context) -> Context:
        raise NotImplementedError()

    @abstractmethod
    def ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def ids_to_text(self, ids: torch.Tensor) -> str:
        raise NotImplementedError()
