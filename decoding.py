from dataclasses import dataclass
from typing import List

import torch

from language_models import LanguageModel
from token_samplers import PureSampling, TokenSampler


@dataclass
class DecodingOutput:
    ids: torch.Tensor
    tokens: List[List[str]]
    text: List[str]
    logits: torch.Tensor


@torch.no_grad()
def decoding(
    language_model: LanguageModel,
    context_text: str = None,
    token_sampler: TokenSampler = None,
    batch_size: int = 1,
    max_num_tokens: int = 900,
) -> DecodingOutput:

    token_sampler = token_sampler or PureSampling()
    token_sampler.reset()

    context = language_model.prepare_context(context_text, batch_size=batch_size)
    sampled_ids = []
    logits_list = []
    for _ in range(max_num_tokens):
        logits, context = language_model.forward(context)
        logits_list.append(logits)

        sampled_id = token_sampler(logits)
        sampled_ids.append(sampled_id.squeeze(-1))

        context = language_model.update_context(sampled_id, context)
    sampled_ids = torch.stack(sampled_ids, dim=1)
    logits_list = torch.stack(logits_list, dim=1)
    return DecodingOutput(
        ids=sampled_ids,
        tokens=[language_model.ids_to_tokens(ids) for ids in sampled_ids],
        text=[language_model.ids_to_text(ids) for ids in sampled_ids],
        logits=logits_list,
    )
