from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

from .language_model import Context, LanguageModel


class TransformersLM(LanguageModel):
    def __init__(self, model_name: str, device: torch.device = torch.device("cpu")):
        super().__init__()
        if "rinna" in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.device = device
        self.model.to(self.device)

    def prepare_context(self, text: str = None, batch_size: int = 1) -> Context:
        context_text = text or self.tokenizer.bos_token
        context_tokens = self.tokenizer.encode(context_text, add_special_tokens=False)
        input_ids = torch.tensor([context_tokens for _ in range(batch_size)]).to(self.device)

        return {"input_ids": input_ids, "past_key_values": None}

    def forward(self, context: Context) -> Tuple[torch.Tensor, Context]:
        forward = self.model(return_dict=True, **context)
        logits = forward.logits[:, -1, :]
        return logits, {"past_key_values": forward.past_key_values}

    def update_context(self, last_generate_id: torch.Tensor, context: Context):
        context["input_ids"] = last_generate_id.unsqueeze(-1)
        return context

    def ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_text(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids)
