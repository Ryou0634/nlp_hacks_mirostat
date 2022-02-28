import math
from typing import List

import torch

from token_samplers.mirostat import (
    Mirostat,
    compute_k,
    estimate_zipf_coefficient,
    mask_logits_top_ks,
)


def estimate_zipf_coefficient_original_implementation(prob: List[float]) -> float:
    prob = sorted(prob, key=lambda x: -x)
    num = 0
    den = 0
    for i in range(100):
        b = prob[i] / prob[i + 1]
        t = (i + 2) / (i + 1)
        num += math.log(b) * math.log(t)
        den += math.log(t) ** 2
    return num / den


def compute_k_original(vocab_size: int, zipf_coefficient: float, max_surprise: float) -> int:
    eps = zipf_coefficient - 1
    try:
        k = ((eps * (2 ** (max_surprise))) / (1 - vocab_size ** (-eps))) ** (1 / zipf_coefficient)
        k = round(k)
    except OverflowError:
        k = vocab_size
    return k


def test_estimate_zipf_coefficient():
    prob = torch.softmax(torch.arange(101, 0, -1).float(), dim=-1)
    assert math.isclose(
        estimate_zipf_coefficient_original_implementation(prob.tolist()),
        estimate_zipf_coefficient(prob).item(),
        rel_tol=1e-6,
    )


def test_estimate_zipf_coefficient_with_batch():

    probs = torch.softmax(torch.stack([torch.arange(101, 0, -1), torch.arange(101, 0, -1) / 2]).float(), dim=-1)
    original_values = [estimate_zipf_coefficient_original_implementation(p.tolist()) for p in probs]

    assert [
        math.isclose(original, ours, rel_tol=1e-6)
        for original, ours in zip(original_values, estimate_zipf_coefficient(probs).tolist())
    ]


def test_compute_k_with_batch():
    vocab_size = 10000
    original_values = [
        compute_k_original(vocab_size=vocab_size, zipf_coefficient=1.2, max_surprise=3.0),
        compute_k_original(vocab_size=vocab_size, zipf_coefficient=1.5, max_surprise=4.0),
    ]

    assert (
        original_values
        == compute_k(
            vocab_size=vocab_size, zipf_coefficient=torch.tensor([1.2, 1.5]), max_surprise=torch.tensor([3.0, 4.0])
        ).tolist()
    )


def test_mask_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    assert (
        mask_logits_top_ks(logits, ks=torch.tensor([3, 2]))
        == torch.tensor([[-torch.inf, -torch.inf, 2, 3, 4], [-torch.inf, 4, -torch.inf, 3, -torch.inf]])
    ).all()


def test_sampling_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    sampler = Mirostat(target_surprisal=3.0)
    assert sampler(logits).size() == (2,)
