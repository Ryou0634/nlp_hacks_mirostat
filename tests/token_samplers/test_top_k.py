import torch

from token_samplers.top_k import TopKSampling, mask_logits_top_k


def test_mask_with_single_instance():
    logits = torch.arange(5).float()
    assert (mask_logits_top_k(logits, k=3) == torch.tensor([-torch.inf, -torch.inf, 2, 3, 4])).all()


def test_mask_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    assert (
        mask_logits_top_k(logits, k=3)
        == torch.tensor([[-torch.inf, -torch.inf, 2, 3, 4], [-torch.inf, 4, -torch.inf, 3, 2]])
    ).all()


def test_sampling_with_single_instance():
    logits = torch.tensor([0, 4, 1, 3, 2]).float()
    sampler = TopKSampling(k=1)
    assert (sampler(logits) == torch.tensor(1)).all()


def test_sampling_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    sampler = TopKSampling(k=1)
    assert (sampler(logits) == torch.tensor([4, 1])).all()
