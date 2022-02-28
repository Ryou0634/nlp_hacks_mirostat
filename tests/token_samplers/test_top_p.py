import torch

from token_samplers.top_p import TopPSampling, mask_logits_top_p


def test_mask_with_single_instance():
    logits = torch.arange(5).float()
    # >>> torch.softmax(torch.arange(5).float(), dim=-1)
    # tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])
    assert (mask_logits_top_p(logits, p=0.9) == torch.Tensor([-torch.inf, -torch.inf, 2, 3, 4])).all()


def test_mask_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    assert (
        mask_logits_top_p(logits, p=0.9)
        == torch.Tensor([[-torch.inf, -torch.inf, 2, 3, 4], [-torch.inf, 4, -torch.inf, 3, 2]])
    ).all()


def test_sampling_with_single_instance():
    logits = torch.tensor([0, 4, 1, 3, 2]).float()
    sampler = TopPSampling(p=0.0)
    assert (sampler(logits) == torch.tensor(1)).all()


def test_sampling_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    sampler = TopPSampling(p=0.0)
    assert (sampler(logits) == torch.tensor([4, 1])).all()
