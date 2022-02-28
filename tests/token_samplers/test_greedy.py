import torch

from token_samplers.greedy import GreedyDecoding, mask_logits_greedy


def test_mask_with_single_instance():
    logits = torch.arange(5).float()
    assert (mask_logits_greedy(logits) == torch.Tensor([-torch.inf, -torch.inf, -torch.inf, -torch.inf, 4])).all()


def test_mask_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    assert (
        mask_logits_greedy(logits)
        == torch.Tensor(
            [[-torch.inf, -torch.inf, -torch.inf, -torch.inf, 4], [-torch.inf, 4, -torch.inf, -torch.inf, -torch.inf]]
        )
    ).all()

    # make sure this is not an in-place operation
    assert (logits == torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()).all()


def test_sampling_with_single_instance():
    logits = torch.tensor([0, 4, 1, 3, 2]).float()
    sampler = GreedyDecoding()
    assert (sampler(logits) == torch.tensor(1)).all()


def test_sampling_with_batched_instance():
    logits = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 1, 3, 2]]).float()
    sampler = GreedyDecoding()
    assert (sampler(logits) == torch.tensor([4, 1])).all()
