import torch
from toy_sae.dictionary_score import get_dictionary_score


def test_basic():
    gt_vectors = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(torch.float32)
    gt_vectors = gt_vectors / torch.norm(gt_vectors, dim=1).reshape(-1, 1)
    weights = gt_vectors.clone()
    score = get_dictionary_score(weights, gt_vectors)
    assert score == 1.

def test_permutation():
    gt_vectors = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(torch.float32)
    gt_vectors = gt_vectors / torch.norm(gt_vectors, dim=1).reshape(-1, 1)

    weights = torch.tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]]).to(torch.float32)
    weights = weights / torch.norm(weights, dim=1).reshape(-1, 1)
    score = get_dictionary_score(weights, gt_vectors)
    assert score == 1.

def test_extra():
    gt_vectors = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).to(torch.float32)
    gt_vectors = gt_vectors / torch.norm(gt_vectors, dim=1).reshape(-1, 1)

    weights = torch.tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]]).to(torch.float32)
    weights = weights / torch.norm(weights, dim=1).reshape(-1, 1)
    score = get_dictionary_score(weights, gt_vectors)
    assert score == 1.

def test_negative():
    gt_vectors = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(torch.float32)
    gt_vectors = gt_vectors / torch.norm(gt_vectors, dim=1).reshape(-1, 1)

    weights = torch.tensor([[10, 5, 6], [7, 8, 9], [1, 2, 3]]).to(torch.float32)
    weights = weights / torch.norm(weights, dim=1).reshape(-1, 1)
    score = get_dictionary_score(weights, gt_vectors)
    assert score < 1.