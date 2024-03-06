
import torch
from toy_sae.dataset_generation import Dataset, DatasetGenerator


def test_shapes():
    n_dims = 10
    n_surplus = 5
    n_examples = 4
    gen = DatasetGenerator(n_dims=n_dims, n_surplus=n_surplus)
    dataset = gen.generate_dataset(n_examples=n_examples, sparse_fraction=0.1)
    assert dataset.gt_binary_vectors.shape == (n_examples, n_dims + n_surplus)
    assert dataset.dense_vectors.shape == (n_examples, n_dims)
    assert dataset.gt_dictionary.shape == (n_dims + n_surplus, n_dims)

def test_loading():
    n_dims = 10
    n_surplus = 5
    n_examples = 4
    gen = DatasetGenerator(n_dims=n_dims, n_surplus=n_surplus)
    dataset = gen.generate_dataset(n_examples=n_examples, sparse_fraction=0.1)
    dataset.to_file("/tmp/test_dataset.pt")
    dataset2 = Dataset.from_file("/tmp/test_dataset.pt")
    assert torch.allclose(dataset2.gt_binary_vectors, dataset.gt_binary_vectors)
    assert torch.allclose(dataset2.dense_vectors, dataset.dense_vectors)
    assert torch.allclose(dataset2.gt_dictionary, dataset.gt_dictionary)

