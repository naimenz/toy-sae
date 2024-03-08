
import torch
from toy_sae.dataset_generation import Dataset, DatasetGenerator, _binary_vector_to_index_vector
from hypothesis import given
import hypothesis.strategies as st


def test_shapes():
    n_dims = 10
    n_surplus = 5
    n_examples = 4
    gen = DatasetGenerator(n_dims=n_dims, n_surplus=n_surplus)
    dataset = gen.generate_dataset(n_examples=n_examples, sparse_fraction=0.1)
    assert dataset.gt_binary_vectors.shape == (n_examples, n_dims + n_surplus)
    assert dataset.dense_vectors.shape == (n_examples, n_dims)
    assert dataset.gt_dictionary.shape == (n_dims + n_surplus, n_dims)

def test_unit_binary_to_index():
    bv = torch.tensor([1, 0, 1, 0, 1, 0])
    iv = _binary_vector_to_index_vector(bv)
    assert torch.all(iv == torch.tensor([0, 2, 4]))

@given(st.lists(st.integers(min_value=0, max_value=1), min_size=1))
def test_property_binary_to_index(binary_list):
    bv = torch.tensor(binary_list).to(torch.int32)
    inverted_bv = (~(bv.to(torch.bool))).to(torch.int32)
    iv = _binary_vector_to_index_vector(bv)
    inverted_iv = _binary_vector_to_index_vector(inverted_bv)
    assert torch.all(bv[iv] == 1)
    assert torch.all(bv[inverted_iv] == 0)  

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

