from dataclasses import dataclass
import json
import torch

@dataclass

class Dataset:
    """An SAE dataset"""
    gt_binary_vectors: torch.Tensor
    dense_vectors: torch.Tensor
    gt_dictionary: torch.Tensor
    sparsity: float

    def to_file(self, path: str):
        """Write the dataset to a pt file."""
        data = {
            "gt_binary_vectors": self.gt_binary_vectors,
            "dense_vectors": self.dense_vectors,
            "gt_dictionary": self.gt_dictionary,
            "sparsity": self.sparsity,
        }
        torch.save(data, path)
    
    @classmethod
    def from_file(cls, path: str):
        """Read the dataset from a pt file."""
        with open(path, "rb") as f:
            data = torch.load(f)
        return cls(
            gt_binary_vectors=data["gt_binary_vectors"],
            dense_vectors=data["dense_vectors"],
            gt_dictionary=data["gt_dictionary"],
            sparsity=data["sparsity"],
        )


class DatasetGenerator:
    """Generate a toy dataset for testing the SAE model.

    We generate a set of n + k dense vectors of dimension n and then sample
    sparse binary vectors of dim n, and construct examples by adding
    the associated dense vectors."""

    def __init__(self, n_dims: int, n_surplus: int, seed: int = 42):
        """
        Args:
            n_dims: dimension of the dense vectors
            n_surplus: number of surplus dense vectors
            seed: random seed
        """
        self.n_dims = n_dims
        self.n_surplus = n_surplus
        self.seed = seed
        self.rng = torch.random.manual_seed(self.seed)
        self.gt_dictionary = self._generate_gt_dictionary(n_dims, n_surplus)

    def generate_dataset(self, n_examples: int, sparse_fraction: float):
        """Generate a dataset of examples.

        Args:
            n_examples: number of examples to generate
            sparse_fraction: fraction of entries in the binary vectors that are non-zero
        """
        n_dict, n_dims = self.gt_dictionary.shape
        binary_vectors = torch.rand(n_examples, n_dict, generator=self.rng)
        binary_vectors = (binary_vectors < sparse_fraction).int()

        examples = []
        for binary_vector in binary_vectors:
            index_vector = _binary_vector_to_index_vector(binary_vector)
            selected_vectors = self.gt_dictionary[index_vector]
            example = torch.sum(selected_vectors, dim=0)
            examples.append(example)
        tensor_examples = torch.stack(examples)
        # post-conditions
        assert tensor_examples.shape == (n_examples, n_dims), "shape incorrect"
        return Dataset(
            gt_binary_vectors=binary_vectors,
            dense_vectors=tensor_examples,
            gt_dictionary=self.gt_dictionary,
            sparsity=sparse_fraction,
        )

    def _generate_gt_dictionary(self, n_dims: int, n_surplus: int):
        """Generate n_dims + n_surplus dense vectors of dimension n_dims.

        TODO: change this to generate nearly-orthogonal vectors more
        thoughtfully."""
        gt_dictionary = 2 * (-0.5 + torch.rand(n_dims + n_surplus, n_dims, generator=self.rng))
        # rescale each vector to be unit length
        gt_dictionary = gt_dictionary / torch.norm(gt_dictionary, dim=1).reshape(-1, 1)

        # post-conditions
        assert gt_dictionary.shape == (n_dims + n_surplus, n_dims), "shape incorrect"
        assert torch.allclose(
            torch.norm(gt_dictionary, dim=1), torch.ones(n_dims + n_surplus)
        ), "not unit length"
        return gt_dictionary

def _binary_vector_to_index_vector(binary_vector: torch.Tensor) -> torch.Tensor:
    """Convert a binary vector indicating which vectors to select into an indexing
    vector that can retrieve them.
    
    Example:
        binary_vector = torch.tensor([0, 1, 0, 1])
        index_vector = torch.tensor([1, 3])
    """
    # pre-conditions
    assert binary_vector.ndim == 1, "binary_vector must be 1D"
    assert binary_vector.dtype in [torch.int32, torch.int64], "binary_vector must be int"
    assert torch.all((binary_vector == 0) | (binary_vector == 1)), "binary_vector must be binary"

    index_vector = torch.nonzero(binary_vector).reshape(-1)
    assert index_vector.ndim == 1, "index_vector must be 1D"
    return index_vector


def main():
    n_dims = 10
    n_surplus = 5

    gen = DatasetGenerator(n_dims=n_dims, n_surplus=n_surplus)
    print(gen.gt_dictionary.shape)
    for i in range(10):
        print(i, torch.dot(gen.gt_dictionary[i], gen.gt_dictionary[i]))
        for j in range(i + 1, 10):
            print(i, j, torch.dot(gen.gt_dictionary[i], gen.gt_dictionary[j]))


if __name__ == "__main__":
    main()
