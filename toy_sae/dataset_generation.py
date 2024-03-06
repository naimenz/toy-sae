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
        self.gt_dictionary = self._generate_dense_vectors(n_dims, n_surplus)

    def generate_dataset(self, n_examples: int, sparse_fraction: float):
        """Generate a dataset of examples.

        Args:
            n_examples: number of examples to generate
            sparse_fraction: fraction of entries in the binary vectors that are non-zero
        """
        rng = torch.random.manual_seed(self.seed)
        n_dict, n_dims = self.gt_dictionary.shape
        binary_vectors = torch.rand(n_examples, n_dict, generator=rng)
        binary_vectors = (binary_vectors < sparse_fraction).int()

        examples = []
        for binary_vector in binary_vectors:
            selected_vectors = self.gt_dictionary[binary_vector]
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

    def _generate_dense_vectors(self, n_dims: int, n_surplus: int):
        """Generate n_dims + n_surplus dense vectors of dimension n_dims.

        TODO: change this to generate nearly-orthogonal vectors more
        thoughtfully."""
        rng = torch.random.manual_seed(self.seed)
        dense_vectors = 2 * (-0.5 + torch.rand(n_dims + n_surplus, n_dims, generator=rng))
        # rescale each vector to be unit length
        dense_vectors = dense_vectors / torch.norm(dense_vectors, dim=1).reshape(-1, 1)

        # post-conditions
        assert dense_vectors.shape == (n_dims + n_surplus, n_dims), "shape incorrect"
        assert torch.allclose(
            torch.norm(dense_vectors, dim=1), torch.ones(n_dims + n_surplus)
        ), "not unit length"
        return dense_vectors


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
