# %% 
import torch

n_examples = 5
n_vectors = 100
n_dims = 10
fraction = 0.1
x = torch.arange(n_vectors*n_dims).reshape(n_vectors, n_dims)

binary_vectors = torch.rand(n_examples, n_vectors)
binary_vectors = (binary_vectors < fraction)
print(binary_vectors.shape)
for b in binary_vectors[-1:]:
    print(b.int())
    # print(x[b])
    # print(x[b].sum(dim=0))

# %%

import torch
x = torch.tensor([1, -100, 4, 5])
l1 = torch.sum(torch.abs(x))
print(l1)
# %%
from toy_sae.dataset_generation import Dataset
ds = Dataset.from_file("/tmp/test_dataset.pt")
print(ds)
# %%
import torch
bv = torch.tensor([1, 0, 1, 0, 1, 0])
print(bv)
print(torch.nonzero(bv).squeeze())
print(torch.tensor([1]).ndim)

# %%
import torch
from toy_sae.dataset_generation import Dataset, _binary_vector_to_index_vector
bv = torch.tensor([1])
inverted_bv = (~(bv.to(torch.bool))).to(torch.int)

iv = _binary_vector_to_index_vector(bv)
inverted_iv = _binary_vector_to_index_vector(inverted_bv)
print(bv, iv)
print(bv[iv])
print(bv[inverted_iv])

# %%
