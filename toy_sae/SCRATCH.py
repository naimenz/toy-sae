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
