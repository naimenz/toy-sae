import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    """A simple sparse autoencoder model."""
    def __init__(self, n_dims: int, n_hidden: int):
        """
        Args:
            n_dims: dimension of the input vectors
            n_hidden: number of hidden units
        """
        super().__init__()
        # using a tied weight matrix
        scale = 1.0 / torch.sqrt(torch.tensor(n_dims))
        self.W = nn.Parameter(scale * 2 * (-0.5 + torch.rand(n_dims, n_hidden)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # pre-conditions
        assert x.shape[1] == self.W.shape[0], "input shape incorrect"
        raw_hidden = x @ self.W
        hidden_activations = F.relu(raw_hidden)
        out = hidden_activations @ self.W.t()
        # post-conditions
        assert out.shape == x.shape, "output shape incorrect"
        return out