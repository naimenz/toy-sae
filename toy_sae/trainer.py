from dataclasses import dataclass
import random
from toy_sae.dataset_generation import Dataset
from toy_sae.sae import SAE
import torch
import wandb


@dataclass
class TrainingConfig:
    learning_rate: float
    sparsity_penalty: float
    n_epochs: int
    batch_size: int


class Trainer:
    """Trainer to run SAE training."""

    def __init__(self, model: SAE, dataset: Dataset):
        """
        Args:
            model: the SAE model
            dataset: the dataset
            learning_rate: the learning rate
            sparsity_penalty: the sparsity penalty (coefficient of the L1 penalty)
        """
        self.model = model
        self.dataset = dataset

    def train(self, training_config: TrainingConfig):
        self._start_wandb_run(training_config)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=training_config.learning_rate)
        for epoch in range(training_config.n_epochs):
            self._train_epoch(self.model, self.dataset, optimizer, epoch, training_config)

    def _train_epoch(
        self,
        model: SAE,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        training_config: TrainingConfig,
    ):
        """Train a single epoch."""
        all_batch_indices = self._get_batch_indices(
            training_config.batch_size, dataset.dense_vectors.shape[0]
        )
        for i, batch_indices in enumerate(all_batch_indices):
            batch_inputs = dataset.dense_vectors[batch_indices]
            train_loss = self._train_batch(
                model, optimizer, batch_inputs, training_config.sparsity_penalty
            )
            wandb.log({"train-loss": train_loss})
            print(f"Epoch {epoch}, batch {i}, train loss: {train_loss}")

    def _train_batch(
        self,
        model: SAE,
        optimizer: torch.optim.Optimizer,
        batch_inputs: torch.Tensor,
        sparsity_penalty: float,
    ):
        out = model(batch_inputs)
        # since it's an autoencoder, the input is the target
        loss = self._loss(out, batch_inputs, model, sparsity_penalty)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _loss(
        self, out: torch.Tensor, batch_inputs: torch.Tensor, model: SAE, sparsity_penalty: float
    ):
        """Loss is an MSE reconstruction loss plus a sparsity penalty."""
        mse_part = torch.mean((out - batch_inputs) ** 2)
        # NOTE: anthropic uses sum while Logan uses mean;
        # mean makes more sense to me
        l1_part = sparsity_penalty * torch.mean(torch.abs(model.W))
        loss = mse_part + l1_part
        return loss

    def _get_batch_indices(self, batch_size: int, n_examples: int):
        """Return a shuffled list of indices for all batches in an epoch."""
        indices = list(range(n_examples))
        random.shuffle(indices)
        return [indices[i : i + batch_size] for i in range(0, n_examples, batch_size)]

    def _start_wandb_run(self, training_config: TrainingConfig):
        config = {
            "dataset_size": self.dataset.dense_vectors.shape[0],
            "dataset_n_dims": self.dataset.dense_vectors.shape[1],
            "dataset_sparsity_fraction": self.dataset.sparsity,
            "model_n_hidden": self.model.W.shape[1],
            "training_learning_rate": training_config.learning_rate,
            "training_sparsity_penalty": training_config.sparsity_penalty,
            "training_n_epochs": training_config.n_epochs,
            "training_batch_size": training_config.batch_size,
        }
        wandb.init(project="toy-sae", entity="naimenz", config=config, reinit=True)
        wandb.watch(self.model)

    def _finish_wandb_run(self):
        wandb.finish()
