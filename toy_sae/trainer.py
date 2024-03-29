from dataclasses import dataclass
import random
from typing import Optional
from toy_sae.dataset_generation import Dataset
from toy_sae.dictionary_score import get_dictionary_score
from toy_sae.sae import SAE
import torch
import wandb
from tqdm import tqdm
from wandb import sdk as wandb_sdk

Optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}


@dataclass
class TrainingConfig:
    learning_rate: float
    sparsity_penalty: float
    n_epochs: int
    batch_size: int
    optimizer: str = "SGD"


class Trainer:
    """Trainer to run SAE training."""

    def __init__(self, model: SAE, dataset: Dataset, valid_dataset: Optional[Dataset]):
        """
        Args:
            model: the SAE model
            dataset: the dataset
            valid_dataset: the validation dataset
        """
        self.model = model
        self.dataset = dataset
        self.valid_dataset = valid_dataset

    def train(self, training_config: TrainingConfig | None):
        config = self._start_wandb_run(training_config)
        optim_f = Optimizers[config.optimizer]
        optimizer = optim_f(self.model.parameters(), lr=config.learning_rate)
        for epoch in tqdm(range(config.n_epochs)):
            if self.valid_dataset is not None:
                valid_loss, (valid_mse, valid_sparsity) = self._compute_validation_loss(
                    self.model
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "val_loss": valid_loss,
                        "val_mse_loss": valid_mse,
                        "val_sparsity_loss": valid_sparsity,
                    }
                )

            self._train_epoch(self.model, self.dataset, optimizer, epoch, config)
        self._log_final_metrics()
    
    def _log_final_metrics(self):
        if self.valid_dataset is not None:
            valid_loss, (valid_mse, valid_sparsity) = self._compute_validation_loss(self.model)
            wandb.summary["final_val_loss"] = valid_loss
            wandb.summary["final_val_mse_loss"] = valid_mse
            wandb.summary["final_val_sparsity_loss"] = valid_sparsity
        # compute the final dictionary score
        dictionary_score = get_dictionary_score(self.model.W.T, self.dataset.gt_dictionary)
        wandb.summary["final_dictionary_score"] = dictionary_score


    def _compute_validation_loss(self, model: SAE) -> tuple[float, tuple[float, float]]:
        assert self.valid_dataset is not None
        with torch.no_grad():
            valid_dataset = self.valid_dataset.dense_vectors
            out = model(valid_dataset)
            loss, (mse_loss, sparsity_loss) = self._loss(out, valid_dataset, model, 0.0)
            return loss.item(), (mse_loss.item(), sparsity_loss.item())

    def _train_epoch(
        self,
        model: SAE,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: wandb_sdk.Config,
    ):
        """Train a single epoch."""
        all_batch_indices = self._get_batch_indices(
            config.batch_size, dataset.dense_vectors.shape[0]
        )
        n_batches = len(all_batch_indices)
        batch_offset = epoch * n_batches
        for i, batch_indices in enumerate(all_batch_indices):
            batch_inputs = dataset.dense_vectors[batch_indices]
            train_loss, (mse_loss, sparsity_loss) = self._train_batch(
                model, optimizer, batch_inputs, config.sparsity_penalty
            )
            dictionary_score = get_dictionary_score(model.W.T, dataset.gt_dictionary)
            wandb.log(
                {
                    "batch": batch_offset + i,
                    "train_mse_loss": mse_loss,
                    "train_sparsity_loss": sparsity_loss,
                    "train_loss": train_loss,
                    "train_dictionary_score": dictionary_score,
                }
            )

    def _train_batch(
        self,
        model: SAE,
        optimizer: torch.optim.Optimizer,
        batch_inputs: torch.Tensor,
        sparsity_penalty: float,
    ) -> tuple[float, tuple[float, float]]:
        out = model(batch_inputs)
        # since it's an autoencoder, the input is the target
        loss, (mse_loss, sparsity_loss) = self._loss(out, batch_inputs, model, sparsity_penalty)
        loss.backward()
        optimizer.step()
        return loss.item(), (mse_loss.item(), sparsity_loss.item())

    def _loss(
        self, out: torch.Tensor, batch_inputs: torch.Tensor, model: SAE, sparsity_penalty: float
    ):
        """Loss is an MSE reconstruction loss plus a sparsity penalty."""
        mse_loss = torch.mean((out - batch_inputs) ** 2)
        # NOTE: anthropic uses sum while Logan uses mean;
        # mean makes more sense to me
        sparsity_loss = sparsity_penalty * torch.mean(torch.abs(model.W))
        loss = mse_loss + sparsity_loss
        return loss, (mse_loss, sparsity_loss)

    def _get_batch_indices(self, batch_size: int, n_examples: int):
        """Return a shuffled list of indices for all batches in an epoch."""
        indices = list(range(n_examples))
        random.shuffle(indices)
        return [indices[i : i + batch_size] for i in range(0, n_examples, batch_size)]

    def _start_wandb_run(self, training_config: TrainingConfig | None) -> wandb_sdk.Config:
        config = {
            "dataset_size": self.dataset.dense_vectors.shape[0],
            "dataset_n_dims": self.dataset.dense_vectors.shape[1],
            "dataset_sparsity_fraction": self.dataset.sparsity,
            "model_n_hidden": self.model.W.shape[1],
        }
        # optionally add training_config params (not necessary if e.g. in a sweep)
        if training_config is not None:
            config.update(
                {
                    "training_learning_rate": training_config.learning_rate,
                    "training_sparsity_penalty": training_config.sparsity_penalty,
                    "training_n_epochs": training_config.n_epochs,
                    "training_batch_size": training_config.batch_size,
                    "training_optimizer": training_config.optimizer,
                }
            )
        wandb.init(project="toy-sae", entity="naimenz", config=config, reinit=True)
        wandb.watch(self.model)
        return wandb.config

    def _finish_wandb_run(self):
        wandb.finish()
