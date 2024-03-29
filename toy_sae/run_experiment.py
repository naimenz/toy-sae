import fire

from toy_sae.dataset_generation import DatasetGenerator
from toy_sae.sae import SAE
from toy_sae.trainer import Trainer, TrainingConfig


def run_exp(
        n_dims: int,
        n_surplus: int,
        n_examples: int,
        sparse_fraction: float,
        n_hidden: int,
        learning_rate: float,
        sparsity_penalty: float,
        n_epochs: int,
        batch_size: int,
        optimizer: str,
        seed: int,
        **kwargs,  # needed for making a partial and using wandb.sweep
):
    gen = DatasetGenerator(n_dims, n_surplus, seed)
    dataset = gen.generate_dataset(n_examples, sparse_fraction)
    valid_dataset = gen.generate_dataset(n_examples, sparse_fraction)
    model = SAE(n_dims, n_hidden)
    trainer = Trainer(model, dataset, valid_dataset)
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        sparsity_penalty=sparsity_penalty,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    trainer.train(training_config)

if __name__ == "__main__":
    fire.Fire(run_exp)