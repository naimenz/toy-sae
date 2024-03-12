import fire
import wandb
from functools import partial
from toy_sae.dataset_generation import DatasetGenerator

from toy_sae.run_experiment import run_exp
from toy_sae.sae import SAE
from toy_sae.trainer import Trainer, TrainingConfig


def main(
    sweep_count: int,
    sweep_name: str,
    sweep_method: str,
    n_dims: int,
    n_surplus: int,
    n_examples: int,
    sparse_fraction: float,
    n_hidden: int,
    seed: int,
):
    wandb.login()
    sparsity_penalty_values = [
        0.0,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
        10.0,
    ]
    params_dict = {
        "learning_rate": {"values": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4], "distribution": "categorical"},
        "sparsity_penalty": {
            "values": sparsity_penalty_values, "distribution": "categorical",
        },
        "n_epochs": {"values": [32, 64, 128], "distribution": "categorical"},
        "batch_size": {"value": 256, "distribution": "constant"},
        "optimizer": {"value": "adam", "distribution": "constant"},
    }

    sweep_configuration = {
        "name": sweep_name,
        "method": sweep_method,
        "metric": {"goal": "maximize", "name": "final_dictionary_score"},
        "parameters": params_dict,
    }

    def sweep_fn():
        gen = DatasetGenerator(n_dims, n_surplus, seed)
        dataset = gen.generate_dataset(n_examples, sparse_fraction)
        valid_dataset = gen.generate_dataset(n_examples, sparse_fraction)
        model = SAE(n_dims, n_hidden)
        trainer = Trainer(model, dataset, valid_dataset)
        trainer.train(None)

    sweep_id = wandb.sweep(sweep_configuration, entity="naimenz", project="toy-sae")
    wandb.agent(sweep_id, sweep_fn, entity="naimenz", project="toy-sae", count=sweep_count)


if __name__ == "__main__":
    fire.Fire(main)
