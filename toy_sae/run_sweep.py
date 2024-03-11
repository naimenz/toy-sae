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

    params_dict = {
        "learning_rate": {"min": 1e-8, "max": 1e-1, "distribution": "log_uniform_values"},
        "sparsity_penalty": {"min": 1e-8, "max": 1e2, "distribution": "log_uniform_values"},
        "n_epochs": {"min": 1, "max": 500, "distribution": "int_uniform"},
        "batch_size": {"values": [32, 64, 128, 256, 512, 1024], "distribution": "categorical"},
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