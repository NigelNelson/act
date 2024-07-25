import wandb
import subprocess
from wandb_imitate_episodes import main
import os

sweep_configuration = {
    "method": "random",
    "metric": {
        "name": "val_loss",
        "goal": "maximize"
    },
    "parameters": {
        "chunk_size": {
            "values" : [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
        },
        "kl_weight": {
            "values": [1, 5, 10, 20, 30, 50, 70, 100]
        }, 
        "batch_size": {
            "values" : [8, 16, 32, 64]
        },
        "learning_rate": {
            "values": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        },
        "hidden_dim": {
            "values": [512, 1024]
        }
    }
}
sweep_id = os.environ["SWEEP_ID"]

wandb.agent(sweep_id, main)  