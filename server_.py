# TypeError: evaluate() missing 2 required positional arguments: 'parameters' and 'config'

"""Flower server example."""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import flwr as fl
import cifar

USE_FEDBN: bool = True
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load data
trainloader, testloader, num_examples = cifar.load_data()
# Load model
model = cifar.Net().to(DEVICE).train()
# Create strategy
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)
def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=2,
        min_available_clients=10,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080", config={"num_rounds": 4}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    x_train,y_train, *c = trainset

    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
