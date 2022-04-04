# server.py
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


def main() -> None:
    # Load model
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cifar.Net().to(DEVICE).train() 
  
    strategy = fl.server.strategy.FedAvg(
        eval_fn=get_eval_fn(model),   
    )

    fl.server.start_server("localhost:8080", config={"num_rounds": 1}, strategy=strategy)

def get_eval_fn(model):
  
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        print("Accuracy: ", accuracy)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

    return evaluate 


if __name__ == "__main__":
    main()
