from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer


def train_data_load(batch_size, ts, DATA_DIR):
    training_dataloader = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=ts),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
    return training_dataloader

def val_data_load(batch_size, ts, DATA_DIR):
    validation_dataloader = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )
    return validation_dataloader

    