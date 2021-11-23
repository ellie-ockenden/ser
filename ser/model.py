from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

def make_model(learning_rate):
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer