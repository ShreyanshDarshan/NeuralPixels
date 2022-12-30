import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset import ImageDataset
import os, sys
import numpy as np
import skimage.io as io
import positional_encoding as pos_encode


class NeuPix(nn.Module):

    def __init__(self, layers, encoding_dim=10):
        super(NeuPix, self).__init__()

        if layers[0] % 2 * encoding_dim != 0:
            raise ValueError("Input size must be a multiple of 2*encoding_dim")
        self.encoding_dim = encoding_dim

        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        encoded = pos_encode.encode(x, self.encoding_dim)
        out = self.sequential(encoded)
        return out
