import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from image_dataset import ImageDataset
import os, sys
import numpy as np
import skimage.io as io


class NeuPix(nn.Module):

    def __init__(self, layers):
        super(NeuPix, self).__init__()
        # use torch.nn.sequential
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.sequential(x)
        return out
