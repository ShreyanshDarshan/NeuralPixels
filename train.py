import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
from tqdm import tqdm
from dataset import ImageDataset
import os
import sys
# import numpy as np
from model import NeuPix
import utils
import positional_encoding as pos_encode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a random tensor of size 2
x = torch.rand(3, 3, 2)
print(x)
pos_encode.encode(x)
# sys.exit(0)

# model params
dim = 10
input_size = 2 * (2 * dim)
output_size = 1
hidden_layers = [128] * 2
layers = [input_size] + hidden_layers + [output_size]

# training params
batch_size = (128)**2
num_epochs = 300
learning_rate = 0.01
image_path = os.path.join(os.path.dirname(__file__), 'data', 'apple_128.png')

train_dataset = ImageDataset(image_path)
train_dataset.visualize()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset[0]

# test_dataset = ImageDataset()
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = NeuPix(layers).to(device)
utils.plot_output(model)

# utils.check_accuracy(train_loader, model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        data = torch.reshape(data, (data.shape[0], -1))
        target = torch.reshape(target, (target.shape[0], 1))

        preds = model(data)

        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

utils.check_accuracy(train_loader, model)
utils.plot_output(model)
# check_accuracy(test_loader, model)