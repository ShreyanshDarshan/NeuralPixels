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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model params
dim = 30
input_size = 2 * (2 * dim)
output_size = 1
hidden_layers = [128] * 5
layers = [input_size] + hidden_layers + [output_size]

# training params
batch_size = (256)**2
num_epochs = 400
learning_rate = 0.001
image_path = os.path.join(os.path.dirname(__file__), 'data', 'apple_128.png')

train_dataset = ImageDataset(image_path)
utils.draw(train_dataset.get_image())
image_resolution = train_dataset.get_image_resolution()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset[0]

model = NeuPix(layers, dim).to(device)
utils.draw(utils.predict_image(model))

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

    if epoch % 20 == 0:
        utils.draw_non_blocking(utils.predict_image(model, image_resolution))

utils.check_accuracy(train_loader, model)
utils.draw(utils.predict_image(model))