import torch
import matplotlib.pyplot as plt
from model import NeuPix
from torch.utils.data import DataLoader
import positional_encoding as pos_encode
import numpy as np


def draw_non_blocking(image: np.ndarray):
    plt.imshow(image, cmap='gray')
    plt.draw()
    plt.pause(0.0000001)


def draw(image: np.ndarray):
    plt.imshow(image, cmap='gray')
    plt.show()


def predict_image(model: torch.nn.Module, resolution: list = [128, 128]):
    device = torch.device(
        'cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
    output_image = None
    with torch.no_grad():
        xs = torch.linspace(0, 1, steps=resolution[0]).to(device)
        ys = torch.linspace(0, 1, steps=resolution[1]).to(device)
        grid = torch.cartesian_prod(xs, ys)

        vals = model(grid)

        output_image = torch.reshape(
            vals, (resolution[1], resolution[0])).cpu().numpy()
    return output_image


def check_accuracy(loader: DataLoader, model: torch.nn.Module):
    device = torch.device(
        'cuda' if next(model.parameters()).is_cuda else 'cpu')

    model.eval()
    with torch.no_grad():
        mse = 0
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            data = torch.reshape(data, (data.shape[0], -1))
            target = torch.reshape(target, (target.shape[0], 1))

            preds = model(data)
            mse += torch.mean(torch.square(preds - target))

        mse /= float(loader.batch_size)
    print('MSE: %.2f%%', (mse))
