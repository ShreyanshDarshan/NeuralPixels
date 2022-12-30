import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import positional_encoding as pos_encode


class ImageDataset(Dataset):

    def __init__(self, image_path):
        self.image = plt.imread(image_path)
        self.image = self.image[:, :, 0]
        if self.image is None:
            raise ValueError("Image not found")
        else:
            print("Image loaded. Image shape: ", self.image.shape)

        self.image = self.image * 2.0 - 1.0
        self.x_res = self.image.shape[1]
        self.y_res = self.image.shape[0]

        self.data = torch.tensor(self.image, dtype=torch.float32)
        self.data = self.data.flatten(0)

        xs = torch.linspace(0, 1, steps=self.image.shape[1])
        ys = torch.linspace(0, 1, steps=self.image.shape[0])
        self.grid = torch.cartesian_prod(xs, ys).reshape(-1, 2)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return (self.grid[index], self.data[index])

    def get_image(self):
        return self.image

    def get_image_resolution(self):
        return [self.x_res, self.y_res]