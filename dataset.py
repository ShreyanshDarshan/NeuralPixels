import os
import torch
from torch.utils.data import Dataset
from skimage import io
import positional_encoding as pos_encode


class ImageDataset(Dataset):

    def __init__(self, image_path):
        self.image = io.imread(image_path, True)
        if self.image is None:
            raise ValueError("Image not found")
        else:
            print ("Image loaded. Image shape: ", self.image.shape)

        self.image = self.image * 2.0 - 1.0

        self.data = torch.tensor(self.image, dtype=torch.float32)
        self.data = self.data.flatten(0)

        xs = torch.linspace(0, 1, steps=self.image.shape[1])
        ys = torch.linspace(0, 1, steps=self.image.shape[0])
        self.grid = torch.cartesian_prod(xs, ys).reshape(-1, 2)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return (self.grid[index], self.data[index])

    def visualize(self):
        io.imshow(self.image * 2 - 1)
        io.show()