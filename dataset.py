import os
import torch
from torch.utils.data import Dataset
from skimage import io
import positional_encoding as pos_encode


class ImageDataset(Dataset):

    def __init__(self, image_path):
        self.image = io.imread(image_path, True)
        self.image = self.image * 2.0 - 1.0

        self.data = torch.tensor(self.image, dtype=torch.float32)
        self.data = self.data.flatten(0)

        xs = torch.linspace(0, 1, steps=self.image.shape[1])
        ys = torch.linspace(0, 1, steps=self.image.shape[0])
        grid = torch.cartesian_prod(xs, ys)
        self.encoded = pos_encode.encode(grid)
        self.encoded = self.encoded.reshape(-1, 40)
        print  ("encoded shape ", self.encoded.shape)

    def __len__(self):
        return self.image.size

    def __getitem__(self, index):
        # y_coord = index // self.image.shape[1]
        # x_coord = index % self.image.shape[1]
        # coords = torch.tensor([x_coord, y_coord]) / torch.tensor(
        #     [self.image.shape[1], self.image.shape[0]])
        # coords = coords.type(torch.float32) * 2.0 - 1.0
        return (self.encoded[index], self.data[index])

    def visualize(self):
        io.imshow(self.image * 2 - 1)
        io.show()