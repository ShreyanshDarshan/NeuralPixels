import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import positional_encoding as pos_encode


class ImageDataset(Dataset):

    def __init__(self, image_path):
        self.image = io.imread(image_path, True)

    def __len__(self):
        return self.image.size

    def __getitem__(self, index):
        y_coord = index // self.image.shape[1]
        x_coord = index % self.image.shape[1]
        grey_color = torch.tensor(self.image[y_coord, x_coord],
                                  dtype=torch.float32)
        coords = torch.tensor([x_coord, y_coord]) / torch.tensor(
            [self.image.shape[1], self.image.shape[0]])
        coords = coords.type(torch.float32)
        return (pos_encode.encode(coords), grey_color)

    def visualize(self):
        io.imshow(self.image)
        io.show()