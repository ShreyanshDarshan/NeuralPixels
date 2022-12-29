import torch
import skimage.io as io
from model import NeuPix
from torch.utils.data import DataLoader
import positional_encoding as pos_encode


def plot_output(model: torch.nn.Module):
    device = torch.device(
        'cuda' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
    res = 512
    output_image = None
    with torch.no_grad():
        xs = torch.linspace(0, 1, steps=res).to(device)
        ys = torch.linspace(0, 1, steps=res).to(device)

        # grid_x, grid_y = torch.meshgrid(xs, ys)

        grid = torch.cartesian_prod(xs, ys)
        encoded_grid = pos_encode.encode(grid)
        vals = model(encoded_grid)
        output_image = torch.reshape(vals,
                                     (res, res)).cpu().numpy().transpose()

    io.imshow(output_image)
    io.show()


def check_accuracy(loader: DataLoader, model: torch.nn.Module):
    # if loader.dataset.train:
    #     print('Checking accuracy on training set')
    # else:
    #     print('Checking accuracy on test set')
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
            if (batch_idx == 0):
                print(data.shape)

            preds = model(data)
            mse += torch.mean(torch.square(preds - target))

        mse /= float(loader.batch_size)
    print('MSE: %.2f%%', (mse))
