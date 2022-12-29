import torch


def encode(t: torch.Tensor, dim: int = 10):
    """Encode a tensor with a sinusoidal positional encoding.
    Args:
        t: tensor to encode
        dim: dimension of the encoding
    Returns:
        encoded tensor
    """
    # if dim % 2 != 0:
    #     raise ValueError("dim must be even")
    t_dim = t.dim()
    device = t.device
    pi_fac = torch.pow(2, torch.arange(0, dim)) * torch.pi
    broadcast_t = t.unsqueeze(t_dim)  #.broadcast_to(-1, dim)  #.repeat(1, dim)
    scaled_t = broadcast_t @ pi_fac.unsqueeze(0).to(device)
    sin_t = torch.sin(scaled_t)
    cos_t = torch.cos(scaled_t)
    stacked_t = torch.stack((sin_t, cos_t), -1)
    return stacked_t.flatten(t_dim - 1, -1)
