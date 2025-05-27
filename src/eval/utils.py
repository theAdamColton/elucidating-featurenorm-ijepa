import torch


def scale_to_zero_one(x, q=0.99):
    max = torch.quantile(x, q)
    min = torch.quantile(x, 1 - q)
    return ((x - min) / (max - min)).clip(0, 1)
