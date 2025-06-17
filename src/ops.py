import torch


def masked_mean_along_sequence_dim(x, mask):
    """
    x: b s d
    mask: b s

    equivalent to:
    torch.stack([xi[mi].mean(0) for xi,mi in zip(x, mask)])

    but outputs zeros instead of nans
    """
    denom = torch.sum(mask, -1, keepdim=True)
    denom.clip_(1)
    x = x * mask.unsqueeze(-1)
    x = x.sum(1) / denom
    return x
