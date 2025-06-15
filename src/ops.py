import torch
import einx


def masked_mean_over_sequence_dim(x, mask):
    b, s, d = x.shape

    if not torch.compiler.is_compiling():
        assert einx.matches("b s", mask, b=b, s=s)

    x = einx.multiply("b s d, b s", x, mask)
    x = einx.sum("b [s] d", x)

    num_masked = einx.sum("b [s]", mask)
    num_masked.clip_(1)
    x = einx.divide("b d, b", x, num_masked)
    return x
