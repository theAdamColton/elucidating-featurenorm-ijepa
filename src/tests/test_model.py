import unittest

import torch
import torch.nn.functional as F

from src.model import masked_mean_var


class TestModel(unittest.TestCase):
    def test_masked_mean_var(self):
        b, s, d = 128, 64, 8
        x = torch.randn(b, s, d) * 3 + 2
        mags = x.abs().sum(-1)
        q = mags.quantile(0.1)
        mask = mags > q

        mean = x[mask].mean(0)
        var = x[mask].var(0)
        std = var**0.5

        mean_hat, var_hat = masked_mean_var(x, mask)
        mean_hat = mean_hat
        var_hat = var_hat
        std_hat = var_hat**0.5

        self.assertLess(F.mse_loss(mean, mean_hat).item(), 0.01)
        self.assertLess(F.mse_loss(std, std_hat).item(), 0.1)
