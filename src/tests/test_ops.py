import torch
import unittest

from src.ops import masked_mean_over_sequence_dim


class TestOps(unittest.TestCase):
    def test_masked_mean_over_sequence_dim_equivalent(self):
        seed = 42
        rng = torch.Generator().manual_seed(seed)
        b, s, d = 8, 128, 8
        x = torch.randn(b, s, d, generator=rng) * 5
        mask = torch.randn(b, s, generator=rng) > -0.5
        mean = x[mask].mean(0).mean()
        mean_hat = masked_mean_over_sequence_dim(x, mask).mean()
        self.assertAlmostEqual(mean.item(), mean_hat.item(), places=4)
