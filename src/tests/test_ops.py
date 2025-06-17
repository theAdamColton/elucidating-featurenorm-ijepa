import torch
import unittest

from src.ops import masked_mean_along_sequence_dim


class TestOps(unittest.TestCase):
    def test_masked_mean_over_sequence_dim_equivalent(self):
        seed = 42
        rng = torch.Generator().manual_seed(seed)
        b, s, d = 32, 256, 32
        x = torch.randn(b, s, d, generator=rng) * 5
        # mostly masked
        mask = torch.randn(b, s, generator=rng) > -0.5
        # but set all of batch 0 to be excluded
        mask[0] = 0

        # b s d -> b d
        mean_hat = masked_mean_along_sequence_dim(x, mask)

        for i in range(b):
            # n d -> d
            mean = x[i][mask[i]].mean(0)

            if mean.mean().isnan():
                continue

            self.assertAlmostEqual(
                mean.mean().item(), mean_hat[i].mean().item(), places=5
            )
            self.assertTrue(torch.allclose(mean, mean_hat[i], rtol=0.001, atol=0.001))
