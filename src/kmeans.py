import einx
import torch


def random_sample(x, k):
    n, _ = x.shape
    idx = torch.randperm(n, device=x.device)[:k]
    return x[idx]


def kmeans(
    samples,
    num_clusters=8,
    num_iters=10,
):
    n, d = samples.shape
    dtype, device = (
        samples.dtype,
        samples.device,
    )

    means = random_sample(samples, num_clusters)

    for _ in range(num_iters):
        # L2 similarity
        sims = einx.subtract("k d, n d -> k n d", means, samples)
        sims = sims**2
        sims = einx.mean("k n [d]", sims)

        # k n -> n
        buckets = torch.argmin(sims, dim=0)
        bins = torch.zeros(num_clusters, device=device, dtype=dtype)
        values = torch.ones(n, device=device, dtype=dtype)
        bins.scatter_add_(0, buckets, values)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = torch.zeros(num_clusters, d, device=device, dtype=dtype)
        new_means.scatter_add_(0, einx.rearrange("n -> n d", buckets, d=d), samples)
        new_means = einx.divide("n d, n", new_means, bins_min_clamped)

        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, buckets, bins
