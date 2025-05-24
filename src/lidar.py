"""
LiDAR: SENSING LINEAR PROBING PERFORMANCE IN
JOINT EMBEDDING SSL ARCHITECTURES

https://openreview.net/forum?id=f3g5XpL9Kb

PDF - https://openreview.net/notes/edits/attachment?id=47njBeMVsL&name=pdf
"""

import torch
import einx


def compute_lidar_score(features, delta=1e-6, eps=1e-6):
    """
    Compute LiDAR score

    Args:
        features: Tensor of shape (n, q, d) where:
            - n = number of samples (~1000)
            - q = number of augmentations per sample (~50)
            - d = feature dimension
        delta: Small constant for regularization of Σ_w
        eps: Small constant for entropy calculation

    Returns:
        lidar_score: The LiDAR score
    """
    n, q, d = features.shape

    # Step 1: Compute per-sample means from augmented features
    # μ_x = E[e(x̃)] for each clean sample x
    # Shape: (n, d)
    augmented_means = einx.mean("n [q] d", features)

    # Step 2: Compute global mean μ = E[μ_x]
    # Shape: (d,)
    global_mean = einx.mean("[n] d", augmented_means)

    # Step 3: Compute between-class covariance Σ_b
    # Σ_b = E[(μ_x - μ)(μ_x - μ)^T]
    # This measures how much the per-sample means vary from the global mean
    centered_sample_means = einx.subtract("n d, d", augmented_means, global_mean)
    sigma_b = einx.dot(
        "n d1, n d2 -> d1 d2", centered_sample_means, centered_sample_means
    ) / (n - 1)

    # Step 4: Compute within-class covariance Σ_w
    # Σ_w = E[E[(e(x̃) - μ_x)(e(x̃) - μ_x)^T | x]] + δI
    # This measures how much augmented features vary around their per-sample means

    # Center each augmented feature around its corresponding per-sample mean
    # Shape: (n, q, d)
    centered_augmented = einx.subtract("n q d, n d", features, augmented_means)

    # Compute within-class scatter: sum over all augmented samples
    # Reshape to (n*q, d) for efficient computation
    centered_flat = einx.rearrange("n q d -> (n q) d", centered_augmented)
    sigma_w = einx.dot("nq d1, nq d2 -> d1 d2", centered_flat, centered_flat) / (
        n * (q - 1)
    )

    # Add regularization to ensure positive definiteness
    sigma_w = sigma_w + delta * torch.eye(d, device=features.device)

    # Step 5: Compute LiDAR matrix Σ_lidar = Σ_w^(-1/2) Σ_b Σ_w^(-1/2)
    # This is the generalized eigenvalue problem solution

    # Compute Σ_w^(-1/2) using eigendecomposition
    eigenvals_w, eigenvecs_w = torch.linalg.eigh(sigma_w)
    eigenvals_w = torch.clamp(eigenvals_w, min=delta)  # Ensure positive definiteness

    # Σ_w^(-1/2) = Q Λ^(-1/2) Q^T
    sigma_w_inv_sqrt = einx.dot(
        "d1 k, k, k d2 -> d1 d2",
        eigenvecs_w,
        1.0 / torch.sqrt(eigenvals_w),
        eigenvecs_w.T,
    )

    # Compute the LiDAR matrix
    sigma_lidar = einx.dot(
        "d1 d2, d2 d3, d3 d4 -> d1 d4", sigma_w_inv_sqrt, sigma_b, sigma_w_inv_sqrt
    )

    # Step 6: Compute eigenvalues and LiDAR score
    eigenvals_lidar = torch.linalg.eigvals(sigma_lidar).real
    eigenvals_lidar = torch.clamp(eigenvals_lidar, min=0)

    # Normalize eigenvalues to get probabilities p_i = λ_i / ||λ||_1 + ε
    eigenvals_sum = eigenvals_lidar.sum()
    if eigenvals_sum > eps:
        probs = eigenvals_lidar / eigenvals_sum + eps
    else:
        # If all eigenvalues are essentially zero, return minimum score
        return torch.tensor(1.0, device=features.device)

    # Compute LiDAR score: exp(-Σ p_i log p_i)
    entropy = -torch.sum(probs * torch.log(probs))
    lidar_score = torch.exp(entropy)

    return lidar_score


def test_lidar():
    """Test the LiDAR implementation with different scenarios."""

    num_samples = 1000
    num_augmentations = 50
    feature_size = 64

    print("=== Testing LiDAR Implementation ===")

    # Test 1: Completely random features (should give low LiDAR score)
    print("\n1. Random features:")
    features = torch.randn(num_samples, num_augmentations, feature_size)

    lidar_score = compute_lidar_score(features)
    print(f"   LiDAR Score: {lidar_score:.4f}")

    # Test 2: Features that are more correlated with some ideal clean features
    print("\n2. Structured features:")
    clean_features = torch.randn(num_samples, feature_size)
    noise = torch.randn(num_samples, num_augmentations, feature_size) * 1e-2
    features = einx.add("n d, n q d", clean_features, noise)

    lidar_score_structured = compute_lidar_score(features)
    print(f"   LiDAR Score: {lidar_score_structured:.4f}")

    # Test 3: Low rank random features
    print("\n3. Low rank random features:")
    low_rank_features = torch.randn(num_samples, num_augmentations, feature_size // 16)
    w = torch.randn(feature_size, feature_size // 16) * 0.02
    features = einx.dot("n q dd, d dd -> n q d", low_rank_features, w)

    lidar_score_low_rank = compute_lidar_score(features)
    print(f"   LiDAR Score: {lidar_score_low_rank:.4f}")

    # Test 4: Low rank correlated features
    print("\n3. Low rank correlated features:")
    low_rank_clean_features = torch.randn(num_samples, feature_size // 16)
    noise = torch.randn(num_samples, num_augmentations, feature_size // 16) * 1e-2
    low_rank_features = einx.add("n dd, n q dd", low_rank_clean_features, noise)
    w = torch.randn(feature_size, feature_size // 16) * 0.02
    features = einx.dot("n q dd, d dd -> n q d", low_rank_features, w)
    lidar_score = compute_lidar_score(features)
    print(f"   LiDAR Score: {lidar_score:.4f}")


if __name__ == "__main__":
    test_lidar()
