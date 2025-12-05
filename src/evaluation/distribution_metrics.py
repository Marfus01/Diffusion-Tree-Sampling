import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics.pairwise import rbf_kernel
from typing import Tuple, Optional

# def compute_mmd(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
#     """
#     Compute Maximum Mean Discrepancy (MMD) between two sets of samples.
    
#     Args:
#         x: Array of shape (n_samples_x, n_features) containing samples from first distribution
#         y: Array of shape (n_samples_y, n_features) containing samples from second distribution
#         sigma: Bandwidth parameter for RBF kernel
        
#     Returns:
#         MMD value between the two distributions
#     """    
#     # Compute kernel matrices
#     kxx = rbf_kernel(x, x, gamma=1.0/(2*sigma**2))
#     kyy = rbf_kernel(y, y, gamma=1.0/(2*sigma**2))
#     kxy = rbf_kernel(x, y, gamma=1.0/(2*sigma**2))
    
#     # Compute MMD
#     mmd = np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)
#     return np.sqrt(mmd)


def compute_mmd(x, y):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y.
    """
    # From Gretton et al. 2012
    # sigma = np.median(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1))
    sigma = 1.0

    x_sqnorms = np.sum(x * x, axis=1, keepdims=True)
    y_sqnorms = np.sum(y * y, axis=1, keepdims=True)

    gamma = 1.0 / (2 * (sigma**2))
    k_xx_dist = -2 * np.dot(x, x.T) + x_sqnorms + x_sqnorms.T
    k_xx = np.exp(-gamma * k_xx_dist)

    k_yy_dist = -2 * np.dot(y, y.T) + y_sqnorms + y_sqnorms.T
    k_yy = np.exp(-gamma * k_yy_dist)

    k_xy_dist = -2 * np.dot(x, y.T) + x_sqnorms + y_sqnorms.T
    k_xy = np.exp(-gamma * k_xy_dist)

    mmd_val = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    return mmd_val * 10

def plot_reward_distributions(
    rewards_x: np.ndarray,
    rewards_y: np.ndarray,
    labels: Tuple[str, str] = ('Distribution X', 'Distribution Y'),
    bins: int = 50,
    alpha: float = 0.6,
    title: str = 'Reward Distribution Comparison',
    filename: str = 'reward_distribution.png'
) -> None:
    """
    Plot overlapping histograms of reward values from two distributions.
    
    Args:
        rewards_x: Array of reward values from first distribution
        rewards_y: Array of reward values from second distribution
        labels: Tuple of labels for the two distributions
        bins: Number of bins for histogram
        alpha: Transparency of histograms
        title: Title of the plot
        filename: Name of the file to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rewards_x, bins=bins, alpha=alpha, label=labels[0], density=False)
    plt.hist(rewards_y, bins=bins, alpha=alpha, label=labels[1], density=False)
    plt.xlabel('Reward Value')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)

def compute_2d_kl_divergence(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 500,
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence between two 2D distributions by discretizing the space.
    
    Args:
        x: Array of shape (n_samples_x, 2) containing samples from first distribution
        y: Array of shape (n_samples_y, 2) containing samples from second distribution
        n_bins: Number of bins for discretization in each dimension
        epsilon: Small value to avoid division by zero
        
    Returns:
        KL divergence value
    """
    # Find the bounds of the data
    x_min = min(x[:, 0].min(), y[:, 0].min())
    x_max = max(x[:, 0].max(), y[:, 0].max())
    y_min = min(x[:, 1].min(), y[:, 1].min())
    y_max = max(x[:, 1].max(), y[:, 1].max())
    
    # Create 2D histogram bins
    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_bins + 1)
    
    # Compute 2D histograms
    hist_x, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[x_bins, y_bins], density=True)
    hist_y, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=[x_bins, y_bins], density=True)
    
    # Add epsilon to avoid division by zero
    hist_x = hist_x + epsilon
    hist_y = hist_y + epsilon
    
    # Normalize histograms
    hist_x = hist_x / hist_x.sum()
    hist_y = hist_y / hist_y.sum()
    
    # Compute KL divergence
    kl_div = np.sum(hist_x * np.log(hist_x / hist_y))
    return kl_div

def compute_jensen_shannon_divergence(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 500,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Jensen-Shannon divergence between two 2D distributions.
    
    Args:
        x: Array of shape (n_samples_x, 2) containing samples from first distribution
        y: Array of shape (n_samples_y, 2) containing samples from second distribution
        n_bins: Number of bins for discretization in each dimension
        epsilon: Small value to avoid division by zero
        
    Returns:
        Jensen-Shannon divergence value
    """
    # Find the bounds of the data
    x_min = min(x[:, 0].min(), y[:, 0].min())
    x_max = max(x[:, 0].max(), y[:, 0].max())
    y_min = min(x[:, 1].min(), y[:, 1].min())
    y_max = max(x[:, 1].max(), y[:, 1].max())
    
    # Create 2D histogram bins
    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_bins + 1)
    
    # Compute 2D histograms
    hist_x, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[x_bins, y_bins], density=True)
    hist_y, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=[x_bins, y_bins], density=True)
    
    # Add epsilon to avoid division by zero
    hist_x = hist_x + epsilon
    hist_y = hist_y + epsilon
    
    # Normalize histograms
    hist_x = hist_x / hist_x.sum()
    hist_y = hist_y / hist_y.sum()
    
    # Compute average distribution
    m = 0.5 * (hist_x + hist_y)
    
    # Compute Jensen-Shannon divergence
    js_div = 0.5 * (entropy(hist_x.flatten(), m.flatten()) + 
                    entropy(hist_y.flatten(), m.flatten()))
    return js_div

def plot_2d_distributions(
    x: np.ndarray,
    y: np.ndarray,
    labels: Tuple[str, str] = ('Distribution X', 'Distribution Y'),
    n_bins: int = 500,
    title: str = '2D Distribution Comparison',
    filename: str = '2d_distribution_comparison.png'
) -> None:
    """
    Plot 2D distributions using scatter plots and contour plots.
    
    Args:
        x: Array of shape (n_samples_x, 2) containing samples from first distribution
        y: Array of shape (n_samples_y, 2) containing samples from second distribution
        labels: Tuple of labels for the two distributions
        n_bins: Number of bins for contour plot
        title: Title of the plot
        filename: Name of the file to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.5, label=labels[0])
    plt.scatter(y[:, 0], y[:, 1], s=5, alpha=0.5, label=labels[1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Contour plot
    plt.subplot(122)
    x_min = min(x[:, 0].min(), y[:, 0].min())
    x_max = max(x[:, 0].max(), y[:, 0].max())
    y_min = min(x[:, 1].min(), y[:, 1].min())
    y_max = max(x[:, 1].max(), y[:, 1].max())
    
    # Create grid for contour plot
    # Use n_bins for the number of grid points, not bin edges
    x_grid = np.linspace(x_min, x_max, n_bins)
    y_grid = np.linspace(y_min, y_max, n_bins)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute 2D histograms
    # Use n_bins-1 for histogram bins to match the contour grid dimensions
    hist_x, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=n_bins-1, 
                                             range=[[x_min, x_max], [y_min, y_max]], 
                                             density=True)
    hist_y, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=n_bins-1,
                                 range=[[x_min, x_max], [y_min, y_max]],
                                 density=True)
    
    # Create new meshgrid that matches histogram dimensions
    X_hist = (x_edges[:-1] + x_edges[1:]) / 2
    Y_hist = (y_edges[:-1] + y_edges[1:]) / 2
    X_hist, Y_hist = np.meshgrid(X_hist, Y_hist)
    
    # Plot contours with matching dimensions
    plt.contour(X_hist, Y_hist, hist_x.T, alpha=0.5, label=labels[0])
    plt.contour(X_hist, Y_hist, hist_y.T, alpha=0.5, label=labels[1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)