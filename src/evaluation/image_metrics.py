import torch
import numpy as np
import math
import tqdm
from itertools import combinations
from torch import cosine_similarity
from torch.nn.functional import adaptive_avg_pool2d
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance


@torch.inference_mode()
def get_inception_features(dataloader, num_samples, model, device, channels=3, desc=""):
    """Extracts Inception features for a given number of samples from a dataloader."""
    features_list = []
    count = 0
    # Ensure num_samples does not exceed dataset size if loader has finite length
    dataset_size = len(dataloader.dataset)
    if num_samples > dataset_size:
        print(f"Warning: Requested {num_samples} samples, but dataloader only has {dataset_size}. Using {dataset_size}.")
        num_samples = dataset_size

    if num_samples == 0:
        raise ValueError(f"Cannot extract features for 0 samples ({desc}).")

    # num_batches = int(math.ceil(num_samples / batch_size)) # TQDM handles total
    pbar = tqdm.tqdm(total=num_samples, desc=f"Extracting Inception features ({desc})", leave=False)
    dataloader_iter = iter(dataloader)
    while count < num_samples:
        batch_data = next(dataloader_iter)
        batch = batch_data['images'].to(device)

        # Handle cases where last batch might be smaller than needed
        samples_in_batch = batch.size(0)
        samples_needed = num_samples - count
        if samples_in_batch > samples_needed:
            batch = batch[:samples_needed]
            samples_in_batch = batch.size(0) # Update actual batch size

        # Input to Inception expects [0, 1] if normalize_input=True
        # Dataloader gives [-1, 1], so convert
        # batch = (batch / 2 + 0.5).clamp(0, 1)

        if channels == 1:
             batch = repeat(batch, "b 1 ... -> b c ...", c=3)

        pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        if pred.size(2) != 1 or pred.size(3) != 1:
             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        features = rearrange(pred, "... 1 1 -> ...")
        features_list.append(features.cpu())
        count += samples_in_batch
        pbar.update(samples_in_batch)
    pbar.close()
    return torch.cat(features_list, dim=0)


@torch.inference_mode()
def get_inception_features_from_tensors(tensor_list, model, device, batch_size, channels=3, desc=""):
    """Extracts Inception features for a list/tensor of image tensors."""
    if isinstance(tensor_list, list):
        all_tensors = torch.cat(tensor_list, dim=0)
    else:
        all_tensors = tensor_list

    num_samples = all_tensors.shape[0]
    if num_samples == 0:
        print(f"\nWarning: Input tensor for {desc} has 0 samples. Returning empty features.")
        feature_dim = model.fc.in_features
        return torch.empty((0, feature_dim), dtype=torch.float32)


    features_list = []
    num_batches = int(math.ceil(num_samples / batch_size))
    pbar = tqdm.tqdm(total=num_samples, desc=f"Extracting Inception features ({desc})", leave=False)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, num_samples)
        batch = all_tensors[start:end].to(device)

        # Input to Inception expects [0, 1] if normalize_input=True
        # Input tensors are assumed [-1, 1], so convert
        # batch = (batch / 2 + 0.5).clamp(0, 1)

        if channels == 1:
             batch = repeat(batch, "b 1 ... -> b c ...", c=3)

        pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        if pred.size(2) != 1 or pred.size(3) != 1: 
             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        features = rearrange(pred, "... 1 1 -> ...")
        features_list.append(features.cpu())
        pbar.update(batch.size(0))
    pbar.close()
    return torch.cat(features_list, dim=0)


def compute_fid(real_features, fake_features):
    """Compute FID for generated images."""
    mu_real = np.mean(real_features.numpy(), axis=0)
    sigma_real = np.cov(real_features.numpy(), rowvar=False)
    mu_fake = np.mean(fake_features.numpy(), axis=0)
    sigma_fake = np.cov(fake_features.numpy(), rowvar=False)

    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value


def compute_diversity(fake_features, config):
    """Compute diversity for generated images."""
    num_fake_eval = fake_features.shape[0]
    
    # Limit pairs for performance if too many images
    num_pairs = min(config.eval.num_fid_samples, num_fake_eval * (num_fake_eval - 1) // 2)

    indices = np.array(list(combinations(range(num_fake_eval), 2)))
    if indices.shape[0] > num_pairs:
        np.random.shuffle(indices)
        selected_indices = indices[:num_pairs]
    else:
        selected_indices = indices

    sim = cosine_similarity(fake_features[selected_indices[:, 0]], fake_features[selected_indices[:, 1]], dim=1)
    sim_clamped = torch.clamp(sim, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(2 * (1. - sim_clamped), min=0.0) + 1e-9)
    avg_cosine_distance = dist.mean().item()

    return avg_cosine_distance    


def gaussian_rbf_kernel(x, y, sigma=None):
    """Compute the Gaussian RBF kernel between two sets of samples."""
    beta = 1. / (2. * (sigma ** 2)) if sigma is not None else 1.0
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-beta * dist_sq)

def compute_mmd2(real_features, fake_features, sigma=10.):
    """
    Compute the Maximum Mean Discrepancy (MMD) using the RBF kernel.

    Uses the unbiased V-statistic estimator.
    Args:
        real_features (torch.Tensor): Features from the real dataset (N, D).
        fake_features (torch.Tensor): Features from the generated dataset (M, D).
        sigma (float, optional): RBF kernel bandwidth. If None, estimated using
                                 the median heuristic. Defaults to None.

    Returns:
        float: The computed MMD value.
    """
    nx = real_features.shape[0]
    ny = fake_features.shape[0]

    if nx < 2 or ny < 2:
        print("Warning: Need at least 2 samples from each distribution for unbiased MMD.")
        # Fallback to biased estimator or return NaN/error? Let's return NaN for now.
        return float('nan')

    # Estimate sigma using median heuristic if not provided
    if sigma is None:
        all_features = torch.cat([real_features, fake_features], dim=0)
        # Calculate pairwise distances efficiently, avoiding sqrt
        # Use torch.pdist for distances within the combined set
        pairwise_dist_sq = torch.pdist(all_features, p=2).pow(2)
        # Filter out zero distances (self-comparisons)
        non_zero_dists_sq = pairwise_dist_sq[pairwise_dist_sq > 1e-9]
        if non_zero_dists_sq.numel() > 0:
             median_dist_sq = torch.median(non_zero_dists_sq)
             sigma = torch.sqrt(median_dist_sq / 2.) # Heuristic sigma = sqrt(median_dist_sq / 2)
        else:
             sigma = 1.0 # Fallback if all distances are zero
        sigma = sigma.item()
        # print(f"Estimated MMD sigma: {sigma:.4f}")

    # Calculate kernel matrices
    Kxx = gaussian_rbf_kernel(real_features, real_features, sigma)
    Kyy = gaussian_rbf_kernel(fake_features, fake_features, sigma)
    Kxy = gaussian_rbf_kernel(real_features, fake_features, sigma)

    # Calculate unbiased MMD^2 statistic (V-statistic)
    # Term 1: Kxx - sum diagonals (i!=j)
    mmd2_term1 = (Kxx.sum() - Kxx.diag().sum()) / (nx * (nx - 1))
    # Term 2: Kyy - sum diagonals (i!=j)
    mmd2_term2 = (Kyy.sum() - Kyy.diag().sum()) / (ny * (ny - 1))
    # Term 3: Kxy
    mmd2_term3 = Kxy.mean() * 2

    mmd2 = mmd2_term1 + mmd2_term2 - mmd2_term3

    # # MMD is the square root of MMD^2. Clamp at 0 to avoid sqrt of small negative numbers.
    # mmd = torch.sqrt(torch.relu(mmd2))

    return 1000 * mmd2.item()

def compute_mmd(x, y, sigma=10.):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    x_sqnorms = torch.sum(x * x, dim=1)
    y_sqnorms = torch.sum(y * y, dim=1)

    gamma = 1.0 / (2 * (sigma**2))
    k_xx_dist = -2 * torch.mm(x, x.t()) + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0)
    k_xx = torch.exp(-gamma * k_xx_dist)

    k_yy_dist = -2 * torch.mm(y, y.t()) + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
    k_yy = torch.exp(-gamma * k_yy_dist)

    k_xy_dist = -2 * torch.mm(x, y.t()) + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
    k_xy = torch.exp(-gamma * k_xy_dist)

    mmd_val = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    return mmd_val.item() * 1000


@torch.inference_mode()
def get_classifier_features_from_dataloader(dataloader, num_samples, classifier_model, device, desc=""):
    """Extracts features from the MNIST CNN classifier model for a given number of samples from a dataloader."""
    # Assumes classifier_model is the CNN defined in src/models/classifier.py
    feature_extractor = classifier_model.embed
    feature_extractor.eval() # Ensure dropout/batchnorm are in eval mode

    features_list = []
    count = 0
    dataset_size = len(dataloader.dataset)
    if num_samples > dataset_size:
        print(f"Warning: Requested {num_samples} classifier samples, but dataloader only has {dataset_size}. Using {dataset_size}.")
        num_samples = dataset_size

    if num_samples == 0:
        raise ValueError(f"Cannot extract classifier features for 0 samples ({desc}).")

    pbar = tqdm.tqdm(total=num_samples, desc=f"Extracting Classifier features ({desc})", leave=False)
    dataloader_iter = iter(dataloader)
    while count < num_samples:
        batch_data = next(dataloader_iter)
        batch = batch_data['images'].to(device)

        samples_in_batch = batch.size(0)
        samples_needed = num_samples - count
        if samples_in_batch > samples_needed:
            batch = batch[:samples_needed]
            samples_in_batch = batch.size(0)

        # Get features from the convolutional part
        features = feature_extractor(batch)
        if features.size(2) != 1 or features.size(3) != 1: 
             features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        #features = features.reshape(features.size(0), -1)

        features_list.append(features.cpu())
        count += samples_in_batch
        pbar.update(samples_in_batch)
    pbar.close()
    return torch.cat(features_list, dim=0)


@torch.inference_mode()
def get_classifier_features_from_tensors(tensor_list, classifier_model, device, batch_size, desc=""):
    """Extracts features from the MNIST CNN classifier model for a list/tensor of image tensors."""
    # Assumes classifier_model is the CNN defined in src/models/classifier.py
    feature_extractor = classifier_model.embed
    feature_extractor.eval()

    if isinstance(tensor_list, list):
        all_tensors = torch.cat(tensor_list, dim=0)
    else:
        all_tensors = tensor_list

    num_samples = all_tensors.shape[0]
    if num_samples == 0:
        print(f"\nWarning: Input tensor for classifier features ({desc}) has 0 samples. Returning empty features.")
        # Get feature dimension from the classifier's out layer
        feature_dim = classifier_model.out.in_features
        return torch.empty((0, feature_dim), dtype=torch.float32)

    features_list = []
    num_batches = int(math.ceil(num_samples / batch_size))
    pbar = tqdm.tqdm(total=num_samples, desc=f"Extracting Classifier features ({desc})", leave=False)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, num_samples)
        batch = all_tensors[start:end].to(device)

        # Get features from the convolutional part
        features = feature_extractor(batch)
        if features.size(2) != 1 or features.size(3) != 1: 
             features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        #features = features.reshape(features.size(0), -1)

        features_list.append(features.cpu())
        pbar.update(batch.size(0))
    pbar.close()
    return torch.cat(features_list, dim=0)    