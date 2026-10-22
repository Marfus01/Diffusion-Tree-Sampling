import logging
import torch
from torchvision import transforms
import sklearn.datasets as skld
import numpy as np
from pathlib import Path

from src.core.config import get_default_toy_config, process_toy_config

logger = logging.getLogger(__name__)


# --- Image Datasets ---

class ScaleTransform:
    """Scale the tensor values to [-1, 1]."""
    def __call__(self, tensor):
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        return (tensor * 2) - 1

class _DictDataset(torch.utils.data.Dataset):
    """Wraps a torchvision dataset so that batches are dicts {x_tensor: images, y_tensor: labels},
    the layout consumed by the run scripts, the classifier training and the metrics."""

    def __init__(self, base, indices, x_key, y_key):
        self.base = base
        self.indices = indices
        self.x_key = x_key
        self.y_key = y_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return {self.x_key: x, self.y_key: torch.tensor(y, dtype=torch.long)}


def get_img_dataset(
    config,
    splits=None,
    filter_labels=None,
    augment=True,
):
    """Returns dataloaders over the torchvision copy of MNIST / CIFAR-10 in config.dataset.data_path.

    Keys: '<split>_loader' for each split and, with filter_labels, '<split>_class_loader'
    restricted to those labels. Images are in [-1, 1]. With augment=True CIFAR-10 images get
    the original pipeline's resize-crop-flip augmentation (used for classifier training and
    the code's own FID); augment=False gives the unmodified images.
    """
    import torchvision

    data = {}
    dataset_name = config.dataset.name.lower()
    data_path = config.dataset.data_path
    batch_size = config.dataset.batch_size
    image_size = config.dataset.image_size
    workers = config.dataset.workers
    x_tensor = config.dataset.x_tensor
    y_tensor = config.dataset.y_tensor
    splits = config.dataset.splits if splits is None else splits

    if dataset_name not in ['mnist', 'cifar10']:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    tfs = []
    if image_size is not None:
        tfs += [transforms.Resize((image_size, image_size))]
        if dataset_name == 'cifar10' and augment:
            tfs += [transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip()]
    tfs += [transforms.ToTensor(), ScaleTransform()]
    transform_composition = transforms.Compose(tfs)

    dataset_cls = torchvision.datasets.CIFAR10 if dataset_name == 'cifar10' else torchvision.datasets.MNIST
    for split in splits:
        base = dataset_cls(root=data_path, train=(split == 'train'), download=False, transform=transform_composition)
        targets = np.asarray(base.targets)
        all_indices = np.arange(len(base))
        data[f'{split}_data'] = base
        data[f'{split}_loader'] = torch.utils.data.DataLoader(
            _DictDataset(base, all_indices, x_tensor, y_tensor),
            batch_size=batch_size, shuffle=True, num_workers=workers)
        if filter_labels is not None:
            if isinstance(filter_labels, int):
                filter_labels = [filter_labels]
            class_indices = all_indices[np.isin(targets, list(filter_labels))]
            data[f'{split}_class_loader'] = torch.utils.data.DataLoader(
                _DictDataset(base, class_indices, x_tensor, y_tensor),
                batch_size=batch_size, shuffle=True, num_workers=workers) if len(class_indices) else None
        logger.info(f"Loaded {dataset_name} {split}: {len(base)} images from {data_path}")

    return data

def cycle(dl):
    """Utility function to cycle indefinitely through a dataloader."""
    while True:
        for data in dl:
            if isinstance(data, dict):
                yield data['images']
            else:
                yield data


# --- 2D Datasets ---

class Dataset2D:
    """Base class for 2D datasets with reward functions and product distributions."""
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.name = "base"
        self.config = config

    def load_gt_samples(self):
        sample_load_path = Path(self.config.paths.checkpoint_dir) / "samples.npy"
        rewards_load_path = Path(self.config.paths.checkpoint_dir) / "rewards.npy"
        if sample_load_path.exists() and rewards_load_path.exists():
            self.gt_samples = np.load(sample_load_path)[:self.config.sampling.population_size]
            self.gt_rewards = np.load(rewards_load_path)[:self.config.sampling.population_size]
        else:
            self.gt_samples = None
            self.gt_rewards = None
        
    def sample(self, n_samples=10000):
        """Sample from the base distribution p(x)"""
        raise NotImplementedError
        
    def calculate_reward(self, samples):
        """Calculate reward r(x) for given samples"""
        raise NotImplementedError
        
    def sample_product_distribution(self, n_samples=10_000, oversample_factor=10):
        """
        Importance–resample from p(x) exp(r(x)).
        Not exact but unbiased if oversample_factor→∞.
        """
        with torch.no_grad():
            # 1. draw proposal points from p
            proposals = self.sample(n_samples * oversample_factor)  # [M,2]
            # 2. compute unnormalised weights
            rewards_np = self.calculate_reward(proposals)           # NumPy, shape [M]
            logw = torch.from_numpy(rewards_np).to(proposals)       # convert to tensor
            # 3. numerically stable normalisation
            w = torch.exp(logw - logw.max())
            w = w / w.sum()
            # 4. multinomial resampling
            idx = torch.multinomial(w, n_samples, replacement=True)
            return proposals[idx], rewards_np[idx.cpu().numpy()]
        
    def sample_product_distribution_diff(self, n_samples=10_000, oversample_factor=10):
        """
        Importance–resample from p(x) exp(r(x)).
        Not exact but unbiased if oversample_factor→∞.
        """
        with torch.no_grad():
            # 1. draw proposal points from p
            proposals = self.sample(n_samples * oversample_factor)  # [M,2]
            # 2. compute unnormalised weights
            rewards_np = self.calculate_reward_diff(proposals)           # NumPy, shape [M]
            logw = rewards_np       # convert to tensor
            # 3. numerically stable normalisation
            w = torch.exp(logw - logw.max())
            w = w / w.sum()
            # 4. multinomial resampling
            idx = torch.multinomial(w, n_samples, replacement=True)
            return proposals[idx], rewards_np[idx].cpu().numpy()


class GaussianMixture(Dataset2D):
    """
    Mixture of 8 Gaussians arranged in a circle.
    Rewards are assigned based on proximity to centers, increasing anti-clockwise from East.
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', reward_temp=1.0):
        super().__init__(config, device)
        self.name = "gaussian_mixture"
        self.std = 0.3
        self.radius = 4.0
        self.reward_temp = reward_temp # Temperature for differentiable reward calculation
        
        # Define the 8 centers anti-clockwise starting from East (angle 0)
        self.centers = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            center = [np.cos(angle) * self.radius, np.sin(angle) * self.radius]
            self.centers.append(center)
        self.centers_np = np.array(self.centers)
        self.centers_pt = torch.tensor(self.centers_np, dtype=torch.float32, device=self.device)
        
        # Define base reward values for each center (1.0 to 8.0 anti-clockwise from East)
        base_rewards_np = np.arange(1.0, 9.0) * 1.5
        # Apply exponential scaling and store
        self.center_reward_values_np = np.exp(base_rewards_np)
        # Store log-rewards on the correct device
        self.center_log_rewards_pt = torch.log(torch.tensor(self.center_reward_values_np, dtype=torch.float32, device=self.device))

        self.load_gt_samples()

    def sample(self, n_samples=10000):
        """Sample from mixture of 8 Gaussians with equal weights"""
        samples_per_gaussian = n_samples // 8
        samples = []
        
        for center in self.centers_np: # Use numpy centers here
            gaussian_samples = center + np.random.randn(samples_per_gaussian, 2) * self.std
            samples.append(gaussian_samples)
        
        # Combine and shuffle
        samples = np.vstack(samples)
        np.random.shuffle(samples)
        
        # Trim to exact n_samples
        samples = samples[:n_samples]
        
        return torch.FloatTensor(samples).to(self.device)
    
    def calculate_reward(self, samples):
        """Calculate reward based on the closest Gaussian center (non-differentiable)."""
        if isinstance(samples, np.ndarray):
            samples = torch.tensor(samples, dtype=torch.float32, device=self.device)
        elif samples.device != self.device:
            samples = samples.to(self.device)
            
        # Calculate squared distances: (N, 1, 2) - (1, 8, 2) -> (N, 8, 2) -> sum -> (N, 8)
        dist_sq = torch.sum((samples.unsqueeze(1) - self.centers_pt.unsqueeze(0))**2, dim=2)
        
        # Find the index of the closest center for each sample
        closest_center_indices = torch.argmin(dist_sq, dim=1) # Shape (N,)
        
        # Get the rewards corresponding to the closest centers
        rewards = self.center_log_rewards_pt[closest_center_indices]
        
        # Return as numpy array
        return rewards.detach().cpu().numpy()

    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate reward based on distance proximity using softmax (differentiable)."""
        if samples.device != self.device:
            samples = samples.to(self.device)
            
        # Calculate squared distances: (N, 1, 2) - (1, 8, 2) -> (N, 8, 2) -> sum -> (N, 8)
        dist_sq = torch.sum((samples.unsqueeze(1) - self.centers_pt.unsqueeze(0))**2, dim=2)
        dist_sq = torch.where(torch.isnan(dist_sq), torch.ones_like(dist_sq) * 1e10, dist_sq)

        # Convert distances to similarity scores (closer distance -> higher score)
        # Using reward_temp to control softness. Smaller temp -> harder assignment.
        # Negative distance is used because softmax expects higher values to be more likely.
        similarities = -dist_sq / self.reward_temp
        
        # # Apply softmax over the similarity scores for each sample
        # # softmax_weights shape: [N, 8]
        softmax_weights = torch.softmax(similarities, dim=1)
        
        # # Calculate weighted average of log rewards
        # # Weighted sum: (N, 8) * (8,) -> (N,)
        avg_log_rewards = torch.matmul(softmax_weights, torch.exp(self.center_log_rewards_pt))
        return torch.log(avg_log_rewards + 1e-10)

        # x_squared = torch.clamp((samples - torch.tensor([0, 1], device=self.device))**2, max=10000)
        # return - x_squared[:, 0] - x_squared[:, 1] / 10

    def sample_product_distribution(self, n_samples=10000):
        """Sample from p(x)r(x) for mixture of Gaussians (Now based on Octants)"""
        # Use octant rewards as mixture weights for sampling
        mixture_weights = self.center_reward_values_np / self.center_reward_values_np.sum()

        # --- This part needs to change --- 
        # The original code sampled from specific Gaussians. Now we need to sample
        # uniformly and then assign rewards based on octant, or sample angles from weighted octants.
        # Let's use importance resampling from a uniform base distribution (e.g., uniform square or large Gaussian)
        # This matches the base class sample_product_distribution logic.

        oversample_factor = 10 # Use the default from base class
        with torch.no_grad():
            # 1. draw proposal points from a simple base (e.g., uniform square or Gaussian)
            # Let's use uniform sampling in a box, e.g., [-6, 6] x [-6, 6]
            proposals_np = self.sample(n_samples * oversample_factor)
            proposals = torch.tensor(proposals_np, dtype=torch.float32, device=self.device)

            # 2. compute unnormalised weights using the NON-DIFFERENTIABLE octant reward
            rewards_np = self.calculate_reward(proposals) # Get log rewards
            logw = torch.from_numpy(rewards_np).to(proposals)       # convert log rewards to tensor
            
            # 3. numerically stable normalisation using log-sum-exp for log weights
            log_w_max = torch.max(logw)
            w_unnorm = torch.exp(logw - log_w_max)
            w = w_unnorm / torch.sum(w_unnorm)
            
            # 4. multinomial resampling
            idx = torch.multinomial(w, n_samples, replacement=True)
            
            # Return the resampled proposals and their corresponding *original* log rewards
            # Need to get the log rewards corresponding to the sampled indices
            final_log_rewards_np = logw[idx].cpu().numpy()

            return proposals[idx], final_log_rewards_np
    
    def sample_product_distribution_diff(self, n_samples=10000):
        """Sample from p(x)r(x) for mixture of Gaussians (Now based on Octants)"""
        # Use octant rewards as mixture weights for sampling
        mixture_weights = self.center_reward_values_np / self.center_reward_values_np.sum()

        # --- This part needs to change --- 
        # The original code sampled from specific Gaussians. Now we need to sample
        # uniformly and then assign rewards based on octant, or sample angles from weighted octants.
        # Let's use importance resampling from a uniform base distribution (e.g., uniform square or large Gaussian)
        # This matches the base class sample_product_distribution logic.

        oversample_factor = 10 # Use the default from base class
        with torch.no_grad():
            # 1. draw proposal points from a simple base (e.g., uniform square or Gaussian)
            # Let's use uniform sampling in a box, e.g., [-6, 6] x [-6, 6]
            proposals_np = self.sample(n_samples * oversample_factor)
            proposals = torch.tensor(proposals_np, dtype=torch.float32, device=self.device)

            # 2. compute unnormalised weights using the NON-DIFFERENTIABLE octant reward
            rewards_np = self.calculate_reward_diff(proposals) # Get log rewards
            logw = rewards_np       # convert log rewards to tensor
            
            # 3. numerically stable normalisation using log-sum-exp for log weights
            log_w_max = torch.max(logw)
            w_unnorm = torch.exp(logw - log_w_max)
            w = w_unnorm / torch.sum(w_unnorm)
            
            # 4. multinomial resampling
            idx = torch.multinomial(w, n_samples, replacement=True)
            
            # Return the resampled proposals and their corresponding *original* log rewards
            # Need to get the log rewards corresponding to the sampled indices
            final_log_rewards_np = logw[idx].cpu().numpy()

            return proposals[idx], final_log_rewards_np


class TwoMoons(Dataset2D):
    """
    Two moons dataset with position-dependent rewards - higher at the tips of the moons.
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', noise=0.1):
        super().__init__(config, device)
        self.name = "two_moons"
        self.noise = noise
        # Define the tips of the moons
        self.upper_moon_tips = np.array([[-4, 1], [1, 2]])
        self.lower_moon_tips = np.array([[2, -1], [-2.5, -1.5]])
        self.upper_moon_tips_pt = torch.tensor(self.upper_moon_tips, dtype=torch.float32, device=self.device)
        self.lower_moon_tips_pt = torch.tensor(self.lower_moon_tips, dtype=torch.float32, device=self.device)
        self.load_gt_samples()
        
    def sample(self, n_samples=10000):
        """Sample from two moons distribution"""
        X, y = skld.make_moons(n_samples=n_samples, noise=self.noise)
        # Scale to fit within -6 to 6 range and center at (0,0)
        X = X * 3 - np.array([1.5, 0.25])  # Scale and center at origin
        return torch.FloatTensor(X).to(self.device)
    
    def calculate_reward(self, samples):
        """Calculate reward based on sum of distances from tips"""
        if isinstance(samples, torch.Tensor):
            points = samples.detach().cpu().numpy()
        else:
            points = samples
        
        rewards = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Calculate distances to all tips
            dists = []
            for tip in np.vstack([self.upper_moon_tips, self.lower_moon_tips]):
                dist = np.sqrt(np.sum((point - tip)**2))
                dists.append(dist)
            
            # Sum of distances
            total_dist = np.sum(dists)
            rewards[i] = -1. * total_dist
            
        return rewards
    
    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate reward based on sum of distances from tips (differentiable)"""
        if samples.device != self.device:
            samples = samples.to(self.device)
            
        # Calculate distances to all tips
        all_tips = torch.cat([self.upper_moon_tips_pt, self.lower_moon_tips_pt], dim=0)
        # Shape: [N, 4, 2] - [N, 1, 2] - [1, 4, 2] -> [N, 4, 2] -> [N, 4]
        dists = torch.norm(samples.unsqueeze(1) - all_tips.unsqueeze(0), dim=2)
        
        # Sum of distances
        total_dist = torch.sum(dists, dim=1)
        
        return -1. * total_dist

    def sample_product_distribution(self, n_samples=10000):
        """Sample from p(x)r(x) for two moons with emphasis on tips using rejection sampling"""
        samples = []
        rewards_list = []
        
        # Find maximum reward to normalize for rejection sampling
        # Generate a batch of samples to estimate max reward
        X_test, y_test = skld.make_moons(n_samples=1000, noise=self.noise)
        X_test = X_test * 3 - np.array([1.5, 0.25])
        test_samples = torch.FloatTensor(X_test).to(self.device)
        test_rewards = np.exp(self.calculate_reward(test_samples))
        max_reward = np.max(test_rewards) * 1.1  # Add 10% margin for safety
        
        # Generate samples until we have enough
        total_attempts = 0
        max_attempts = n_samples * 50  # Safety limit
        
        while len(samples) < n_samples and total_attempts < max_attempts:
            # Generate a batch of candidate samples
            batch_size = min(n_samples * 2, 10000)
            X_batch, y_batch = skld.make_moons(n_samples=batch_size, noise=self.noise)
            X_batch = X_batch * 3 - np.array([1.5, 0.25])
            
            # Calculate rewards
            batch_samples = torch.FloatTensor(X_batch).to(self.device)
            batch_rewards = np.exp(self.calculate_reward(batch_samples))
            
            # Rejection sampling
            for i in range(len(X_batch)):
                if len(samples) >= n_samples:
                    break
                    
                # Accept with probability proportional to reward
                acceptance_prob = batch_rewards[i] / max_reward
                if np.random.random() < acceptance_prob:
                    samples.append(X_batch[i])
                    rewards_list.append(batch_rewards[i])
                
                total_attempts += 1
                if total_attempts >= max_attempts:
                    break
        
        # If we didn't get enough samples, just use what we have
        if len(samples) < n_samples:
            logger.warning(f"Could only generate {len(samples)} samples using rejection sampling")
        
        # Convert to numpy arrays and then to torch tensors
        samples_array = np.array(samples[:n_samples])
        rewards_array = np.array(rewards_list[:n_samples])
        
        samples_tensor = torch.FloatTensor(samples_array).to(self.device)
        # rewards_tensor = torch.FloatTensor(rewards_array).to(self.device)
        
        return samples_tensor, np.log(rewards_array + 1e-10)


class Spiral(Dataset2D):
    """
    Spiral dataset with rewards based on different sections of the spiral.
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', noise=0.1):
        super().__init__(config, device)
        self.name = "spiral"
        self.noise = noise
        self.n_spirals = 1
        self.max_radius = 4.0
        
        # Define sections of the spiral with different rewards
        # Each section is defined by an angle range (in radians)
        self.sections = [
            {"angle_range": (0, 2*np.pi), "reward": np.exp(1.0)},
            {"angle_range": (2*np.pi, 4*np.pi), "reward": np.exp(3.0)},
            {"angle_range": (4*np.pi, 6*np.pi), "reward": np.exp(5.0)}
        ]
        
        # PyTorch versions for diff calculation
        self.section_rewards_pt = torch.tensor([s['reward'] for s in self.sections], dtype=torch.float32, device=self.device)
        self.section_angles_min_pt = torch.tensor([s['angle_range'][0] for s in self.sections], dtype=torch.float32, device=self.device)
        self.section_angles_max_pt = torch.tensor([s['angle_range'][1] for s in self.sections], dtype=torch.float32, device=self.device)
        self.min_radius_pt = torch.tensor(0.2, dtype=torch.float32, device=self.device) # Consistent with _get_section_from_point
        self.max_radius_pt = torch.tensor(self.max_radius, dtype=torch.float32, device=self.device)
        self.load_gt_samples()

    def _generate_spirals(self, n_samples):
        """Helper function to generate distinct spiral data"""
        samples_per_spiral = n_samples // self.n_spirals
        
        X = np.zeros((n_samples, 2))
        
        for i in range(self.n_spirals):
            start_idx = i * samples_per_spiral
            end_idx = start_idx + samples_per_spiral
            
            # Generate spiral points with more turns for better visibility
            theta = np.linspace(0, 2 * np.pi * 3, samples_per_spiral)  # 3 rounds for more distinct spirals
            r = np.linspace(0.2, self.max_radius, samples_per_spiral)  # Start closer to center
            
            # Add phase shift for different spirals
            theta = theta + (i * np.pi / self.n_spirals)  # Larger phase shift
            
            # Convert to cartesian coordinates
            X[start_idx:end_idx, 0] = r * np.cos(theta)
            X[start_idx:end_idx, 1] = r * np.sin(theta)
            
            # Add controlled noise proportional to radius to maintain spiral shape
            noise_scale = self.noise * (r / self.max_radius)[:, np.newaxis]
            X[start_idx:end_idx] += np.random.randn(samples_per_spiral, 2) * noise_scale
        
        return X
    
    def _get_section_from_point(self, point):
        """Determine which section a point belongs to based on its position in the spiral"""
        # Calculate polar coordinates
        x, y = point
        r = np.sqrt(x**2 + y**2)
        
        # In _generate_spirals, we have:
        # - theta = np.linspace(0, 2 * np.pi * 3, samples_per_spiral)
        # - r = np.linspace(0.2, self.max_radius, samples_per_spiral)
        # 
        # There's a perfect linear relationship between theta and r:
        # - When r = 0.2, theta = 0
        # - When r = 4.0, theta = 6π
        
        # Convert radius directly to unwrapped angle
        if r <= 0.2:
            # Below minimum radius
            unwrapped_angle = 0
        elif r >= self.max_radius:
            # Above maximum radius
            unwrapped_angle = 6 * np.pi
        else:
            # Linear mapping from radius to angle
            unwrapped_angle = (r - 0.2) / (self.max_radius - 0.2) * 6 * np.pi
        
        # Find which section this angle belongs to
        for section in self.sections:
            min_angle, max_angle = section["angle_range"]
            if min_angle <= unwrapped_angle < max_angle:
                return section
        
        # Default to the last section if no match found
        return self.sections[-1]
    
    def sample(self, n_samples=10000):
        """Sample from spiral distribution"""
        X = self._generate_spirals(n_samples)
        np.random.shuffle(X)  # Shuffle the samples
        return torch.FloatTensor(X).to(self.device)
    
    def calculate_reward(self, samples):
        """Calculate reward based on which section of the spiral the point belongs to"""
        if isinstance(samples, torch.Tensor):
            points = samples.detach().cpu().numpy()
        else:
            points = samples
        
        rewards = np.zeros(len(points))
        
        for i, point in enumerate(points):
            section = self._get_section_from_point(point)
            rewards[i] = section["reward"]
        
        # if isinstance(samples, torch.Tensor):
        #     rewards = torch.tensor(rewards, device=samples.device, dtype=torch.float32)
        
        return np.log(rewards + 1e-10)
         
    def _generate_spiral_section(self, n_samples, angle_min, angle_max):
        """
        Generate spiral points specifically within a given angle range.
        
        Args:
            n_samples: Number of samples to generate
            angle_min: Minimum angle in radians
            angle_max: Maximum angle in radians
            
        Returns:
            Array of points within the specified section of the spiral
        """
        # Generate theta values within the specified range
        theta = np.linspace(angle_min, angle_max, n_samples)
        
        # Convert angles to radii based on the relationship in _get_section_from_point
        min_radius = 0.2 + (angle_min / (6 * np.pi)) * (self.max_radius - 0.2)
        max_radius = 0.2 + (angle_max / (6 * np.pi)) * (self.max_radius - 0.2)
        r = np.linspace(min_radius, max_radius, n_samples)
        
        # Convert to cartesian coordinates
        X = np.zeros((n_samples, 2))
        X[:, 0] = r * np.cos(theta)
        X[:, 1] = r * np.sin(theta)
        
        # Add controlled noise proportional to radius to maintain spiral shape
        noise_scale = self.noise * (r / self.max_radius)[:, np.newaxis]
        X += np.random.randn(n_samples, 2) * noise_scale
        
        return X
    
    def calculate_reward_diff(self, samples: torch.Tensor, softmax_temp: float = 0.1) -> torch.Tensor:
        """Calculate reward based on spiral section using softmax (differentiable)"""
        x = samples[:, 0]
        y = samples[:, 1]
        
        # Convert radius directly to unwrapped angle using torch operations
        r = torch.sqrt(x**2 + y**2 + 1e-6) # Add epsilon for stability if x=y=0
        
        # Convert radius directly to unwrapped angle using torch operations
        # Matches the logic in _get_section_from_point
        unwrapped_angle = torch.zeros_like(r)
        mask_mid = (r > self.min_radius_pt) & (r < self.max_radius_pt)
        mask_above = (r >= self.max_radius_pt)
        
        # Linear mapping for points within the main radius range
        unwrapped_angle[mask_mid] = (r[mask_mid] - self.min_radius_pt) / (self.max_radius_pt - self.min_radius_pt) * 6 * torch.pi
        # Cap angle for points above max radius
        unwrapped_angle[mask_above] = 6 * torch.pi
        # Angle is 0 for points at or below min radius (already initialized)
        
        # Calculate "scores" based on distance to center of each section's angle range
        # section_centers shape: [num_sections]
        section_centers = (self.section_angles_min_pt + self.section_angles_max_pt) / 2.0
        # section_widths shape: [num_sections]
        section_widths = (self.section_angles_max_pt - self.section_angles_min_pt)

        # Calculate negative squared distance to section centers, normalized by width (heuristic score)
        # dist_to_centers shape: [N, num_sections]
        dist_to_centers_sq = (unwrapped_angle.unsqueeze(1) - section_centers.unsqueeze(0)).pow(2)
        # Use a negative score, higher means closer/better fit. Add epsilon to width.
        # score shape: [N, num_sections]
        score = -dist_to_centers_sq / (section_widths.unsqueeze(0) + 1e-6)

        # Softmax over scores
        # softmax_weights shape: [N, num_sections]
        softmax_weights = torch.softmax(score / softmax_temp, dim=1)

        # Weighted average of log rewards
        log_reward_values = torch.log(self.section_rewards_pt)
        avg_log_rewards = torch.matmul(softmax_weights, log_reward_values)

        return avg_log_rewards

    def sample_product_distribution(self, n_samples=10000):
        """Sample from p(x)r(x) for spiral using a direct section-based approach"""
        # Calculate total weight for each section
        total_reward = sum(section["reward"] for section in self.sections)
        section_probs = [section["reward"] / total_reward for section in self.sections]
        
        # Determine how many samples to draw from each section
        section_samples = np.random.multinomial(n_samples, section_probs)
        
        # Debugging - print expected samples per section
        # print("Expected samples per section based on rewards:")
        # for i, section in enumerate(self.sections):
        #     print(f"Section {i} (angle range {section['angle_range']}): {section_samples[i]} samples, reward={section['reward']}")
        
        all_samples = []
        all_rewards = []
        
        # Generate samples for each section directly using the section angle ranges
        for i, section in enumerate(self.sections):
            n_section_samples = section_samples[i]
            if n_section_samples == 0:
                continue
                
            # Get angle range for this section
            angle_min, angle_max = section["angle_range"]
            
            # Generate points specifically for this section
            section_points = self._generate_spiral_section(n_section_samples, angle_min, angle_max)
            
            # Add points and rewards to our lists
            all_samples.extend(section_points)
            all_rewards.extend([section["reward"]] * len(section_points))
        
        # Convert to numpy arrays
        samples = np.array(all_samples)
        rewards = np.array(all_rewards)
        
        # Shuffle to mix the sections
        idx = np.random.permutation(len(samples))
        samples = samples[idx]
        rewards = rewards[idx]
        
        # Convert to torch tensors
        samples = torch.FloatTensor(samples).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        
        return samples, np.log(rewards + 1e-10)


class Checkerboard(Dataset2D):
    """
    Checkerboard dataset where the base distribution is uniform on filled squares.
    Reward decays exponentially from top-right to bottom-left across the whole domain.
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', n_squares=4, alpha=0.5):
        super().__init__(config, device)
        self.name = "checkerboard"
        self.n_squares = n_squares  # Number of squares in each dimension
        self.scale = 8.0 / self.n_squares  # Scale to fit within [-4, 4]
        self.alpha = alpha # Exponential decay rate
        
        # Precompute alpha tensor
        self.alpha_pt = torch.tensor(self.alpha, device=self.device, dtype=torch.float32)
        self.load_gt_samples()

    def sample(self, n_samples=10000):
        """Sample uniformly from the filled squares of the checkerboard pattern."""
        samples = []
        while len(samples) < n_samples:
            # Sample uniformly from [-4, 4] x [-4, 4]
            point = np.random.uniform(-4, 4, (1, 2))
            
            # Calculate which square the point falls in
            # Ensure indexing starts from 0
            x_square = int((point[0, 0] + 4) // self.scale)
            y_square = int((point[0, 1] + 4) // self.scale)
            
            # Clamp indices to be safe, although uniform sampling should be within bounds
            x_square = np.clip(x_square, 0, self.n_squares - 1)
            y_square = np.clip(y_square, 0, self.n_squares - 1)
            
            # Only accept points in alternating squares (e.g., where i+j is even)
            if (x_square + y_square) % 2 == 0: 
                samples.append(point[0])
        
        samples = np.array(samples)
        return torch.tensor(samples, dtype=torch.float32, device=self.device)
    
    def calculate_reward(self, samples):
        """Calculate reward = exp(alpha * (x + y)) (non-differentiable)."""
        if isinstance(samples, torch.Tensor):
            points = samples.detach().cpu().numpy()
        else:
            points = samples

        # x_plus_y = points[:, 0] + points[:, 1]
        # return self.alpha * x_plus_y
        return - self.alpha * (points[:, 0] ** 2 + points[:, 1] ** 2)
    
    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate reward = exp(alpha * (x + y)) (differentiable)."""
        if samples.device != self.device:
            samples = samples.to(self.device)
            
        # x_plus_y = samples[:, 0] + samples[:, 1]
        # return self.alpha_pt * x_plus_y
        return - self.alpha_pt * (samples[:, 0] ** 2 + samples[:, 1] ** 2)


class SwissRoll(Dataset2D):
    """
    Swiss roll dataset with rewards based on different sections of the roll.
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', noise=0.5):
        super().__init__(config, device)
        self.name = "swiss_roll"
        self.noise = noise
        
        # Define sections of the swiss roll with different rewards
        # Each section is defined by a range of the unrolled parameter t
        self.sections = [
            {"t_range": (0, np.pi), "reward": np.exp(2.0)},
            {"t_range": (np.pi, 2*np.pi), "reward": np.exp(4.0)},
            {"t_range": (2*np.pi, 3*np.pi), "reward": np.exp(6.0)}
        ]
        
    def _get_section_from_point(self, point):
        """Determine which section a point belongs to based on its position in the swiss roll"""
        x, y = point
        
        # Estimate the unrolled parameter t
        # For swiss roll, t is related to the angle in polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Convert to positive angle
        if theta < 0:
            theta += 2 * np.pi
        
        # Estimate t based on distance from origin and angle
        # This is an approximation that works for our visualization
        t = theta + r
        
        # Find which section this t belongs to
        for section in self.sections:
            min_t, max_t = section["t_range"]
            if min_t <= t < max_t:
                return section
        
        # Default to the last section if no match found
        return self.sections[-1]
    
    def sample(self, n_samples=10000):
        """Sample from swiss roll distribution"""
        # Generate swiss roll data
        X, _ = skld.make_swiss_roll(n_samples=n_samples, noise=self.noise)
        # Only use first two dimensions and scale
        X = X[:, [0, 2]] / 3.0  # Scale to roughly [-4, 4] range
        return torch.FloatTensor(X).to(self.device)
    
    def calculate_reward(self, samples):
        """Calculate reward based on which section of the swiss roll the point belongs to"""
        if isinstance(samples, torch.Tensor):
            points = samples.detach().cpu().numpy()
        else:
            points = samples
        
        rewards = np.zeros(len(points))
        
        for i, point in enumerate(points):
            section = self._get_section_from_point(point)
            rewards[i] = section["reward"]
        
        # if isinstance(samples, torch.Tensor):
        #     rewards = torch.tensor(rewards, device=samples.device, dtype=torch.float32)
        
        return np.log(rewards + 1e-10)
    
    def calculate_reward_diff(self, samples: torch.Tensor, softmax_temp: float = 0.1) -> torch.Tensor:
        """Calculate reward based on swiss roll section using softmax (differentiable)"""
        x = samples[:, 0]
        y = samples[:, 1]
        
        # Estimate the unrolled parameter t using torch operations
        r = torch.sqrt(x**2 + y**2) # torch.norm(samples, dim=1)
        theta = torch.atan2(y, x)
        
        # Convert to positive angle [0, 2pi]
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        
        # Estimate t (approximation)
        t = theta + r 

        # Calculate "scores" based on distance to center of each section's t range
        section_centers = (self.section_t_min_pt + self.section_t_max_pt) / 2.0
        section_widths = (self.section_t_max_pt - self.section_t_min_pt)

        # Calculate negative squared distance to section centers, normalized by width
        # dist_to_centers_sq shape: [N, num_sections]
        dist_to_centers_sq = (t.unsqueeze(1) - section_centers.unsqueeze(0)).pow(2)
        # score shape: [N, num_sections]
        score = -dist_to_centers_sq / (section_widths.unsqueeze(0) + 1e-6)

        # Softmax over scores
        # softmax_weights shape: [N, num_sections]
        softmax_weights = torch.softmax(score / softmax_temp, dim=1)

        # Weighted average of log rewards
        log_reward_values = torch.log(self.section_rewards_pt)
        avg_log_rewards = torch.matmul(softmax_weights, log_reward_values)

        return avg_log_rewards

    def sample_product_distribution(self, n_samples=10000):
        """Sample from p(x)r(x) for swiss roll using a mixture approach"""
        # Calculate total weight for each section
        total_reward = sum(section["reward"] for section in self.sections)
        section_probs = [section["reward"] / total_reward for section in self.sections]
        
        # Determine how many samples to draw from each section
        section_samples = np.random.multinomial(n_samples, section_probs)
        
        all_samples = []
        all_rewards = []
        
        # Generate samples for each section
        for i, section in enumerate(self.sections):
            n_section_samples = section_samples[i]
            if n_section_samples == 0:
                continue
                
            # Generate more samples than needed
            oversample_factor = 3
            X, t = skld.make_swiss_roll(n_samples=n_section_samples * oversample_factor, noise=self.noise)
            candidates = X[:, [0, 2]] / 3.0  # Scale to match our range
            
            # Filter points that belong to this section based on t
            min_t, max_t = section["t_range"]
            section_indices = np.where((min_t <= t) & (t < max_t))[0]
            
            if len(section_indices) >= n_section_samples:
                # Take random subset if we have enough points
                selected_indices = np.random.choice(section_indices, size=n_section_samples, replace=False)
                section_points = candidates[selected_indices]
            else:
                # If we don't have enough points, generate more specifically for this section
                section_points = []
                while len(section_points) < n_section_samples:
                    # Generate more swiss roll data
                    X_more, t_more = skld.make_swiss_roll(n_samples=n_section_samples, noise=self.noise)
                    points_more = X_more[:, [0, 2]] / 3.0
                    
                    # Filter for this section
                    indices_more = np.where((min_t <= t_more) & (t_more < max_t))[0]
                    if len(indices_more) > 0:
                        section_points.extend(points_more[indices_more])
                
                # Trim to exact number needed
                section_points = np.array(section_points[:n_section_samples])
            
            all_samples.extend(section_points)
            all_rewards.extend([section["reward"]] * len(section_points))
        
        # Convert to numpy arrays
        samples = np.array(all_samples)
        rewards = np.array(all_rewards)
        
        # Shuffle to mix the sections
        idx = np.random.permutation(len(samples))
        samples = samples[idx]
        rewards = rewards[idx]
        
        # Convert to torch tensors
        samples = torch.FloatTensor(samples).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        
        return samples, np.log(rewards + 1e-10)


# ------------------------------------------------------------------------- #
# 2. Checkerboard Stripes
# ------------------------------------------------------------------------- #
class CheckerboardStripe(Dataset2D):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', beta=2.0):
        super().__init__(config, device)
        self.name = "checkerboard_stripe"
        self.beta = beta
        self.load_gt_samples()
    def sample(self, n_samples=10_000):
        samples = np.random.uniform(-2, 2, size=(n_samples, 2))
        return torch.tensor(samples, dtype=torch.float32, device=self.device)

    def calculate_reward(self, samples):
        if isinstance(samples, torch.Tensor):
            x = samples.detach().cpu().numpy()
        else:
            x = samples
        stripe = np.sin(3 * np.pi * (x[:, 0] + x[:, 1]))
        return self.beta * stripe

    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate checkerboard stripe reward (differentiable)"""
        # Use torch.pi and torch operations
        stripe = torch.sin(3 * torch.pi * (samples[:, 0] + samples[:, 1]))
        return self.beta * stripe

# ------------------------------------------------------------------------- #
# 3. Two-Moons Bridge
# ------------------------------------------------------------------------- #
class TwoMoonsBridge(Dataset2D):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', noise=0.05, beta=5.0, sigma_r=0.3):
        super().__init__(config, device)
        self.name = "two_moons_bridge"
        self.noise, self.beta, self.sigma_r = noise, beta, sigma_r
        self.load_gt_samples()
    
    # def sample(self, n_samples=10_000):
    #     # generate two moons with radius 1 and distance between centers = 0.5
    #     n = n_samples // 2
    #     theta = np.random.rand(n) * np.pi
    #     moon1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    #     moon2 = np.stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5], axis=1)
    #     samples = np.concatenate([moon1, moon2], axis=0)
    #     samples += np.random.randn(*samples.shape) * self.noise
    #     return torch.tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, n_samples=10000):
        """Sample from two moons distribution"""
        X, y = skld.make_moons(n_samples=n_samples, noise=self.noise)
        # Scale to fit within -6 to 6 range and center at (0,0)
        X = X * 3 - np.array([1.5, 0.25])  # Scale and center at origin
        return torch.FloatTensor(X).to(self.device)

    def calculate_reward(self, samples):
        if isinstance(samples, torch.Tensor):
            x = samples.detach().cpu().numpy()
        else:
            x = samples
        r2 = np.sum(x ** 2, axis=1)
        return self.beta * np.exp(-0.5 * r2 / (self.sigma_r ** 2))

    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate two-moons bridge reward (differentiable)"""
        r2 = torch.sum(samples ** 2, dim=1)
        return self.beta * torch.exp(-0.5 * r2 / (self.sigma_r ** 2))

# ------------------------------------------------------------------------- #
# 4. Banana (Rosenbrock)
# ------------------------------------------------------------------------- #
class BananaRosenbrock(Dataset2D):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu', alpha=0.05, sigma_base=1.0):
        super().__init__(config, device)
        self.name = "banana_rosenbrock"
        self.alpha = alpha
        self.sigma_base = sigma_base
        self.load_gt_samples()

    def sample(self, n_samples=10_000):
        samples = np.random.randn(n_samples, 2) * self.sigma_base
        return torch.tensor(samples, dtype=torch.float32, device=self.device)

    def calculate_reward(self, samples):
        if isinstance(samples, torch.Tensor):
            x = samples.detach().cpu().numpy()
        else:
            x = samples
        term1 = (1 - x[:, 0]) ** 2
        term2 = 100 * (x[:, 1] - x[:, 0] ** 2) ** 2
        return -self.alpha * (term1 + term2)

    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate Banana (Rosenbrock) reward (differentiable)"""
        term1 = (1 - samples[:, 0]) ** 2
        term2 = 100 * (samples[:, 1] - samples[:, 0] ** 2) ** 2
        return -self.alpha * (term1 + term2)

DATASETS = {
    "gaussian_mixture": GaussianMixture,
    "two_moons": TwoMoons,
    "spiral": Spiral,
    "checkerboard": Checkerboard,
    "checkerboard_stripe": CheckerboardStripe,
    "two_moons_bridge": TwoMoonsBridge,
    "banana_rosenbrock": BananaRosenbrock
}


def get_2d_dataset(config, name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Helper function to get a dataset by name"""
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} not found. Available datasets: {list(DATASETS.keys())}")
    return DATASETS[name](config, device=device)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    config = get_default_toy_config()
    config = process_toy_config(config)
    for dataset in DATASETS:
        if dataset in ["swiss_roll"]:
            continue
        if dataset not in ["checkerboard"]:
            continue
        # if dataset not in ['gaussian_mixture', 'checkerboard']:
        #     continue
        print(dataset)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        dataset = get_2d_dataset(config, dataset)
        samples, rewards = dataset.sample_product_distribution_diff(n_samples=10000)
        samples = samples.cpu().numpy()
        rewards = rewards

        save_path = f"experiments/toy/checkpoints/{dataset.name}/samples.npy"
        np.save(save_path, samples)
        save_path = f"experiments/toy/checkpoints/{dataset.name}/rewards.npy"
        np.save(save_path, rewards)
        # plot samples
        axs[0].scatter(samples[:, 0], samples[:, 1], s=1)
        axs[0].set_title(dataset.name)
        samples, rewards = dataset.sample_product_distribution(n_samples=1000)
        samples = samples.cpu().numpy()
        axs[1].scatter(samples[:, 0], samples[:, 1], s=1, c=rewards)
        axs[1].set_title(dataset.name)
        samples, rewards = dataset.sample_product_distribution_diff(n_samples=1000)
        samples = samples.cpu().numpy()
        axs[2].scatter(samples[:, 0], samples[:, 1], s=1, c=rewards)
        axs[2].set_title(dataset.name)
        plt.savefig(f"{dataset.name}.png")
        plt.close()