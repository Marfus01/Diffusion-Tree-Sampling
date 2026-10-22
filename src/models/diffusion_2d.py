import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import argparse
import random
import os
import sys
import time
from src.utils.datasets import get_2d_dataset, DATASETS
from src.core.config import get_default_toy_config, process_toy_config

# Import the new plot utilities
from src.utils.plotting_utils import (
    visualize_dataset_comparison,
    visualize_oneshot_denoising,
    visualize_diffusion_process,
    plot_reward_heatmap,
    visualize_reward_and_distributions
)

# Neural network to predict noise
class NoisePredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=64):
        super().__init__()
        
        # Sinusoidal position embedding for time
        self.time_dim = time_dim
        
        # MLP to further process the sinusoidal embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        # Create sinusoidal time embeddings
        half_dim = self.time_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Process time embeddings through MLP
        t_emb = self.time_mlp(emb)
        
        # Concatenate input and time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # Predict noise
        return self.net(x_t)

# Diffusion Model class
class DiffusionModel:
    def __init__(self, input_dim=2, hidden_dim=128, n_timesteps=500, beta_start=1e-4, beta_end=0.02, lr=1e-3, device='cuda'):
        self.n_timesteps = n_timesteps
        self.input_dim = input_dim
        self.device = device
        
        # Define noise schedule (linearly increasing variance)
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Initialize noise predictor network
        self.noise_predictor = NoisePredictor(input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.noise_predictor.parameters(), lr=lr)
        # Add a learning rate scheduler for faster convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000, eta_min=1e-6)
    
    # Function to visualize noise schedule for different beta values
    def visualize_noise_schedule(self, beta_values=None, save_path=None):
        """
        Visualize the diffusion process sigma values as a function of time.
        
        Args:
            beta_values: List of (beta_start, beta_end) tuples to compare
            save_path: Path to save the visualization (optional)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if beta_values is None:
            # Default comparison values
            beta_values = [
                (1e-3, 0.05),  # Current model values
                (1e-3, 0.06),
                (1e-3, 0.07),
            ]
        
        plt.figure(figsize=(12, 8))
        timesteps = np.arange(self.n_timesteps)
        
        # Plot other noise schedules for comparison
        for bs, be in beta_values:                
            betas = torch.linspace(bs, be, self.n_timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            
            plt.plot(timesteps, sqrt_one_minus_alphas_cumprod.numpy(), 
                    label=f'β_start={bs}, β_end={be}', 
                    linestyle='--')
        
        plt.xlabel('Timestep t')
        plt.ylabel('Noise Scale σ(t)')
        plt.title('Diffusion Process Noise Scale Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            # Create filename based on beta values if not specified
            if save_path.endswith('/') or os.path.isdir(save_path):
                # If save_path is a directory, create filename with beta values
                os.makedirs(save_path, exist_ok=True)
                filename = f"noise_schedule_beta.png"
                full_path = os.path.join(save_path, filename)
            else:
                # If save_path includes a filename, use it as is
                full_path = save_path
            
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
        else:
            # If no save_path provided, save in current directory with beta values in name
            filename = f"noise_schedule_beta.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.close()
    
    def extract(self, a, t, x_shape):
        """Helper function to extract timestep-specific values."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def sample_initial_noise(self, batch_size=1):
        """Sample pure noise as starting point for reverse diffusion."""
        return torch.randn(batch_size, self.input_dim).to(self.device)
    
    def q_sample(self, x_0, t, noise=None):
        # Convert t to tensor if it's an integer
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.device).long()
        
        # Ensure t has the right shape for batch processing
        if t.shape[0] == 1 and x_0.shape[0] > 1:
            t = t.expand(x_0.shape[0])

        """Forward diffusion process: add noise to the clean data."""
        noise = torch.randn_like(x_0) if noise is None else noise
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    # def p_sample(self, x_t, t):
    #     """Single step of reverse diffusion process using DDPM sampling."""
    #     self.noise_predictor.eval()
        
    #     # Convert t to tensor if it's an integer
    #     if not isinstance(t, torch.Tensor):
    #         t = torch.tensor([t], device=self.device).long()
        
    #     # Ensure t has the right shape for batch processing
    #     if t.shape[0] == 1 and x_t.shape[0] > 1:
    #         t = t.expand(x_t.shape[0])
        
    #     with torch.no_grad():
    #         # Predict noise
    #         predicted_noise = self.noise_predictor(x_t, t.float() / self.n_timesteps)
            
    #         # Get predicted x_0
    #         sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
    #         sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    #         predicted_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            
    #         # Get the coefficients for the posterior mean
    #         betas_t = self.extract(self.betas, t, x_t.shape)
    #         alpha_t = self.extract(self.alphas, t, x_t.shape)
    #         alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x_t.shape)
    #         alpha_cumprod_prev_t = self.extract(self.alphas_cumprod_prev, t, x_t.shape)
            
    #         # Calculate posterior mean
    #         posterior_mean = (alpha_cumprod_prev_t.sqrt() * betas_t / (1 - alpha_cumprod_t)) * predicted_x0 + \
    #                         ((1 - alpha_cumprod_prev_t) * alpha_t.sqrt() / (1 - alpha_cumprod_t)) * x_t
            
    #         # Get posterior variance
    #         variance = self.extract(self.posterior_variance, t, x_t.shape)
            
    #         # Add noise only if t > 0, handling vectorized case
    #         noise = torch.zeros_like(x_t)
    #         mask = (t > 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
    #         if mask.sum() > 0:
    #             noise = torch.randn_like(x_t) * mask
            
    #         # Sample from posterior
    #         x_t_prev = posterior_mean + variance.sqrt() * noise
            
    #         return x_t_prev

    def p_sample(self, x_t, t, eta=1.0, guidance_grad=None, guidance_scale=1.0, return_log_prob=False, use_smc=False,
                 noise_pred=None):
        """
        Single step of reverse diffusion process using DDIM sampling.

        Args:
            x_t (torch.Tensor): The noisy input tensor at timestep t.
            t (Union[int, torch.Tensor]): The current timestep index (or indices).
            eta (float): Controls the stochasticity of the sampling (0 = DDIM, 1 = DDPM-like).
            noise_pred (torch.Tensor, optional): noise prediction at (x_t, t) already computed by the
                caller (e.g. inside the guidance gradient); the network is then not called again.

        Returns:
            torch.Tensor: The denoised tensor at timestep t-1. With return_log_prob, also the
            log-densities of x_{t-1} under the base kernel and under the guided proposal; both are
            0 for the deterministic step t=0 (zero posterior variance).
        """
        self.noise_predictor.eval()

        # Convert t to tensor if it's an integer
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.device).long()

        # Ensure t has the right shape for batch processing
        if t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])

        with torch.no_grad():
            if noise_pred is None:
                predicted_noise = self.noise_predictor(x_t, t.float() / self.n_timesteps)
            else:
                predicted_noise = noise_pred.detach()

            alpha_cumprod_t_prev = self.extract(self.alphas_cumprod_prev, t, x_t.shape) # Handles t=0 correctly

            # Predict x_0 using formula (12) from DDPM paper appendix
            sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

            predicted_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t

            # Use precomputed posterior variance: variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
            posterior_variance_t = self.extract(self.posterior_variance, t, x_t.shape)
            std_dev_t = eta * (posterior_variance_t ** 0.5)

            # Calculate the coefficient for the predicted noise term (direction pointing to x_t)
            dir_xt_coeff = (1 - alpha_cumprod_t_prev - std_dev_t**2).clamp(min=0).sqrt() # Clamp to avoid sqrt of negative

            # Calculate the mean prediction for x_{t-1}
            pred_prev_sample_mean = alpha_cumprod_t_prev.sqrt() * predicted_x0 + dir_xt_coeff * predicted_noise

            # Add noise if eta > 0
            noise = torch.zeros_like(x_t)
            if eta > 0:
                noise = torch.randn_like(x_t)

            # # Combine mean and noise term
            x_t_prev = pred_prev_sample_mean + std_dev_t * noise

            # noise = torch.randn_like(x_t)
            # predicted_noise = noise

            # posterior_variance_t = self.extract(self.posterior_variance, t, x_t.shape)
            # std_dev_t = eta * (posterior_variance_t ** 0.5)
        
            # sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            # sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            
            # # x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise
            # pred_prev_sample_mean = sqrt_alphas_cumprod_t * x_t
            # x_t_prev = sqrt_alphas_cumprod_t * x_t + sqrt_one_minus_alphas_cumprod_t * noise

            if guidance_grad is not None:
                guidance_grad = guidance_grad.to(self.device)
                x_t_prev = x_t_prev + std_dev_t ** 2 * guidance_scale * guidance_grad

            if return_log_prob:
                var = torch.broadcast_to(std_dev_t ** 2, x_t_prev.shape)
                stochastic = var > 0
                safe_var = torch.where(stochastic, var, torch.ones_like(var))
                log_norm = -0.5 * torch.log(2 * math.pi * safe_var)
                log_prob_diffusion = -0.5 * (x_t_prev - pred_prev_sample_mean).pow(2) / safe_var + log_norm
                log_prob_diffusion = torch.where(stochastic, log_prob_diffusion, torch.zeros_like(log_prob_diffusion)).sum(dim=-1)
                if guidance_grad is not None:
                    shift = std_dev_t ** 2 * guidance_scale * guidance_grad
                    log_prob_proposal = -0.5 * (x_t_prev - pred_prev_sample_mean - shift).pow(2) / safe_var + log_norm
                    log_prob_proposal = torch.where(stochastic, log_prob_proposal, torch.zeros_like(log_prob_proposal)).sum(dim=-1)
                else:
                    log_prob_proposal = log_prob_diffusion
                return x_t_prev, log_prob_diffusion, log_prob_proposal, predicted_noise
            else:
                return x_t_prev, predicted_noise
    
    def predict_noise(self, x_t, t):
        self.noise_predictor.eval()

        # Convert t to tensor if it's an integer
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.device).long()

        # Ensure t has the right shape for batch processing
        if t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])

        with torch.no_grad():
            predicted_noise = self.noise_predictor(x_t, t.float() / self.n_timesteps)

        return predicted_noise

    def scheduler_step(self, noise_pred, t, x_t, eta=1.0):

        # Convert t to tensor if it's an integer
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.device).long()

        # Ensure t has the right shape for batch processing
        if t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])

        alpha_cumprod_t_prev = self.extract(self.alphas_cumprod_prev, t, x_t.shape) # Handles t=0 correctly

        # Predict x_0 using formula (12) from DDPM paper appendix
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        predicted_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t

        # Use precomputed posterior variance: variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        posterior_variance_t = self.extract(self.posterior_variance, t, x_t.shape)
        std_dev_t = eta * (posterior_variance_t ** 0.5)

        # Calculate the coefficient for the predicted noise term (direction pointing to x_t)
        dir_xt_coeff = (1 - alpha_cumprod_t_prev - std_dev_t**2).clamp(min=0).sqrt() # Clamp to avoid sqrt of negative

        # Calculate the mean prediction for x_{t-1}
        pred_prev_sample_mean = alpha_cumprod_t_prev.sqrt() * predicted_x0 + dir_xt_coeff * noise_pred

        # Add noise if eta > 0
        noise = torch.zeros_like(x_t)
        if eta > 0:
            noise = torch.randn_like(x_t)

        # # Combine mean and noise term
        x_t_prev = pred_prev_sample_mean + std_dev_t * noise
        return x_t_prev, predicted_x0

    def ddim_sample(self, n_samples=1000, n_steps=100):
        """Generate samples using DDIM sampling."""
        self.noise_predictor.eval()
        
        # Start from pure noise
        x = self.sample_initial_noise(n_samples)
        
        # Get the subset of timesteps to use for sampling
        timesteps = list(range(0, self.n_timesteps, self.n_timesteps // n_steps))
        timesteps = sorted(timesteps, reverse=True)  # Start from T, go to 0
        
        for i, t in enumerate(timesteps):
            # If at last timestep, the previous timestep is 0
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            
            # Convert to tensor
            t_tensor = torch.ones(n_samples, dtype=torch.long, device=self.device) * t
            t_prev_tensor = torch.ones(n_samples, dtype=torch.long, device=self.device) * t_prev
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.noise_predictor(x, t_tensor.float() / self.n_timesteps)
            
            # Get alpha values for current and previous timesteps
            alpha_cumprod_t = self.extract(self.alphas_cumprod, t_tensor, x.shape)
            alpha_cumprod_t_prev = self.extract(self.alphas_cumprod, t_prev_tensor, x.shape)
            
            # Get predicted x_0
            sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
            sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t_tensor, x.shape)
            predicted_x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            
            # DDIM update formula
            sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
            sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1 - alpha_cumprod_t_prev)
            
            # Deterministic update (DDIM)
            x = sqrt_alpha_cumprod_t_prev * predicted_x0 + sqrt_one_minus_alpha_cumprod_t_prev * predicted_noise
        
        return x.cpu().numpy()
    
    def ddim_denoise(self, x_t, t):
        """Predict the clean sample directly from any noisy sample (one-shot denoising)."""
        self.noise_predictor.eval()
        
        # Convert t to tensor if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.device).expand(x_t.shape[0]).long()
        
        # Normalize t for the model (model expects t in range [0, 1])
        t_normalized = t.float() / self.n_timesteps
        
        # Predict the noise component
        predicted_noise = self.noise_predictor(x_t, t_normalized)
        
        # Extract the alphas for timestep t
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # predicted_noise = sqrt_one_minus_alphas_cumprod_t * x_t
        
        # Reverse the noise addition process to get x_0
        predicted_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        return predicted_x0, predicted_noise
    
    def train(self, dataloader, dataset_name, n_epochs=5000, save_interval=500):
        """Train the diffusion model."""
        # Create dataset-specific folders
        checkpoint_dir = os.path.join('checkpoints', dataset_name)
        plots_dir = os.path.join('plots', dataset_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        losses = []
        
        for epoch in tqdm.tqdm(range(n_epochs)):
            self.noise_predictor.train()
            epoch_loss = 0
            
            for batch_idx, (x_0,) in enumerate(dataloader):
                x_0 = x_0.to(self.device)
                
                # Randomly sample timesteps
                t = torch.randint(0, self.n_timesteps, (x_0.shape[0],), device=self.device).long()
                
                # Add noise to the input
                x_t, noise = self.q_sample(x_0, t)
                
                # Predict noise
                predicted_noise = self.noise_predictor(x_t, t.float() / self.n_timesteps)
                
                # Calculate loss (mean squared error between actual and predicted noise)
                loss = nn.MSELoss()(predicted_noise, noise)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Update learning rate based on loss
            self.scheduler.step()
            
            # Print progress and generate samples every save_interval epochs
            if (epoch + 1) % save_interval == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # # Generate samples using DDIM (with fewer steps for speed)
                # samples = self.ddim_sample(n_samples=500, n_steps=100)
                
                # # Visualize real data and generated samples
                # plt.figure(figsize=(12, 5))
                
                # plt.subplot(1, 2, 1)
                # clean_data = next(iter(dataloader))[0].cpu().numpy()
                # plt.scatter(clean_data[:, 0], clean_data[:, 1], s=5, alpha=0.5)
                # plt.title("Real Data")
                # plt.xlim(-6, 6)
                # plt.ylim(-6, 6)
                
                # plt.subplot(1, 2, 2)
                # plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
                # plt.title(f"Generated Samples (Epoch {epoch+1})")
                # plt.xlim(-6, 6)
                # plt.ylim(-6, 6)
                
                # plt.savefig(os.path.join(plots_dir, f"samples_epoch_{epoch+1}.png"))
                # plt.close()
                
                # # Save checkpoint
                # checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                # torch.save(self.noise_predictor.state_dict(), checkpoint_path)
        
        # Save the final trained model
        final_model_path = os.path.join(checkpoint_dir, "diffusion_model.pt")
        torch.save(self.noise_predictor.state_dict(), final_model_path)
        
        # Plot the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title(f"Training Loss - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(plots_dir, "training_loss.png"))
        plt.close()
        
        return losses

# Utility function for sampling from unnormalized probabilities
def sample_from_unnormalized_probs(items, unnormalized_probs):
    """
    Sample an item from a list according to unnormalized probabilities.
    This function doesn't require probabilities to sum to 1.
    
    Args:
        items: List of items to sample from
        unnormalized_probs: List of unnormalized probabilities (can be any positive numbers)
        
    Returns:
        One sampled item from the list
    """
    if len(items) != len(unnormalized_probs):
        raise ValueError("Items and probabilities must have the same length")
    
    if not items:
        raise ValueError("Items list cannot be empty")
    
    # Handle edge cases
    if len(items) == 1:
        return items[0]
    
    # Filter out any negative probabilities and handle all-zero case
    valid_indices = [i for i, p in enumerate(unnormalized_probs) if p > 0]
    if not valid_indices:
        # If all probabilities are zero or negative, sample uniformly
        return random.choice(items)
    
    # Create a list of (item, prob) tuples with only valid probabilities
    valid_items = [items[i] for i in valid_indices]
    valid_probs = [unnormalized_probs[i] for i in valid_indices]
    
    # Compute cumulative sum of probabilities
    total = sum(valid_probs)
    cumulative = [sum(valid_probs[:i+1]) / total for i in range(len(valid_probs))]
    
    # Sample a random number between 0 and 1
    r = random.random()
    
    # Find the index where r falls in the cumulative distribution
    for i, cum_prob in enumerate(cumulative):
        if r <= cum_prob:
            return valid_items[i]
    
    # Fallback (should almost never happen due to floating-point precision)
    return valid_items[-1]

# Utility function for sampling multiple items according to unnormalized probabilities
def sample_multiple(items, unnormalized_probs, n_samples=1, replacement=True):
    """
    Sample multiple items from a list according to unnormalized probabilities.
    
    Args:
        items: List of items to sample from
        unnormalized_probs: List of unnormalized probabilities (any positive numbers)
        n_samples: Number of samples to draw
        replacement: Whether to sample with replacement
        
    Returns:
        List of sampled items
    """
    if not replacement and n_samples > len(items):
        raise ValueError("Cannot sample more items without replacement than are available")
    
    if not replacement:
        # For sampling without replacement, use a different approach
        result = []
        remaining_items = items.copy()
        remaining_probs = unnormalized_probs.copy()
        
        for _ in range(n_samples):
            if not remaining_items:
                break
                
            item = sample_from_unnormalized_probs(remaining_items, remaining_probs)
            result.append(item)
            
            # Remove the selected item
            idx = remaining_items.index(item)
            remaining_items.pop(idx)
            remaining_probs.pop(idx)
        
        return result
    else:
        # For sampling with replacement, just call the single-item sampling function repeatedly
        return [sample_from_unnormalized_probs(items, unnormalized_probs) for _ in range(n_samples)]

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
    random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 256  # Increased from 128 for faster training
    n_epochs = 5000   # Reduced from 10000 to speed up training
    lr = 3e-3         # Increased from 1e-3 for faster convergence
    n_timesteps = 100  # Reduced from 1000 to speed up sampling
    beta_start = 1e-3
    beta_end = 0.07

    parser = argparse.ArgumentParser(description='Train a diffusion model or run one-shot denoising example')
    parser.add_argument('--dataset', type=str, default='gaussian_mixture', 
                        choices=list(DATASETS.keys()),
                        help='Dataset to use for training/testing')
    parser.add_argument('--epochs', type=int, default=n_epochs, help='Training epochs (cosine LR schedule spans all epochs).')
    parser.add_argument('--base_dir', type=str, default=None, help='Experiment root; the checkpoint is written to <base_dir>/toy/checkpoints/<dataset>/.')
    parser.add_argument('--plot-reward', action='store_true', help='Plot the reward function heatmap')
    parser.add_argument('--no-progress-bar', action='store_true', help='Disable tqdm progress bars and use simple progress messages')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    n_epochs = args.epochs
    use_progress_bar = not args.no_progress_bar
    print(f"Using dataset: {dataset_name}")

    config = get_default_toy_config()
    config.dataset.name = dataset_name
    if args.base_dir is not None:
        config.common.base_dir = args.base_dir
    config = process_toy_config(config)
    
    # 1. Get the dataset from datasets.py
    dataset_obj = get_2d_dataset(config, dataset_name, device='cpu')
    dataset = dataset_obj.sample(n_samples=10000)
    dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size, shuffle=True, 
                            num_workers=4 if torch.cuda.is_available() else 0)
    
    # 2. Initialize the diffusion model
    model = DiffusionModel(input_dim=2, hidden_dim=128, n_timesteps=n_timesteps, 
                            beta_start=beta_start, beta_end=beta_end, lr=lr, device=device)
    model.scheduler = optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=n_epochs, eta_min=1e-6)
    
    # Try to load the pre-trained model if available
    checkpoint_dir = config.paths.checkpoint_dir
    model_path = os.path.join(checkpoint_dir, "diffusion_model.pt")
    
    # Check if the file exists before attempting to load it
    if os.path.exists(model_path):
        model.noise_predictor.load_state_dict(torch.load(model_path))
        print(f"Loaded pre-trained diffusion model for {dataset_name}.")
    else:
        print(f"No pre-trained model found for {dataset_name}. Training a new model...")
        # Custom training loop to handle progress bar option
        
        plots_dir = config.paths.plot_dir
        losses = []
        epoch_report_interval = max(1, n_epochs // 20)  # Report progress ~20 times
        
        epoch_iterator = range(n_epochs)
        if use_progress_bar:
            epoch_iterator = tqdm.tqdm(epoch_iterator)
        
        start_time = time.time()
        for epoch in epoch_iterator:
            model.noise_predictor.train()
            epoch_loss = 0
            
            for batch_idx, (x_0,) in enumerate(dataloader):
                x_0 = x_0.to(device)
                
                # Randomly sample timesteps
                t = torch.randint(0, model.n_timesteps, (x_0.shape[0],), device=device).long()
                
                # Add noise to the input
                x_t, noise = model.q_sample(x_0, t)
                
                # Predict noise
                predicted_noise = model.noise_predictor(x_t, t.float() / model.n_timesteps)
                
                # Calculate loss (mean squared error between actual and predicted noise)
                loss = nn.MSELoss()(predicted_noise, noise)
                
                # Backpropagation
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Update learning rate based on loss
            model.scheduler.step()
            
            # Print progress at regular intervals if not using progress bar
            if not use_progress_bar and (epoch + 1) % epoch_report_interval == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, LR: {model.optimizer.param_groups[0]['lr']:.6f}")
                sys.stdout.flush()  # Ensure progress is visible in logs
            
            # Update progress bar description if using tqdm
            if use_progress_bar:
                epoch_iterator.set_description(f"Loss: {avg_loss:.4f}, LR: {model.optimizer.param_groups[0]['lr']:.6f}")
        
        end_time = time.time()
        print(f"Training completed {n_epochs} epochs in {end_time - start_time:.2f} seconds")

        # Save the final trained model
        final_model_path = os.path.join(checkpoint_dir, "diffusion_model.pt")
        torch.save(model.noise_predictor.state_dict(), final_model_path)
        
        # Plot the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title(f"Training Loss - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(plots_dir, "training_loss.png"))
        plt.close()
    
    plots_dir = config.paths.plot_dir
    
    # 3. Generate samples from the trained model using p_sample (DDPM) instead of ddim_sample
    print("Generating samples using DDPM (p_sample)...")
    n_samples = 10000
    # Start from pure noise
    x = model.sample_initial_noise(n_samples)
    
    # Configure the timestep iterator based on progress bar preference
    timesteps = range(model.n_timesteps - 1, -1, -1)
    report_steps = max(1, model.n_timesteps // 10)
    if use_progress_bar:
        timesteps = tqdm.tqdm(timesteps)

    # Step-by-step denoising
    for i, t in enumerate(timesteps):
        x, _ = model.p_sample(x, t)
        
        # Print progress message at regular intervals if not using progress bar
        if not use_progress_bar and (i + 1) % report_steps == 0:
            progress = (i + 1) / model.n_timesteps * 100
            print(f"Sampling progress: {progress:.1f}% (step {i+1}/{model.n_timesteps})")
            sys.stdout.flush()  # Ensure progress is visible in logs
    
    # Convert to numpy for visualization
    samples = x.cpu().numpy()
    
    # 4. Get samples from product distribution p(x)r(x)
    product_samples, _ = dataset_obj.sample_product_distribution(n_samples=10000)
    
    # 5. Visualize comparison between original data and generated samples (without product distribution)
    visualize_dataset_comparison(
        original_data=dataset,
        generated_samples=samples,
        filename="diffusion_samples",
        dataset_name=dataset_name,
        plots_dir=plots_dir
    )
    
    # 6. Plot reward and distributions together if requested
    if args.plot_reward:
        print("Plotting reward function and distributions...")
        visualize_reward_and_distributions(
            original_data=dataset,
            product_samples=product_samples,
            reward_fn=dataset_obj.calculate_reward,
            title=f"Reward and Distributions - {dataset_name}",
            filename=os.path.join(plots_dir, "product_distribution.png"),
            xlim=(-6, 6),
            ylim=(-6, 6)
        )
    
    # 7. Visualize one-shot denoising at different timesteps
    clean_samples = dataset_obj.sample(n_samples=10000).to(device)
    timesteps = [20, 40, 60, 80, 100]  # 5 different timestep values (adjusted for n_timesteps=100)
    
    # Prepare dictionaries for visualization
    noisy_samples_dict = {}
    predicted_samples_dict = {}
    
    for t_val in timesteps:
        t = torch.ones(clean_samples.shape[0], dtype=torch.long, device=device) * (t_val - 1)
        
        # Add noise to the clean samples
        noisy_samples, _ = model.q_sample(clean_samples, t)
        
        # Predict clean samples directly
        with torch.no_grad():
            predicted_clean, _ = model.ddim_denoise(noisy_samples, t)
        
        noisy_samples_dict[t_val] = noisy_samples
        predicted_samples_dict[t_val] = predicted_clean
    
    # Visualize one-shot denoising results
    visualize_oneshot_denoising(
        clean_samples=clean_samples,
        noisy_samples_dict=noisy_samples_dict,
        predicted_samples_dict=predicted_samples_dict,
        dataset_name=dataset_name,
        plots_dir=plots_dir
    )

    model.visualize_noise_schedule(beta_values=None, save_path=plots_dir)

if __name__ == "__main__":
    main()
