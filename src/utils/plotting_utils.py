import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import os

def density_plot_2d(points, ax=None, cmap='viridis', contour_levels=15, 
                    xlim=(-6, 6), ylim=(-6, 6), bw_method=0.15, 
                    grid_size=100, alpha=0.8, colorbar=True, 
                    scatter_points=False, scatter_alpha=0.1):
    """
    Create a 2D density plot using kernel density estimation.
    
    Args:
        points: numpy array of shape (n_samples, 2) containing 2D points
        ax: matplotlib axis for plotting (creates new one if None)
        cmap: colormap to use for the density plot
        contour_levels: number of contour levels to show
        xlim, ylim: plot limits
        bw_method: bandwidth for kernel density estimation
        grid_size: resolution of the density grid
        alpha: transparency of the density plot
        colorbar: whether to add a colorbar
        scatter_points: whether to overlay scatter plot of points
        scatter_alpha: transparency of scattered points
        
    Returns:
        matplotlib axis with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    
    # Create a grid for density estimation
    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Remove NaNs or infinite values
    valid_points = np.isfinite(points).all(axis=1)
    clean_points = points[valid_points]
    
    if len(clean_points) < 5:
        ax.text(0, 0, "Not enough valid points for density estimation", 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax
    
    # Compute kernel density estimate
    try:
        kde = gaussian_kde(clean_points.T, bw_method=bw_method)
        Z = kde(grid_positions).reshape(X.shape)
    except Exception as e:
        print(f"Error in density estimation: {e}")
        ax.text(0, 0, "Density estimation failed", ha='center', va='center', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap, alpha=alpha)
    
    # Optionally overlay scatter plot
    if scatter_points:
        ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, alpha=scatter_alpha, color='k')
    
    # Add colorbar
    if colorbar:
        plt.colorbar(contour, ax=ax)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    return ax

def comparison_density_plots(data_dict, title=None, filename=None, figsize=(15, 6),
                            xlim=(-6, 6), ylim=(-6, 6), cmaps=None,
                            scatter_overlay=False, grid_size=100, bw_method=0.15):
    """
    Create a side-by-side comparison of density plots.
    
    Args:
        data_dict: Dictionary of {name: points}, where points is a numpy array of shape (n_samples, 2)
        title: Overall figure title (optional)
        filename: Path to save the figure (optional)
        figsize: Size of the figure
        xlim, ylim: Plot limits
        cmaps: List of colormaps to use (one per plot)
        scatter_overlay: Whether to overlay scatter plots
        grid_size: Resolution of the density grid
        bw_method: Bandwidth for KDE
        
    Returns:
        matplotlib figure
    """
    n_plots = len(data_dict)
    if n_plots == 0:
        return None
    
    # Use a single colormap for all plots for consistent comparison
    cmap = 'viridis'
    if cmaps is None:
        cmaps = [cmap] * n_plots
    else:
        # Override with the same colormap for all plots
        cmaps = [cmap] * n_plots
    
    # Create figure and axes
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Find global min/max density for consistent color scaling
    all_points = []
    for points in data_dict.values():
        points_np = points.cpu().numpy() if hasattr(points, 'cpu') else points
        valid_points = np.isfinite(points_np).all(axis=1)
        if np.sum(valid_points) >= 5:
            all_points.append(points_np[valid_points])
    
    # Create a grid for global density estimation
    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Calculate global min/max density if possible
    global_min_density = None
    global_max_density = None
    if all_points:
        try:
            for points_np in all_points:
                kde = gaussian_kde(points_np.T, bw_method=bw_method)
                Z = kde(grid_positions).reshape(X.shape)
                
                if global_min_density is None or np.min(Z) < global_min_density:
                    global_min_density = np.min(Z)
                if global_max_density is None or np.max(Z) > global_max_density:
                    global_max_density = np.max(Z)
        except:
            pass
    
    # Create plot for each dataset
    for i, (name, points) in enumerate(data_dict.items()):
        points_np = points.cpu().numpy() if hasattr(points, 'cpu') else points
        
        # Create density plot
        ax = density_plot_2d(
            points_np, ax=axes[i], cmap=cmaps[i], 
            xlim=xlim, ylim=ylim, 
            scatter_points=scatter_overlay,
            grid_size=grid_size,
            bw_method=bw_method,
            colorbar=False  # We'll add a single colorbar later
        )
        
        # If we have global min/max, set consistent color scale
        if global_min_density is not None and global_max_density is not None:
            try:
                valid_points = np.isfinite(points_np).all(axis=1)
                clean_points = points_np[valid_points]
                if len(clean_points) >= 5:
                    kde = gaussian_kde(clean_points.T, bw_method=bw_method)
                    Z = kde(grid_positions).reshape(X.shape)
                    
                    # Clear previous contours and redraw with consistent levels
                    ax.clear()
                    levels = np.linspace(global_min_density, global_max_density, 15)
                    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmaps[i], alpha=0.8)
                    
                    if scatter_overlay:
                        ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, alpha=0.1, color='k')
                    
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            except:
                pass
        
        axes[i].set_title(name)
    
    # Add a single colorbar for the entire figure
    if global_min_density is not None and global_max_density is not None:
        norm = plt.Normalize(global_min_density, global_max_density)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Density')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    
    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def plot_density_evolution(samples_dict, title=None, filename=None, figsize=(15, 10),
                          xlim=(-6, 6), ylim=(-6, 6), cmaps=None, grid_size=100,
                          bw_method=0.15, nrows=2, show_scatter=False):
    """
    Plot density evolution across different timesteps or conditions.
    
    Args:
        samples_dict: Dictionary of {timestep/condition: points}
        title: Overall figure title
        filename: Path to save the figure
        figsize: Size of the figure
        xlim, ylim: Plot limits
        cmaps: List of colormaps to use
        grid_size: Resolution of the density grid
        bw_method: Bandwidth for KDE
        nrows: Number of rows in the subplot grid
        show_scatter: Whether to include scatter plots along with density plots
        
    Returns:
        matplotlib figure
    """
    n_plots = len(samples_dict)
    if n_plots == 0:
        return None
    
    # Calculate grid layout
    if nrows is None:
        nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))
    
    # Use a single colormap for all plots
    cmap = 'viridis'
    if cmaps is None:
        cmaps = [cmap] * n_plots
    else:
        # Override with the same colormap for all plots
        cmaps = [cmap] * n_plots
    
    # Adjust figure layout based on whether we're showing scatter plots
    if show_scatter:
        # Create a figure with 2 rows per timestep (density + scatter)
        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(figsize[0], figsize[1] * 1.5))
        axes = axes.reshape(2 * nrows, ncols)
    else:
        # Create regular figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
    
    # Sort samples by timestep (assuming numeric keys)
    sorted_items = sorted(samples_dict.items(), 
                         key=lambda x: float(x[0]) if isinstance(x[0], (int, float, str)) else 0)
    
    # Find global min/max density for consistent color scaling
    all_points = []
    for _, points in sorted_items:
        points_np = points.cpu().numpy() if hasattr(points, 'cpu') else points
        valid_points = np.isfinite(points_np).all(axis=1)
        if np.sum(valid_points) >= 5:
            all_points.append(points_np[valid_points])
    
    # Create a grid for global density estimation
    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Calculate global min/max density if possible
    global_min_density = None
    global_max_density = None
    if all_points:
        try:
            for points_np in all_points:
                kde = gaussian_kde(points_np.T, bw_method=bw_method)
                Z = kde(grid_positions).reshape(X.shape)
                
                if global_min_density is None or np.min(Z) < global_min_density:
                    global_min_density = np.min(Z)
                if global_max_density is None or np.max(Z) > global_max_density:
                    global_max_density = np.max(Z)
        except:
            pass
    
    # Create density plot for each timestep
    for i, (timestep, points) in enumerate(sorted_items):
        row_idx = i // ncols
        col_idx = i % ncols
        
        if row_idx < nrows and col_idx < ncols:
            points_np = points.cpu().numpy() if hasattr(points, 'cpu') else points
            valid_points = np.isfinite(points_np).all(axis=1)
            clean_points = points_np[valid_points]
            
            if show_scatter:
                # For scatter plots, we use two rows per timestep
                density_ax = axes[row_idx * 2, col_idx]
                scatter_ax = axes[row_idx * 2 + 1, col_idx]
                
                # Create density plot
                if len(clean_points) >= 5 and global_min_density is not None and global_max_density is not None:
                    try:
                        kde = gaussian_kde(clean_points.T, bw_method=bw_method)
                        Z = kde(grid_positions).reshape(X.shape)
                        
                        # Create contour plot with consistent levels
                        levels = np.linspace(global_min_density, global_max_density, 15)
                        contour = density_ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
                    except:
                        # Fallback to default density plot if KDE fails
                        density_plot_2d(
                            clean_points, ax=density_ax, cmap=cmap,
                            xlim=xlim, ylim=ylim, grid_size=grid_size, bw_method=bw_method,
                            colorbar=False
                        )
                else:
                    # Fallback to default density plot
                    density_plot_2d(
                        clean_points, ax=density_ax, cmap=cmap,
                        xlim=xlim, ylim=ylim, grid_size=grid_size, bw_method=bw_method,
                        colorbar=False
                    )
                
                # Create scatter plot
                scatter_ax.scatter(clean_points[:, 0], clean_points[:, 1], s=2, alpha=0.3, color='k')
                scatter_ax.set_xlim(xlim)
                scatter_ax.set_ylim(ylim)
                
                # Add titles
                density_ax.set_title(f"t={timestep} - Density")
                scatter_ax.set_title(f"t={timestep} - Scatter")
            else:
                # Regular density plot only
                ax = axes[row_idx, col_idx]
                
                if len(clean_points) >= 5 and global_min_density is not None and global_max_density is not None:
                    try:
                        kde = gaussian_kde(clean_points.T, bw_method=bw_method)
                        Z = kde(grid_positions).reshape(X.shape)
                        
                        # Create contour plot with consistent levels
                        levels = np.linspace(global_min_density, global_max_density, 15)
                        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
                    except:
                        # Fallback to default density plot if KDE fails
                        density_plot_2d(
                            clean_points, ax=ax, cmap=cmap,
                            xlim=xlim, ylim=ylim, grid_size=grid_size, bw_method=bw_method,
                            colorbar=False
                        )
                else:
                    # Fallback to default density plot
                    density_plot_2d(
                        clean_points, ax=ax, cmap=cmap,
                        xlim=xlim, ylim=ylim, grid_size=grid_size, bw_method=bw_method,
                        colorbar=False
                    )
                
                # Add timestep as title
                ax.set_title(f"t={timestep}")
    
    # Hide unused subplots
    if show_scatter:
        for i in range(n_plots, nrows * ncols):
            row_idx = (i // ncols) * 2
            col_idx = i % ncols
            if row_idx < 2 * nrows and col_idx < ncols:
                axes[row_idx, col_idx].axis('off')
                axes[row_idx + 1, col_idx].axis('off')
    else:
        for i in range(n_plots, nrows * ncols):
            row_idx = i // ncols
            col_idx = i % ncols
            if row_idx < nrows and col_idx < ncols:
                axes[row_idx, col_idx].axis('off')
    
    # Add colorbar for the whole figure
    if global_min_density is not None and global_max_density is not None:
        norm = plt.Normalize(global_min_density, global_max_density)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Density')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    
    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def plot_reward_heatmap(grid_size=100, xlim=(-6, 6), ylim=(-6, 6), 
                       reward_fn=None, title="Reward Function", 
                       filename=None, figsize=(8, 7), cmap='viridis'):
    """
    Plot a heatmap of the reward function over a 2D grid.
    
    Args:
        grid_size: Resolution of the grid
        xlim, ylim: Plot limits
        reward_fn: Function that takes 2D points and returns reward values
        title: Plot title
        filename: Path to save the figure
        figsize: Size of the figure
        cmap: Colormap for the heatmap
        
    Returns:
        matplotlib figure
    """
    if reward_fn is None:
        return None
    
    # Create the grid
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Create an array of grid points
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Compute rewards
    try:
        import torch
        # Convert to tensor if reward_fn expects tensors
        grid_tensor = torch.FloatTensor(grid_points)
        rewards = reward_fn(grid_tensor)
        # Convert back to numpy if needed
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
    except:
        # Fallback to direct numpy input
        rewards = np.array([reward_fn(p) for p in grid_points])
    
    # Reshape to grid
    Z = rewards.reshape(X.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    
    # Add contour lines for better visualization
    contour = ax.contour(X, Y, Z, colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Add colorbar with proper positioning
    fig.subplots_adjust(right=0.85)  # Make room for colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Reward')
    
    # Set limits and title
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9)
    
    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_diffusion_process(diffusion_samples, dataset_name, plots_dir=None):
    """
    Visualize the diffusion process with samples from different timesteps.
    
    Args:
        diffusion_samples: Dictionary mapping timesteps to samples
        dataset_name: Name of the dataset
        plots_dir: Directory to save plots
        
    Returns:
        None
    """
    # Default plots directory if not provided
    if plots_dir is None:
        plots_dir = os.path.join('plots', dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
    
    # Plot the diffusion process
    plot_density_evolution(
        diffusion_samples,
        title=f"Diffusion Process - {dataset_name}",
        filename=os.path.join(plots_dir, "diffusion_process.png"),
        figsize=(15, 10),
        xlim=(-6, 6),
        ylim=(-6, 6)
    )

def visualize_dataset_comparison(original_data, generated_samples, filename=None,
                               dataset_name="dataset", plots_dir=None, show_scatter=True):
    """
    Visualize comparison between original data, generated samples
    
    Args:
        original_data: Tensor or numpy array of original data points
        generated_samples: Tensor or numpy array of generated samples
        product_samples: Tensor or numpy array of product distribution samples (optional)
        dataset_name: Name of the dataset
        plots_dir: Directory to save plots
        show_scatter: Whether to include scatter plots along with density plots (default: True)
        
    Returns:
        None
    """
    # Default plots directory if not provided
    if plots_dir is None:
        plots_dir = os.path.join('plots', dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
    
    # Prepare data dictionary
    data_dict = {
        "Original Data": original_data,
        "Generated Samples": generated_samples
    }
    
    # Convert data to numpy if needed
    data_np = {}
    for key, data in data_dict.items():
        data_np[key] = data.cpu().numpy() if hasattr(data, 'cpu') else data
    
    if show_scatter:
        # Create a 2x2 figure with density plots on top and scatter plots on bottom
        n_plots = len(data_dict)
        fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))
        
        # Plot density plots in the top row
        for i, (name, data) in enumerate(data_np.items()):
            density_plot_2d(
                data, ax=axes[0, i], cmap='viridis',
                xlim=(-6, 6), ylim=(-6, 6), grid_size=100,
                bw_method=0.15, colorbar=True
            )
            axes[0, i].set_title(f"{name} - Density")
        
        # Plot scatter plots in the bottom row
        for i, (name, data) in enumerate(data_np.items()):
            axes[1, i].scatter(data[:, 0], data[:, 1], s=2, alpha=0.3)
            axes[1, i].set_xlim((-6, 6))
            axes[1, i].set_ylim((-6, 6))
            axes[1, i].set_title(f"{name} - Scatter")
        
        plt.suptitle(f"Original Distribution and Generated Samples - {dataset_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the figure
        # Use the dataset_name to create a more specific filename if provided
        filename = os.path.join(plots_dir, f"{filename}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        # Create comparison plot with density plots only
        comparison_density_plots(
            data_dict,
            title=f"Original Distribution and Generated Samples - {dataset_name}",
            filename=os.path.join(plots_dir, f"{filename}.png"),
            figsize=(15, 6),
            xlim=(-6, 6),
            ylim=(-6, 6),
            grid_size=100,
            bw_method=0.15
        )

def visualize_reward_and_distributions(original_data, product_samples, reward_fn,
                                      title="Reward and Distributions",
                                      filename=None, xlim=(-6, 6), ylim=(-6, 6),
                                      figsize=(18, 6), grid_size=100):
    """
    Create a three-panel plot showing original distribution, product distribution, and reward heatmap.
    
    Args:
        original_data: Tensor or numpy array of original data points
        product_samples: Tensor or numpy array of product distribution samples
        reward_fn: Function that takes 2D points and returns reward values
        title: Overall figure title
        filename: Path to save the figure
        xlim, ylim: Plot limits
        figsize: Size of the figure
        grid_size: Resolution of the density grid
        
    Returns:
        matplotlib figure
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert data to numpy if needed
    original_np = original_data.cpu().numpy() if hasattr(original_data, 'cpu') else original_data
    product_np = product_samples.cpu().numpy() if hasattr(product_samples, 'cpu') else product_samples
    
    # 1. Plot original data distribution
    density_plot_2d(
        original_np, ax=axes[0], cmap='viridis',
        xlim=xlim, ylim=ylim, grid_size=grid_size,
        bw_method=0.15, colorbar=True
    )
    axes[0].set_title("Original Distribution p(x)")
    
    # 2. Plot product distribution
    density_plot_2d(
        product_np, ax=axes[1], cmap='viridis',
        xlim=xlim, ylim=ylim, grid_size=grid_size,
        bw_method=0.15, colorbar=True
    )
    axes[1].set_title("Product Distribution p(x)r(x)")
    
    # 3. Create reward heatmap
    # Create the grid
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Create an array of grid points
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Compute rewards
    try:
        import torch
        # Convert to tensor if reward_fn expects tensors
        grid_tensor = torch.FloatTensor(grid_points)
        rewards = reward_fn(grid_tensor)
        # Convert back to numpy if needed
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
    except:
        # Fallback to direct numpy input
        rewards = np.array([reward_fn(p) for p in grid_points])
    
    # Reshape to grid
    Z = rewards.reshape(X.shape)
    
    # Plot heatmap
    im = axes[2].pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    
    # Add contour lines for better visualization
    contour = axes[2].contour(X, Y, Z, colors='k', linewidths=0.5, alpha=0.5)
    axes[2].clabel(contour, inline=True, fontsize=8)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[2])
    cbar.set_label('Reward')
    
    # Set limits and title
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    axes[2].set_title("Reward Function r(x)")
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Save figure
    if filename:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_oneshot_denoising(clean_samples, noisy_samples_dict, predicted_samples_dict,
                              dataset_name="dataset", plots_dir=None, show_scatter=True):
    """
    Visualize one-shot denoising at different timesteps.
    
    Args:
        clean_samples: Tensor or numpy array of original clean samples
        noisy_samples_dict: Dictionary mapping timesteps to noisy samples
        predicted_samples_dict: Dictionary mapping timesteps to predicted clean samples
        dataset_name: Name of the dataset
        plots_dir: Directory to save plots
        show_scatter: Whether to overlay scatter plots on density plots (default: True)
        
    Returns:
        None
    """
    # Default plots directory if not provided
    if plots_dir is None:
        plots_dir = os.path.join('plots', dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
    
    # Convert clean samples to numpy if needed
    clean_np = clean_samples.cpu().numpy() if hasattr(clean_samples, 'cpu') else clean_samples
    
    # Sort timesteps
    timesteps = sorted(noisy_samples_dict.keys())
    
    # Calculate number of plots needed (clean samples + timesteps)
    n_plots = 1 + len(timesteps)
    nrows = 2  # Fixed number of rows
    ncols = int(np.ceil(n_plots / nrows))
    
    # Create figure for visualization
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Define consistent limits for all plots
    xlim = (-6, 6)
    ylim = (-6, 6)
    cmap = 'viridis'
    grid_size = 100
    bw_method = 0.15
    
    # Create a grid for global density estimation
    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Collect all points for global min/max density calculation
    all_points = [clean_np]
    for t in timesteps:
        noisy = noisy_samples_dict[t].cpu().numpy() if hasattr(noisy_samples_dict[t], 'cpu') else noisy_samples_dict[t]
        predicted = predicted_samples_dict[t].cpu().numpy() if hasattr(predicted_samples_dict[t], 'cpu') else predicted_samples_dict[t]
        all_points.append(noisy)
        all_points.append(predicted)
    
    # Calculate global min/max density
    global_min_density = None
    global_max_density = None
    for points_np in all_points:
        valid_points = np.isfinite(points_np).all(axis=1)
        clean_points = points_np[valid_points]
        if len(clean_points) >= 5:
            try:
                kde = gaussian_kde(clean_points.T, bw_method=bw_method)
                Z = kde(grid_positions).reshape(X.shape)
                
                if global_min_density is None or np.min(Z) < global_min_density:
                    global_min_density = np.min(Z)
                if global_max_density is None or np.max(Z) > global_max_density:
                    global_max_density = np.max(Z)
            except:
                pass
    
    # Plot original clean samples
    valid_points = np.isfinite(clean_np).all(axis=1)
    clean_points = clean_np[valid_points]
    
    if len(clean_points) >= 5 and global_min_density is not None and global_max_density is not None:
        try:
            kde = gaussian_kde(clean_points.T, bw_method=bw_method)
            Z = kde(grid_positions).reshape(X.shape)
            
            # Create contour plot with consistent levels
            levels = np.linspace(global_min_density, global_max_density, 15)
            contour = axes[0].contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
            
            # Add scatter plot if requested
            if show_scatter:
                # Sample a subset of points if there are too many
                max_scatter_points = 1000
                if len(clean_points) > max_scatter_points:
                    indices = np.random.choice(len(clean_points), max_scatter_points, replace=False)
                    scatter_points = clean_points[indices]
                else:
                    scatter_points = clean_points
                axes[0].scatter(scatter_points[:, 0], scatter_points[:, 1], s=5, color='black', alpha=0.5)
        except:
            # Fallback to default density plot
            density_plot_2d(clean_points, ax=axes[0], cmap=cmap, xlim=xlim, ylim=ylim, colorbar=False)
            # Add scatter plot if requested
            if show_scatter:
                max_scatter_points = 1000
                if len(clean_points) > max_scatter_points:
                    indices = np.random.choice(len(clean_points), max_scatter_points, replace=False)
                    scatter_points = clean_points[indices]
                else:
                    scatter_points = clean_points
                axes[0].scatter(scatter_points[:, 0], scatter_points[:, 1], s=5, color='black', alpha=0.5)
    else:
        # Fallback to default density plot
        density_plot_2d(clean_points, ax=axes[0], cmap=cmap, xlim=xlim, ylim=ylim, colorbar=False)
        # Add scatter plot if requested
        if show_scatter:
            max_scatter_points = 1000
            if len(clean_points) > max_scatter_points:
                indices = np.random.choice(len(clean_points), max_scatter_points, replace=False)
                scatter_points = clean_points[indices]
            else:
                scatter_points = clean_points
            axes[0].scatter(scatter_points[:, 0], scatter_points[:, 1], s=5, color='black', alpha=0.5)
    
    axes[0].set_title(f"Original Clean Samples - {dataset_name}")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    
    # Plot noisy and denoised samples for each timestep
    for i, t in enumerate(timesteps):
        if i < len(axes) - 1:  # Ensure we don't exceed available axes
            ax_idx = i + 1  # Skip the first axis (used for clean samples)
            
            # Get samples
            noisy = noisy_samples_dict[t].cpu().numpy() if hasattr(noisy_samples_dict[t], 'cpu') else noisy_samples_dict[t]
            predicted = predicted_samples_dict[t].cpu().numpy() if hasattr(predicted_samples_dict[t], 'cpu') else predicted_samples_dict[t]
            
            # Get valid points for both noisy and predicted samples
            valid_noisy_points = np.isfinite(noisy).all(axis=1)
            clean_noisy = noisy[valid_noisy_points]
            
            valid_pred_points = np.isfinite(predicted).all(axis=1)
            clean_predicted = predicted[valid_pred_points]
            
            if len(clean_predicted) >= 5 and global_min_density is not None and global_max_density is not None:
                try:
                    # Create density plot of the predicted samples (not noisy samples)
                    kde = gaussian_kde(clean_predicted.T, bw_method=bw_method)
                    Z = kde(grid_positions).reshape(X.shape)
                    
                    # Create contour plot with consistent levels
                    levels = np.linspace(global_min_density, global_max_density, 15)
                    contour = axes[ax_idx].contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
                    
                    # Add scatter plot if requested
                    if show_scatter:
                        # Sample a subset of points if there are too many
                        max_scatter_points = 500  # Fewer points for each type to avoid overcrowding
                        
                        # Plot noisy samples
                        if len(clean_noisy) > max_scatter_points:
                            indices = np.random.choice(len(clean_noisy), max_scatter_points, replace=False)
                            scatter_noisy = clean_noisy[indices]
                        else:
                            scatter_noisy = clean_noisy
                        axes[ax_idx].scatter(scatter_noisy[:, 0], scatter_noisy[:, 1], s=5, color='black', alpha=0.5, label='Noisy')
                        
                        # Plot predicted samples
                        if len(clean_predicted) > max_scatter_points:
                            indices = np.random.choice(len(clean_predicted), max_scatter_points, replace=False)
                            scatter_predicted = clean_predicted[indices]
                        else:
                            scatter_predicted = clean_predicted
                        axes[ax_idx].scatter(scatter_predicted[:, 0], scatter_predicted[:, 1], s=5, color='red', alpha=0.5, label='Denoised')
                        
                        # Add legend
                        axes[ax_idx].legend(loc='upper right', fontsize=8)
                except:
                    # Fallback to default density plot
                    density_plot_2d(clean_predicted, ax=axes[ax_idx], cmap=cmap, xlim=xlim, ylim=ylim, colorbar=False)
                    
                    # Add scatter plot if requested
                    if show_scatter:
                        max_scatter_points = 500
                        
                        # Plot noisy samples
                        if len(clean_noisy) > max_scatter_points:
                            indices = np.random.choice(len(clean_noisy), max_scatter_points, replace=False)
                            scatter_noisy = clean_noisy[indices]
                        else:
                            scatter_noisy = clean_noisy
                        axes[ax_idx].scatter(scatter_noisy[:, 0], scatter_noisy[:, 1], s=5, color='black', alpha=0.5, label='Noisy')
                        
                        # Plot predicted samples
                        if len(clean_predicted) > max_scatter_points:
                            indices = np.random.choice(len(clean_predicted), max_scatter_points, replace=False)
                            scatter_predicted = clean_predicted[indices]
                        else:
                            scatter_predicted = clean_predicted
                        axes[ax_idx].scatter(scatter_predicted[:, 0], scatter_predicted[:, 1], s=5, color='red', alpha=0.5, label='Denoised')
                        
                        # Add legend
                        axes[ax_idx].legend(loc='upper right', fontsize=8)
            else:
                # Fallback to default density plot
                density_plot_2d(clean_predicted, ax=axes[ax_idx], cmap=cmap, xlim=xlim, ylim=ylim, colorbar=False)
                
                # Add scatter plot if requested
                if show_scatter:
                    max_scatter_points = 500
                    
                    # Plot noisy samples
                    if len(clean_noisy) > max_scatter_points:
                        indices = np.random.choice(len(clean_noisy), max_scatter_points, replace=False)
                        scatter_noisy = clean_noisy[indices]
                    else:
                        scatter_noisy = clean_noisy
                    axes[ax_idx].scatter(scatter_noisy[:, 0], scatter_noisy[:, 1], s=5, color='black', alpha=0.5, label='Noisy')
                    
                    # Plot predicted samples
                    if len(clean_predicted) > max_scatter_points:
                        indices = np.random.choice(len(clean_predicted), max_scatter_points, replace=False)
                        scatter_predicted = clean_predicted[indices]
                    else:
                        scatter_predicted = clean_predicted
                    axes[ax_idx].scatter(scatter_predicted[:, 0], scatter_predicted[:, 1], s=5, color='red', alpha=0.5, label='Denoised')
                    
                    # Add legend
                    axes[ax_idx].legend(loc='upper right', fontsize=8)
            
            axes[ax_idx].set_title(f"Noisy (t={t}) & Denoised")
            axes[ax_idx].set_xlim(xlim)
            axes[ax_idx].set_ylim(ylim)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    # Add a single colorbar for the entire figure
    if global_min_density is not None and global_max_density is not None:
        norm = plt.Normalize(global_min_density, global_max_density)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Density')
    
    fig.suptitle(f"One-shot Denoising - {dataset_name}", fontsize=14)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    
    if plots_dir:
        # Ensure directory exists
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "oneshot_denoising.png"), dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    # Example usage
    print("This module provides plotting utilities for diffusion models.") 