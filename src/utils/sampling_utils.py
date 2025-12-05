import random

def sample_from_unnormalized_probs(items, unnormalized_probs):
    """
    Sample an item from a list according to unnormalized probabilities.
    This function doesn't require probabilities to sum to 1.
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


# Population-based resampling function
def resample_population(population, value_fn=None, values=None):
    """
    Resamples a population according to their value function, maintaining the same population size.
    
    Args:
        population: List of items to be resampled
        value_fn: Function that computes a value/fitness for each item (higher is better)
                  Only used if values are not directly provided
        values: Pre-computed values/fitness scores (optional)
        
    Returns:
        A new list with the same length as population, where items with higher values appear more frequently
    """
    if not population:
        return []
        
    population_size = len(population)
    
    # Get values if not provided
    if values is None:
        if value_fn is None:
            raise ValueError("Either value_fn or values must be provided")
        values = [value_fn(item) for item in population]
    
    # Ensure all values are positive for resampling proportions
    # Add a small constant to avoid zero probabilities
    min_val = min(values)
    if min_val <= 0:
        # Shift values so they're all positive
        values = [v - min_val + 1e-6 for v in values]
    
    # Create a new population by sampling with replacement
    new_population = []
    for _ in range(population_size):
        item = sample_from_unnormalized_probs(population, values)
        new_population.append(item)
        
    return new_population
