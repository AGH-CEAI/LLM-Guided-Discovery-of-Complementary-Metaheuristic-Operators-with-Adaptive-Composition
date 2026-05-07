**Idea: Entropy-KL Divergence Inertia Adaptation**
One-line description: Adapt inertia weight by measuring entropy of fitness distribution and KL divergence of spatial distribution from uniform, signaling when population has converged and needs more exploration (Category C: Information-theoretic / distributional).

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using information-theoretic signals.
    
    Category C: Treats the population as a probability distribution.
    - Entropy of normalized fitness values: measures fitness diversity
    - KL divergence of spatial distribution from uniform: measures exploration coverage
    Low entropy + high KL = converged population = increase inertia for exploration.
    """
    # --- Fitness entropy: measures convergence via fitness diversity ---
    # Normalize fitness to probability distribution
    f_min = np.min(self.current_fitness)
    p_fit = self.current_fitness - f_min + 1e-10
    p_fit = p_fit / (np.sum(p_fit) + 1e-10)
    
    # Shannon entropy of fitness distribution (normalized)
    entropy_fit = -np.sum(p_fit * np.log(p_fit + 1e-10))
    max_entropy = np.log(self.np) if self.np > 1 else 1.0
    normalized_entropy_fit = np.clip(entropy_fit / (max_entropy + 1e-10), 0.0, 1.0)
    
    # --- Spatial KL divergence: measures deviation from uniform coverage ---
    # For each dimension, compare empirical distribution to uniform via histogram
    kl_div = 0.0
    spatial_entropy = 0.0
    n_bins = min(10, max(3, self.np // 5))  # adaptive bin count
    
    for d in range(self.dim):
        hist, _ = np.histogram(self.population[:, d], bins=n_bins, density=True)
        hist = np.clip(hist, 1e-10, None)
        
        # KL divergence from uniform distribution
        uniform_prob = 1.0 / n_bins
        kl_div += np.sum(hist * np.log(hist / (uniform_prob + 1e-10)))
        
        # Spatial entropy per dimension
        spatial_entropy -= np.mean(hist * np.log(hist))
    
    kl_div = kl_div / (self.dim + 1e-10)  # average across dimensions
    spatial_entropy = spatial_entropy / (self.dim + 1e-10)
    
    # Normalize KL: higher = more concentrated = less exploration
    kl_normalized = np.clip(kl_div / 5.0, 0.0, 1.0)
    
    # --- Combine information-theoretic signals ---
    # Low entropy_fit + high kl_normalized = converged = need exploration
    # exploration_signal in [0, 1]: 1 = fully diverse, 0 = fully converged
    exploration_signal = normalized_entropy_fit * (1.0 - kl_normalized)
    exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
    
    # Inertia target: high when converged (exploration_signal low), low when diverse
    target_inertia = 0.95 - 0.55 * exploration_signal
    
    # --- Generation-based schedule for long-term convergence ---
    decay = 0.729 - 0.15 * (self.generation / 1000)
    target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)
    
    # EMA smoothing to avoid jitter
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```