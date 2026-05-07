**Idea: Heavy-Tailed Covariance Mixture Sampling**

The worst unsolved tasks (errors ~10²-10⁰) suggest the algorithm gets trapped in local optima or fails to explore effectively due to poorly conditioned covariance. Standard Gaussian sampling in eigenspace cannot escape these traps. This variant mixes standard Gaussian sampling with a heavy-tailed (Cauchy-like) distribution along principal axes, enabling long jumps that can escape local optima and explore different basins.

```python
def _sample_population_batch(self):
    """
    Sample population using a mixture of standard Gaussian and heavy-tailed
    distributions in the eigenspace of C. Heavy-tailed component helps escape
    local optima on ill-conditioned/deceptive tasks.
    """
    pop_size = self.pop_size
    dim = self.dim
    
    # Mixture weight: more heavy-tailed for exploration when stuck
    heavy_tail_prob = min(0.4, 0.1 + 0.05 * min(self.generation, 50))
    
    # Determine how many from each component
    n_heavy = int(pop_size * heavy_tail_prob)
    n_standard = pop_size - n_heavy
    
    # Standard Gaussian component (in eigenspace of C)
    z_std = np.random.randn(n_standard, dim)
    scaled_std = z_std * self.D[np.newaxis, :]
    rotated_std = scaled_std @ self.B.T
    pop_std = self.mean[np.newaxis, :] + self.sigma * rotated_std
    
    # Heavy-tailed component: t-distribution with nu=3 degrees of freedom
    # Equivalent to sqrt(nu/chi2) * standard_normal, scaled by D
    # This gives heavier tails than Gaussian, enabling long jumps
    nu = 3.0
    z_heavy = np.random.randn(n_heavy, dim)
    # Sample chi2 and ensure positive (numerical robustness)
    chi2_samples = np.random.chisquare(nu, size=n_heavy)
    chi2_samples = np.maximum(chi2_samples, 1e-10)
    # t-distribution scaling: sqrt(nu/chi2) gives heavy tails
    t_scale = np.sqrt(nu / chi2_samples)[:, np.newaxis]
    z_heavy_tailed = z_heavy * t_scale
    
    # Transform heavy-tailed samples through covariance structure
    scaled_heavy = z_heavy_tailed * (self.D * 2.0)[np.newaxis, :]  # 2x scaling for longer jumps
    rotated_heavy = scaled_heavy @ self.B.T
    pop_heavy = self.mean[np.newaxis, :] + self.sigma * rotated_heavy
    
    # Combine components
    population = np.vstack([pop_std, pop_heavy])
    
    # Clip to bounds
    population = np.clip(population, self.lb, self.ub)
    
    # Also clip to prevent numerical overflow
    population = np.clip(population, -1e+15, 1e+15)
    
    return population, None
```