**Idea: Momentum-Directed Sampling with Stagnation-Adaptive Bias**

The worst unsolved tasks (16, 19, 20, 23 with errors 10-500x) are likely stuck in deceptive local optima or have narrow basins. The standard CMA-ES sampling samples purely from N(mean, σ²C), which cannot escape when the mean is trapped. This variant injects **directional momentum bias** from the evolution path to actively steer sampling toward unexplored regions, with strength scaled by stagnation severity.

```python
def _sample_population_batch(self):
    """
    Sample population with momentum-directed exploration bias.
    When stagnation is detected, inject directional bias from evolution path
    to escape local optima and explore new basins.
    """
    # Stagnation-adaptive momentum strength
    stagnation_ratio = min(self.stagnation_counter / max(1, 20 + self.dim), 1.0)
    momentum_strength = 0.5 + 2.0 * stagnation_ratio  # Range: 0.5 to 2.5
    
    # Compute momentum direction from evolution path (normalized)
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    if p_sigma_norm > 1e-20:
        momentum_dir = self.p_sigma / p_sigma_norm
    else:
        momentum_dir = np.zeros(self.dim)
    
    # Sample standard normals
    z = np.random.randn(self.pop_size, self.dim)
    
    # Inject momentum bias into half the population (exploration)
    n_biased = self.pop_size // 2
    for i in range(n_biased):
        # Blend random direction with momentum
        random_dir = np.random.randn(self.dim)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-20)
        biased_dir = (1.0 - momentum_strength * 0.3) * random_dir + momentum_strength * 0.3 * momentum_dir
        biased_dir = biased_dir / (np.linalg.norm(biased_dir) + 1e-20)
        z[i] = biased_dir * np.abs(np.random.randn()) + 0.5 * np.random.randn(self.dim)
    
    # Transform samples through covariance structure
    scaled = z * self.D[np.newaxis, :]
    rotated = scaled @ self.B.T
    population = self.mean[np.newaxis, :] + self.sigma * rotated
    
    # Add explicit mean shift for biased individuals (jump toward momentum direction)
    for i in range(n_biased):
        jump_size = self.sigma * (0.3 + 1.0 * stagnation_ratio)
        population[i] += jump_size * momentum_dir
    
    return population, z
```