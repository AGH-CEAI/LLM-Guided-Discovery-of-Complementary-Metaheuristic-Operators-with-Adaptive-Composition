**Idea: Restart-Triggered Aggressive Exploration Sampling**

The worst tasks (16, 17) have errors ~100× and ~50× larger than the easiest tasks. This suggests the current multivariate normal sampling around the mean is getting trapped in local optima. The key insight: when the algorithm is stagnating or has low diversity, the sampling should become radically more exploratory — not just increase sigma, but fundamentally reshape the sampling distribution to escape the current basin.

```python
def _sample_trials_batch(self):
    """Sample with restart-triggered aggressive exploration for trapped populations."""
    # Check if we're stuck or losing diversity
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = pop_variance / (expected_var + 1e-10)
    
    is_stagnated = self.stagnation_counter > self.max_stagnation // 2
    is_low_diversity = diversity_ratio < 0.05
    should_explore_aggressively = is_stagnated or is_low_diversity
    
    if should_explore_aggressively:
        # Split: 60% far-uniform, 40% anisotropic around best
        n_far = int(0.6 * self.NP)
        n_local = self.NP - n_far
        
        # Far-uniform: sample from the entire bounds hypercube
        far_samples = np.random.uniform(self.lb, self.ub, (n_far, self.dim))
        
        # Local: anisotropic sampling around best solution with large sigma
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Use 5x sigma for local exploration to escape basin
        local_sigma = self.sigma * 5.0
        z_local = np.random.randn(n_local, self.dim)
        local_samples = self.x_opt + local_sigma * (z_local @ L.T)
        local_samples = self._clip_to_bounds(local_samples)
        
        # Combine both pools
        self.trials = np.vstack([far_samples, local_samples])
        
        # Perturb mean to break symmetry
        mean_perturb = np.random.randn(self.dim) * 0.3 * np.sqrt(pop_variance + 1e-10)
        self.mean = np.clip(self.mean + mean_perturb, self.lb, self.ub)
    else:
        # Standard CMA-ES sampling with adaptive covariance
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
```