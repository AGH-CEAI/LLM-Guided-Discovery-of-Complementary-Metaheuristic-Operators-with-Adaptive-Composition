Looking at the priority targets with errors in the 1e+01 to 1e+02 range, the algorithm is severely failing to find the global optimum. The current Cholesky-based sampling is a standard CMA-ES approach that may not provide enough exploration when stuck at such high errors. I need a fundamentally different sampling strategy.

**Idea: Antithetic Pair Sampling with Covariance Perturbation**

This approach uses correlated antithetic pairs (sampling z and -z) to reduce variance AND adds periodic random perturbations to the covariance matrix to escape local optima basins that standard CMA-ES cannot break out of.

```python
def _sample_trials_batch(self):
    """Sample using antithetic pairs with covariance perturbation for escape."""
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-8:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
    # Periodic covariance perturbation to escape local optima
    if self.generation > 0 and self.generation % 20 == 0:
        perturb_scale = 0.1 * self.sigma
        random_perturb = np.random.randn(self.dim, self.dim)
        random_perturb = 0.5 * (random_perturb + random_perturb.T)
        self.C = self.C + perturb_scale * random_perturb
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(self.C)
    except np.linalg.LinAlgError:
        diag_C = np.diag(self.C)
        diag_C = np.maximum(diag_C, 1e-10)
        L = np.diag(np.sqrt(diag_C))
    
    # Generate base samples
    z = np.random.randn(self.NP // 2, self.dim)
    
    # Create antithetic pairs: (z, -z) for variance reduction
    z_pairs = np.vstack([z, -z])
    
    # Scale sigma adaptively for exploration
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Boost exploration when diversity is low
    sigma_scale = 1.0 + 2.0 * (1.0 - diversity)
    sigma_scale = np.clip(sigma_scale, 1.0, 5.0)
    
    # Transform and create trials
    self.trials = self.mean + self.sigma * sigma_scale * (z_pairs @ L.T)
    
    # Handle odd NP
    if self.trials.shape[0] < self.NP:
        extra = np.random.randn(1, self.dim)
        extra_trial = self.mean + self.sigma * sigma_scale * (extra @ L.T)
        self.trials = np.vstack([self.trials, extra_trial])
    
    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)
```