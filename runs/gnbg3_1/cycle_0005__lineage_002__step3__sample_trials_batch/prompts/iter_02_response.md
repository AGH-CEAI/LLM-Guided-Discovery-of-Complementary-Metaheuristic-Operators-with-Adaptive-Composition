**Idea: Elite-Focused Anisotropic Sampling with Adaptive Radius**

Instead of sampling around the global mean (which may be trapped in a local optimum on hard tasks), this variant explicitly samples around the top-k elite solutions using their anisotropic covariance structure. It also scales the exploration radius by the population diversity ratio to escape local optima on the worst tasks (16, 17, 18, etc.).

```python
def _sample_trials_batch(self):
    """Sample using elite-focused anisotropic perturbations with adaptive exploration radius."""
    # Identify top-k elite solutions
    sorted_indices = np.argsort(self.fitness)
    k = max(3, self.NP // 10)
    elite_indices = sorted_indices[:k]
    elite_pop = self.population[elite_indices]
    
    # Compute elite covariance (captures local landscape structure)
    elite_mean = np.mean(elite_pop, axis=0)
    elite_cov = np.cov(elite_pop.T) + 1e-8 * np.eye(self.dim)
    
    # Ensure positive definiteness for Cholesky
    min_eig = np.min(np.linalg.eigvalsh(elite_cov))
    if min_eig < 1e-8:
        elite_cov += (1e-7 - min_eig) * np.eye(self.dim)
    
    try:
        L_elite = np.linalg.cholesky(elite_cov)
    except np.linalg.LinAlgError:
        diag_elite = np.diag(elite_cov)
        diag_elite = np.maximum(diag_elite, 1e-10)
        L_elite = np.diag(np.sqrt(diag_elite))
    
    # Compute population diversity for adaptive radius
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Adaptive exploration: increase radius when diversity is low
    exploration_boost = 1.0 + 2.0 * (1.0 - diversity_ratio)
    exploration_boost = np.clip(exploration_boost, 1.0, 5.0)
    
    # Mix between global mean and elite mean
    mix_weight = 0.7 * diversity_ratio + 0.3
    adaptive_mean = mix_weight * self.mean + (1.0 - mix_weight) * elite_mean
    
    # Sample trials: 60% from elite-focused, 40% from global mean
    n_elite_trials = int(0.6 * self.NP)
    n_global_trials = self.NP - n_elite_trials
    
    # Elite-focused sampling with anisotropic perturbations
    z_elite = np.random.randn(n_elite_trials, self.dim)
    trials_elite = elite_mean + self.sigma * exploration_boost * (z_elite @ L_elite.T)
    
    # Global mean sampling (baseline CMA-ES style)
    try:
        L_global = np.linalg.cholesky(self.C)
    except np.linalg.LinAlgError:
        diag_global = np.diag(self.C)
        diag_global = np.maximum(diag_global, 1e-10)
        L_global = np.diag(np.sqrt(diag_global))
    
    z_global = np.random.randn(n_global_trials, self.dim)
    trials_global = self.mean + self.sigma * (z_global @ L_global.T)
    
    # Combine and clip to bounds
    self.trials = np.vstack([trials_elite, trials_global])
    self.trials = self._clip_to_bounds(self.trials)
```