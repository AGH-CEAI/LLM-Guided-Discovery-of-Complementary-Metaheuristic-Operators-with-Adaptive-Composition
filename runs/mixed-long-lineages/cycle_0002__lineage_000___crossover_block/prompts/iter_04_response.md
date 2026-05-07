**Idea: Success-Guided Adaptive Crossover**
Per-dimension success tracking with weighted crossover probabilities to target tasks where block crossover is too uniform.

```python
def _crossover_block(self, population, trial_population):
    """Op 3: Success-guided adaptive crossover."""
    n_trials = self.NP
    dim = self.dim
    
    # Initialize per-dimension success tracking
    if not hasattr(self, 'dim_improve_count'):
        self.dim_improve_count = np.zeros(dim)
        self.dim_attempt_count = np.zeros(dim)
        self.prev_trial_pop = np.zeros((n_trials, dim))
    
    # Track which dimensions led to improvements (comparing consecutive trials)
    if self.prev_trial_pop.shape[0] == n_trials:
        improved_mask = trial_population < self.prev_trial_pop
        self.dim_improve_count += np.sum(improved_mask, axis=0)
        self.dim_attempt_count += n_trials
    
    self.prev_trial_pop = trial_population.copy()
    
    # Compute per-dimension success rates (avoid division by zero)
    dim_success_rate = np.zeros(dim)
    valid_dims = self.dim_attempt_count > 10
    dim_success_rate[valid_dims] = (
        self.dim_improve_count[valid_dims] / self.dim_attempt_count[valid_dims]
    )
    
    # Adaptive CR per individual based on population diversity
    pop_std = np.std(population, axis=0)
    max_spread = self.upper - self.lower
    diversity_factor = np.clip(np.mean(pop_std) / (max_spread + 1e-10), 0.05, 0.95)
    
    # Base CR from diversity, modulated by dimension success
    base_cr = 0.3 + 0.4 * diversity_factor
    
    # Per-dimension CR: higher for historically successful dimensions
    dim_weights = np.clip(dim_success_rate, 0.1, 0.9)
    dim_weights_norm = dim_weights / (np.sum(dim_weights) + 1e-10)
    
    # Generate crossover masks
    cross_mask = np.zeros((n_trials, dim), dtype=bool)
    for i in range(n_trials):
        # Sample number of dimensions to cross from distribution
        n_cross = max(1, int(np.clip(
            base_cr * dim * (0.5 + 0.5 * self.rng.random()),
            1, dim
        )))
        
        # Weighted selection of dimensions (prefer historically successful ones)
        probs = dim_weights_norm.copy()
        selected_dims = set()
        attempts = 0
        while len(selected_dims) < n_cross and attempts < n_cross * 3:
            d = self.rng.choice(dim, p=probs)
            selected_dims.add(d)
            attempts += 1
        selected_dims = list(selected_dims)
        
        # Also ensure at least one random dimension gets crossed
        if len(selected_dims) == 0:
            selected_dims = [self.rng.integers(0, dim)]
        
        cross_mask[i, selected_dims] = True
    
    # Ensure at least one dimension crosses for each individual
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        if not cross_mask[i, :].any():
            cross_mask[i, j_rand[i]] = True
    
    # Apply crossover with weighted blending for successful dimensions
    blend_weights = 0.7 + 0.3 * dim_weights
    blended_trial = (
        blend_weights[np.newaxis, :] * trial_population +
        (1 - blend_weights[np.newaxis, :]) * population[:n_trials]
    )
    
    offspring = np.where(cross_mask, blended_trial, population[:n_trials])
    return offspring
```