**Idea: Parent-Preserving Confidence Crossover**

A fundamentally different strategy that tracks per-dimension improvement history and selectively preserves high-performing parent dimensions while only applying crossover to dimensions that consistently fail to improve. This protects good building blocks on deceptive/non-separable landscapes.

```python
def _crossover_block(self, population, trial_population):
    """Op 6: Confidence-weighted parent preservation crossover."""
    n_trials = self.NP
    dim = self.dim
    
    # Initialize per-dimension tracking if needed
    if not hasattr(self, '_dim_success_count') or self._dim_success_count is None or len(self._dim_success_count) != dim:
        self._dim_success_count = np.zeros(dim)
        self._dim_attempt_count = np.zeros(dim)
        self._dim_history_fitness = np.full(dim, np.inf)
    
    # Track which dimensions improved in trial vectors (global, not per-individual)
    trial_better = (trial_population < population[:n_trials]).astype(float)
    dim_signal = np.mean(trial_better, axis=0)
    
    # Update per-dimension statistics
    self._dim_attempt_count += 1
    update_mask = self._dim_attempt_count > 0
    alpha = 0.1
    self._dim_success_count[update_mask] = (
        alpha * dim_signal[update_mask] * float(n_trials) +
        (1 - alpha) * self._dim_success_count[update_mask]
    )
    
    # Compute confidence: higher success rate = higher confidence to preserve parent
    success_rate = np.zeros(dim)
    valid = self._dim_attempt_count > 5
    success_rate[valid] = self._dim_success_count[valid] / self._dim_attempt_count[valid]
    
    # Normalize confidence to probability of preserving parent
    conf_preserve = np.clip(success_rate, 0.05, 0.95)
    
    # For each trial, decide per-dimension whether to keep parent or take trial
    cross_mask = np.zeros((n_trials, dim), dtype=bool)
    for i in range(n_trials):
        # Dimensions with high confidence: preserve parent
        # Dimensions with low confidence: take trial vector
        preserve_prob = conf_preserve + 0.2 * (self.CR[i] - 0.5)
        preserve_prob = np.clip(preserve_prob, 0.0, 1.0)
        
        rand_vals = self.rng.uniform(size=dim)
        cross_mask[i] = rand_vals >= preserve_prob
    
    # Ensure at least one dimension takes from trial (diversity guarantee)
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        if not cross_mask[i, :].any():
            cross_mask[i, j_rand[i]] = True
    
    # Create offspring
    offspring = np.where(cross_mask, trial_population, population[:n_trials])
    return offspring
```