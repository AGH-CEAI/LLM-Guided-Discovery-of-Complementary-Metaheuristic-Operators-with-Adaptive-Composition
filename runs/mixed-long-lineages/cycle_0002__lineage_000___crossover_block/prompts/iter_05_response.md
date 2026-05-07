**Idea: Dimension-Success Guided Crossover**

A crossover operator that learns which dimensions lead to improvements and preferentially preserves those dimensions, targeting tasks where mutation finds good directions but crossover destroys them.
```python
def _crossover_block(self, population, trial_population):
    """Op 5: Dimension-success guided crossover (adaptive per-dim preservation)."""
    n_trials = self.NP
    dim = self.dim
    
    # Initialize tracking arrays (lazy initialization)
    if not hasattr(self, 'dim_success_count') or self.dim_success_count is None or len(self.dim_success_count) != dim:
        self.dim_success_count = np.zeros(dim)
        self.dim_attempt_count = np.zeros(dim)
    
    # Compute per-dimension improvement signal: is trial better than parent at each dim?
    trial_better = np.abs(trial_population - population[:n_trials]) > 1e-10
    trial_improved_fitness = trial_population < population[:n_trials]
    
    # Count dimension-level improvements only when overall fitness improved
    for i in range(n_trials):
        if trial_improved_fitness[i]:
            self.dim_success_count += trial_better[i]
    
    self.dim_attempt_count += 1
    
    # Compute probability of preserving each dimension (based on historical success)
    dim_success_rate = self.dim_success_count / (self.dim_attempt_count + 1e-10)
    
    # Bias toward preserving dimensions that historically lead to improvements
    # Use sqrt to amplify signal while keeping probabilities reasonable
    dim_probs = np.clip(dim_success_rate ** 0.5, 0.05, 0.95)
    
    # Normalize to get per-dimension crossover probabilities
    dim_probs = dim_probs / (np.sum(dim_probs) + 1e-10)
    
    # Generate crossover mask based on learned dimension probabilities
    cross_mask = self.rng.uniform(size=(n_trials, dim)) < dim_probs[np.newaxis, :]
    
    # Ensure at least one dimension is selected (random fallback)
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        if not cross_mask[i, :].any():
            cross_mask[i, j_rand[i]] = True
    
    # Create offspring: preserve parent where mask is False, take trial where True
    offspring = np.where(cross_mask, trial_population, population[:n_trials])
    return offspring
```