**Idea: Success-Rate Guided Crossover**
Uses per-dimension historical success rates to probabilistically inherit trial vector components, with an adaptive mixing factor that shifts conservative when parents perform well.

```python
def _crossover_block(self, population, trial_population):
    """Op 3: Success-rate guided crossover with adaptive mixing."""
    n_trials = self.NP
    dim = self.dim
    
    # Initialize per-dimension success tracking if needed
    if not hasattr(self, 'dim_cx_success') or self.dim_cx_success is None or len(self.dim_cx_success) != dim:
        self.dim_cx_success = np.zeros(dim)
        self.dim_cx_total = np.zeros(dim) + 1e-10
    
    # Compute per-dimension success rates for crossover
    success_rates = self.dim_cx_success / (self.dim_cx_total + 1e-10)
    
    # Global diversity measure
    pop_std = np.std(population[:n_trials], axis=0)
    max_spread = self.upper - self.lower
    diversity = np.mean(pop_std / (max_spread + 1e-10))
    
    # Adaptive mixing factor: conservative when diverse, aggressive when clustered
    mix_base = np.clip(0.3 + 0.2 * diversity, 0.15, 0.5)
    
    # Per-individual adaptive mixing based on relative fitness
    fitness = self.population_fitness[:n_trials] if hasattr(self, 'population_fitness') else np.zeros(n_trials)
    best_fit = np.min(fitness) if len(fitness) > 0 and np.any(np.isfinite(fitness)) else 0.0
    worst_fit = np.max(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else 1.0
    fit_range = worst_fit - best_fit + 1e-10
    
    # Individuals worse than median get higher mixing (more exploration)
    median_fit = np.median(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else best_fit
    mix_factor = np.clip(
        mix_base + 0.2 * np.maximum(0, (median_fit - fitness) / fit_range),
        0.1, 0.6
    )
    
    # Create blended population (weighted average)
    mix_expanded = mix_factor[:, np.newaxis]
    blended = (
        mix_expanded * trial_population +
        (1.0 - mix_expanded) * population[:n_trials]
    )
    
    # Per-dimension inheritance probability based on historical success
    inherit_prob = np.clip(success_rates, 0.2, 0.8)
    
    # Generate crossover mask using success-rate guided probabilities
    cross_mask = self.rng.uniform(size=(n_trials, dim)) < inherit_prob[np.newaxis, :]
    
    # Ensure at least one dimension from trial is taken
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        if not cross_mask[i, :].any():
            cross_mask[i, j_rand[i]] = True
    
    # Create offspring from blended population
    offspring = np.where(cross_mask, blended, population[:n_trials])
    
    # Update success tracking based on trial performance
    trial_better = (trial_population < population[:n_trials]).astype(float)
    dim_improved = np.mean(trial_better, axis=0)
    self.dim_cx_total += 1
    alpha = 0.1
    self.dim_cx_success = alpha * dim_improved * n_trials + (1 - alpha) * self.dim_cx_success
    
    return offspring
```