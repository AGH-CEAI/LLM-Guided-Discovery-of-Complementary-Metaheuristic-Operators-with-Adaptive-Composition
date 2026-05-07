**Idea: Rank-Perc Adaptive Selection**
Replacement using fitness percentile ranks and adaptive thresholding based on population rank distribution. Category D: pure fitness-signal reasoning with no spatial/covariance computations.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Rank-based greedy selection with adaptive thresholding.
    
    Uses only fitness signals (no distances/covariance):
    - Population rank percentiles to assess convergence
    - Rank improvement as acceptance signal
    - Adaptive threshold based on population rank spread
    - Secondary fallback threshold for diversity maintenance
    """
    np_pop = len(population)
    
    # Compute population rank percentiles (0=best, 100=worst)
    pop_ranks = np.argsort(np.argsort(fitness)) / max(np_pop - 1, 1) * 100
    trial_ranks = np.zeros(np_pop)
    sorted_trial = np.argsort(trial_fitness)
    for i, idx in enumerate(sorted_trial):
        trial_ranks[idx] = i / max(np_pop - 1, 1) * 100
    
    # Rank improvement: negative means trial is better-ranked
    rank_improvement = trial_ranks - pop_ranks
    
    # Convergence signal from rank distribution spread
    rank_spread = np.ptp(pop_ranks)
    converged = rank_spread < 10.0
    
    # Adaptive threshold: tighten when converged, relax when scattered
    if converged:
        base_threshold = -2.0  # Require rank improvement
    else:
        base_threshold = -1.0  # More lenient when population is scattered
    
    # Primary acceptance: rank improvement exceeds threshold
    accept_mask = rank_improvement < base_threshold
    
    # If nothing accepted, use secondary threshold for diversity
    if not np.any(accept_mask):
        secondary_threshold = base_threshold * 1.5
        accept_mask = rank_improvement < secondary_threshold
    
    # Apply replacements
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    return new_population, new_fitness, accept_mask
```