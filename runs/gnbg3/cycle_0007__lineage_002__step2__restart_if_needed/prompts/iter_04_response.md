**Idea: Fitness-Rank Correlation and Percentile Spread Restart**

This approach uses fitness-rank dynamics across generations: computing Spearman rank correlation to detect evolutionary stagnation, and analyzing fitness percentile spread to detect premature convergence. When either signal crosses thresholds, the population is reinitialized with rank-informed diversity injection.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on fitness-rank dynamics: Spearman correlation and percentile spread."""
    from scipy.stats import spearmanr
    
    # Compute current fitness ranks (normalized 0-1)
    current_ranks = self._compute_fitness_ranking(fitness)
    
    # Initialize rank history if needed
    if not hasattr(self, '_rank_history'):
        self._rank_history = []
        self._fitness_percentiles_history = []
    
    # Store current state
    self._rank_history.append(current_ranks.copy())
    self._fitness_percentiles_history.append(np.percentile(fitness, [10, 25, 50, 75, 90]))
    
    # Keep bounded history
    if len(self._rank_history) > 8:
        self._rank_history.pop(0)
    if len(self._fitness_percentiles_history) > 8:
        self._fitness_percentiles_history.pop(0)
    
    restart_triggered = False
    restart_reason = ""
    
    # Signal 1: Spearman rank correlation between consecutive generations
    if len(self._rank_history) >= 2:
        prev_ranks = self._rank_history[-2]
        corr, _ = spearmanr(prev_ranks, current_ranks)
        if not np.isnan(corr) and corr > 0.88:
            restart_triggered = True
            restart_reason = f"rank_stagnation_corr={corr:.3f}"
    
    # Signal 2: Fitness percentile spread (detects premature convergence)
    if len(self._fitness_percentiles_history) >= 3:
        recent_percentiles = np.array(self._fitness_percentiles_history[-3:])
        # Compute spread of spread: variance of IQR across recent generations
        p75 = recent_percentiles[:, 3]
        p25 = recent_percentiles[:, 1]
        iqr = p75 - p25
        iqr_shrink_rate = (iqr[0] - iqr[-1]) / (max(iqr[0], 1e-10))
        if iqr_shrink_rate > 0.6 and iqr[-1] < 0.1 * (np.max(fitness) - np.min(fitness) + 1e-10):
            restart_triggered = True
            restart_reason = f"percentile_shrink_rate={iqr_shrink_rate:.3f}"
    
    # Signal 3: Top-rank individuals dominating (low rank entropy)
    rank_entropy = -np.sum(current_ranks * np.log(current_ranks + 1e-10)) / np.log(self.NP)
    if rank_entropy < 0.15:
        restart_triggered = True
        restart_reason = f"low_rank_entropy={rank_entropy:.3f}"
    
    if not restart_triggered:
        return (None, None, None)
    
    # Restart: preserve best, reinitialize others with rank-informed perturbation
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Compute fitness-based weights for mutation intensity
    # Worst individuals get more perturbation
    rank_weights = 1.0 - current_ranks  # Higher weight for worse ranks
    rank_weights /= rank_weights.sum() + 1e-10
    
    new_population = np.empty_like(population)
    new_population[0] = best_individual
    
    # Use success-history weighted sampling for donor selection
    if hasattr(self, '_ema_success_rate'):
        success_weights = self._ema_success_rate + 0.1
        success_weights /= success_weights.sum()
        # Select donors based on historical success
        n_donors = min(4, self.NP - 1)
        donor_indices = np.random.choice(self.NP, size=n_donors, p=success_weights, replace=False)
        donor_pool = population[donor_indices]
    else:
        donor_pool = population
    
    for i in range(1, self.NP):
        # Rank-informed mutation: worse individuals get more diverse donors
        mutation_scale = 20.0 + 30.0 * rank_weights[i]
        
        if len(donor_pool) > 0:
            donor_idx = np.random.randint(len(donor_pool))
            donor = donor_pool[donor_idx]
        else:
            donor = np.random.uniform(-50, 50, self.dim)
        
        noise = np.random.randn(self.dim) * mutation_scale
        new_population[i] = np.clip(best_individual + noise, -100.0, 100.0)
    
    # Reset stagnation counter
    self.stagnation_counter = 0
    
    return (new_population, best_fitness, best_individual)
```