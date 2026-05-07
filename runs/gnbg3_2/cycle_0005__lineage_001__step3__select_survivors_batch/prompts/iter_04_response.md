**Idea: Adaptive Rank-Greedy Hybrid with Spearman Landscape Detection**

Using fitness-rank correlation and temporal success-rate EMAs to detect when greedy selection causes premature convergence on rugged/deceptive landscapes, then switching to rank-based selection that preserves diversity.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """
    Adaptive selection: greedy when improving, rank-based when stagnant.
    Uses Spearman correlation and temporal EMA for regime detection.
    Category D: Fitness-landscape / rank-based reasoning only.
    """
    # Compute raw improvement rate
    improved_mask_greedy = trial_fitness < fitness
    improvement_rate = np.mean(improved_mask_greedy)
    
    # Initialize EMA state for temporal regime detection
    if not hasattr(self, 'sel_ewma_short'):
        self.sel_ewma_short = improvement_rate
        self.sel_ewma_long = improvement_rate
    
    # Update short/long EMAs with different smoothing (temporal dynamics)
    self.sel_ewma_short = 0.3 * improvement_rate + 0.7 * self.sel_ewma_short
    self.sel_ewma_long = 0.1 * improvement_rate + 0.9 * self.sel_ewma_long
    
    # Compute Spearman rank correlation between parent fitness and trial fitness
    # Low correlation indicates rugged/deceptive landscape where greedy fails
    n = len(fitness)
    if n > 3:
        ranks_f = np.argsort(np.argsort(fitness)) + 1
        ranks_tf = np.argsort(np.argsort(trial_fitness)) + 1
        # Handle edge case where all values identical
        std_f = np.std(fitness)
        std_tf = np.std(trial_fitness)
        if std_f > 1e-12 and std_tf > 1e-12:
            cov = np.mean((fitness - np.mean(fitness)) * (trial_fitness - np.mean(trial_fitness)))
            spearman_proxy = cov / (std_f * std_tf + 1e-12)
        else:
            spearman_proxy = 1.0
    else:
        spearman_proxy = 1.0
    
    # Regime detection using fitness-landscape signals
    short_trending_up = self.sel_ewma_short > self.sel_ewma_long
    both_stagnant = self.sel_ewma_short < 0.08 and self.sel_ewma_long < 0.12
    low_correlation = spearman_proxy < 0.25  # Rugged landscape signal
    
    # Switch to rank-based selection when: declining success OR stagnant OR rugged
    use_rank_selection = (not short_trending_up) or both_stagnant or low_correlation
    
    if use_rank_selection:
        # Rank-based survival: combine population + trials, select best NP by rank
        # This maintains diversity by potentially keeping inferior-but-different parents
        combined_pop = np.vstack([population, trials])
        combined_fitness = np.concatenate([fitness, trial_fitness])
        
        n_pop = len(population)
        
        # Compute ranks: 1 = best (lowest fitness)
        rank_order = np.argsort(combined_fitness)
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(len(rank_order)) + 1
        
        # Select top NP individuals
        selected_indices = np.argsort(ranks)[:n_pop]
        
        new_population = combined_pop[selected_indices]
        new_fitness = combined_fitness[selected_indices]
        improved_mask = selected_indices >= n_pop  # True if selected individual was from trials
        
        return new_population, new_fitness, improved_mask
    else:
        # Standard greedy selection when population is making consistent progress
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        improved_mask = trial_fitness < fitness
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
```