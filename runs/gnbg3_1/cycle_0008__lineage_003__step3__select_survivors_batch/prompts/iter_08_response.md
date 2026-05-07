**Idea: Fitness-Diversity Biased Selection**
A multi-objective survivor selection that explicitly balances raw fitness against a diversity contribution score (mean distance to other candidates), ensuring the population doesn't collapse to a single basin and can escape local optima on the hardest multimodal tasks.

```python
def _select_survivors_batch(self):
    """Select survivors balancing raw fitness with diversity contribution."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    # Ensure valid fitness values
    valid_mask = np.isfinite(combined_fit)
    combined_pop = combined_pop[valid_mask]
    combined_fit = combined_fit[valid_mask]
    
    n = len(combined_fit)
    if n == 0:
        return
    
    # Normalize fitness to [0, 1] range for fair combination with diversity
    fit_min, fit_max = combined_fit.min(), combined_fit.max()
    if fit_max > fit_min + 1e-15:
        fit_norm = (combined_fit - fit_min) / (fit_max - fit_min)
    else:
        fit_norm = np.zeros(n)
    
    # Compute diversity contribution: mean distance to all other individuals
    # Higher = more isolated = more valuable for exploration
    pop_center = np.mean(combined_pop, axis=0)
    dist_to_center = np.linalg.norm(combined_pop - pop_center, axis=1)
    global_scale = np.mean(dist_to_center) + 1e-10
    
    diversity_score = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
        # Exclude self (distance 0)
        other_dists = dists[np.arange(n) != i]
        if len(other_dists) > 0:
            diversity_score[i] = np.mean(other_dists) / global_scale
    
    # Adaptive weighting: more diversity pressure when best error is large
    # This helps escape local optima on the hardest tasks
    alpha = 0.3  # diversity weight
    combined_score = (1 - alpha) * fit_norm + alpha * (1 - diversity_score)
    
    # Greedy selection of NP survivors
    sorted_indices = np.argsort(combined_score)
    survivor_indices = sorted_indices[:self.NP]
    
    self.population = combined_pop[survivor_indices]
    self.fitness = combined_fit[survivor_indices]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```