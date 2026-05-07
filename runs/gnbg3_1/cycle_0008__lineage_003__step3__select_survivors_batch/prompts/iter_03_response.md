**Idea: Crowding-Distance Survivor Selection**
Replace pure elitist selection with a multi-objective approach that balances fitness with spatial crowding-distance diversity, using adaptive weighting to maintain exploration on deceptive landscapes.

```python
def _select_survivors_batch(self):
    """Select survivors balancing fitness with crowding-distance diversity."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    n_candidates = len(combined_fit)
    n_select = self.NP
    
    # Compute normalized fitness rank (higher = better)
    sorted_fit_indices = np.argsort(combined_fit)
    fit_rank = np.zeros(n_candidates)
    fit_rank[sorted_fit_indices] = np.linspace(1.0, 0.0, n_candidates)
    
    # Compute crowding distance for diversity
    crowding = np.zeros(n_candidates)
    if n_candidates > 2:
        pop_norm = np.zeros_like(combined_pop)
        for d in range(self.dim):
            d_min, d_max = combined_pop[:, d].min(), combined_pop[:, d].max()
            rng = d_max - d_min + 1e-10
            pop_norm[:, d] = (combined_pop[:, d] - d_min) / rng
        
        for i in range(n_candidates):
            dists = np.linalg.norm(pop_norm - pop_norm[i], axis=1)
            nearest_indices = np.argsort(dists)[1:min(10, n_candidates)]
            if len(nearest_indices) > 0:
                crowding[i] = np.mean(dists[nearest_indices])
        
        max_crowd = crowding.max() + 1e-10
        crowding /= max_crowd
    
    # Adaptive diversity weight: more diversity pressure when converged
    pop_variance = np.mean(np.var(combined_pop, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    diversity_weight = 0.3 + 0.5 * (1.0 - diversity_ratio)
    
    # Combined score
    scores = (1.0 - diversity_weight) * fit_rank + diversity_weight * crowding
    
    # Select top-N by combined score
    selected_indices = np.argsort(scores)[-n_select:]
    
    self.population = combined_pop[selected_indices]
    self.fitness = combined_fit[selected_indices]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```