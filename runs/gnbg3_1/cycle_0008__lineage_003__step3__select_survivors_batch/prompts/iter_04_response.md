Looking at the benchmark results, the worst tasks (16, 17, 20, 23, 10, 12, 18, 19) are all stuck at errors around 10-100, suggesting the algorithm is getting trapped in local optima or losing diversity too quickly. The current survivor selection uses pure elitist (μ, λ)-selection which can lead to premature convergence.

**Idea: Diversity-Aware Survivor Selection with Adaptive Pressure**
This approach explicitly balances fitness-based selection with diversity preservation by using a weighted combination of normalized fitness rank and population spread. When diversity is low (convergence detected), the selection pressure toward diversity increases to escape local optima. This is fundamentally different from pure elitism.

```python
def _select_survivors_batch(self):
    """Select survivors with adaptive fitness-diversity balancing."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    n_total = len(combined_fit)
    if n_total == 0 or self.NP <= 0:
        return
    
    fitness_rank = np.argsort(np.argsort(combined_fit)) / max(n_total - 1, 1)
    fitness_rank = np.clip(fitness_rank, 0.0, 1.0)
    
    pop_std = np.std(combined_pop, axis=0) + 1e-10
    expected_std = (self.ub[0] - self.lb[0]) / 6.0
    norm_std = np.clip(pop_std / expected_std, 0.0, 1.0)
    diversity_score = np.mean(norm_std)
    
    diversity_weight = 0.1 + 0.9 * min(diversity_score * 2.0, 1.0)
    
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
    improvement = max(0.0, self.prev_f_opt - np.min(combined_fit))
    if improvement < 1e-12:
        diversity_weight = min(diversity_weight * 1.5, 0.95)
    self.prev_f_opt = np.min(combined_fit)
    
    combined_diversity = np.zeros(n_total)
    for i in range(n_total):
        dists = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
        combined_diversity[i] = np.mean(dists)
    
    max_dist = max(np.max(combined_diversity), 1e-10)
    combined_diversity = combined_diversity / max_dist
    
    fitness_component = (1.0 - diversity_weight) * (1.0 - fitness_rank)
    diversity_component = diversity_weight * combined_diversity
    combined_score = fitness_component + diversity_component
    
    sorted_indices = np.argsort(-combined_score)
    self.population = combined_pop[sorted_indices[:self.NP]]
    self.fitness = combined_fit[sorted_indices[:self.NP]]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```