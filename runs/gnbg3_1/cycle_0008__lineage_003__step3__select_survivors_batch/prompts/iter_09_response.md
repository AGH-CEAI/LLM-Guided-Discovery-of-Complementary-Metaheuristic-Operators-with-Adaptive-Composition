Looking at the benchmark data, the worst tasks (16, 17, 20, 12, 23, 18, 19, 10) have errors stuck at ~1e+01 to ~1e+02, suggesting the algorithm is trapped in poor local optima on highly multimodal functions. All 7 previous variants focused on covariance adaptation, but the survivor selection remains pure elitist (μ, λ)-selection that always keeps the fittest NP individuals, which promotes premature convergence on multimodal landscapes.

**Idea: Diversity-Preserving Niching Selection**
Replace pure elitist selection with fitness-sharing niching that explicitly maintains multiple subpopulations in different basins of attraction. This forces exploration of alternative optima instead of collapsing to a single region.

```python
def _select_survivors_batch(self):
    """Select survivors using fitness-sharing niching to preserve diversity."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    n_combined = len(combined_fit)
    sigma_share = 0.1 * (self.ub[0] - self.lb[0]) / self.NP
    
    niche_counts = np.zeros(n_combined)
    for i in range(n_combined):
        distances = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
        niche_counts[i] = np.sum(distances < sigma_share) + 1e-10
    
    shared_fitness = combined_fit * niche_counts
    
    sorted_indices = np.argsort(shared_fitness)
    self.population = combined_pop[sorted_indices[:self.NP]]
    self.fitness = combined_fit[sorted_indices[:self.NP]]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```