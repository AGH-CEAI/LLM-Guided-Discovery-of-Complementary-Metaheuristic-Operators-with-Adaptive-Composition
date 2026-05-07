**Idea: Diversity-Penalized Fitness Selection**
Replace pure elitist sorting with a composite score that combines fitness rank + spatial diversity, using exponential ranking and stochastic sampling to preserve genetically distinct individuals alongside top performers.

```python
def _select_survivors_batch(self):
    """Select survivors via diversity-penalized fitness ranking."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    n_combined = len(combined_fit)
    if n_combined <= self.NP:
        self.population = combined_pop[:self.NP]
        self.fitness = combined_fit[:self.NP]
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
        return
    
    # Exponential ranking of raw fitness
    sorted_fit_indices = np.argsort(combined_fit)
    ranks = np.zeros(n_combined)
    ranks[sorted_fit_indices] = np.arange(1, n_combined + 1)
    exp_fitness = np.exp(-0.01 * ranks)
    
    # Diversity factor: inverse distance to nearest neighbor in population
    diversity = np.ones(n_combined)
    for i in range(n_combined):
        dists = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
        dists[i] = np.inf
        if np.isfinite(dists).any():
            nn_dist = np.min(dists)
            pop_range = np.linalg.norm(self.ub - self.lb)
            diversity[i] = 1.0 + 5.0 * (nn_dist / (pop_range + 1e-10))
    
    diversity = np.clip(diversity, 1.0, 10.0)
    
    # Composite score: balance fitness (exponential rank) with diversity
    composite = exp_fitness * diversity
    
    # Stochastic selection weighted by composite score
    probs = composite / np.sum(composite)
    probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    probs = probs / np.sum(probs)
    
    selected_indices = np.random.choice(n_combined, size=self.NP, replace=False, p=probs)
    
    self.population = combined_pop[selected_indices]
    self.fitness = combined_fit[selected_indices]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```