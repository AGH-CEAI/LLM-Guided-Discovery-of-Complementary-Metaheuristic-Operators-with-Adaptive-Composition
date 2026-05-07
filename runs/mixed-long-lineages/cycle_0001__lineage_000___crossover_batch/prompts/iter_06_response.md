**Idea: Archive-Driven Adaptive Crossover with Temperature Scaling**

Uses mutation vector differences to compute per-dimension crossover probabilities, with temperature annealing based on recent improvement rate. Targets tasks requiring fine-grained exploration control (8, 12, 15, 18, 20, 21, 22).

```python
def _crossover_batch(self, population, trial_population):
    n_trials = self.NP
    dim = self.dim
    cr_batch = self.CR.copy()
    
    # Compute mutation vector differences (magnitude per individual)
    diff_mag = np.linalg.norm(trial_population - population, axis=1, keepdims=True)
    diff_mag = np.maximum(diff_mag, 1e-10)
    
    # Normalize per-dimension differences -> exploration probability
    diff_norm = np.abs(trial_population - population) / diff_mag
    
    # Temperature inversely correlated with recent improvement rate
    if hasattr(self, 'last_improvement_rate'):
        temp = max(0.05, 2.0 * (1.0 - self.last_improvement_rate))
    else:
        temp = 1.0
    
    # Softmax over dimensions for each trial vector
    exp_diff = np.exp(diff_norm / (temp + 1e-10))
    denom = np.sum(exp_diff, axis=1, keepdims=True) + 1e-10
    explore_prob = exp_diff / denom
    
    # Combine base CR with exploration probability
    cr_expanded = cr_batch[:, np.newaxis]
    cross_prob = cr_expanded * explore_prob + (1 - cr_expanded) * (1.0 / dim)
    cross_prob = np.clip(cross_prob, 1e-6, 1.0)
    
    # Generate crossover mask
    cross_mask = self.rng.uniform(size=(n_trials, dim)) < cross_prob
    
    # Ensure at least one dimension from trial
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        cross_mask[i, j_rand[i]] = True
    
    offspring = np.where(cross_mask, trial_population, population)
    return offspring
```