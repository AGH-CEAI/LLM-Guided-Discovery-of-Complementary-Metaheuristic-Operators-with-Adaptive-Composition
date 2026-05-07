**Idea: Hybrid Binomial-Exponential Crossover**
Uses a random blend of binomial and exponential crossover operators, where exponential crossover makes smaller, localized changes that preserve more parent structure—potentially better for tasks where aggressive crossover degrades performance.
```python
def _crossover_batch(self, population, trial_population):
    cr_batch = self.CR.copy()
    n_trials = self.NP
    dim = self.dim
    
    cross_mask = np.zeros((n_trials, dim), dtype=bool)
    
    for i in range(n_trials):
        if self.rng.random() < 0.5:
            # Binomial crossover (standard)
            mask = self.rng.uniform(size=dim) < cr_batch[i]
            j_rand = self.rng.integers(0, dim)
            mask[j_rand] = True
        else:
            # Exponential crossover: copy a contiguous segment from trial
            j_start = self.rng.integers(0, dim)
            mask = np.zeros(dim, dtype=bool)
            L = 0
            while L < dim and self.rng.random() < cr_batch[i]:
                j = (j_start + L) % dim
                mask[j] = True
                L += 1
            if not np.any(mask):
                mask[j_start] = True
        
        cross_mask[i] = mask
    
    offspring = np.where(cross_mask, trial_population, population)
    return offspring
```