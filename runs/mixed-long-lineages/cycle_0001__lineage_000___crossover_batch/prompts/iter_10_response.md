**Idea: Dimension-Learned Success-Biased Crossover**

Uses per-dimension historical improvement rates to probabilistically bias which variables are inherited from trial vectors. Instead of jADE's global CR adaptation (which fails on Task 8), this learns which dimensions tend to produce improvements and preferentially selects those. This directly addresses Task 8's failure by decoupling crossover decisions from the jADE adaptation that hurts it.

```python
def _crossover_batch(self, population, trial_population):
    n_trials = self.NP
    dim = self.dim
    
    # Initialize per-dimension tracking on first call
    if not hasattr(self, 'dim_success_count'):
        self.dim_success_count = np.zeros(dim)
        self.dim_attempt_count = np.zeros(dim)
    
    # Compute trial vs parent improvement for each dimension
    trial_improved = trial_population < population
    self.dim_attempt_count += 1
    self.dim_success_count += np.any(trial_improved, axis=0)
    
    # Compute per-dimension success probability
    eps = 1.0 / max(1, n_trials)
    dim_prob = (self.dim_success_count + eps) / (self.dim_attempt_count + eps)
    dim_prob = np.clip(dim_prob, 0.01, 0.99)
    
    # Normalize to probability distribution
    dim_prob = dim_prob / (np.sum(dim_prob) + 1e-30)
    
    # Build crossover mask using dimension-learned probabilities
    cross_mask = self.rng.uniform(size=(n_trials, dim)) < dim_prob[np.newaxis, :]
    
    # Ensure at least one dimension from trial (preserves DE property)
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        cross_mask[i, j_rand[i]] = True
    
    offspring = np.where(cross_mask, trial_population, population)
    return offspring
```