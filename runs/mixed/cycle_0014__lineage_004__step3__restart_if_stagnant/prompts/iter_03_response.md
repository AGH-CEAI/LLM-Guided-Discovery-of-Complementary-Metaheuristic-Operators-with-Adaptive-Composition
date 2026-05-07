**Idea: Multi-Strategy Diversity Restart with Opposition and Latin Hypercube**

Instead of small perturbations around x_best, this restart uses three complementary strategies: (1) random restart in the full bounds, (2) opposition-based sampling (reflecting solutions through the search space center), and (3) Latin Hypercube sampling for space-filling coverage. This targets the worst tasks by aggressively escaping any local optimum basin that x_best may be trapped in.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
    NP = len(population)
    new_pop = population.copy()
    dim = self.dim
    lower, upper = self.lower, self.upper
    
    # Strategy 1: Random uniform restart (1/3 of replacements)
    n_random = NP // 3
    random_indices = np.random.choice(NP, n_random, replace=False)
    for idx in random_indices:
        new_pop[idx] = np.random.uniform(lower, upper, dim)
    
    # Strategy 2: Opposition-based restart (1/3 of replacements)
    n_oppose = NP // 3
    oppose_indices = np.random.choice([i for i in range(NP) if i not in random_indices], 
                                       min(n_oppose, NP - n_random), replace=False)
    for idx in oppose_indices:
        # Reflect through search space center
        new_pop[idx] = (lower + upper) / 2.0 + (upper - lower) / 2.0 * np.random.uniform(-1, 1, dim)
    
    # Strategy 3: Latin Hypercube Sampling for remaining
    remaining = [i for i in range(NP) if i not in random_indices and i not in oppose_indices]
    if len(remaining) > 0:
        n_lhs = len(remaining)
        # Generate LHS samples in [0, 1] and scale to bounds
        lhs_samples = np.zeros((n_lhs, dim))
        for d in range(dim):
            samples = np.linspace(0, 1, n_lhs + 1)[:-1] + np.random.uniform(0, 1/n_lhs, n_lhs)
            lhs_samples[:, d] = samples
        np.random.shuffle(lhs_samples)
        lhs_scaled = lower + lhs_samples * (upper - lower)
        for i, idx in enumerate(remaining):
            new_pop[idx] = lhs_scaled[i]
    
    new_pop = self._clip_to_bounds_batch(new_pop)
    return new_pop
```