**Idea: Restart-Based Mutation with Stagnation Escape**
When stagnation_count exceeds 20, this operator progressively replaces mutation targets with uniformly random vectors (probability scales from 50% to 95%), completely abandoning the best-guided direction that traps the population in local optima. This is fundamentally different from all prior variants which always use pbest as mutation anchor.

```python
def _mutate_original(self, population, fitness):
    """Restart-based mutation: aggressive escape from local optima via random restarts"""
    sorted_idx = np.argsort(fitness)
    n_best = max(1, int(self.NP * self.p_best_rate))
    best_indices = sorted_idx[:n_best]
    p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
    p_best = population[p_best_idx]

    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]

    base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

    if self.stagnation_count > 20:
        restart_prob = min(0.50 + 0.05 * (self.stagnation_count - 20), 0.95)
        restart_mask = np.random.rand(self.NP) < restart_prob
        n_restart = int(np.sum(restart_mask))
        if n_restart > 0:
            restart_vectors = np.random.uniform(self.lower, self.upper, (n_restart, self.dim))
            base[restart_mask] = restart_vectors

    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                               for i in range(self.NP)])
        sim_std = np.std(similarity) + 1e-10
        weights_culture = np.exp(-similarity / sim_std)
        weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
        cultural_offset = cultural_contribution * 0.15
        donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
    else:
        donors = base

    return np.clip(donors, self.lower, self.upper)
```