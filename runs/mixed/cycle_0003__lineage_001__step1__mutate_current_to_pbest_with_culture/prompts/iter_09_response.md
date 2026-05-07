**Idea: Historical-Best Anchored Mutation with Adaptive Escape Scaling**

This operator replaces the current-to-pbest structure with a fundamentally different paradigm: it **anchors mutations to the global best solution ever found** (across all generations), combined with adaptive scaling that increases mutation magnitude when the population is stuck far from the optimum. This directly targets the worst tasks (12, 8, 21, 13) where errors >1e+01 indicate the algorithm is trapped in distant local optima and needs to make large, informed jumps to escape.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
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

    if np.isfinite(self.best_fitness) and self.generation > 5:
        current_best = np.min(fitness)
        improvement_ratio = (current_best - self.best_fitness) / (abs(self.best_fitness) + 1e-30)

        # Adaptive F: increase when stuck far from optimum
        F_escape = np.clip(self.F * (1.0 + max(0, -improvement_ratio) * 0.5), 0.5, 2.0)

        # Direction weight: favor historical best when population is stuck
        w_hist = np.clip(0.7 + max(0, -improvement_ratio) * 0.2, 0.5, 0.9)
        w_curr = 1.0 - w_hist

        # Compute cultural centroid for direction guidance
        if np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_centroid = np.dot(weights_norm, self.cultural_memory)
            # Blend historical best with cultural knowledge
            target = w_hist * self.best_solution + w_curr * p_best + 0.1 * cultural_centroid
        else:
            target = w_hist * self.best_solution + w_curr * p_best

        donors = population + F_escape * (target - population) + self.F * (population[r1] - population[r2])
    else:
        donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

    return donors
```