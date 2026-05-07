**Idea: Restart-Driven Opposition-Based Mutation**

For the worst unsolved tasks (16, 12, 8, 21, 23 with errors 1e+2–1e+3), all variants converge to the same local optima region. This variant triggers an **opposition-based restart** when stagnant: it reflects the population across the search-space center to generate diametrically-opposite candidates, then mutates from these opposition points. This provides a fundamentally different exploration direction vs. all prior variants that only perturb within or near the current basin.

```python
def _mutate_original(self, population, fitness):
    """Restart-Driven Opposition-Based Mutation: escape local optima via reflection"""
    bounds_center = (self.lower + self.upper) / 2.0

    # Detect deep stagnation: stuck far from target
    is_stuck = self.stagnation_count > 15

    if is_stuck:
        # Opposition-based exploration: reflect population across bounds center
        # This jumps to the opposite side of the search space
        opposition_pop = 2.0 * bounds_center - population
        opposition_pop = np.clip(opposition_pop, self.lower, self.upper)

        # Mutate from opposition points instead of current population
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

        # Mutation from OPPOSITION population with amplified F
        F_oppose = min(self.F * 1.5, 2.0)
        opp_base = opposition_pop + F_oppose * (p_best - opposition_pop) + F_oppose * (population[r1] - population[r2])

        # Blend: 60% opposition-mutated, 40% standard mutation from current
        F_std = self.F
        std_base = population + F_std * (p_best - population) + F_std * (population[r1] - population[r2])

        donors = 0.6 * opp_base + 0.4 * std_base
        return np.clip(donors, self.lower, self.upper)

    # Standard mutation (same as original for non-stuck state)
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