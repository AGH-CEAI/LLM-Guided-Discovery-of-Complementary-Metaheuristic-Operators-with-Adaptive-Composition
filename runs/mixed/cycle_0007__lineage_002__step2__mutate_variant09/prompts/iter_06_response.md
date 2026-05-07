**Idea: Stagnation-Triggered Random Direction Super-Mutation**

The worst unsolved tasks (errors 1e+0 to 1e+3) are stuck in local optima. Current operators all rely on p_best/best_solution guidance, which is counterproductive when those are trapped in a local basin. This variant completely abandons guided directions when stagnant and uses a large-F random direction push to escape.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Stagnation-triggered random direction with super-mutation escape"""
    # Super-mutation escape when stagnant
    if self.stagnation_count > 20:
        # Use random direction with very large F to escape local optima
        # Completely ignore p_best when trapped - it's part of the problem
        F_super = 1.5 + min(0.5 * self.stagnation_count, 3.0)
        F_super = min(F_super, 5.0)
        
        idx_r = np.random.randint(0, self.NP, (self.NP, 3))
        idx_r[:, 1:] = np.where(idx_r[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_r[:, 1:] + 1) % self.NP, idx_r[:, 1:])
        idx_r[:, 2] = np.where(idx_r[:, 2] == idx_r[:, 1],
                               (idx_r[:, 2] + 1) % self.NP, idx_r[:, 2])
        r1, r2 = idx_r[:, 1], idx_r[:, 2]
        
        donors = population + F_super * (population[r1] - population[r2])
        
    elif self.stagnation_count > 10:
        # Medium escape with some guided component
        F_escape = 1.0 + 0.1 * (self.stagnation_count - 10)
        F_escape = min(F_escape, 2.0)
        
        idx_r = np.random.randint(0, self.NP, (self.NP, 3))
        idx_r[:, 1:] = np.where(idx_r[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_r[:, 1:] + 1) % self.NP, idx_r[:, 1:])
        idx_r[:, 2] = np.where(idx_r[:, 2] == idx_r[:, 1],
                               (idx_r[:, 2] + 1) % self.NP, idx_r[:, 2])
        r1, r2 = idx_r[:, 1], idx_r[:, 2]
        
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]
        
        # Blend: 60% random direction, 40% guided
        donors = population + F_escape * (0.6 * (population[r1] - population[r2]) + 0.4 * (p_best - population))
        
    else:
        # Normal operation: use p_best guidance
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

        donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

    return np.clip(donors, self.lower, self.upper)
```