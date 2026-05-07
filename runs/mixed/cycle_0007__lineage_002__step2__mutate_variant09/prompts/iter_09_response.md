**Idea: Opposition-Based DE with Centroid Reflection**
Uses mirrored population members (opposition learning) around the centroid to escape deep local optima — fundamentally different from all prior variants that only explore in the direction of best solutions.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Opposition-Based DE with centroid reflection and restart injection"""
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

    # Standard DE mutation
    base_mutants = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

    # Opposition learning: reflect around centroid
    centroid = np.mean(population, axis=0)
    lower_arr = np.array(self.lower)
    upper_arr = np.array(self.upper)
    range_arr = upper_arr - lower_arr

    # Opposition candidates (mirrored across search space)
    opposition = lower_arr + upper_arr - base_mutants

    # Blend DE mutant with opposition candidate
    alpha = np.random.rand(self.NP, 1)
    donors = alpha * base_mutants + (1 - alpha) * opposition

    # Inject random restart when severely stagnant
    if self.stagnation_count > 20:
        n_inject = max(1, self.NP // 4)
        inject_idx = np.random.randint(0, self.NP, size=n_inject)
        inject = np.random.uniform(self.lower, self.upper, (n_inject, self.dim))
        donors[inject_idx] = inject

    # Dimension-wise perturbation for escaping local traps
    if self.stagnation_count > 10:
        perturb_mask = np.random.rand(self.NP, self.dim) < 0.3
        perturb = np.random.randn(self.NP, self.dim) * 0.5 * (self.stagnation_count / 50.0)
        donors = donors + perturb_mask * perturb

    return np.clip(donors, self.lower, self.upper)
```