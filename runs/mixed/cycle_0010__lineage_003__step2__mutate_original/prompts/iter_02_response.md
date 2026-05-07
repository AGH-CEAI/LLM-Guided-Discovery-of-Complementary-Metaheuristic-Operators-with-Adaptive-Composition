Looking at the task errors, the worst unsolved tasks (12, 16, 8, 21, 23) have errors stuck at ~1e+01 to 1e+03, suggesting the population is converging to local optima and the mutation operators lack sufficient diversity mechanisms to escape. The original operator uses cultural memory but lacks explicit historical best guidance. Variant_01_idea_0 uses JADE-style archive but performs worse on several hard tasks.

**Idea: Historical Best-Guided Mutation with Convergence-Aware Exploration**

This variant combines the current-to-pbest/1 structure with:
1. **Historical best guidance**: Uses `self.best_solution` (the global best found across ALL generations) as an additional direction, unlike the original which only uses current p_best and cultural memory
2. **Convergence-aware perturbation**: When population clusters too tightly (detected via centroid distance), a stronger random perturbation kicks in to escape local optima

This is fundamentally different because it explicitly uses the global historical best to pull solutions toward the true optimum rather than just the current best or cultural memory.

```python
def _mutate_original(self, population, fitness):
    """Historical Best-Guided Mutation with Convergence-Aware Exploration"""
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

    # Historical best guidance (NEW: uses global best from ALL generations)
    if self.best_solution is not None:
        hist_best = self.best_solution
        # Compute distance-based weights: closer individuals get stronger pull
        hist_dist = np.linalg.norm(population - hist_best, axis=1) + 1e-10
        hist_weights = 1.0 / hist_dist
        hist_weights = hist_weights / (np.sum(hist_weights) + 1e-10)
        # Direction toward historical best, weighted by convergence level
        pop_center = np.mean(population, axis=0)
        convergence = np.linalg.norm(population - pop_center, axis=1) / (np.linalg.norm(hist_best - pop_center) + 1e-10)
        convergence = np.clip(convergence, 0, 1)
        # Stronger pull when more converged (population clustered away from best)
        hist_strength = 0.3 * (1.0 - convergence)
        hist_direction = (hist_best - population)
        base = base + hist_strength[:, np.newaxis] * hist_direction

    # Convergence-aware perturbation (NEW: escapes local optima)
    pop_center = np.mean(population, axis=0)
    pop_dispersion = np.linalg.norm(population - pop_center, axis=1)
    avg_dispersion = np.mean(pop_dispersion) + 1e-10
    normalized_dispersion = pop_dispersion / avg_dispersion

    # When dispersion is low (population clustered), add stronger perturbation
    low_disp_mask = normalized_dispersion < 0.5
    if np.any(low_disp_mask):
        perturbation_scale = 0.5 * self.F * (1.0 - normalized_dispersion[low_disp_mask])
        perturbation = np.random.randn(np.sum(low_disp_mask), self.dim) * perturbation_scale[:, np.newaxis]
        base[low_disp_mask] = base[low_disp_mask] + perturbation

    return np.clip(base, self.lower, self.upper)
```