Looking at the worst unsolved tasks (16, 12, 8, 21, 23, 13, 14, 20), they're all stuck at specific error magnitudes (5.0, 50.0, 100+, 500+) which strongly suggests they're trapped in local optima basins. All existing variants use directional mutations guided by the current population, but when the population converges to a local optimum, these directional mutations can't escape because they always point inward toward the basin.

**Analysis:**
- Error magnitudes stuck at specific values indicate local optima traps
- All variants (original, JADE, variant_07, variant_09) use variants of `p_best - population` or similar guided directions
- When population converges, these directions become useless (all point to same basin)
- Need a mechanism that actively moves AWAY from the converged region

**Idea: Reflective Diversification with Bounce Effect**

This is fundamentally different: instead of following directional guidance from the stuck population, it computes the population centroid, finds the direction each individual is from the centroid, and then BOUNCES off the centroid by reflecting in the opposite direction. This creates a systematic escape mechanism when stuck.

```python
def _mutate_original(self, population, fitness):
    """Reflective Diversification: escape local optima by bouncing away from population centroid"""
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

    # Compute population centroid and direction vectors from centroid
    centroid = np.mean(population, axis=0)
    diff_from_centroid = population - centroid
    dist_from_centroid = np.linalg.norm(diff_from_centroid, axis=1, keepdims=True) + 1e-10
    direction_from_centroid = diff_from_centroid / dist_from_centroid

    # Reflected direction (opposite to cluster direction)
    reflected_direction = -direction_from_centroid

    # Compute reflection strength based on stagnation and convergence
    convergence_ratio = np.std(population, axis=0) / (self.upper - self.lower + 1e-10)
    avg_convergence = np.mean(convergence_ratio)

    if self.stagnation_count > 10 or avg_convergence < 0.05:
        # Strong reflection when stuck or highly converged
        reflection_strength = min(0.5, 0.25 + 0.025 * self.stagnation_count)
    elif self.stagnation_count > 3:
        # Moderate reflection when starting to stagnate
        reflection_strength = 0.15 + 0.02 * self.stagnation_count
    else:
        reflection_strength = 0.0

    if reflection_strength > 0:
        # Scale jump based on search space and distance from centroid
        jump_scale = (self.upper - self.lower) * 0.15
        reflected_points = centroid + reflected_direction * jump_scale
        donors = (1 - reflection_strength) * base + reflection_strength * reflected_points
    else:
        donors = base

    return np.clip(donors, self.lower, self.upper)
```