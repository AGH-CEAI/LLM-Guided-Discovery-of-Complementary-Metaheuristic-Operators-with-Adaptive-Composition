**Idea: Centroid Repulsion Mutation**

One-line description: Escape from local optima by mutating AWAY from the population centroid toward the "anti-centroid" (reflected best position), using large perturbation scales to explore the opposite side of the search space.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Centroid Repulsion — escape local optima by mutating away from population center"""
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

    # Compute centroid (center of mass of current population)
    centroid = np.mean(population, axis=0)

    # Compute search space diameter for scale normalization
    space_diameter = np.linalg.norm(self.upper - self.lower) + 1e-10

    # Distance from centroid to best individual (normalized)
    centroid_to_best = self.best_solution - centroid
    dist_centroid_best = np.linalg.norm(centroid_to_best) + 1e-10

    # Anti-centroid: reflect best through centroid to explore opposite side
    # anti_centroid = centroid - (best - centroid) = 2*centroid - best
    anti_centroid = 2.0 * centroid - self.best_solution

    # Stagnation-driven scale: larger when stuck, to escape basin of attraction
    stagnation = float(self.stagnation_count)
    repulsion_scale = 1.5 + min(stagnation * 0.05, 2.0)  # range [1.5, 3.5]
    repulsion_scale = min(repulsion_scale, 3.5)

    # Compute repulsion vector: move FROM centroid toward anti-centroid
    # The farther the best is from centroid, the stronger the repulsion
    repulsion_direction = anti_centroid - centroid
    repulsion_norm = np.linalg.norm(repulsion_direction) + 1e-10
    unit_repulsion = repulsion_direction / repulsion_norm

    # Target = centroid + scaled repulsion vector
    target = centroid + unit_repulsion * dist_centroid_best * repulsion_scale

    # Clip target to bounds (edge case: handle out-of-bounds reflection)
    target = np.clip(target, self.lower, self.upper)

    # Large perturbation scale for exploration
    F_large = np.clip(0.8 + min(stagnation * 0.03, 1.0), 0.8, 2.0)
    F_secondary = np.clip(self.F + 0.2, 0.5, 2.0)

    # Primary mutation: move toward anti-centroid (away from best)
    donors = population + F_large * (target - population) + F_secondary * (population[r1] - population[r2])

    # Additional diversity injection when stagnant
    if stagnation > 10:
        diversity_noise = np.random.randn(self.NP, self.dim)
        diversity_scale = min(0.2 * np.log1p(stagnation), 2.0)
        donors = donors + diversity_scale * diversity_noise

    return np.clip(donors, self.lower, self.upper)
```