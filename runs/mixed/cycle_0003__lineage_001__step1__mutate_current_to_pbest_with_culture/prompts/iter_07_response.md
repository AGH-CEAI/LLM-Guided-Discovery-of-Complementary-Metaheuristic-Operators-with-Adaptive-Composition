Looking at the priority targets, Tasks 12, 8, and 21 are stuck at extremely high errors (~10^3 to 10^2), which suggests the algorithm is failing to converge at all. The current implementation samples `p_best` from only the top 10% of individuals, which provides insufficient selection pressure when stuck in local optima. The fixed 0.7 scaling factor and conservative 15% cultural blend aren't aggressive enough for these deceptive landscapes.

**Idea: Current-to-Global-Best with Adaptive Escalating Perturbation**

Replace the conservative pbest-sampling with a global-best approach combined with adaptive scaling and escalating random perturbation when stagnant. This fundamentally shifts from exploitative "best-of-top-10%" to aggressive "global-best" attraction with escape mechanisms.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
    sorted_idx = np.argsort(fitness)
    n_best = max(1, int(self.NP * self.p_best_rate))
    best_indices = sorted_idx[:n_best]

    # Use GLOBAL best instead of random pbest from top 10%
    global_best_idx = best_indices[0]
    global_best = population[global_best_idx]

    # Generate unique random indices for each target individual
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None], 
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1], 
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]

    # Adaptive scaling: increase F when stagnant to escape local optima
    F_adapted = self.F
    if self.stagnation_count > 10:
        F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count - 10)), 2.0)

    # Core mutation: current-to-global-best/1 (stronger than current-to-pbest)
    base = population + F_adapted * (global_best - population) + F_adapted * (population[r1] - population[r2])

    # Escalating random perturbation when stagnant (NEW: adds diversity on demand)
    if self.stagnation_count > 5:
        perturbation_scale = min(0.3 * np.log1p(self.stagnation_count), 2.0)
        random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
        base = base + random_perturbation

    # Cultural memory blending (same as before)
    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        similarity = np.array([np.linalg.norm(population[i] - cultural_contribution) 
                               for i in range(self.NP)])
        weights_culture = np.exp(-similarity / (np.std(similarity) + 1e-10))
        weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
        cultural_offset = cultural_contribution * 0.15
        donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
    else:
        donors = base

    return donors
```