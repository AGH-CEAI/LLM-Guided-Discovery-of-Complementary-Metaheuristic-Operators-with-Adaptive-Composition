**Idea: Absolute-Best Archive-Driven Mutation with Dimensional Scaling**

Instead of using p_best (top 10%) with scalar F and weak cultural influence (15%), this variant uses the **absolute best individual** for all mutations, adapts F **per-dimension** based on archive statistics, and applies **periodic jump mutations** to escape local optima traps. This targets the worst tasks (12, 8, 21, 13) that are stuck at errors >10 because the current operator's weak guidance and scalar F cannot drive search toward the global optimum.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
    sorted_idx = np.argsort(fitness)
    
    # Use ABSOLUTE best (not p_best) for stronger exploitation
    best_idx = sorted_idx[0]
    best_solution = population[best_idx]
    
    # Build archive of top performers for adaptive scaling
    archive_size = max(3, self.NP // 5)
    archive_indices = sorted_idx[:archive_size]
    archive = population[archive_indices]
    
    # Per-dimension adaptive F based on archive spread
    archive_std = np.std(archive, axis=0)
    mean_std = np.mean(archive_std) + 1e-10
    F_base = np.clip(self.F, 0.3, 1.5)
    F_per_dim = F_base * (1.0 + archive_std / mean_std)
    F_per_dim = np.clip(F_per_dim, 0.1, 2.0)
    
    # Generate random indices avoiding current individual
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]
    
    # current-to-absolute-best/1 mutation with per-dimension F
    diff_best = best_solution - population
    diff_rand = population[r1] - population[r2]
    base = population + F_per_dim * diff_best + F_per_dim * diff_rand
    
    # Periodic jump mutations to escape local optima (every 7 generations or when stagnant)
    if self.generation % 7 == 0 or self.stagnation_count > self.stagnation_limit // 4:
        archive_mean = np.mean(archive, axis=0)
        jump_scale = 1.5 + np.random.rand() * 2.0
        jump_donors = population + jump_scale * (archive_mean - population + 
                                                  np.random.randn(self.NP, self.dim) * (archive_std + 1e-10))
        donors = 0.25 * jump_donors + 0.75 * base
    else:
        donors = base
    
    # Cultural memory influence (reduced to 10% since best is now strongest)
    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        donors = 0.90 * donors + 0.10 * (population + cultural_contribution * 0.15)
    
    return donors
```