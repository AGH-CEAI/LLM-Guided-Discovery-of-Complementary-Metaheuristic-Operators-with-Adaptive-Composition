**Idea: Extreme-Direction Diversity Mutation**

This strategy attacks the worst unsolved tasks (errors 10^1–10^3) by escaping local optima traps through mutation vectors derived from the *diversity extrema* of the population, combined with stagnation-adaptive scaling. Unlike p-best-directed variants that converge toward a single target, this uses the full population diversity spectrum—pulling from both the best AND worst individuals' differences to generate orthogonal exploration directions, with dramatically increased scaling during stagnation to break free from deceptive basins.

```python
def _mutate_original(self, population, fitness):
    """Extreme-Direction Diversity Mutation: escape local optima via population extrema"""
    sorted_idx = np.argsort(fitness)
    n_elite = max(1, int(self.NP * 0.1))
    elite_indices = sorted_idx[:n_elite]
    
    # Select elite target for each individual
    elite_idx = elite_indices[np.random.randint(0, n_elite, size=self.NP)]
    elite = population[elite_idx]
    
    # Generate r1, r2, r3 indices (all different from each other and from current)
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2, r3 = idx_a[:, 0], idx_a[:, 1], idx_a[:, 2]
    
    # === CORE DIFFERENCE: Use EXTREME population members for diversity vectors ===
    # Get bottom performers (worst individuals) - these represent unexplored/bad regions
    n_worst = max(1, int(self.NP * 0.2))
    worst_indices = sorted_idx[-n_worst:]
    worst_idx = worst_indices[np.random.randint(0, n_worst, size=self.NP)]
    worst = population[worst_idx]
    
    # Get second-best performers for additional diversity
    second_best_idx = sorted_idx[1:max(2, n_elite + 1)]
    second_idx = second_best_idx[np.random.randint(0, len(second_best_idx), size=self.NP)]
    second = population[second_idx]
    
    # === Build three orthogonal difference vectors ===
    # V1: Elite-to-worst (explores opposite of bad regions)
    diff_worst = worst - elite
    # V2: Random pair difference (standard exploration)
    diff_rand = population[r2] - population[r1]
    # V3: Second-best to random (additional diversity channel)
    diff_second = second - population[r3]
    
    # === Stagnation-adaptive scaling: dramatically increase exploration when stuck ===
    stagnation_level = min(self.stagnation_count / 30.0, 3.0)
    base_scale = self.F * (1.0 + 0.3 * np.random.rand(self.NP))
    stagnation_scale = 1.0 + stagnation_level * 2.0
    F_scaled = np.clip(base_scale * stagnation_scale, 0.1, 3.0)
    
    # === Combine three difference vectors ===
    donors = population + F_scaled[:, np.newaxis] * (
        0.5 * diff_worst + 
        0.3 * diff_rand + 
        0.2 * diff_second
    )
    
    # === Add directional perturbation during stagnation ===
    if self.stagnation_count > 5:
        perturb_scale = min(5.0 * np.log1p(self.stagnation_count), 20.0)
        perturbation = np.random.randn(self.NP, self.dim) * perturb_scale
        donors = donors + perturbation
    
    return np.clip(donors, self.lower, self.upper)
```