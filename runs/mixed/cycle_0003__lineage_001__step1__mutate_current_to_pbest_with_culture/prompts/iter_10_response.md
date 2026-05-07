**Idea: Adaptive Ensemble Mutation with Elite Diversity Escaping**

Instead of a single mutation vector, generate multiple candidates using different strategies (elite-guided, diversity-driven, long-jump exploration) and select via weighted probability. When stuck (high stagnation), inject aggressive exploration by mutating toward random directions with large scaling. This breaks out of the wrong basins that trap all current variants on Tasks 8, 12, 13, 21.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
    elite_count = max(1, int(self.NP * self.p_best_rate))
    sorted_idx = np.argsort(fitness)
    elite_indices = sorted_idx[:elite_count]
    elite_pop = population[elite_indices]
    best_idx = elite_indices[0]
    best = population[best_idx]
    
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None], 
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1], 
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]
    
    # Standard DE/current-to-pbest mutation
    p_best_idx = elite_indices[np.random.randint(0, elite_count, size=self.NP)]
    p_best = population[p_best_idx]
    de_mutant = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])
    
    # Elite-guided mutation: move toward best-known elite
    elite_idx_per_ind = np.random.randint(0, elite_count, size=self.NP)
    elite_guide = elite_pop[elite_idx_per_ind]
    elite_mutant = population + self.F * (elite_guide - population) + self.F * (population[r1] - population[r2])
    
    # Diversity escape: large random perturbation around best
    escape_scale = max(self.F * 2.0, 1.5)
    random_dir = np.random.randn(self.NP, self.dim)
    random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-10)
    jump_magnitude = np.random.uniform(0.1, 1.0, size=(self.NP, 1)) * escape_scale * np.abs(self.upper - self.lower).mean()
    diversity_mutant = best + random_dir * jump_magnitude
    
    # Ensemble selection with stagnation-aware weighting
    if self.stagnation_count > 15:
        # Stuck: heavily favor diversity escape
        weights = np.array([0.15, 0.15, 0.70])
    elif self.stagnation_count > 5:
        weights = np.array([0.30, 0.30, 0.40])
    else:
        weights = np.array([0.50, 0.30, 0.20])
    
    weights = weights / weights.sum()
    selector = np.random.rand(self.NP)
    donors = np.where(selector < weights[0], de_mutant,
                      np.where(selector < weights[0] + weights[1], elite_mutant, diversity_mutant))
    
    return donors
```