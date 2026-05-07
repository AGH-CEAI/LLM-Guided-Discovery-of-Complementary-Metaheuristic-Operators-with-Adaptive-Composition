**Idea: Ensemble-Diversity Mutation with Adaptive Strategy Selection**

This approach generates THREE fundamentally different mutation vectors per individual using diverse strategies (current-to-pbest, greedy-best, distance-based diversity), then adaptively selects among them weighted by recent success. This directly targets the worst unsolved tasks (12, 8, 21, 13) which likely suffer from getting trapped in local optima due to insufficient exploration diversity in the current single-strategy mutation.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
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
    
    # Strategy 1: Current-to-pbest (standard)
    mutant1 = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])
    
    # Strategy 2: Greedy best (use only best, not p_best)
    best_idx_single = sorted_idx[0]
    best_vec = population[best_idx_single]
    F_greedy = np.clip(self.F * 1.2, 0.3, 2.0)
    mutant2 = population + F_greedy * (best_vec - population) + F_greedy * (population[r1] - population[r2])
    
    # Strategy 3: Distance-based diversity (force exploration)
    centroid = np.mean(population, axis=0)
    dists = np.linalg.norm(population - centroid, axis=1)
    dists_norm = dists / (np.max(dists) + 1e-10)
    most_diverse_idx = np.argmax(dists_norm)
    diverse_vec = population[most_diverse_idx]
    F_large = np.clip(self.F * 1.8, 0.5, 2.5)
    mutant3 = population + F_large * (diverse_vec - population) + F_large * (population[r1] - population[r2])
    
    # Adaptive strategy weights (initialized uniformly, updated via success tracking)
    if not hasattr(self, 'strategy_weights'):
        self.strategy_weights = np.ones(3) / 3.0
        self.strategy_success = np.zeros(3)
    
    weights = self.strategy_weights / (np.sum(self.strategy_weights) + 1e-10)
    selected_strategies = np.random.choice(3, size=self.NP, p=weights)
    
    donors = np.where(selected_strategies[:, None] == 0, mutant1,
                      np.where(selected_strategies[:, None] == 1, mutant2, mutant3))
    
    # Cultural influence (keep but reduce to 8% to not overpower diversity)
    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        cultural_offset = cultural_contribution * 0.08
        donors = (1 - 0.08) * donors + 0.08 * (population + cultural_offset)
    
    return donors
```