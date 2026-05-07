**Idea: Rank-IQR + Population Improvement Rate**
Adapt inertia weight using fitness rank distribution (IQR of ranks) and population-level improvement rate. Both signals are rank-based and scale-independent, targeting convergence detection for the worst unsolved tasks.
```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using fitness rank distribution and success history.
    
    Category D: Fitness-landscape / rank-based
    - Uses fitness ranks (not raw distances or covariance)
    - Combines rank-spread signal with population improvement rate
    - Scale-independent, robust to fitness landscape shape
    """
    # Compute fitness ranks (scale-independent, [0, 1])
    sorted_indices = np.argsort(self.current_fitness)
    ranks = np.zeros(self.np)
    ranks[sorted_indices] = np.arange(self.np) / max(1, self.np - 1)
    
    # Rank-based IQR: measures spread of fitness ranks in population
    q75_idx = int(0.75 * (self.np - 1))
    q25_idx = int(0.25 * (self.np - 1))
    sorted_ranks = np.sort(ranks)
    rank_iqr = sorted_ranks[q75_idx] - sorted_ranks[q25_idx]
    rank_iqr = np.clip(rank_iqr, 0.01, 1.0)
    
    # Normalized rank spread (target ~0.5 for uniform distribution)
    rank_spread = rank_iqr / 0.5
    
    # Population-level improvement rate (fitness-based, not distance-based)
    if not hasattr(self, '_prev_fitness_for_inertia'):
        self._prev_fitness_for_inertia = self.current_fitness.copy()
    
    prev_fitness = self._prev_fitness_for_inertia
    improved_mask = self.current_fitness < prev_fitness
    improvement_rate = np.sum(improved_mask) / max(1, self.np)
    
    self._prev_fitness_for_inertia = self.current_fitness.copy()
    
    # Combine signals multiplicatively
    # High rank_spread -> higher target inertia (more exploration)
    # High improvement_rate -> higher target inertia (exploiting good directions)
    rank_spread_clamped = np.clip(rank_spread, 0.1, 2.0)
    improvement_clamped = np.clip(improvement_rate, 0.01, 1.0)
    
    target_inertia = 0.4 + 0.55 * (rank_spread_clamped * improvement_clamped)
    
    # EMA smoothing
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```