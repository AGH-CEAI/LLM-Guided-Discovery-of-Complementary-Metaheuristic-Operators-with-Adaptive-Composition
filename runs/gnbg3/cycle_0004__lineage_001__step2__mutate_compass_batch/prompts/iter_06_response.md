**Idea: Temporal Momentum Compass**
Track an exponential moving average (EMA) of successful mutation direction vectors across generations. When selecting the best compass direction, blend it with the momentum vector (30% weight) to exploit temporal autocorrelation in the search trajectory. This gives the algorithm "inertia" that carries successful directional trends forward.

```python
def _mutate_compass_batch(self, population, fitness):
    """Temporal momentum compass: blend current direction with EMA of past successes."""
    NP = self.NP
    
    # Initialize temporal momentum tracking
    if not hasattr(self, '_dir_ema'):
        self._dir_ema = np.zeros((NP, self.dim))
        self._ema_alpha = 0.3
    
    fitness_ranks = self._compute_fitness_ranking(fitness)
    mutants = np.empty_like(population)
    
    for i in range(NP):
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        i_rank = fitness_ranks[i]
        
        # Direction A: DE/rand/1 style
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        # Direction B: DE/best/1 style
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
        # Direction C: pbest with archive
        p = 0.1
        top_p = int(np.ceil(p * NP))
        pbest_candidates = np.argsort(fitness_ranks)[:top_p]
        pbest = population[np.random.choice(pbest_candidates)]
        dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
        score_C = i_rank - fitness_ranks[pbest_candidates].min()
        
        # Select best direction for this individual
        candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
        best_dir, _ = max(candidates, key=lambda x: x[1])
        
        # Blend with temporal momentum (EMA of past directions)
        # Higher momentum weight when EMA is stronger (more consistent direction)
        ema_strength = 1.0 / (1.0 + np.linalg.norm(self._dir_ema[i]))
        momentum_weight = 0.3 * ema_strength
        mutants[i] = (1 - momentum_weight) * best_dir + momentum_weight * self._dir_ema[i]
        
        # Update EMA: track direction toward better fitness
        if hasattr(self, '_prev_population') and self._prev_population is not None:
            if fitness[i] < getattr(self, '_prev_fitness', fitness)[i]:
                delta_dir = best_dir - self._dir_ema[i]
                self._dir_ema[i] = (1 - self._ema_alpha) * self._dir_ema[i] + self._ema_alpha * delta_dir
    
    # Store for next generation comparison
    self._prev_population = population.copy()
    self._prev_fitness = fitness.copy()
    
    return mutants
```