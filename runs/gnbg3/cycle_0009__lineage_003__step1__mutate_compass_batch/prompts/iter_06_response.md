Looking at the task, I need a temporal/dynamical approach for `_mutate_compass_batch` that tracks quantities across generations to beat 2.251e-06 on task 5.

**Idea: Temporal Momentum with Trend Detection**
Track population drift and improvement momentum across generations using exponential moving averages. When stagnation is detected, boost exploration by amplifying momentum vectors and adding orthogonal perturbation based on recent centroid drift. This gives the mutation "inertia" that carries forward successful directional trends while escaping local optima when stuck.

```python
def _mutate_compass_batch(self, population, fitness):
    """Temporal momentum-guided compass mutation.
    
    Tracks population drift and improvement momentum across generations (F: temporal/dynamical).
    Uses EMA of centroid movement and fitness improvement direction to bias mutations.
    Detects stagnation via windowed fitness variance and boosts exploration when stuck.
    """
    NP, dim = self.dim if hasattr(self, 'dim') else population.shape
    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = np.zeros(dim)
        self._ema_fitness_improvement = 0.0
        self._centroid_history = []
        self._fitness_history = []
        self._momentum_vec = np.zeros(dim)
        self._stagnation_window = []
        self._alpha = 0.3
        self._momentum_scale = 0.5
    
    # Compute current centroid and fitness stats
    centroid = population.mean(axis=0)
    current_best_fitness = np.min(fitness)
    
    # Update centroid EMA (temporal tracking)
    self._ema_centroid = (1 - self._alpha) * self._ema_centroid + self._alpha * centroid
    
    # Track centroid history for drift computation
    self._centroid_history.append(centroid.copy())
    if len(self._centroid_history) > 10:
        self._centroid_history.pop(0)
    
    # Compute centroid drift (rate of change across generations)
    if len(self._centroid_history) >= 3:
        drift = self._centroid_history[-1] - self._centroid_history[0]
        drift_norm = np.linalg.norm(drift) + 1e-10
        drift_direction = drift / drift_norm
    else:
        drift_direction = np.zeros(dim)
        drift_norm = 0.0
    
    # Update fitness EMA and stagnation window
    self._fitness_history.append(current_best_fitness)
    if len(self._fitness_history) > 10:
        self._fitness_history.pop(0)
    
    self._stagnation_window.append(current_best_fitness)
    if len(self._stagnation_window) > 8:
        self._stagnation_window.pop(0)
    
    # Detect stagnation via fitness variance in window
    if len(self._stagnation_window) >= 4:
        fitness_variance = np.var(self._stagnation_window)
        is_stagnating = fitness_variance < 1e-10
    else:
        fitness_variance = 1.0
        is_stagnating = False
    
    # Compute momentum vector: direction of recent centroid movement
    if len(self._centroid_history) >= 2:
        recent_movement = self._centroid_history[-1] - max(0, len(self._centroid_history) - 3) * np.mean(self._centroid_history[:max(1, len(self._centroid_history)-3)], axis=0)
        self._momentum_vec = (1 - self._alpha) * self._momentum_vec + self._alpha * recent_movement
    
    # Stagnation detection via fitness improvement rate
    if len(self._fitness_history) >= 3:
        recent_improvement = self._fitness_history[0] - self._fitness_history[-1]
        improvement_rate = max(0, recent_improvement) / (len(self._fitness_history) - 1)
    else:
        improvement_rate = 0.0
        recent_improvement = 0.0
    
    # Compute stagnation score (0 = no stagnation, 1 = fully stuck)
    stagnation_score = 0.0
    if len(self._stagnation_window) >= 4:
        window_range = max(self._stagnation_window) - min(self._stagnation_window)
        if window_range < 1e-8:
            stagnation_score = 1.0
        else:
            stagnation_score = 1.0 - np.clip(improvement_rate / (window_range + 1e-10), 0, 1)
    
    # Adaptive scaling based on stagnation
    exploration_boost = 1.0 + 2.0 * stagnation_score
    
    # Compute fitness ranks for direction selection
    fitness_ranks = self._compute_fitness_ranking(fitness)
    
    # Generate mutations per individual
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
        
        # Direction D: Momentum-based direction (temporal/dynamical innovation)
        # Use EMA centroid drift to bias mutation direction
        momentum_direction = self._momentum_vec.copy()
        if np.linalg.norm(momentum_direction) > 1e-10:
            momentum_direction = momentum_direction / np.linalg.norm(momentum_direction)
        
        # Combine drift and momentum for exploration
        temporal_bias = drift_direction * drift_norm * 0.1 + momentum_direction * self._momentum_scale
        
        # Apply stagnation-triggered exploration boost
        temporal_bias *= exploration_boost
        
        # Add orthogonal perturbation to drift direction
        if drift_norm > 1e-10:
            perp = np.random.randn(dim)
            perp -= np.dot(perp, drift_direction) * drift_direction
            perp_norm = np.linalg.norm(perp) + 1e-10
            perp /= perp_norm
            temporal_bias += perp * np.random.rand() * drift_norm * 0.2 * (1 + stagnation_score)
        
        dD = population[i] + self.F * temporal_bias + self.F * (population[r1] - population[r2])
        score_D = stagnation_score * 0.8 + (1 - stagnation_score) * 0.2  # Boost when stagnating
        
        # Select best direction
        candidates = [(dA, score_A), (dB, score_B), (dC, score_C), (dD, score_D)]
        best_dir, best_score = max(candidates, key=lambda x: x[1])
        
        mutants[i] = best_dir
    
    # Clamp to bounds
    mutants = np.clip(mutants, -100.0, 100.0)
    return mutants
```