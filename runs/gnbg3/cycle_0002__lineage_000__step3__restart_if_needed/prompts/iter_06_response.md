**Idea: Temporal Drift-Aware Restart with EMA Momentum**

This implementation tracks the exponential moving average of the best fitness and the centroid drift velocity across generations to distinguish between productive convergence and stagnation. A restart is triggered when the fitness EMA shows minimal improvement over a time window while centroid movement indicates the population is wandering without progress.

```python
def _restart_if_needed(self, population, fitness):
    """Temporal/dynamical restart: detect stagnation via EMA fitness drift and centroid momentum."""
    # Initialize temporal tracking attributes
    if not hasattr(self, '_ema_best_fitness'):
        self._ema_best_fitness = np.inf
        self._fitness_ema_history = []
        self._centroid_history = []
        self._ema_alpha = 0.2
        self._window_size = 15
        self._restart_count = 0
    
    # Compute current best fitness
    current_best = np.min(fitness)
    
    # Update EMA of best fitness (exponential smoothing for temporal signal)
    if np.isfinite(current_best):
        if np.isfinite(self._ema_best_fitness):
            self._ema_best_fitness = (1 - self._ema_alpha) * self._ema_best_fitness + self._ema_alpha * current_best
        else:
            self._ema_best_fitness = current_best
    
    # Store history for drift detection
    self._fitness_ema_history.append(self._ema_best_fitness)
    if len(self._fitness_ema_history) > self._window_size:
        self._fitness_ema_history.pop(0)
    
    # Compute centroid and track its drift (temporal movement signal)
    centroid = np.mean(population, axis=0)
    self._centroid_history.append(centroid.copy())
    if len(self._centroid_history) > self._window_size:
        self._centroid_history.pop(0)
    
    # Calculate centroid drift: magnitude of net displacement over window
    if len(self._centroid_history) >= 3:
        drift_vector = self._centroid_history[-1] - self._centroid_history[0]
        centroid_drift_magnitude = np.linalg.norm(drift_vector)
        avg_step = centroid_drift_magnitude / (len(self._centroid_history) - 1)
    else:
        centroid_drift_magnitude = 0.0
        avg_step = 0.0
    
    # Compute fitness improvement rate over the time window
    fitness_improved = False
    if len(self._fitness_ema_history) >= 5:
        recent_improvement = self._fitness_ema_history[0] - self._fitness_ema_history[-1]
        fitness_improved = recent_improvement > 1e-6
    
    # Restart trigger logic: EMA stagnation + wandering without progress
    # High centroid drift but low fitness improvement = wasted exploration = restart
    should_restart = False
    if len(self._fitness_ema_history) >= 5:
        ema_trend = self._fitness_ema_history[-1] - self._fitness_ema_history[-5]
        if ema_trend > -1e-5 and centroid_drift_magnitude > 5.0:
            should_restart = True
        elif ema_trend > -1e-7 and self._restart_count > 0:
            should_restart = True
    
    if not should_restart:
        return (None, None, None)
    
    # Execute restart: preserve elite, regenerate rest with spatial perturbation
    self._restart_count += 1
    
    # Keep top 15% as elite anchors
    sorted_indices = np.argsort(fitness)
    n_elite = max(2, int(0.15 * self.NP))
    elite_indices = sorted_indices[:n_elite]
    elite_pop = population[elite_indices].copy()
    elite_fitness = fitness[elite_indices].copy()
    
    # Generate new population: mix of random and elite-perturbed
    new_pop = np.empty((self.NP, self.dim))
    
    # Perturb elite with moderate noise
    for i in range(n_elite):
        noise_scale = 15.0 / np.sqrt(self._restart_count + 1)  # Reduce noise with restart count
        new_pop[i] = elite_pop[i] + np.random.randn(self.dim) * noise_scale
    
    # Fill remainder with random initialization + occasional archive injection
    for i in range(n_elite, self.NP):
        if len(self.archive) > 0 and np.random.rand() < 0.2:
            # Inject from archive with perturbation
            archive_concat = np.vstack(self.archive)
            donor = archive_concat[np.random.randint(len(archive_concat))]
            new_pop[i] = donor + np.random.randn(self.dim) * 20.0
        else:
            # Fresh random initialization with wider bounds
            new_pop[i] = np.random.uniform(-100.0, 100.0, self.dim)
    
    new_pop = np.clip(new_pop, -100.0, 100.0)
    
    # Compute new fitness
    new_fitness = func(new_pop) if 'func' in dir() else np.full(self.NP, np.inf)
    
    # Find best in new population
    best_new_idx = np.argmin(new_fitness)
    best_new_fitness = new_fitness[best_new_idx]
    best_new_x = new_pop[best_new_idx].copy()
    
    # Reset temporal tracking for fresh convergence signal
    self._ema_best_fitness = best_new_fitness
    self._fitness_ema_history = [best_new_fitness]
    self._centroid_history = [np.mean(new_pop, axis=0)]
    
    return (new_pop, best_new_fitness, best_new_x)
```