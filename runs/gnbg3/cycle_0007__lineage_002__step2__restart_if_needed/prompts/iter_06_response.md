**Idea: Temporal Velocity-Adaptive Restart with Multi-Signal Confidence**

This approach tracks fitness improvement velocity, population diversity drift, and centroid movement across generations using exponential moving averages and rate-of-change signals. A confidence-weighted combination determines restart urgency, addressing the "fallback random pick" issue by providing meaningful temporal decision signals.

```python
def _restart_if_needed(self, population, fitness):
    """
    Temporal/dynamical restart: track fitness velocity, diversity drift, and
    centroid movement across generations using EMAs and rate-of-change signals.
    A multi-signal confidence score determines restart urgency.
    """
    # Initialize temporal tracking attributes
    if not hasattr(self, '_fitness_velocity_history'):
        self._fitness_velocity_history = []
        self._diversity_ema = None
        self._diversity_velocity_history = []
        self._centroid_history = []
        self._ema_alpha = 0.2  # EMA smoothing factor
        self._restart_confidence = 0.0
    
    # Signal 1: Fitness improvement velocity (rate of change of best fitness)
    best_fit = np.min(fitness)
    if hasattr(self, '_prev_best_fitness'):
        fitness_delta = self._prev_best_fitness - best_fit
        self._fitness_velocity_history.append(fitness_delta)
    else:
        fitness_delta = 0.0
        self._fitness_velocity_history.append(0.0)
    self._prev_best_fitness = best_fit
    
    # Keep bounded history
    if len(self._fitness_velocity_history) > 15:
        self._fitness_velocity_history.pop(0)
    
    # Compute smoothed fitness velocity using EMA
    if len(self._fitness_velocity_history) >= 3:
        velocity_ema = self._ema_alpha * self._fitness_velocity_history[-1]
        if len(self._fitness_velocity_history) > 1:
            velocity_ema += (1 - self._ema_alpha) * np.mean(self._fitness_velocity_history[-3:-1])
    else:
        velocity_ema = np.mean(self._fitness_velocity_history) if self._fitness_velocity_history else 0.0
    
    # Signal 2: Population diversity drift (EMA of diversity with velocity)
    current_diversity = self._compute_diversity(population)
    if self._diversity_ema is None:
        self._diversity_ema = current_diversity
    else:
        self._diversity_ema = (1 - self._ema_alpha) * self._diversity_ema + self._ema_alpha * current_diversity
    
    # Diversity velocity: rate of change
    if len(self._diversity_velocity_history) > 0:
        diversity_delta = current_diversity - self._diversity_velocity_history[-1]
    else:
        diversity_delta = 0.0
    self._diversity_velocity_history.append(current_diversity)
    if len(self._diversity_velocity_history) > 15:
        self._diversity_velocity_history.pop(0)
    
    # Signal 3: Centroid drift (movement of population center over time)
    centroid = population.mean(axis=0)
    if len(self._centroid_history) > 0:
        centroid_displacement = np.linalg.norm(centroid - self._centroid_history[-1])
    else:
        centroid_displacement = 0.0
    self._centroid_history.append(centroid.copy())
    if len(self._centroid_history) > 15:
        self._centroid_history.pop(0)
    
    # Compute centroid drift velocity
    if len(self._centroid_history) >= 3:
        recent_disp = [np.linalg.norm(self._centroid_history[i] - self._centroid_history[i-1]) 
                       for i in range(1, min(4, len(self._centroid_history)))]
        centroid_drift_velocity = np.mean(recent_disp) if recent_disp else 0.0
    else:
        centroid_drift_velocity = 0.0
    
    # Normalize signals to [0, 1] scale for confidence combination
    # Low velocity, low diversity, low drift = stagnation = high restart confidence
    
    # Fitness stagnation score (0 = stagnant, 1 = improving)
    vel_range = max(1e-6, np.percentile(np.abs(self._fitness_velocity_history), 90)) if self._fitness_velocity_history else 1e-6
    fitness_stagnation = 1.0 - np.clip(np.abs(velocity_ema) / vel_range, 0.0, 1.0)
    
    # Diversity collapse score (0 = collapsed, 1 = healthy)
    div_range = max(1e-6, self._diversity_ema * 2) if self._diversity_ema > 0 else 1.0
    diversity_collapse = 1.0 - np.clip(current_diversity / div_range, 0.0, 1.0)
    
    # Centroid freeze score (0 = frozen, 1 = moving)
    drift_range = max(1e-6, centroid_drift_velocity * 3) if centroid_drift_velocity > 0 else 1.0
    centroid_freeze = 1.0 - np.clip(centroid_drift_velocity / drift_range, 0.0, 1.0)
    
    # Confidence-weighted combination with temporal decay memory
    self._restart_confidence = (0.5 * self._restart_confidence + 
                                 0.5 * (0.5 * fitness_stagnation + 
                                        0.3 * diversity_collapse + 
                                        0.2 * centroid_freeze))
    
    # Decision threshold with hysteresis
    restart_threshold = 0.75
    min_stagnation_gens = 8
    
    # Track consecutive high-confidence stagnation generations
    if not hasattr(self, '_high_conf_stagnation_count'):
        self._high_conf_stagnation_count = 0
    
    if self._restart_confidence > restart_threshold:
        self._high_conf_stagnation_count += 1
    else:
        self._high_conf_stagnation_count = max(0, self._high_conf_stagnation_count - 1)
    
    should_restart = (self._high_conf_stagnation_count >= min_stagnation_gens)
    
    if should_restart:
        # Generate new population with temporal awareness
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        
        # Adaptive spread based on how long we've been stagnant
        spread_factor = np.clip(50.0 + self._high_conf_stagnation_count * 5.0, 50.0, 200.0)
        
        new_pop = np.random.randn(self.NP, self.dim) * spread_factor + best_solution
        new_pop = np.clip(new_pop, -100.0, 100.0)
        
        # Inject best solution to preserve progress
        new_pop[0] = best_solution
        np.random.shuffle(new_pop)
        new_pop[0] = best_solution
        
        # Reset temporal tracking
        self._fitness_velocity_history = []
        self._diversity_ema = None
        self._diversity_velocity_history = []
        self._centroid_history = []
        self._restart_confidence = 0.0
        self._high_conf_stagnation_count = 0
        
        return (new_pop, best_fit, best_solution)
    
    return (None, None, None)
```