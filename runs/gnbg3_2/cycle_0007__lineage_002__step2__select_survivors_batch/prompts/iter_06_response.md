**Idea: Temporal Stagnation-Responsive Selection**

Use exponential moving averages of improvement magnitude and acceptance rate across generations to dynamically modulate selection pressure. When EMA signals indicate stagnation (low acceptance, declining momentum), reduce pressure to accept more exploratory trials; when momentum is positive, tighten pressure for exploitation. This directly targets the severe stagnation causing 10^2–10^5 errors on tasks 17, 16, 11, 19, 6.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Temporal selection: EMA-based stagnation detection with adaptive selection pressure."""
    improved_mask = trial_fitness < fitness
    improvement = fitness - trial_fitness
    
    # Initialize temporal state
    if not hasattr(self, '_temp_improvement_ema'):
        self._temp_improvement_ema = 0.0
        self._temp_acceptance_ema = 0.0
        self._temp_stagnation_counter = 0
        self._temp_prev_fitness = np.mean(fitness)
        self._temp_momentum = 0.0
    
    # Current generation metrics
    current_improvement = np.sum(improvement[improved_mask]) / max(np.sum(improved_mask), 1)
    current_acceptance = np.mean(improved_mask)
    
    # Update EMAs with different smoothing for short-term vs long-term trends
    alpha_fast, alpha_slow = 0.4, 0.15
    self._temp_improvement_ema = alpha_fast * current_improvement + (1 - alpha_fast) * self._temp_improvement_ema
    self._temp_acceptance_ema = alpha_slow * current_acceptance + (1 - alpha_slow) * self._temp_acceptance_ema
    
    # Stagnation detection: count generations without ANY improvement
    if np.sum(improved_mask) == 0:
        self._temp_stagnation_counter += 1
    else:
        self._temp_stagnation_counter = max(0, self._temp_stagnation_counter - 2)
    
    # Momentum: detect rate-of-change of improvement (improving or degrading?)
    if current_improvement > self._temp_improvement_ema:
        self._temp_momentum = 0.3 * self._temp_momentum + 0.7  # positive momentum
    else:
        self._temp_momentum = 0.3 * self._temp_momentum - 0.3  # negative momentum
    self._temp_momentum = np.clip(self._temp_momentum, -1.0, 1.0)
    
    # Fitness drift: track if population mean is moving (good) or stuck (bad)
    current_mean_fit = np.mean(fitness)
    fitness_drift = abs(current_mean_fit - self._temp_prev_fitness)
    self._temp_prev_fitness = current_mean_fit
    
    # Compute adaptive selection pressure based on temporal signals
    stagnation_penalty = min(self._temp_stagnation_counter / 20.0, 1.0)
    acceptance_bonus = self._temp_acceptance_ema  # higher acceptance → more confident
    momentum_bonus = self._temp_momentum  # positive = improving, negative = degrading
    
    # Pressure factor: lower when stagnant, higher when improving
    base_pressure = 1.0
    pressure_adjustment = 0.5 * stagnation_penalty * (-1.0) + 0.3 * momentum_bonus + 0.2 * (acceptance_bonus - 0.3)
    selection_pressure = base_pressure + pressure_adjustment
    selection_pressure = np.clip(selection_pressure, 0.1, 2.0)
    
    # Compute acceptance threshold using temporal improvement baseline
    fit_range = np.ptp(fitness)
    fit_range = max(fit_range, 1e-10)
    
    # EMA-based threshold: when EMA is high, be selective; when low (stagnation), be permissive
    ema_normalized = self._temp_improvement_ema / max(fit_range, 1e-10)
    base_threshold = 0.005 / selection_pressure
    temporal_threshold = base_threshold * np.exp(-2.0 * ema_normalized)
    temporal_threshold = np.clip(temporal_threshold, 1e-6, 0.5)
    
    # Scaled improvement
    scaled_improvement = improvement / fit_range
    
    # Accept if improved OR if scaled improvement exceeds temporal threshold
    accept_mask = improved_mask | (scaled_improvement > temporal_threshold)
    
    # Build new population
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    return new_population, new_fitness, accept_mask
```