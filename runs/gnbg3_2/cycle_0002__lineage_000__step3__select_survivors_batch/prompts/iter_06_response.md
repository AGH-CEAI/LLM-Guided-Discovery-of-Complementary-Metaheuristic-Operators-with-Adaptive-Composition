**Idea: EMA Momentum Survivor Selection**
Use exponential moving averages of fitness and improvement velocity tracking to make selection decisions that favor consistent improvers and prevent premature convergence on hard tasks.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Temporal-aware selection using EMA of fitness and rate-of-improvement tracking."""
    
    # Initialize temporal state on first call
    if not hasattr(self, 'fitness_ema'):
        self.fitness_ema = fitness.copy()
        self.fitness_ema_alpha = 0.3
        self.prev_fitness = fitness.copy()
        self.improvement_streak = np.zeros(len(fitness))
        self.generation_age = np.zeros(len(fitness))
    
    # Track generation age (survival time in population)
    self.generation_age += 1
    
    # Compute rate of change (velocity) of current fitness
    fitness_velocity = fitness - self.prev_fitness  # Negative means improving
    
    # Adaptive EMA alpha: consistently improving individuals get more responsive EMA
    improvement_boost = np.clip(self.improvement_streak / 10.0, 0, 0.2)
    adaptive_alpha = self.fitness_ema_alpha - improvement_boost
    adaptive_alpha = np.clip(adaptive_alpha, 0.1, 0.5)
    
    # Update EMA of current fitness (per-individual)
    new_ema = np.zeros_like(fitness)
    for i in range(len(fitness)):
        new_ema[i] = adaptive_alpha[i] * fitness[i] + (1 - adaptive_alpha[i]) * self.fitness_ema[i]
    
    # Evaluate trial against EMA (smoothed reference) rather than raw fitness
    ema_improved_mask = trial_fitness < self.fitness_ema
    
    # Compute projected EMA for trials
    trial_ema = np.array([adaptive_alpha[i] * trial_fitness[i] + 
                          (1 - adaptive_alpha[i]) * self.fitness_ema[i] 
                          for i in range(len(trial_fitness))])
    
    # Velocity bonus: reward individuals showing consistent improvement trend
    # Negative velocity = fitness decreasing = improving
    velocity_bonus = np.zeros(len(fitness))
    improving_mask = fitness_velocity < 0
    velocity_bonus[improving_mask] = 0.1 * np.abs(fitness_velocity[improving_mask])
    velocity_bonus = np.clip(velocity_bonus, 0, 0.5)
    
    # Age penalty: reduce selection pressure for older individuals to maintain diversity
    age_penalty = np.clip(self.generation_age / 50.0, 0, 0.3)
    
    # Combined selection: EMA comparison with velocity bonus and age penalty
    combined_score_current = self.fitness_ema - velocity_bonus + age_penalty
    combined_score_trial = trial_ema
    
    temporal_select_mask = combined_score_trial < combined_score_current
    
    # Fallback: also accept if raw fitness significantly improves (ensures no regression)
    raw_improved_mask = trial_fitness < fitness
    
    # Combined decision: accept if temporal trend improves OR raw fitness strictly improves
    accept_mask = temporal_select_mask | raw_improved_mask
    
    # Build new population
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    # Update temporal state for accepted individuals
    self.fitness_ema[accept_mask] = trial_ema[accept_mask]
    self.improvement_streak[accept_mask] += 1
    self.improvement_streak[~accept_mask] = np.maximum(0, self.improvement_streak[~accept_mask] - 1)
    self.generation_age[accept_mask] = 0  # Reset age on acceptance
    
    # Update previous fitness for next generation
    self.prev_fitness = fitness.copy()
    
    return new_population, new_fitness, accept_mask
```