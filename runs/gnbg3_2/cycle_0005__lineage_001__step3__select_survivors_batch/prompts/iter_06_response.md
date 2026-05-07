**Idea: Temporal Regime-Adaptive Selection Pressure**

This variant tracks the temporal dynamics of improvement signals across generations using EMA-based regime detection. When improvement has been stagnating over a time window (detected via slow EMA divergence), selection pressure is relaxed to maintain diversity. When improvement is accelerating, greedy selection is enforced. This directly addresses the worst unsolved tasks (17, 16, 11, 6) where the algorithm appears stuck in local optima due to premature selection pressure convergence.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Temporal regime-adaptive selection with EMA-based stagnation detection."""
    improved_mask = trial_fitness < fitness
    
    # Initialize temporal state on first call
    if not hasattr(self, 'gen_improvement_ema'):
        self.gen_improvement_ema = np.zeros(self.np)  # Per-individual EMA of improvement
        self.pop_improvement_ema_short = 0.0
        self.pop_improvement_long = 0.0
        self.temporal_regime = 'neutral'  # 'improving', 'stagnant', 'declining'
        self.regime_patience = 0
        self.stagnation_window = []
    
    # Compute per-individual improvement magnitude (negative = improvement)
    improvement_mag = trial_fitness - fitness
    improvement_mag = np.where(improved_mask, improvement_mag, 0.0)
    
    # Update per-individual EMA of improvement magnitude
    alpha_indiv = 0.3
    self.gen_improvement_ema = alpha_indiv * improvement_mag + (1 - alpha_indiv) * self.gen_improvement_ema
    
    # Compute population-level improvement signal
    pop_improvement = np.sum(np.where(improved_mask, fitness - trial_fitness, 0.0))
    
    # Update population EMAs with different time constants
    alpha_short = 0.4
    alpha_long = 0.1
    self.pop_improvement_ema_short = alpha_short * pop_improvement + (1 - alpha_short) * self.pop_improvement_ema_short
    self.pop_improvement_long = alpha_long * pop_improvement + (1 - alpha_long) * self.pop_improvement_long
    
    # Maintain rolling window of improvement signals
    self.stagnation_window.append(np.mean(improved_mask))
    if len(self.stagnation_window) > 12:
        self.stagnation_window.pop(0)
    
    # Temporal regime detection via EMA divergence and window statistics
    window_mean = np.mean(self.stagnation_window)
    ema_trend = self.pop_improvement_ema_short - self.pop_improvement_long
    
    if ema_trend > 0.01 and window_mean > 0.25:
        self.temporal_regime = 'improving'
        self.regime_patience = 0
    elif ema_trend < -0.005 or window_mean < 0.15:
        self.regime_patience += 1
        if self.regime_patience > 5:
            self.temporal_regime = 'stagnant'
    else:
        self.temporal_regime = 'neutral'
        self.regime_patience = max(0, self.regime_patience - 1)
    
    # Compute temporal fitness: blend current fitness with EMA-based history
    # Individuals that consistently improve get a "temporal bonus" (lower effective fitness)
    temporal_bonus = 0.1 * self.gen_improvement_ema  # Positive EMA = consistent improver
    effective_fitness = fitness - temporal_bonus
    effective_trial_fitness = trial_fitness.copy()
    
    # Determine selection pressure based on temporal regime
    if self.temporal_regime == 'improving':
        # High selection pressure: trust current generation's greedy signal
        selection_mask = trial_fitness < fitness
    elif self.temporal_regime == 'stagnant':
        # Low selection pressure: use temporal fitness, favor consistent improvers
        # Allow some non-improving trials if they have good temporal history
        temporal_improvement_mask = effective_trial_fitness < effective_fitness
        # Also accept trials that improve individuals with poor temporal history
        poor_temporal_mask = self.gen_improvement_ema < np.percentile(self.gen_improvement_ema, 30)
        selection_mask = temporal_improvement_mask | (improved_mask & poor_temporal_mask)
    else:
        # Neutral: standard greedy with slight temporal weighting
        selection_mask = trial_fitness < fitness
    
    # Apply selection
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    new_population[selection_mask] = trials[selection_mask]
    new_fitness[selection_mask] = trial_fitness[selection_mask]
    
    return new_population, new_fitness, improved_mask
```