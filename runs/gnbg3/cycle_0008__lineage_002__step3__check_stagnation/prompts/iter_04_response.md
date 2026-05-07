Looking at the context, I need to implement `_check_stagnation` using Category D (Fitness-landscape / rank-based) signals. The key is to detect stagnation using only fitness signals — rank correlations, success history, and fitness percentiles — without any spatial/distance computations.

**Idea: Spearman Rank Correlation + Fitness Spread**
Track rank stability via Spearman correlation between consecutive generations and monitor fitness spread. High rank correlation (>0.9) indicates stagnation; narrow fitness spread indicates convergence. This is fundamentally different from the geometric approaches used in previous variants.

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation using fitness-landscape / rank-based signals.
    
    Signals used:
    - Spearman rank correlation between consecutive generations
    - Fitness percentile spread (10th-90th percentile)
    - Consecutive generation improvement tracking
    """
    # Initialize tracking attributes
    if not hasattr(self, '_stagnation_counter'):
        self._stagnation_counter = 0
        self._fitness_history = []
        self._prev_pop_fitness = None
    
    # Track best fitness history (F: temporal/dynamical)
    self._fitness_history.append(float(best_fitness))
    max_history = 20
    if len(self._fitness_history) > max_history:
        self._fitness_history.pop(0)
    
    # D: Rank-based stagnation detection using Spearman correlation
    stagnation_detected = False
    
    # Check if population fitness is available for rank analysis
    if hasattr(self, '_current_population_fitness') and self._current_population_fitness is not None:
        pop_fitness = np.asarray(self._current_population_fitness)
        
        # Filter out NaN values
        valid_mask = ~np.isnan(pop_fitness)
        if valid_mask.sum() >= 5:
            pop_fitness = pop_fitness[valid_mask]
            
            if self._prev_pop_fitness is not None and len(self._prev_pop_fitness) == len(pop_fitness):
                prev_valid = ~np.isnan(self._prev_pop_fitness)
                if prev_valid.sum() == len(pop_fitness):
                    # D: Spearman rank correlation (rank-based)
                    try:
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(self._prev_pop_fitness, pop_fitness)
                        # High correlation = stable ranks = stagnation
                        if corr is not None and corr > 0.9:
                            stagnation_detected = True
                    except (ValueError, ImportError):
                        # Manual Spearman fallback
                        pass
            
            self._prev_pop_fitness = pop_fitness.copy()
            
            # D: Fitness percentile spread (fitness-landscape based)
            # Narrow spread indicates population has converged
            p10, p90 = np.percentile(pop_fitness, [10, 90])
            spread = p90 - p10
            mean_fitness = np.mean(pop_fitness)
            if mean_fitness != 0:
                rel_spread = spread / (abs(mean_fitness) + 1e-10)
            else:
                rel_spread = spread
            
            if rel_spread < 1e-8:
                stagnation_detected = True
    else:
        # Fallback: best fitness trajectory analysis
        if len(self._fitness_history) >= 3:
            recent = self._fitness_history[-3:]
            improvement = max(recent) - min(recent)
            if improvement < 1e-10:
                stagnation_detected = True
    
    # Update stagnation counter (F: temporal tracking over generations)
    if stagnation_detected:
        self._stagnation_counter += 1
    else:
        # Reset on detected progress
        if len(self._fitness_history) >= 2:
            if self._fitness_history[-1] < self._fitness_history[-2] - 1e-12:
                self._stagnation_counter = 0
    
    # Threshold for stagnation detection
    stagnation_threshold = 15
    return self._stagnation_counter >= stagnation_threshold
```