**Idea: Success-Rule Step Size Adaptation**
Uses a success-rate based rule inspired by the 1/5th success rule from classical evolution strategies. Tracks whether recent selected individuals improved fitness and adjusts sigma accordingly, fundamentally different from the evolution-path based approach.
```python
def _adapt_step_size(self):
    # Track success rate-based adaptation (fundamentally different from p_sigma path)
    success_window = max(3, self.mu // 2)
    
    # Use recent fitness history if available
    if not hasattr(self, '_recent_successes'):
        self._recent_successes = []
    
    # Simple success heuristic: compare median of selected vs full population
    # This is a direct success-ratio approach, not using evolution paths
    if hasattr(self, '_last_median_fit') and np.isfinite(self._last_median_fit):
        valid_fitness = self.fitness_buffer if hasattr(self, 'fitness_buffer') else np.array([np.inf])
        current_median = np.median(valid_fitness[np.isfinite(valid_fitness)]) if np.any(np.isfinite(valid_fitness)) else self._last_median_fit
        
        # Success ratio: how often do we see improvement?
        if self._last_median_fit > 0:
            ratio = current_median / (self._last_median_fit + 1e-20)
        elif current_median < 0:
            ratio = 1.0 - (current_median / (self._last_best_fit + 1e-20))
        else:
            ratio = 1.0
        
        # Success rate based on relative fitness change
        if ratio < 0.99:  # Improvement detected
            self._recent_successes.append(1)
        else:
            self._recent_successes.append(0)
        
        # Keep window bounded
        if len(self._recent_successes) > success_window:
            self._recent_successes.pop(0)
        
        # Compute success rate
        if len(self._recent_successes) >= success_window:
            success_rate = np.mean(self._recent_successes)
            # Target success rate ~1/5 for CMA-ES like behavior
            target_success = 0.20
            # Adaptation: increase sigma when success rate is high
            adaptation = (success_rate - target_success) / target_success
            self.sigma *= np.exp(0.1 * np.clip(adaptation, -2.0, 2.0))
        else:
            # Insufficient data, use stagnation-based heuristic
            if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 5:
                self.sigma *= 1.5  # Increase diversity when stagnant
    else:
        self._recent_successes = []
    
    # Store for next iteration
    valid_fit = self.fitness_buffer if hasattr(self, 'fitness_buffer') and len(self.fitness_buffer) > 0 else np.array([np.inf])
    if np.any(np.isfinite(valid_fit)):
        self._last_median_fit = float(np.median(valid_fit[np.isfinite(valid_fit)]))
        self._last_best_fit = float(np.min(valid_fit[np.isfinite(valid_fit)]))
    
    # Clip to safe bounds
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```