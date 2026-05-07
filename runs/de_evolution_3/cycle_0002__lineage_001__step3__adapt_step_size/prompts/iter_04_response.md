**Idea: Fitness-Progress Step Size Adaptation**

Replace the evolution-path-based sigma adaptation with a fitness-progress-driven approach that increases sigma when the algorithm stalls in local optima (the dominant failure mode for the worst unsolved tasks with errors > 0.5) and decreases sigma when fitness is consistently improving.

```python
def _adapt_step_size(self):
    # Fitness-progress-based step size adaptation
    # Fundamentally different from evolution-path-based CSA:
    # - Increases sigma when fitness stagnates (escape local optima)
    # - Decreases sigma when fitness improves (fine-tune convergence)
    # This directly addresses the worst tasks (21,17,18,19,22,20,15,14)
    # which are stuck at errors > 0.5, indicating local-optimum trapping
    
    if not hasattr(self, '_prev_best_f_for_sigma'):
        self._prev_best_f_for_sigma = float('inf')
        self._sigma_adjustment_count = 0
    
    prev_f = self._prev_best_f_for_sigma
    curr_f = self.best_f
    
    # Compute relative fitness progress
    if prev_f > 0 and np.isfinite(curr_f):
        progress_ratio = (prev_f - curr_f) / (abs(prev_f) + 1e-20)
    elif prev_f <= 0 and np.isfinite(curr_f):
        progress_ratio = curr_f - prev_f if curr_f < prev_f else 0.0
    else:
        progress_ratio = 0.0
    
    # Progress ratio > 0 means improvement, < 0 means stagnation/worsening
    # Target: exponential decay toward progress_ratio = 0.05 (5% improvement target)
    target_progress = 0.05
    progress_error = progress_ratio - target_progress
    
    # Adaptation: positive error -> decrease sigma, negative error -> increase sigma
    adapt_rate = 0.2  # moderate adaptation rate for stability
    self.sigma *= np.exp(-adapt_rate * progress_error)
    
    # Update tracking
    self._prev_best_f_for_sigma = float(curr_f)
    self._sigma_adjustment_count += 1
    
    # Reset if we had a significant improvement
    if progress_ratio > 0.1:
        self._sigma_adjustment_count = 0
    
    # Clamp to numerical bounds
    self.sigma = float(np.clip(self.sigma, 1e-20, 1e6))
```