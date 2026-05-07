Looking at the unsolved tasks, the worst performers (Tasks 21, 17, 18, 19 with errors ~1-5) are clearly stuck—likely in local optima or slow convergence. The key issue is that the current step size adaptation relies solely on the evolution path norm `p_sigma`, which reflects step consistency but NOT actual fitness improvement. When trapped in local optima, the evolution path can suggest "converged" behavior while fitness stalls.

**Idea: Fitness-Stagnation-Triggered Aggressive Exploration**

This approach adds a direct fitness-improvement feedback loop to sigma adaptation. When recent fitness shows no improvement (stagnation), sigma is boosted exponentially based on stagnation depth. This is fundamentally different from the pure evolution-path approach and should help escape local optima on the worst tasks.

```python
def _adapt_step_size(self):
    # Initialize fitness history tracking
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    if not hasattr(self, '_stagnation_depth'):
        self._stagnation_depth = 0
    
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Track fitness for direct improvement feedback
    self._fitness_history.append(float(self.best_f))
    max_history = 20
    if len(self._fitness_history) > max_history:
        self._fitness_history = self._fitness_history[-max_history:]
    
    # Compute fitness-based stagnation signal
    if len(self._fitness_history) >= 10:
        recent_best = min(self._fitness_history[-5:])
        older_best = min(self._fitness_history[-10:-5]) if len(self._fitness_history) >= 10 else self._fitness_history[0]
        stagnation_detected = (recent_best >= older_best)
    elif len(self._fitness_history) >= 5:
        recent_best = min(self._fitness_history[-5:])
        stagnation_detected = (self._fitness_history[0] <= recent_best)
    else:
        stagnation_detected = False
    
    # Update stagnation depth counter
    if stagnation_detected:
        self._stagnation_depth += 1
    else:
        self._stagnation_depth = max(0, self._stagnation_depth - 1)
    
    # Apply fitness-stagnation triggered exploration boost
    if self._stagnation_depth >= 2:
        # Aggressive sigma increase proportional to stagnation depth
        boost = 1.0 + 0.3 * self._stagnation_depth
        self.sigma *= boost
    else:
        # Standard CMA-ES adaptation with evolution path
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
    
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```