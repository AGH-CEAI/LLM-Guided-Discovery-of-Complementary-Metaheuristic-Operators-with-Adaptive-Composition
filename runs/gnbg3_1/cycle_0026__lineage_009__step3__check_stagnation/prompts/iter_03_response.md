Looking at the priority unsolved tasks (errors ~1e+1 to ~1e+2), the algorithm is getting stuck in local optima. The current stagnation check only tracks single-generation improvement, which is too sensitive to noise and misses long-term trapping patterns. I need a fundamentally different approach that tracks **improvement momentum** over a window and also monitors **covariance ill-conditioning** as an independent stagnation signal.

**Idea: Momentum + Condition-Based Stagnation**
Multi-scale improvement tracking with covariance health monitoring — tracks relative improvement over sliding windows and uses eigenvalue condition number to detect search space degeneracy independently of fitness stagnation.
```python
def _check_stagnation(self):
    """Track stagnation using momentum-based improvement and covariance condition monitoring."""
    if not hasattr(self, 'stagnation_momentum'):
        self.stagnation_momentum = 0.0
        self.fit_history = []
        self.cond_history = []
    
    # Track fitness history for momentum calculation
    self.fit_history.append(self.f_opt)
    if len(self.fit_history) > 20:
        self.fit_history.pop(0)
    
    # Track covariance condition number history
    eigvals = np.linalg.eigvalsh(self.C)
    eigvals = np.clip(eigvals, 1e-15, None)
    cond = np.max(eigvals) / np.min(eigvals)
    self.cond_history.append(cond)
    if len(self.cond_history) > 20:
        self.cond_history.pop(0)
    
    # Compute relative improvement momentum
    if len(self.fit_history) >= 5:
        recent_window = self.fit_history[-5:]
        early_window = self.fit_history[:5] if len(self.fit_history) >= 5 else self.fit_history
        
        # Relative improvement: how much did fitness improve relative to scale?
        scale = max(abs(self.f_opt), 1.0, abs(self.fit_history[0]))
        recent_improvement = (np.mean(early_window) - np.mean(recent_window)) / scale
        recent_improvement = max(recent_improvement, 0.0)
        
        # Momentum: EMA of improvement rate
        self.stagnation_momentum = 0.7 * self.stagnation_momentum + 0.3 * recent_improvement
    else:
        self.stagnation_momentum = 0.0
    
    # Stagnation detected if: no momentum AND condition is problematic
    momentum_threshold = 1e-6
    cond_healthy = np.mean(self.cond_history) < 1e7 if self.cond_history else True
    
    if self.stagnation_momentum < momentum_threshold and not cond_healthy:
        self.stagnation_counter += 3  # Accelerated stagnation on ill-conditioned state
    elif self.stagnation_momentum < momentum_threshold:
        self.stagnation_counter += 1
    else:
        self.stagnation_counter = max(0, self.stagnation_counter - 1)
    
    self.f_opt_prev = self.f_opt
```