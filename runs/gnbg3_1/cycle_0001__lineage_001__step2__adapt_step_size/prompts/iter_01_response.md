**Idea: Success-Rate Step Size Adaptation**
A fundamentally different approach using offspring success rate instead of evolution path cumulation. Tracks how many offspring improve over parents and adjusts sigma proportionally.
```python
def _adapt_step_size(self):
    """Adapt step-size using success rate (alternative to CMA-ES CSA)."""
    # Count successful offspring (better than best parent)
    parent_best = min(self.fitness)
    successes = np.sum(self.trial_fitness < parent_best)
    success_rate = successes / max(1, self.NP)
    
    # Maintain smoothed success history for stability
    if not hasattr(self, 'success_history'):
        self.success_history = []
    self.success_history.append(success_rate)
    if len(self.success_history) > 5 + self.dim // 10:
        self.success_history.pop(0)
    
    # Use exponential moving average for smoothing
    smoothed_success = np.mean(self.success_history) if self.success_history else 0.5
    
    # Target success rate ~1/5 (CMA-ES default for (mu/w, lambda))
    target_rate = 0.2
    success_ratio = smoothed_success / max(target_rate, 1e-10)
    
    # Adapt sigma: increase when too many successes, decrease when too few
    # Use smaller learning rate for stability
    adapt_rate = 0.1 / (1.0 + np.sqrt(self.dim / self.NP))
    
    # Exponential update with bounds
    factor = np.exp(adapt_rate * (success_ratio - 1.0))
    self.sigma *= factor
    
    # Ensure reasonable bounds
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```