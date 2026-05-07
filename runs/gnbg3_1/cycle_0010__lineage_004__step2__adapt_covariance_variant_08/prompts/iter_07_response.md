Looking at the per-task errors, I see a clear pattern: the WORST unsolved tasks (16, 17, 11, 20, 23) have errors in the 10-100 range, indicating the optimizer is getting **stuck in local optima** and cannot escape. The current variant_08 uses multiple evolution paths but lacks any mechanism to break out of these traps.

**Key insight**: All 24 tasks remain unsolved because CMA-ES variants converge too quickly to local basins. The highest-error tasks need a **restart-driven covariance reset** that detects stagnation and aggressively resets the covariance matrix + expands step size to force escape.

**Why this is fundamentally different**: variant_08 passively maintains multiple evolution paths without any restart trigger. This proposal actively monitors fitness history, detects stagnation, and forces a complete covariance reset with 3x step size expansion — a fundamentally different "escape local optima" mechanism.

**Idea: Restart-Triggered Covariance Reset**
Monitors recent fitness history for stagnation; when stuck, resets covariance to identity and expands step size 3x to escape local optima.
```python
def _adapt_covariance_variant_08(self):
    """Restart-triggered covariance reset for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Monitor fitness history for stagnation detection
    if not hasattr(self, 'restart_fitness_history'):
        self.restart_fitness_history = []
    self.restart_fitness_history.append(self.f_opt)
    max_history = max(20, self.dim * 2)
    if len(self.restart_fitness_history) > max_history:
        self.restart_fitness_history.pop(0)
    
    should_restart = False
    if len(self.restart_fitness_history) >= max_history:
        fit_range = max(self.restart_fitness_history) - min(self.restart_fitness_history)
        fit_scale = max(abs(self.f_opt), 1.0, np.median(self.restart_fitness_history))
        if fit_range < 1e-6 * fit_scale:
            should_restart = True
    
    # Also restart on extreme stagnation counter
    if self.stagnation_counter > self.max_stagnation // 2:
        should_restart = True
    
    if should_restart:
        # Full covariance reset to escape local optima
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        
        # Aggressive step size expansion
        bound_range = max(self.ub[0] - self.lb[0], 1.0)
        self.sigma = min(self.sigma * 3.0, 0.1 * bound_range)
        self.sigma = max(self.sigma, 1e-8)
        
        # Clear history after restart
        self.restart_fitness_history = []
        should_restart = False
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```