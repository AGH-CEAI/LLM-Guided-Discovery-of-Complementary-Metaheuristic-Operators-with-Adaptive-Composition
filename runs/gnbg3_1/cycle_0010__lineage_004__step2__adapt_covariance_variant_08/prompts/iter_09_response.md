**Idea: Condition-Aware Covariance with Stagnation Detection**
A fundamentally different approach that monitors improvement velocity and uses condition-number-adaptive learning rates to handle ill-conditioned problems, with explicit stagnation detection triggering aggressive covariance perturbation.
```python
def _adapt_covariance_variant_08(self):
    """Condition-aware covariance adaptation with stagnation-triggered exploration bursts."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Track improvement velocity
    if not hasattr(self, 'improvement_history'):
        self.improvement_history = []
    current_improvement = max(0.0, self.f_opt_prev - self.f_opt)
    self.improvement_history.append(current_improvement)
    if len(self.improvement_history) > 5:
        self.improvement_history.pop(0)
    avg_improvement = np.mean(self.improvement_history) if self.improvement_history else 1e-15
    
    # Detect stagnation: no meaningful improvement in recent generations
    is_stagnant = (avg_improvement < 1e-8 and len(self.improvement_history) >= 5)
    
    # Compute condition number of current covariance
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.maximum(np.min(eigvals), 1e-15)
    eig_max = np.max(eigvals)
    cond = eig_max / eig_min
    
    # Condition-aware learning rate: reduce update speed for ill-conditioned problems
    cond_factor = np.clip(np.log10(cond + 1.0) / 10.0, 0.01, 1.0)
    
    # Stagnation trigger: aggressive exploration burst
    if is_stagnant:
        # Reset improvement history after triggering
        self.improvement_history = []
        
        # Add exploration burst to covariance
        burst_strength = np.clip(0.5 * np.log10(cond + 1.0), 0.1, 2.0)
        self.C = (1.0 - burst_strength) * self.C + burst_strength * np.eye(self.dim)
        
        # Also reset evolution path to allow new direction
        self.pc = np.zeros(self.dim)
    
    # Adaptive learning rates based on condition number
    cc_adapt = np.clip(self.cc * cond_factor, 0.001, 0.3)
    ccov_adapt = np.clip(self.ccov * cond_factor, 1e-10, 0.5)
    
    # Evolution path update with condition-aware rate
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine updates with condition-aware weighting
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * rank_one +
              (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
    
    # Additional diversity injection for ill-conditioned problems
    if cond > 1e6:
        self.C += 0.01 * np.eye(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Condition-based covariance reduction to prevent numerical issues
    eigvals_check = np.linalg.eigvalsh(self.C)
    cond_check = np.max(eigvals_check) / np.maximum(np.min(eigvals_check), 1e-15)
    if cond_check > 1e7:
        self.C *= 0.5
```