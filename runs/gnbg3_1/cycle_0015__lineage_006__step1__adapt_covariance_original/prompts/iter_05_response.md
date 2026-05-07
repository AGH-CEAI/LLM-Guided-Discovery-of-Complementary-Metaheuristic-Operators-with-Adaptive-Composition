**Idea: Covariance Matrix Restart on Stagnation**

A fundamentally different strategy: when the algorithm stagnates (no improvement for extended generations) OR when the covariance matrix becomes extremely ill-conditioned, perform a targeted restart by reinitializing the covariance matrix to identity while preserving the best solution found. This breaks the "frozen" exploration ellipse that traps CMA-ES in local optima, allowing fresh global exploration. Unlike archive-guided perturbations (which add noise), this resets the covariance structure entirely to escape deep local basins.

```python
def _adapt_covariance_original(self):
    """Original covariance adaptation with restart on stagnation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Check for stagnation and trigger covariance restart
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    
    # Restart conditions: extreme ill-conditioning OR stagnation
    should_restart = False
    restart_reason = ""
    
    # Condition-based restart: matrix too elongated
    if cond > 1e6:
        should_restart = True
        restart_reason = "ill_conditioned"
    
    # Stagnation-based restart: best not improving
    if hasattr(self, 'stagnation_counter') and self.stagnation_counter > self.max_stagnation:
        should_restart = True
        restart_reason = "stagnation"
    
    # Spectrum collapse: all eigenvalues similar but tiny
    if eig_max > 1e-10 and eig_min / eig_max < 1e-8:
        should_restart = True
        restart_reason = "spectrum_collapse"
    
    if should_restart:
        # Reinitialize covariance to identity (spherical exploration)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        # Increase step size for wider exploration after restart
        self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.1)
        self.sigma = max(self.sigma, 1e-6)
```