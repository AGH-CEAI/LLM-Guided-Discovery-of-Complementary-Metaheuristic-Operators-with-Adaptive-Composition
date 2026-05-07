Looking at the results, all variants are failing on the hardest tasks (errors of 1e+1 to 1e+2 range). The key insight is that these high errors suggest the algorithm is getting trapped in local optima with collapsed covariance matrices, and the existing restart logic only triggers after extended stagnation. By that point, the covariance has already become degenerate.

The solution is **early restart detection within covariance adaptation** — detecting failure signals (premature stagnation, low diversity, ill-conditioned covariance) and resetting the covariance BEFORE the algorithm gets stuck, enabling escape from local optima on hard tasks.

**Idea: Early Restart Covariance Adaptation**
Detects multiple failure modes (stagnation, low diversity, ill-conditioning) and resets covariance early to escape local optima, combined with adaptive learning rates.
```python
def _adapt_covariance(self):
    """Covariance adaptation with early restart on stagnation/diversity/conditioning signals.
    
    Detects early warning signs BEFORE full stagnation and resets covariance to escape
    local optima, especially targeting hard tasks with errors > 1e+1.
    """
    # Compute population diversity
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = pop_variance / (expected_var + 1e-10)
    
    # Compute covariance condition number
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    
    # Early stagnation: trigger restart BEFORE full stagnation threshold
    early_stagnation = self.stagnation_counter > max(15, self.dim // 2)
    
    # Trigger restart on any early warning sign
    if early_stagnation or diversity < 0.05 or cond > 1e4 or eig_min < 1e-7:
        # Reset covariance to identity (exploration restart)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.L = None
    
    # Proceed with CMA-ES update
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rate: boost when population is converging
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    if rel_var < 0.01:
        ccov_scale = 5.0
    elif rel_var < 0.1:
        ccov_scale = 2.0
    else:
        ccov_scale = 1.0
    
    self.C = ((1.0 - self.ccov * ccov_scale) * self.C + 
              self.ccov * ccov_scale * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * ccov_scale * 2.0 * rank_mu))
    
    # Add exploration boost when stuck
    if rel_var < 0.01:
        self.C += 0.1 * np.eye(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
```