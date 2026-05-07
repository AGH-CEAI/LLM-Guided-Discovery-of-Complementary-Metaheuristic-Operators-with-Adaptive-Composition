Looking at the worst-performing tasks (16, 17, 11, 20, 23 with errors 10-100+), these are likely suffering from **deceptive local optima trapping** and/or **severe ill-conditioning**. The current variant_08 uses multiple evolution paths but doesn't explicitly detect and escape from deceptive basins.

**Idea: Global Archive Escape with Condition-Aware Scaling**
A fundamentally different strategy that explicitly maintains a global exploration path tracking the best-ever directions, detects trapping via fitness stagnation and eigenspread collapse, and injects structured perturbations toward archive bests with adaptive covariance scaling based on condition number.

```python
def _adapt_covariance_variant_08(self):
    """Global archive escape with condition-aware covariance scaling."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Detect trapping: fitness stagnation + eigenspread collapse
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    
    is_trapped = (rel_var < 1e-3) and (eig_spread < 1e-4)
    
    # Initialize global exploration path
    if not hasattr(self, 'p_global'):
        self.p_global = np.zeros(self.dim)
    
    # Global path: tracks direction toward global best
    if not hasattr(self, 'x_global_best'):
        self.x_global_best = self.x_opt.copy()
        self.global_stagnation = 0
    else:
        if np.linalg.norm(self.x_opt - self.x_global_best) > 1e-8 * (self.ub[0] - self.lb[0]):
            self.x_global_best = self.x_opt.copy()
            self.global_stagnation = 0
        else:
            self.global_stagnation += 1
    
    # Update global path
    cg = 0.02
    to_global = self.x_global_best - self.mean
    norm_to = np.linalg.norm(to_global)
    if norm_to > 1e-10:
        to_global /= norm_to
        self.p_global = (1.0 - cg) * self.p_global + np.sqrt(cg * (2.0 - cg)) * to_global
    
    # Adaptive learning rates based on trap detection
    if is_trapped:
        ccov_1 = min(0.5, 5.0 * self.ccov)
        ccov_mu = min(0.3, 5.0 * self.ccov)
        cc_adapt = min(0.3, 2.0 * self.cc)
    else:
        ccov_1 = self.ccov
        ccov_mu = self.ccov
        cc_adapt = self.cc
    
    # Primary evolution path
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Global path rank-one update
    rank_global = np.outer(self.p_global, self.p_global)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    total_w = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_w += abs(w)
    if total_w > 0:
        rank_mu /= total_w
    
    # Combine updates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * (rank_one + 0.3 * rank_global) +
              ccov_mu * rank_mu)
    
    # Inject structured perturbation when trapped
    if is_trapped:
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.maximum(np.min(eigvals), 1e-10)
        if cond > 1e5:
            self.C *= 0.8
        self.C += 0.1 * np.eye(self.dim) * np.mean(eigvals)
        self.p_global *= 0.5
    
    self.C = self._ensure_positive_definite(self.C)
```