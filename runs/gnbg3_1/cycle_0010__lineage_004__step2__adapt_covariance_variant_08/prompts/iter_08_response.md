**Idea: Eigenvalue-Balanced Restart Escape**
Covariance adaptation using eigenvalue monitoring with aggressive condition balancing and restart-triggered exploration when the covariance matrix becomes too elongated, specifically targeting the severely multimodal/deceptive tasks (16, 17, 11, 20, 23) that get trapped in local optima with error ~1e+1 to 1e+2.

```python
def _adapt_covariance_variant_08(self):
    """Eigenvalue-balanced restart escape for severe multimodality."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    total_w = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_w += abs(w)
    if total_w > 0:
        rank_mu /= total_w
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-15)
    eig_spread = eig_min / (eig_max + 1e-15)
    
    # Detect stagnation: best hasn't improved
    is_stagnant = hasattr(self, 'stagnation_counter') and self.stagnation_counter > self.dim * 2
    
    # Detect severe ill-conditioning or collapsed spectrum
    is_ill_conditioned = cond > 1e6 or eig_spread < 1e-6
    
    # Detect convergence trap: low diversity + high condition
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    is_converged_trap = rel_var < 1e-3 and cond > 1e4
    
    needs_escape = is_stagnant or is_ill_conditioned or is_converged_trap
    
    if needs_escape:
        # Aggressively rebalance eigenvalues: shrink large, boost small
        target_cond = min(cond, 1e4)
        eigvals_balanced = np.exp(np.linspace(np.log(eig_min), np.log(eig_min * target_cond), self.dim))
        eigvals_balanced = np.clip(eigvals_balanced, 1e-10, None)
        
        # Diagonalize and rescale
        Q = np.linalg.eig(self.C)[1]
        self.C = Q @ np.diag(eigvals_balanced) @ Q.T
        self.C = 0.5 * (self.C + self.C.T)
        
        # Inject exploration along principal axes
        self.C += 0.2 * np.mean(eigvals) * np.eye(self.dim)
        
        # Boost learning rates temporarily
        ccov_escape = min(self.ccov * 5.0, 0.4)
        cc_escape = min(self.cc * 3.0, 0.25)
    else:
        ccov_escape = self.ccov
        cc_escape = self.cc
    
    # Recompute pc with escaped learning rate if needed
    if needs_escape:
        self.pc = (1.0 - cc_escape) * self.pc + np.sqrt(cc_escape * (2.0 - cc_escape)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
    
    self.C = ((1.0 - ccov_escape) * self.C + 
              ccov_escape * rank_one + 
              (1.0 - 1.0 / self.mueff) * ccov_escape * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Hard reset on extreme conditions
    eigvals_final = np.linalg.eigvalsh(self.C)
    cond_final = np.max(eigvals_final) / np.min(eigvals_final)
    if cond_final > 1e8 or np.any(np.isnan(self.C)):
        self.C = np.eye(self.dim) * np.mean(eigvals_final)
        self.pc = np.zeros(self.dim)
        self.L = None
```