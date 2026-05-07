Looking at the task errors, the worst unsolved tasks (16, 17, 20, 11, 19, 23, 18, 22) are all stuck at errors ~10^1 to ~10^2, which is ~8-10 orders of magnitude above target. This pattern strongly suggests **premature covariance matrix collapse** — the search distribution becomes too narrow, trapping the optimizer in local optima before reaching high precision.

The key evidence: all existing variants (even the best ones) are stuck at similar error magnitudes, indicating a fundamental convergence failure mode. The current `_adapt_covariance_variant_01` uses static learning rates that don't respond to covariance condition number.

**Idea: Eigenvalue Spread Control with Adaptive Learning Rates**

This approach explicitly monitors the condition number of the covariance matrix and dynamically scales learning rates inversely with eigenspread. When eigenvalues become too concentrated (high condition number), it increases learning rates and injects noise to prevent collapse. This directly addresses the failure mode on multimodal tasks where the search distribution collapses prematurely.

```python
def _adapt_covariance_variant_01(self):
    """Eigenvalue spread control with adaptive learning rates."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute eigenvalue spread of covariance matrix
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.maximum(eigvals.min(), 1e-15)
    eig_max = np.maximum(eigvals.max(), 1e-15)
    cond = eig_max / eig_min
    eig_spread = eig_min / (eig_max + 1e-15)
    
    # Adaptive learning rates based on condition number
    log_cond = np.log10(max(cond, 1.0))
    spread_factor = 1.0 + max(0.0, log_cond - 2.0) * 0.5
    
    # Base learning rates from CMA-ES theory
    ccov_base = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    
    # Scale up learning rates when condition number is large
    ccov_adaptive = min(ccov_base * spread_factor, 0.5)
    cc_adaptive = min(self.cc * spread_factor, 0.3)
    
    # Evolution path update
    self.pc = (1.0 - cc_adaptive) * self.pc + \
              np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combined update with adaptive rate
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * (rank_one + (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu))
    
    # Inject exploration when condition number is problematic
    if log_cond > 3.0:
        # Add small random perturbation to prevent eigenvalue collapse
        noise_scale = 0.01 * ccov_adaptive * np.mean(eigvals)
        self.C += noise_scale * np.eye(self.dim)
    
    # Hard reset if covariance becomes pathological
    if cond > 1e8 or np.any(np.isnan(self.C)):
        self.C = np.eye(self.dim) * (self.sigma ** 2)
        self.pc = np.zeros(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
```