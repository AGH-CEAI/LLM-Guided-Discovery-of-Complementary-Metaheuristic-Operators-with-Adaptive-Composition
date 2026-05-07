Looking at the error patterns, I see that tasks 6, 9, 10, 11, 12, 16, 17, 18, 19, 20, 22, 23 are stuck at errors of 10^1 to 10^2 — these are severely ill-conditioned or multimodal problems where the current approach fails to properly explore. The existing variants either perturb the covariance matrix directly or modulate learning rates, but none use the **eigendecomposition-guided step-size adaptation** that is the cornerstone of successful CMA-ES on ill-conditioned problems.

The key insight: maintaining an explicit evolution path `ps` for step-size control (separate from `pc` for covariance) and adapting the covariance matrix shape via eigendecomposition gives precise control over exploration along each eigenvector direction. This is fundamentally different from all prior approaches.

**Idea: Eigendecomposition-Guided CMA-ES with Dual Evolution Paths**
Uses the classic Hansen-style CMA-ES with separate step-size and covariance evolution paths, eigendecomposition for covariance adaptation, and explicit condition number control — the proven approach for ill-conditioned problems.
```python
def _adapt_covariance_variant_09(self):
    """Eigendecomposition-guided CMA-ES with dual evolution paths for ill-conditioned problems."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Initialize step-size evolution path if needed
    if not hasattr(self, 'ps'):
        self.ps = np.zeros(self.dim)
    
    # Local cumulation parameters (standard CMA-ES)
    cs_local = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
    cc_local = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
    ccov_1 = 1.0 / (self.dim + 2.0)
    ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    
    # Norm of y_mean (used for ps update)
    y_norm_sq = np.sum(y_mean ** 2)
    y_norm = np.sqrt(max(y_norm_sq, 1e-10))
    
    # Step-size evolution path (ps) - key for ill-conditioned problems
    self.ps = (1.0 - cs_local) * self.ps + np.sqrt(cs_local * (2.0 - cs_local)) * y_mean / y_norm * y_norm_sq
    
    # Step-size adaptation via path length control
    if self.generation > 2:
        ps_norm_sq = np.sum(self.ps ** 2)
        ps_norm = np.sqrt(max(ps_norm_sq, 1e-10))
        expected_norm = self.dim ** 0.5 * (1.0 - 1.0 / (4.0 * self.dim)) * (1.0 - 1.0 / (21.0 * self.dim ** 2))
        sigma_ratio = ps_norm / (expected_norm + 1e-10)
        damping = 1.0 + cs_local / (1.0 + max(sigma_ratio / 1.0, 0.0))
        self.sigma *= np.exp(min(0.3, cs_local / damping * (sigma_ratio / (1.0 + sigma_ratio) - 0.5)))
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    # Covariance evolution path (pc)
    self.pc = (1.0 - cc_local) * self.pc + np.sqrt(cc_local * (2.0 - cc_local)) * y_mean
    
    # Rank-one and rank-mu updates
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Covariance update with separate rank-one and rank-mu learning rates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu)
    
    # Eigendecomposition for condition number control (key for ill-conditioned tasks)
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    
    if eig_min < 1e-10:
        self.C += (1e-9 - eig_min) * np.eye(self.dim)
        eigvals, eigvecs = np.linalg.eigh(self.C)
    
    # Control condition number to prevent ill-conditioning
    cond = eig_max / max(eig_min, 1e-10)
    cond_threshold = 1e7
    
    if cond > cond_threshold:
        # Damp extreme eigenvalues toward geometric mean
        eig_target = np.sqrt(eig_min * eig_max)
        damp_factor = 0.9
        eigvals = np.where(eigvals < eig_target, eig_target * damp_factor + eigvals * (1 - damp_factor), eigvals)
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Update Cholesky factor L for efficient sampling
    eigvals_L, eigvecs_L = np.linalg.eigh(self.C)
    eigvals_L = np.maximum(eigvals_L, 1e-10)
    sqrt_eig = np.sqrt(eigvals_L)
    self.L = eigvecs_L * sqrt_eig
    
    self.C = self._ensure_positive_definite(self.C)
```