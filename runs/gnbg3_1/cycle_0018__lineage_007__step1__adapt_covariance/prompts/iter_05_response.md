**Idea: Eigendecomposition-Based Condition Number Control**

Directly controls covariance matrix condition number via eigenvalue rescaling. When eigenvalues become too spread (high condition number), exponentially rescales them to prevent elongated sampling distributions that cause premature convergence on ill-conditioned tasks. This is fundamentally different from archive-based perturbations (v01), simple rate scaling (v06), or active CMA-ES weight decomposition (v08).

```python
def _adapt_covariance(self):
    """Eigendecomposition-based condition number control for ill-conditioned tasks."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Eigendecomposition-based condition number control
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-30, None)
        
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-30)
        
        # Exponential rescaling of eigenvalues based on condition number
        # When cond is high, eigenvalues are too spread - rescale toward geometric mean
        if cond > 1e2:
            log_eig = np.log(eigvals + 1e-30)
            log_mean = np.mean(log_eig)
            # Exponential rescaling: push eigenvalues back toward geometric mean
            alpha = min(1.0, np.log(cond) / 20.0)
            log_eig_scaled = log_mean + (log_eig - log_mean) * (1.0 - alpha)
            eigvals = np.exp(np.clip(log_eig_scaled, -30, 30))
        
        # Reconstruct covariance with (potentially) rescaled eigenvalues
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)
        
    except np.linalg.LinAlgError:
        pass
    
    # Standard CMA-ES update on (possibly rescaled) covariance
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # Clip to prevent numerical explosion
    self.C = np.clip(self.C, -1e10, 1e10)
    self.C = self._ensure_positive_definite(self.C)
```