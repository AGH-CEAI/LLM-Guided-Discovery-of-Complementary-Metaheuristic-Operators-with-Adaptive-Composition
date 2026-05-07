Looking at the error magnitudes, the priority tasks (16, 17, 18, 19, 20, 22, 23) are stuck at errors of ~8-96, requiring 9+ orders of magnitude improvement. All existing variants (original, variant_01, variant_02) use similar CMA-ES-style covariance updates that rely on **eigenvalue clipping** to bound the condition number — this prevents the covariance matrix from adapting sufficiently in ill-conditioned directions, causing stagnation far from the global optimum.

The key insight: standard CMA-ES cannot rapidly shrink large eigenvalues to escape wrong basins. The new strategy uses **eigendecomposition with eigenvalue momentum adaptation** — explicitly tracking eigenvalue changes over time and applying multiplicative shrinkage to dominant eigenvectors, bypassing the condition-number floor that limits all prior approaches.

**Idea: Eigendecomposition with Eigenvalue Momentum**
Directly modifies eigenvalues using momentum-based exponential smoothing, enabling rapid shrinkage of dominant (potentially wrong) directions and boosting of minor directions — fundamentally different from clipping-based approaches.
```python
def _adapt_covariance_variant_01(self):
    """Eigendecomposition with eigenvalue momentum for rapid covariance reshaping."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    ccov_1 = 1.0 / (self.dim + 2.0)
    ccov_mu_val = 2.0 / ((self.dim + 2.0) ** 2)
    
    C_raw = ((1.0 - ccov_1 - ccov_mu_val) * self.C + 
             ccov_1 * rank_one + 
             ccov_mu_val * self.mueff * rank_mu)
    
    C_raw = 0.5 * (C_raw + C_raw.T)
    
    try:
        eigvals, eigvecs = np.linalg.eigh(C_raw)
        eigvals = np.maximum(eigvals, 1e-15)
        
        if not hasattr(self, 'eig_momentum'):
            self.eig_momentum = np.ones(self.dim)
        
        log_eig = np.log(eigvals + 1e-15)
        median_log = np.median(log_eig)
        
        target_log = np.where(
            log_eig > median_log,
            median_log + 0.7 * (log_eig - median_log),
            median_log + 1.3 * (log_eig - median_log)
        )
        
        rate = 0.15
        self.eig_momentum = (1.0 - rate) * self.eig_momentum + rate * (target_log - log_eig)
        log_eig += 0.05 * self.eig_momentum
        
        eigvals = np.exp(log_eig)
        
        C_raw = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
    except np.linalg.LinAlgError:
        diag_C = np.diag(C_raw)
        diag_C = np.maximum(diag_C, 1e-10)
        C_raw = np.diag(diag_C)
    
    self.C = self._ensure_positive_definite(C_raw + 1e-8 * np.eye(self.dim))
```