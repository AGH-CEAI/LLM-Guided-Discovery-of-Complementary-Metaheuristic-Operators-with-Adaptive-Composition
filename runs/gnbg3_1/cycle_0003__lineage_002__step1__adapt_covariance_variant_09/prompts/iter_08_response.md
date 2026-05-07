**Idea: Eigenvalue-Drift Control with Active Conditioning**
A fundamentally different approach that monitors and actively controls the condition number of the covariance matrix by applying selective eigenvalue clamping and drift correction, preventing premature convergence on ill-conditioned tasks.

```python
def _adapt_covariance_variant_09(self):
    """Eigenvalue-drift control with active conditioning number management."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    
    pop_spread = np.max(np.std(self.population, axis=0)) / (np.min(np.std(self.population, axis=0)) + 1e-10)
    target_cond = min(1e6, self.dim * 10.0)
    
    if not hasattr(self, 'eigen_history'):
        self.eigen_history = []
    
    eigvals, eigvecs = np.linalg.eigh(self.C)
    self.eigen_history.append(eigvals.copy())
    if len(self.eigen_history) > 5:
        self.eigen_history.pop(0)
    
    if len(self.eigen_history) >= 3:
        eigen_drift = np.max(np.abs(self.eigen_history[-1] / (self.eigen_history[0] + 1e-10) - 1.0))
    else:
        eigen_drift = 0.0
    
    cond_C = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    
    if cond_C > target_cond:
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        target_min = max_eig / target_cond
        
        if min_eig < target_min * 0.9:
            self.C += eigvecs @ np.diag(np.maximum(0, target_min - eigvals)) @ eigvecs.T
    
    ccov_adaptive = base_ccov
    if eigen_drift > 0.5:
        ccov_adaptive *= 0.7
    elif eigen_drift < 0.1 and cond_C < target_cond:
        ccov_adaptive *= 1.3
    
    if pop_spread < 0.1:
        ccov_adaptive *= 1.5
    elif pop_spread > 10.0:
        ccov_adaptive *= 0.8
    
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              ccov_adaptive * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
    eigvals_final = np.linalg.eigvalsh(self.C)
    if np.min(eigvals_final) < 1e-12:
        self.C += (1e-10 - np.min(eigvals_final)) * np.eye(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
```