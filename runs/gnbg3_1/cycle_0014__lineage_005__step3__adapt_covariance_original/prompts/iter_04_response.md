Looking at the task errors, the worst unsolved tasks (16, 17, 23, 20, 12, 11) all have errors in the 1e+1 to 1e+2 range, indicating complete stagnation. The original CMA-ES covariance adaptation relies on slow, incremental updates that cannot escape the basin of attraction when stuck in local optima.

**Idea: Eigendecomposition-Directed Covariance Adaptation**
This approach explicitly controls the eigenvalue spectrum of the covariance matrix. By decomposing C into eigenvectors and eigenvalues, we can exponentially boost small eigenvalues (promoting exploration in poorly-conditioned directions) and damp large eigenvalues (preventing premature convergence). This is fundamentally different from all previous variants which only adjust learning rates or add perturbations uniformly.

```python
def _adapt_covariance_original(self):
    """Eigendecomposition-directed covariance adaptation for escaping stagnation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute population spread and fitness variance for conditioning
    pop_spread = np.mean(np.std(self.population, axis=0))
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Detect stagnation: low variance OR low spread
    is_stagnant = (rel_var < 1e-3) or (pop_spread < 1e-6 * (self.ub[0] - self.lb[0]))
    
    # Evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Base covariance update (standard CMA-ES)
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # Eigendecomposition-directed adaptation
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        
        # Clamp eigenvalues to avoid numerical issues
        eigvals = np.clip(eigvals, 1e-20, None)
        
        # Compute target condition number
        dim = self.dim
        target_cond = 1e6 if is_stagnant else 1e4
        
        # Sort eigenvalues ascending
        idx = np.argsort(eigvals)
        eigvals_sorted = eigvals[idx]
        
        # Compute target spectrum: exponential spacing for controlled condition number
        log_min = np.log(max(eigvals_sorted[0], 1e-20))
        log_max = log_min + np.log(target_cond)
        target_eigvals = np.exp(np.linspace(log_min, log_max, dim))
        
        # Blend current and target eigenvalues (stronger correction when stagnant)
        blend = 0.3 if is_stagnant else 0.05
        eigvals_corrected = (1.0 - blend) * eigvals + blend * target_eigvals
        
        # Reorder to match original positions
        eigvals_final = np.zeros(dim)
        eigvals_final[idx] = eigvals_corrected
        
        # Ensure minimum eigenvalue
        min_eig = np.mean(eigvals_final) * 1e-6
        eigvals_final = np.maximum(eigvals_final, min_eig)
        
        # Reconstruct covariance matrix
        self.C = eigvecs @ np.diag(eigvals_final) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)
        
    except np.linalg.LinAlgError:
        # Fallback: use diagonal approximation
        diag_C = np.diag(self.C)
        diag_C = np.maximum(diag_C, 1e-10)
        self.C = np.diag(diag_C)
    
    self.C = self._ensure_positive_definite(self.C)
```