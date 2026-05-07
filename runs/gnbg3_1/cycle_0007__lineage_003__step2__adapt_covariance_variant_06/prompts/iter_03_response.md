**Idea: Catastrophic Diversification with Eigenvalue Floor**

A fundamentally different strategy that triggers aggressive covariance explosion when the optimizer is "pathologically stuck" (no fitness progress + near-zero diversity). While other variants try to scale learning rates smoothly, this one forces the covariance matrix into a high-diversity state by injecting a large random perturbation along the principal axis and enforcing an eigenvalue floor to prevent premature re-convergence.

```python
def _adapt_covariance_variant_06(self):
    """Catastrophic diversification with eigenvalue floor for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Detect pathological stagnation: no improvement + near-zero diversity
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = pop_variance / (expected_var + 1e-10)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    
    # Pathological convergence detection: fitness stalled AND population collapsed
    pathological = (rel_var < 1e-6) and (diversity_ratio < 1e-5) and (eig_spread < 1e-6)
    
    if pathological:
        # Force covariance explosion: add massive random perturbation
        rand_dir = np.random.randn(self.dim)
        rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)
        
        # Scale perturbation by current max eigenvalue and sigma
        explosion_magnitude = np.max(eigvals) * 50.0 * max(self.sigma, 1.0)
        explosion_matrix = explosion_magnitude * np.outer(rand_dir, rand_dir)
        
        # Blend explosion into covariance
        self.C = 0.5 * self.C + 0.5 * explosion_matrix
        
        # Enforce eigenvalue floor to prevent immediate re-collapse
        min_eigenvalue = (self.ub[0] - self.lb[0]) ** 2 * 1e-6
        for i in range(self.dim):
            if self.C[i, i] < min_eigenvalue:
                self.C[i, i] = min_eigenvalue
        
        # Reset evolution path to remove directional memory
        self.pc = np.zeros(self.dim)
        
        # Boost step size to capitalize on new exploration direction
        self.sigma = min(self.sigma * 3.0, 10.0)
        
        self.C = self._ensure_positive_definite(self.C)
        return
    
    # Standard CMA-ES adaptation when not pathological
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rate based on condition number
    cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
    if cond > 1e6:
        ccov_scale = 0.1
    elif cond > 1e4:
        ccov_scale = 0.3
    elif cond > 1e2:
        ccov_scale = 0.5
    else:
        ccov_scale = 1.0
    
    ccov_adapt = self.ccov * ccov_scale
    
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```