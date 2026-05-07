Looking at the benchmark data, all 24 tasks remain unsolved with the worst tasks (16, 17, 20, 12, 23, 11) having errors in the 1e+01 to 6e+01 range. This suggests a fundamental failure mode: **premature convergence to local optima** where the covariance adaptation gets "stuck" in a narrow basin and cannot escape.

The key observation is that variant_08 (Active CMA-ES) performs best on the hardest tasks but still gets trapped. The issue is that when trapped, the standard CMA-ES update simply shrinks/expands the same covariance structure — it doesn't restructure the search distribution fundamentally.

**Idea: Eigenspace Restart with Orthogonal Perturbation**

The core problem is that all existing variants update covariance incrementally. When trapped, they just scale the same eigenvectors. This approach detects trapping and **restructures the eigenspace** by:
1. Detecting stagnation via relative fitness variance AND improvement stall
2. When trapped: rotate the covariance by mixing eigenvectors + inject exploration noise along the weakest directions
3. This breaks the algorithm out of the local basin by fundamentally changing the sampling geometry

```python
def _adapt_covariance_original(self):
    """Eigenspace restart with orthogonal perturbation for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # Detect trapping: low fitness variance AND stagnation
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    if not hasattr(self, 'prev_f_opt_trap'):
        self.prev_f_opt_trap = self.f_opt
        self.stagnation_gen = 0
    else:
        f_improved = self.prev_f_opt_trap - self.f_opt
        if f_improved > 1e-10:
            self.stagnation_gen = 0
        else:
            self.stagnation_gen += 1
        self.prev_f_opt_trap = self.f_opt
    
    # Trigger eigenspace restart if trapped
    is_trapped = (rel_var < 1e-3) and (self.stagnation_gen > self.dim * 2)
    
    if is_trapped:
        # Eigendecompose current covariance
        eigvals, eigvecs = np.linalg.eigh(self.C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Mix eigenvectors to create new basis (prevents collapse)
        theta = 0.3 * np.pi
        mix_matrix = np.cos(theta) * np.eye(self.dim) + np.sin(theta) * (np.ones((self.dim, self.dim)) / self.dim - np.eye(self.dim))
        eigvecs_mixed = eigvecs @ mix_matrix
        
        # Inject exploration noise along weakest eigendirections
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        eigvals_scaled = eigvals.copy()
        
        # Boost weak directions, cap strong directions
        for i in range(self.dim):
            ratio = eigvals[i] / (max_eig + 1e-10)
            if ratio < 0.1:
                eigvals_scaled[i] = min_eig + 0.2 * (max_eig - min_eig)
            elif ratio > 0.9:
                eigvals_scaled[i] = 0.7 * max_eig
        
        # Reconstruct covariance in mixed basis
        self.C = eigvecs_mixed @ np.diag(eigvals_scaled) @ eigvecs_mixed.T
        self.C = 0.5 * (self.C + self.C.T)
        
        # Reset evolution path to align with new covariance
        self.pc *= 0.1
        self.stagnation_gen = 0
        
        # Increase learning rate temporarily
        ccov_boost = min(self.ccov * 3.0, 0.5)
        self.C = (1.0 - ccov_boost) * self.C + ccov_boost * np.eye(self.dim)
    else:
        self.stagnation_gen = max(0, self.stagnation_gen - 1)
    
    self.C = self._ensure_positive_definite(self.C)
```