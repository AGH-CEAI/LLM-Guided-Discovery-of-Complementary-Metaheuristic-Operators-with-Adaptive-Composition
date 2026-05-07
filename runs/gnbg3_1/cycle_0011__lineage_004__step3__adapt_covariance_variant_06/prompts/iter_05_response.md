**Idea: Adaptive Restart-Driven Covariance Reset with Directional Memory**

This variant detects when the optimizer is stuck in poor local optima (large errors ~10-60) and performs a more aggressive covariance reset that preserves directional memory from the evolution path. Unlike previous variants that use archive tracking or temperature scaling, this approach uses restart-triggered directional exploration where the covariance matrix is explicitly shaped toward unexplored regions by projecting the evolution path onto principal axes and perturbing orthogonal directions.

```python
def _adapt_covariance_variant_06(self):
    """Adaptive restart-driven covariance with directional memory for escaping large local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Track improvement for stagnation detection
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.stagnation_gen = 0
    
    f_improvement = max(0.0, self.prev_f_opt - self.f_opt)
    self.prev_f_opt = self.f_opt
    
    # Detect stagnation: no improvement for extended generations
    stagnation_threshold = max(20, self.dim * 2)
    if f_improvement > 1e-12:
        self.stagnation_gen = 0
    else:
        self.stagnation_gen += 1
    
    is_stagnant = self.stagnation_gen > stagnation_threshold
    
    # Also detect convergence to poor local optima via absolute fitness
    is_trapped = (self.f_opt > 1.0 and self.stagnation_gen > stagnation_threshold // 2)
    
    if is_stagnant or is_trapped:
        # Reset stagnation counter
        self.stagnation_gen = 0
        
        # Perform eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        
        # Project evolution path onto eigenspace
        pc_proj = eigvecs.T @ self.pc
        
        # Perturb along directions where pc has low magnitude (underexplored)
        # and reduce exploration along high-pc directions (already explored)
        perturb_scale = np.zeros(self.dim)
        for i in range(self.dim):
            pc_mag = abs(pc_proj[i])
            pc_normalized = pc_mag / (np.linalg.norm(pc_proj) + 1e-10)
            # Strong perturbation where path hasn't explored
            perturb_scale[i] = 3.0 * (1.0 - min(pc_normalized, 1.0))
        
        # Add scaled rank-one perturbations along underexplored eigenvectors
        for i in range(self.dim):
            strength = perturb_scale[i] * (1.0 - 0.5 * i / self.dim)
            if strength > 0.1:
                v = eigvecs[:, i].reshape(-1, 1)
                self.C += strength * (v @ v.T)
        
        # Also add identity perturbation for uniform exploration
        self.C += 1.0 * np.eye(self.dim)
        
        # Reset evolution path partially to allow new direction
        self.pc *= 0.2
        
        self.C = self._ensure_positive_definite(self.C)
    else:
        # Standard covariance update with adaptive scaling
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        # Adaptive scaling based on fitness variance
        if rel_var < 0.001:
            scale = 5.0
        elif rel_var < 0.01:
            scale = 2.0
        else:
            scale = 1.0
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
```