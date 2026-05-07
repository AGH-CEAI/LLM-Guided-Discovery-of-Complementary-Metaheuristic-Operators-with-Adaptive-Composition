**Idea: Coordinate-Stuck Detection with Anisotropic Covariance Injection**

A fundamentally different approach that detects per-coordinate convergence stagnation (when population variance in a dimension drops too low) and injects anisotropic covariance perturbations specifically along those "stuck" axes, forcing exploration in directions that have stopped improving.
```python
def _adapt_covariance_variant_09(self):
    """Coordinate-stuck detection with anisotropic covariance injection for escaping local optima."""
    # Compute per-coordinate population variance (key diagnostic)
    pop_var = np.var(self.population, axis=0)
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    rel_var = pop_var / (expected_var + 1e-10)
    
    # Identify "stuck" coordinates (very low relative variance)
    stuck_threshold = 1e-4
    stuck_mask = rel_var < stuck_threshold
    num_stuck = np.sum(stuck_mask)
    
    # Track stuck counter for hysteresis (avoid flickering)
    if not hasattr(self, 'stuck_counter'):
        self.stuck_counter = 0
    if num_stuck > self.dim * 0.3:  # >30% stuck triggers counter
        self.stuck_counter += 1
    else:
        self.stuck_counter = max(0, self.stuck_counter - 1)
    
    # Compute condition number of covariance
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    
    # Determine if we're in escape mode
    is_escaping = (self.stuck_counter > 5) or (cond > 1e6)
    
    # Base covariance adaptation
    y_mean = (self.mean - self.old_mean) / self.sigma
    cc_base = self.cc
    ccov_base = self.ccov
    
    if is_escaping:
        # ESCAPE MODE: aggressive exploration
        # 1. Reset evolution path partially
        self.pc = 0.5 * self.pc + np.sqrt(cc_base * (2.0 - cc_base)) * y_mean
        
        # 2. Compute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        
        # 3. Inject anisotropic perturbation along stuck coordinates
        inject_matrix = np.zeros((self.dim, self.dim))
        if num_stuck > 0:
            # Add strong perturbations in stuck coordinate directions
            for i in range(self.dim):
                if stuck_mask[i]:
                    # Strong axis-aligned injection
                    inject_matrix += 3.0 * np.outer(eigvecs[:, i], eigvecs[:, i])
        
        # 4. Also add global exploration boost
        inject_matrix += 0.5 * np.eye(self.dim)
        
        # 5. Apply with higher learning rate
        ccov_escape = min(ccov_base * 5.0, 0.5)
        cc_escape = min(cc_base * 2.0, 0.3)
        
        self.pc = (1.0 - cc_escape) * self.pc + np.sqrt(cc_escape * (2.0 - cc_escape)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_escape) * self.C + 
                  ccov_escape * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_escape * 2.0 * rank_mu +
                  inject_matrix)
        
        # Reset stuck counter after injection
        self.stuck_counter = 0
    else:
        # NORMAL MODE: standard CMA-ES update with coordinate-wise damping
        # Damp pc update when specific coordinates are struggling
        coord_damping = np.ones(self.dim)
        coord_damping[stuck_mask] = 2.0  # Resist collapse on stuck coords
        
        cc_adaptive = cc_base * np.clip(coord_damping, 0.5, 2.0)
        cc_adaptive = np.clip(cc_adaptive, 0.001, 0.3)
        
        self.pc = (1.0 - np.mean(cc_adaptive)) * self.pc + np.sqrt(np.mean(cc_adaptive) * (2.0 - np.mean(cc_adaptive))) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_base) * self.C + 
                  ccov_base * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_base * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```