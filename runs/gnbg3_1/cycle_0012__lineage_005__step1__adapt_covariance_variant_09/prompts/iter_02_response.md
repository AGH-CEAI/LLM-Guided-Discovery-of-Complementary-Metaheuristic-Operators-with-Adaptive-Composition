**Idea: Hierarchical Stagnation Detection with Targeted Covariance Injection**

This approach uses a multi-level stagnation detector (fitness plateau, covariance collapse, path stagnation) to trigger targeted covariance injections along the principal eigendirections. Unlike the temperature-based approach which modulates learning rates smoothly, this method introduces discrete "escape impulses" when stagnation is detected, injecting axis-aligned perturbations scaled to the problem dimension to break out of deceptive basins on the hardest tasks.

```python
def _adapt_covariance_variant_09(self):
    """Hierarchical stagnation detection with targeted eigenspace injection."""
    # Compute population fitness statistics
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Compute covariance condition and spectrum
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-10)
    eig_spread = eig_min / (eig_max + 1e-10)
    
    # Compute expected variance and diversity
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    pop_variance = np.mean(np.var(self.population, axis=0))
    diversity_ratio = pop_variance / (expected_var + 1e-10)
    
    # Initialize stagnation counters if needed
    if not hasattr(self, 'stagnation_levels'):
        self.stagnation_levels = {'fitness': 0, 'cov': 0, 'path': 0, 'diversity': 0}
    
    # Level 1: Fitness stagnation
    if rel_var < 1e-4:
        self.stagnation_levels['fitness'] += 1
    else:
        self.stagnation_levels['fitness'] = max(0, self.stagnation_levels['fitness'] - 2)
    
    # Level 2: Covariance collapse
    if cond > 1e6 or eig_spread < 1e-6:
        self.stagnation_levels['cov'] += 1
    else:
        self.stagnation_levels['cov'] = max(0, self.stagnation_levels['cov'] - 2)
    
    # Level 3: Diversity loss
    if diversity_ratio < 1e-3:
        self.stagnation_levels['diversity'] += 1
    else:
        self.stagnation_levels['diversity'] = max(0, self.stagnation_levels['diversity'] - 2)
    
    # Check if any stagnation level exceeds threshold
    threshold = max(5, self.dim // 2)
    trigger_level = max(self.stagnation_levels.values())
    is_stagnant = trigger_level >= threshold
    
    if is_stagnant:
        # Compute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(self.C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 1e-10)
        eigvecs = eigvecs[:, idx]
        
        # Inject along top k eigendirections with diminishing strength
        k = min(self.dim, max(3, self.dim // 3))
        injection_strength = 0.5 * min(diversity_ratio + 0.1, 1.0)
        
        for i in range(k):
            # Stronger injection for directions with smaller eigenvalues (collapsed axes)
            rel_strength = injection_strength * (k - i) / k
            self.C += rel_strength * np.outer(eigvecs[:, i], eigvecs[:, i])
        
        # Also add isotropic component for uniform exploration boost
        self.C += 0.1 * injection_strength * np.eye(self.dim)
        
        # Reset evolution path to encourage new direction
        self.pc *= 0.2
        
        # Reset stagnation counters
        for key in self.stagnation_levels:
            self.stagnation_levels[key] = 0
        
        self.C = self._ensure_positive_definite(self.C)
    else:
        # Standard covariance update with adaptive learning rates
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Adaptive cc based on condition
        cc_adapt = self.cc * np.clip(1.0 + np.log10(max(cond, 1.0)) * 0.1, 0.5, 2.0)
        cc_adapt = np.clip(cc_adapt, 0.01, 0.3)
        
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        # Adaptive ccov based on diversity
        ccov_scale = np.clip(1.0 + 2.0 * np.log1p(diversity_ratio * 10.0), 0.1, 3.0)
        ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
        
        # Prevent extreme conditioning during normal operation
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.min(eigvals_check)
        if cond_check > 1e7:
            self.C *= 0.5
        
        self.C = self._ensure_positive_definite(self.C)
```