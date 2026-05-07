**Idea: Stagnation-Triggered Mean Shift with Active Covariance Reset**

This approach detects when the population is trapped in a local optimum (high fitness similarity + no improvement) and performs a targeted mean shift to an unexplored region using a heavy-tailed (Cauchy) distribution, combined with active covariance reset that preserves some eigenvalue structure for faster re-convergence.
```python
def _adapt_covariance_variant_06(self):
    """Stagnation-triggered mean shift with active covariance reset for escaping local optima."""
    # Track improvement for stagnation detection
    if not hasattr(self.improvement_ema):
        self.improvement_ema = 1.0
    
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * np.log1p(max(improvement, 1e-15))
    
    # Stagnation detection: low improvement AND low population diversity
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)
    
    is_stagnant = (self.improvement_ema < -5.0) and (rel_var < 0.05)
    
    # Determine if mean shift should occur
    should_shift = is_stagnant or (eig_cond > 1e6) or (rel_var < 1e-4)
    
    if should_shift:
        # Mean shift: move mean toward unexplored region using heavy-tailed perturbation
        # Compute step size based on current covariance scale
        cov_scale = np.mean(eigvals)
        step_scale = 2.0 * np.sqrt(cov_scale) * max(1.0, self.dim / 10.0)
        
        # Generate Cauchy-like direction (heavy-tailed for escaping narrow basins)
        # Use product of t-distribution samples for multivariate heavy-tailed
        z_cauchy = np.random.standard_t(df=2, size=self.dim)
        z_cauchy = z_cauchy / (np.abs(z_cauchy) + 0.1)  # normalize to unit scale
        
        # Blend with Gaussian for stability
        z_gauss = np.random.randn(self.dim)
        direction = 0.3 * z_cauchy + 0.7 * z_gauss
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-10:
            direction /= direction_norm
        
        # Shift mean to new region
        shift_vector = step_scale * direction
        candidate_mean = self.mean + shift_vector
        
        # Clip to bounds
        candidate_mean = np.clip(candidate_mean, self.lb, self.ub)
        
        # Update mean and covariance
        self.old_mean = self.mean.copy()
        self.mean = candidate_mean
        
        # Reset covariance with partial eigenvalue preservation
        # Keep diagonal structure but scale eigenvalues
        preserved_C = 0.2 * np.diag(np.diag(self.C))  # preserve diagonal scaling
        self.C = preserved_C + 0.8 * np.eye(self.dim) * cov_scale
        
        # Reset evolution path to allow new direction learning
        self.pc = np.zeros(self.dim)
        
        # Reset improvement tracking
        self.improvement_ema = 1.0
        
        self.C = self._ensure_positive_definite(self.C)
        return
    
    # Normal covariance adaptation with adaptive learning rates
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Adaptive learning rate based on condition number
    ccov_adapt = self.ccov
    if eig_cond > 1e4:
        ccov_adapt *= 0.5  # reduce update rate when ill-conditioned
    
    cc_adapt = self.cc * min(1.0, 1.0 + 0.5 * self.improvement_ema)
    cc_adapt = np.clip(cc_adapt, 0.001, 0.3)
    
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Active covariance update with negative weight exploration
    neg_weight_sum = 0.0
    for i in range(self.mu):
        if self.weights[i] < 0:
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += abs(self.weights[i]) * np.outer(diff, diff)
            neg_weight_sum += abs(self.weights[i])
    
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))
    
    # Add small exploration component when far from optimum
    if rel_var > 0.5:
        self.C += 0.01 * np.eye(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
```