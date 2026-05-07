Looking at the error patterns, the worst tasks (16-23, 10-12) all have errors ~1e+01 to 8e+01, suggesting the algorithm is getting trapped in local optima with insufficient exploration. The current variant_01 reduces learning rates based on condition number, but this conservative approach prevents the algorithm from escaping these traps. I need a fundamentally different mechanism: **active diversity injection through covariance matrix perturbation when stagnation is detected**.

**Idea: Eigenspace Diversity Injection**
This approach performs eigendecomposition and injects exploration along under-represented eigen-directions when the algorithm stagnates, directly targeting the loss-of-diversity failure mode causing the worst errors.

```python
def _adapt_covariance_variant_01(self):
    """Eigenspace diversity injection with stagnation-triggered exploration bursts."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute eigendecomposition for eigenspace analysis
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    
    # Stagnation detection: track fitness improvement and population diversity
    if not hasattr(self, 'stagnation_adapt_counter'):
        self.stagnation_adapt_counter = 0
        self.stagnation_adapt_best = self.f_opt
    
    fitness_delta = self.stagnation_adapt_best - self.f_opt
    self.stagnation_adapt_counter += 1
    
    if fitness_delta > 1e-8:
        self.stagnation_adapt_counter = 0
        self.stagnation_adapt_best = self.f_opt
    
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Trigger diversity injection on stagnation
    inject_diversity = self.stagnation_adapt_counter > max(10, self.dim // 2) or diversity_ratio < 0.01
    
    if inject_diversity:
        self.stagnation_adapt_counter = 0
        
        # Project mean movement into eigenspace to find under-represented directions
        y_mean_eig = eigvecs.T @ y_mean
        movement_magnitude = np.abs(y_mean_eig)
        
        # Identify eigen-directions with minimal recent movement
        if not hasattr(self, 'eig_movement_history'):
            self.eig_movement_history = np.zeros(self.dim)
        self.eig_movement_history = 0.9 * self.eig_movement_history + 0.1 * movement_magnitude
        
        # Compute exploration scores: high score = needs more exploration
        exploration_score = 1.0 / (self.eig_movement_history + 0.1)
        exploration_score /= (np.sum(exploration_score) + 1e-10)
        
        # Inject diversity proportional to exploration needs
        injection_strength = 0.3 * np.mean(eigvals)
        noise_in_eigspace = np.random.randn(self.dim) * exploration_score
        noise_cov = np.outer(noise_in_eigspace, noise_in_eigspace)
        noise_C = eigvecs @ noise_cov @ eigvecs.T
        
        # Also reset evolution path to allow new exploration directions
        self.pc *= 0.1
        
        # Apply injection
        self.C = self.C + injection_strength * noise_C
        self.C = self._ensure_positive_definite(self.C)
        
        return
    
    # Normal adaptation path with eigenspace-aware learning rate
    cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    
    # Adaptive learning rates: be aggressive on ill-conditioned problems
    if cond > 1e6:
        ccov_scale = 2.0
        cc_scale = 2.0
    elif cond > 1e4:
        ccov_scale = 1.5
        cc_scale = 1.5
    elif cond > 1e2:
        ccov_scale = 1.2
        cc_scale = 1.2
    else:
        ccov_scale = 1.0
        cc_scale = 1.0
    
    # Boost learning when making good progress
    if fitness_delta > 1e-4:
        progress_boost = min(3.0, 1.0 + 10.0 * np.log10(fitness_delta + 1e-15))
        ccov_scale *= progress_boost
        cc_scale *= progress_boost
    
    ccov_adapt = min(self.ccov * ccov_scale, 0.4)
    cc_adapt = min(self.cc * cc_scale, 0.2)
    
    # Evolution path update
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine updates
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```