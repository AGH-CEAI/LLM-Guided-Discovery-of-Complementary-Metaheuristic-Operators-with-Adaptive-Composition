**Idea: Eigenvalue-Directed Covariance Perturbation**

A fundamentally different approach that detects convergence through eigenvalue spread and injects directional perturbations along principal axes to escape local optima. Uses axis-aligned explosions when the covariance matrix becomes too anisotropic, combined with adaptive rank-mu weighting based on individual offspring contribution.

```python
def _adapt_covariance_variant_06(self):
    """Eigenvalue-directed covariance perturbation for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Compute eigenvalue decomposition for convergence detection
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    
    # Fitness variance as secondary convergence signal
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Detect stagnation: low eigenvalue spread + low fitness variance
    is_stagnant = (eig_spread < 1e-4) or (rel_var < 0.001)
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update with adaptive weighting
    rank_mu = np.zeros((self.dim, self.dim))
    total_weight = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        # Weight by both selector weight and individual fitness contribution
        w = self.weights[i]
        # Boost weights for individuals that improved the most
        if hasattr(self, 'trial_fitness') and i < len(self.trial_fitness):
            fitness_contrib = max(0.0, 1.0 - self.trial_fitness[i] / (self.f_opt + 1e-10))
            w = w * (1.0 + fitness_contrib)
        rank_mu += w * np.outer(diff, diff)
        total_weight += abs(w)
    
    if total_weight > 0:
        rank_mu /= total_weight
    
    # Base covariance update
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * rank_one + 
              (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
    
    # Apply eigenvalue-directed perturbation when stagnant
    if is_stagnant:
        # Compute current eigenvalues
        current_eigvals = np.linalg.eigvalsh(self.C)
        max_eig = np.max(current_eigvals)
        min_eig = np.min(current_eigvals)
        
        # Expand along largest eigenvector (most promising direction)
        if max_eig > 1e-10:
            principal_dir = eigvecs[:, -1]  # eigenvector of largest eigenvalue
            
            # Compute exploration strength based on stagnation severity
            stagnation_strength = min(1.0, -np.log10(max(eig_spread, 1e-10)) / 4.0)
            exploration_boost = 1.0 + 2.0 * stagnation_strength
            
            # Add rank-one perturbation along principal axis
            rank_perturb = np.outer(principal_dir, principal_dir)
            self.C += exploration_boost * self.ccov * max_eig * rank_perturb
        
        # Shrink smallest eigenvalues to encourage new direction discovery
        min_eig_current = np.min(np.linalg.eigvalsh(self.C))
        if min_eig_current < 1e-8:
            self.C += (1e-7 - min_eig_current) * np.eye(self.dim)
        
        # Apply diagonal noise to prevent premature re-collapse
        diag_noise = 0.1 * np.mean(np.diag(self.C)) * np.random.randn(self.dim)
        np.fill_diagonal(self.C, np.diag(self.C) + diag_noise)
        
        # Reset evolution path to allow new direction
        self.pc *= 0.3
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
    
    # Safety check: if condition number is extreme, regularize
    eigvals_final = np.linalg.eigvalsh(self.C)
    cond_final = np.max(eigvals_final) / np.maximum(np.min(eigvals_final), 1e-10)
    if cond_final > 1e8:
        self.C = 0.9 * self.C + 0.1 * np.eye(self.dim) * np.mean(eigvals_final)
```