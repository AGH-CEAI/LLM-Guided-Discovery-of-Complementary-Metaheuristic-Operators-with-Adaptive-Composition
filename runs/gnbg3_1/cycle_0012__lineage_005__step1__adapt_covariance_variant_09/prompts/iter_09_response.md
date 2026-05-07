**Idea: Eigenvalue-Weighted Covariance Adaptation with Diversity-Driven Exploration**

A fundamentally different approach that directly analyzes the covariance eigenspectrum and population geometry to make informed adaptation decisions. Uses eigenvalue-weighted rank updates, condition-number-based ccov scaling, and a geometric diversity metric to trigger directional exploration boosts when the algorithm detects convergence to ill-conditioned subspaces.

```python
def _adapt_covariance_variant_09(self):
    """Eigenvalue-weighted covariance adaptation with diversity-driven exploration."""
    # Compute covariance eigendecomposition for geometric analysis
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-15)
    
    # Geometric analysis metrics
    cond = np.max(eigvals) / np.min(eigvals)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-15)
    eig_entropy = -np.sum((eigvals / np.sum(eigvals)) * np.log(eigvals / np.sum(eigvals) + 1e-15))
    max_eig_ratio = np.max(eigvals) / (np.median(eigvals) + 1e-15)
    
    # Population diversity via geometric span
    pop_diffs = self.population[:self.mu] - self.old_mean
    pop_projections = pop_diffs @ eigvecs
    proj_var = np.var(pop_projections, axis=0)
    proj_coverage = np.sum(proj_var > 1e-10 * np.mean(proj_var))
    diversity_ratio = proj_coverage / max(self.dim, 1)
    
    # Detect problematic states
    is_ill_conditioned = cond > 1e5 or eig_spread < 1e-6
    is_low_diversity = diversity_ratio < 0.3
    is_stagnant = self.stagnation_counter > max(20, self.dim * 2)
    needs_exploration = is_ill_conditioned or is_low_diversity or is_stagnant
    
    # Condition-number-adaptive ccov scaling
    cond_scaling = np.clip(np.log10(cond + 1) / 3.0, 0.1, 10.0)
    
    # Eigenvalue-weighted base ccov (emphasize under-represented directions)
    inv_eigvals = 1.0 / (eigvals + 1e-10)
    weights_normalized = inv_eigvals / np.sum(inv_eigvals)
    eig_weight = np.sum(weights_normalized * np.log(eigvals + 1e-10))
    
    base_ccov_1 = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2)
    base_ccov_mu = (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    
    # Scale ccov based on condition and diversity
    if needs_exploration:
        ccov_scale = 2.0 * cond_scaling
        cc_scale = 0.5
    else:
        ccov_scale = 1.0 / (1.0 + eig_weight * 0.5)
        cc_scale = 1.0
    
    ccov_1 = np.clip(base_ccov_1 * ccov_scale, 1e-10, 0.3)
    ccov_mu = np.clip(base_ccov_mu * ccov_scale, 1e-10, 0.3)
    cc_adapt = np.clip(self.cc * cc_scale, 0.001, 0.3)
    
    # Evolution path update
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Rank-one update (eigenvalue-weighted)
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update with eigenvalue-based reweighting
    rank_mu = np.zeros((self.dim, self.dim))
    total_w = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        # Reweight by inverse eigenvalue of that direction
        proj = diff @ eigvecs
        eig_reweight = np.sum((proj ** 2) / (eigvals + 1e-10))
        w_eff = w * np.clip(eig_reweight / (np.sum(proj ** 2) + 1e-10), 0.1, 10.0)
        rank_mu += w_eff * np.outer(diff, diff)
        total_w += abs(w_eff)
    
    if total_w > 0:
        rank_mu /= total_w
    
    # Combine updates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu)
    
    # Diversity-driven exploration boost
    if needs_exploration:
        # Add exploration along top and bottom eigenvectors
        k_perturb = min(self.dim, max(2, self.dim // 3))
        for i in range(k_perturb):
            # Top eigenvectors (strong perturbation to break narrow basins)
            strength_top = 0.5 * (k_perturb - i) / k_perturb
            self.C += strength_top * np.outer(eigvecs[:, -(i + 1)], eigvecs[:, -(i + 1)])
            # Bottom eigenvectors (gentle boost for dormant directions)
            strength_bottom = 0.2 * (i + 1) / k_perturb
            self.C += strength_bottom * np.outer(eigvecs[:, i], eigvecs[:, i])
        
        # Axis-aligned perturbation for uniform coverage
        self.C += 0.3 * np.eye(self.dim)
        
        # Dampen pc to reset biased search direction
        self.pc *= 0.2
        
        # Reset stagnation counter after exploration
        self.stagnation_counter = 0
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
    
    # Emergency restart on extreme ill-conditioning
    eigvals_check = np.linalg.eigvalsh(self.C)
    cond_check = np.max(eigvals_check) / np.min(eigvals_check)
    if cond_check > 1e8:
        self.C = np.eye(self.dim) * np.mean(eigvals_check)
        self.pc = np.zeros(self.dim)
```