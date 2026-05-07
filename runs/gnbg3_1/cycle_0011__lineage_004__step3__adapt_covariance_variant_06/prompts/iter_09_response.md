**Idea: Tangent-Drift Escape with Condition-Aware Eigenspace Inflation**

When stuck in local optima (large errors ~10^1–10^2), the population collapses to a tight region, the evolution path `pc` aligns with the local gradient, and all existing variants continue shrinking the covariance in the same direction. This variant explicitly detects "drift stagnation" (improvement stalls despite active search), computes the tangent direction perpendicular to `pc`, and injects covariance energy along that tangent to break the local orbit and explore a new basin.

```python
def _adapt_covariance_variant_06(self):
    """Tangent-drift escape with condition-aware eigenspace inflation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # --- Drift stagnation detection ---
    # Track consecutive non-improving generations vs improving
    if not hasattr(self, '_drift_improve_count'):
        self._drift_improve_count = 0
        self._drift_total_count = 0
    
    self._drift_total_count += 1
    if self.f_opt < getattr(self, '_drift_f_opt_prev', self.f_opt) - 1e-12:
        self._drift_improve_count = 0
    else:
        self._drift_improve_count += 1
    self._drift_f_opt_prev = self.f_opt
    
    # Drift stagnation = stuck for many generations in a collapsed region
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min, eig_max = np.min(eigvals), np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    eig_spread = eig_min / (eig_max + 1e-10)
    
    is_collapsed = (eig_spread < 1e-4) or (cond > 1e5)
    is_stuck = self._drift_improve_count >= max(20, self.dim * 3)
    is_drift_stuck = is_collapsed and is_stuck
    
    # --- Tangent direction computation ---
    tangent_perturb = np.zeros((self.dim, self.dim))
    escape_strength = 0.0
    
    if is_drift_stuck:
        # Tangent = perpendicular to pc (rotate pc by 90 degrees in random subspace)
        pc_norm = np.linalg.norm(self.pc)
        if pc_norm > 1e-10:
            pc_unit = self.pc / pc_norm
            
            # Build orthonormal basis: pick random vector not parallel to pc
            rand_v = np.random.randn(self.dim)
            rand_v -= np.dot(rand_v, pc_unit) * pc_unit
            rand_v_norm = np.linalg.norm(rand_v)
            if rand_v_norm > 1e-10:
                rand_v /= rand_v_norm
                
                # Gram-Schmidt to get second orthonormal direction
                v2 = rand_v - np.dot(rand_v, pc_unit) * pc_unit
                v2_norm = np.linalg.norm(v2)
                if v2_norm > 1e-10:
                    v2 /= v2_norm
                    
                    # Tangent direction: perpendicular to pc, spanning subspace
                    # Use multiple tangent directions for higher dims
                    for v in [v2]:
                        # Inject along tangent direction (rank-1 matrix)
                        tangent_perturb += 0.5 * np.outer(v, v)
                    
                    escape_strength = min(0.3, 0.1 * np.log1p(self._drift_improve_count))
    
    # --- Condition-aware eigenspace inflation ---
    # When eigenvalues are too spread, inflate the smallest ones
    eigvals_safe = np.maximum(eigvals, 1e-10)
    target_spread = 1e-3  # Target min/max ratio
    inflation = np.zeros(self.dim)
    
    for i in range(self.dim):
        if eigvals_safe[i] < eig_max * target_spread:
            # Inflate small eigenvalues toward target
            inflation[i] = (eig_max * target_spread) - eigvals_safe[i]
    
    if np.any(inflation > 0):
        # Apply inflation along principal axes
        eigvecs = np.linalg.eigh(self.C)[1]
        for i in range(self.dim):
            if inflation[i] > 0:
                tangent_perturb += inflation[i] * np.outer(eigvecs[:, i], eigvecs[:, i])
    
    # --- Standard covariance update with adaptive rates ---
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    if is_drift_stuck:
        ccov_scale = 2.0  # Accelerate covariance update when stuck
        cc_scale = 1.5
    elif rel_var < 0.01:
        ccov_scale = 2.0
        cc_scale = 1.5
    else:
        ccov_scale = 1.0
        cc_scale = 1.0
    
    ccov_adapt = min(self.ccov * ccov_scale, 0.5)
    cc_adapt = min(self.cc * cc_scale, 0.3)
    
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))
    
    # --- Apply tangent-drift escape and eigenspace inflation ---
    if escape_strength > 0:
        self.C += escape_strength * tangent_perturb
    
    if np.any(inflation > 0):
        self.C += tangent_perturb
    
    # Reset drift counter on successful escape
    if is_drift_stuck and escape_strength > 0:
        self._drift_improve_count = 0
    
    self.C = self._ensure_positive_definite(self.C)
```