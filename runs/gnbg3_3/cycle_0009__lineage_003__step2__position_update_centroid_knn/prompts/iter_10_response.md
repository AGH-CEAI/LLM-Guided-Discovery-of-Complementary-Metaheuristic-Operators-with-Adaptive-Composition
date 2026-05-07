**Idea: Rank-Deficiency Pseudoinverse Velocity Correction**

Category B: Spectral / linear-algebraic. Uses pseudoinverse correction when population covariance becomes rank-deficient, detected via singular value threshold. Projects velocity onto principal subspace and amplifies collapsed directions using inverse singular value weighting. Targets worst tasks (17, 16, 6, 11) where population collapses to subspaces.

```python
def _position_update_centroid_knn(self):
    """Pseudoinverse velocity correction for rank-deficient populations (Category B).
    
    Detects when population covariance becomes ill-conditioned (low-rank) using
    singular value thresholding. Uses Moore-Penrose pseudoinverse to project and
    correct velocity onto the principal subspace, amplifying collapsed directions.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        
        singular_values = np.clip(singular_values, 1e-10, None)
        total_var = np.sum(singular_values ** 2) + 1e-10
        
        sv_norm = singular_values / (singular_values[0] + 1e-10)
        
        # Detect rank deficiency: how many singular values are significant
        significant_sv = np.sum(sv_norm > 0.01)
        rank_ratio = significant_sv / len(singular_values)
        
        # Condition number for anisotropic scaling
        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        
        # Effective dimensionality from cumulative variance
        cumvar = np.cumsum(singular_values ** 2) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim
        
        # Inverse singular value weighting: amplify collapsed directions
        # Low sv_norm → high amplification factor
        inv_sv_weight = 1.0 / (sv_norm + 0.05)
        inv_sv_weight = inv_sv_weight / (np.max(inv_sv_weight) + 1e-10)
        
        # Blend: well-conditioned (high eff_dim_ratio) → less correction
        # Ill-conditioned (low eff_dim_ratio) → stronger pseudoinverse correction
        correction_strength = np.clip(1.0 - eff_dim_ratio, 0.0, 0.8)
        
        # Project velocity onto principal axes
        vel_proj = self.velocity @ Vt.T
        
        # Scale each component: inverse SV weighting with blend
        uniform_scale = np.ones(len(singular_values))
        per_comp_scale = uniform_scale * (1.0 - correction_strength) + inv_sv_weight * correction_strength
        
        # Additional condition-number-driven scaling for very ill-conditioned cases
        if cond > 100.0:
            log_cond = np.log1p(cond) / np.log1p(1000.0)
            log_cond = np.clip(log_cond, 0.0, 1.0)
            high_cond_damp = 1.0 - 0.3 * log_cond
            per_comp_scale *= high_cond_damp
        
        vel_scaled = vel_proj * np.clip(per_comp_scale, 0.1, 3.0)
        
        # Reconstruct: velocity corrected via pseudoinverse principle
        corrected_vel = vel_scaled @ Vt
        
        # Combine with base velocity update (inertia already in velocity)
        new_population = self.population + corrected_vel
        
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```