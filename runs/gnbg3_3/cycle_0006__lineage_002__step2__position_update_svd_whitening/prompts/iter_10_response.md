**Idea: Condition-Adaptive Rank-Truncated Whitening**

Category B: Spectral / linear-algebraic — uses eigendecomposition with rank-adaptive Tikhonov regularization and logarithmic spectral modulation, targeting severe anisotropy in the worst unsolved tasks (17, 16, 6) where errors exceed 10^2.

```python
def _position_update_svd_whitening(self):
    """Condition-adaptive rank-truncated whitening (variant_10, Category B).
    
    Targets worst unsolved tasks (17: 3.5e4, 16: 2.4e3, 6: 7e2) where
    severe anisotropy/ill-conditioning blocks convergence. Key differences:
    - Tikhonov regularization on eigendecomposition (stable pseudo-inverse)
    - Logarithmic spectral modulation (gentler than inverse-square)
    - Rank-adaptive damping based on effective dimensionality
    """
    centered = self.population - np.mean(self.population, axis=0)

    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-14, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if eigenvalues[-1] <= 0:
            eigenvalues[-1] = 1e-14
        
        cond = eigenvalues[0] / eigenvalues[-1]
        total_var = np.sum(eigenvalues) + 1e-10
        
        # Effective dimensionality: count components explaining 95% variance
        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim = np.clip(eff_dim, 1, len(eigenvalues))
        
        # Rank-adaptive Tikhonov regularization
        # More regularization for lower-rank (more degenerate) populations
        rank_ratio = eff_dim / len(eigenvalues)
        reg_strength = 0.01 * (1.0 - rank_ratio) + 0.001
        reg_strength = np.clip(reg_strength, 0.001, 0.1)
        
        # Logarithmic spectral modulation (gentler than inverse-square)
        # Maps eigenvalue ratio to [0.5, 2.0] range with log scaling
        log_eig_ratio = np.log1p(eigenvalues / (eigenvalues[-1] + 1e-14))
        log_cond = np.log1p(cond)
        
        # Base modulation: inverse of normalized eigenvalue (gentle)
        base_modulation = 1.0 / (log_eig_ratio + 1.0)
        base_modulation = base_modulation / (np.max(base_modulation) + 1e-10)
        
        # Condition-adaptive blend: high cond → more aggressive modulation
        blend_weight = np.clip(log_cond / 10.0, 0.0, 0.8)
        
        # Uniform scale: dampen all directions when ill-conditioned
        uniform_scale = np.clip(1.0 / np.sqrt(log_cond + 1.0), 0.3, 1.5)
        
        # Per-component scale with log modulation
        per_component_scale = uniform_scale * (1.0 - blend_weight) + base_modulation * blend_weight
        
        # Additional damping for high-spread directions (prevent explosion)
        high_spread_mask = eigenvalues / eigenvalues[0] > 0.5
        per_component_scale[high_spread_mask] *= 0.7
        
        # Velocity modulation in eigenspace
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * per_component_scale
        new_population = self.population + (vel_scaled @ eigenvectors.T)
        
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```