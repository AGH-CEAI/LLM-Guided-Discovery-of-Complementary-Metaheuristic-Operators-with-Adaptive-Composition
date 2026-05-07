**Idea: Eigenvalue-Weighted Principal Subspace Centroid Correction**

Uses spectral decomposition of population covariance to modulate centroid attraction. Key insight: condition number from eigenvalues directly quantifies anisotropy, enabling targeted correction. Velocity is projected onto principal components and scaled by eigenvalue magnitude, damping exploration along collapsed directions while preserving momentum along dominant axes.

```python
def _position_update_centroid_knn(self):
    """Eigenvalue-driven anisotropic centroid correction (Category B).
    
    Uses eigendecomposition of population covariance to:
    1. Detect anisotropy via condition number (eigenvalue ratio)
    2. Modulate centroid attraction strength by condition number
    3. Project centroid direction onto principal subspace
    4. Weight projected direction by eigenvalue magnitude (emphasize major axes)
    5. Scale velocity per principal component proportional to eigenvalue
    
    Targets worst tasks (17, 16, 6, 5) with high anisotropy.
    """
    centroid = np.mean(self.population, axis=0)
    
    # --- Spectral decomposition of population covariance ---
    try:
        centered = self.population - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Condition number: primary anisotropy signal
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 10000.0)
        
        # Effective dimensionality from cumulative eigenvalue variance
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var
        cumvar = np.cumsum(eigenvalues_norm)
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim
        
        # Spectral signals
        cond_signal = np.log1p(cond) / np.log1p(10000.0)
        cond_signal = np.clip(cond_signal, 0.0, 1.0)
        
        # --- Centroid attraction modulated by spectral signals ---
        to_centroid = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_dist
        
        # Project direction onto principal subspace
        proj_coeffs = to_centroid_dir @ eigenvectors
        proj_coeffs_norm = proj_coeffs / (np.linalg.norm(proj_coeffs, axis=1, keepdims=True) + 1e-10)
        
        # Weight by eigenvalue magnitude (emphasize major axes)
        eigen_weights = eigenvalues / (eigenvalues[0] + 1e-10)
        projected_dir = proj_coeffs_norm * eigen_weights
        
        # Reconstruct aligned direction
        aligned_dir = projected_dir @ eigenvectors.T
        aligned_dir = aligned_dir / (np.linalg.norm(aligned_dir, axis=1, keepdims=True) + 1e-10)
        
        # Strength inversely proportional to effective dimensionality ratio
        # Low eff_dim_ratio = concentrated in few axes = stronger correction needed
        correction_strength = 0.5 * (1.0 + cond_signal) * (2.0 - eff_dim_ratio)
        correction_strength = np.clip(correction_strength, 0.2, 3.0)
        
        centroid_correction = correction_strength * aligned_dir
        
        # --- Velocity modulation in principal subspace ---
        vel_proj = self.velocity @ eigenvectors
        vel_scale = 0.5 + 0.5 * eigen_weights
        vel_scale = np.clip(vel_scale, 0.3, 1.5)
        vel_scaled = vel_proj * vel_scale
        
        new_population = self.population + self.inertia_weight * (vel_scaled @ eigenvectors.T) + 0.3 * centroid_correction
        
    except np.linalg.LinAlgError:
        to_centroid = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_dist
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * to_centroid_dir
    
    self.population = self._clip_to_bounds(new_population)
```