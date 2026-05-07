**Idea: Eigenvalue-Weighted Neighborhood Adaptation**

Adapts neighborhood size using condition number and spectral entropy of the population covariance matrix. High condition number (stretched/collapsed subspace) → larger neighborhood for exploration; high spectral entropy (isotropic spread) → smaller neighborhood for exploitation. This gives a fundamentally different angle from diversity-based approaches by directly measuring anisotropy and effective dimensionality of the search landscape.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology neighborhood size using spectral properties of population.
    
    Uses condition number and spectral entropy of population covariance to detect:
    - High condition number → anisotropic/ill-conditioned → larger neighborhood
    - Low spectral entropy → collapsed subspace → larger neighborhood
    - Isotropic distribution → smaller neighborhood for exploitation
    
    This is orthogonal to diversity-based approaches (Category A) because it
    directly measures the eigenvalue structure of the population distribution.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Condition number: ratio of max/min eigenvalue
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 1e6)
        
        # Spectral entropy: how uniformly spread are eigenvalues?
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            eig_norm = eigenvalues / total_var
            eig_norm = eig_norm[eig_norm > 1e-10]  # avoid log(0)
            spectral_entropy = -np.sum(eig_norm * np.log(eig_norm))
            max_entropy = np.log(self.dim)
            entropy_ratio = spectral_entropy / (max_entropy + 1e-10)
        else:
            entropy_ratio = 0.0
        
        # Combined spectral signal: high cond OR low entropy → need larger neighborhood
        cond_signal = np.log1p(cond) / np.log1p(1e6)
        spectral_signal = 0.5 * cond_signal + 0.5 * (1.0 - entropy_ratio)
        
        # Nonlinear mapping: spectral signal -> neighborhood scale factor
        # Range: [0.4, 1.5] (well-conditioned -> ill-conditioned)
        neighborhood_scale = 0.4 + 1.1 * spectral_signal
        
        target_size = int(self.neighborhood_size * neighborhood_scale)
        target_size = np.clip(target_size, 1, self.np // 2)
        
        if target_size > self.neighborhood_size:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif target_size < self.neighborhood_size:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
            
    except np.linalg.LinAlgError:
        # Fallback: use simple diversity-based rule
        diversity = self._compute_diversity()
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
```