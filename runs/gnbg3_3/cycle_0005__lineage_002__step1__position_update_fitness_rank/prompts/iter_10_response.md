**Idea: Eigenvalue Entropy with Rank-Conditioned Velocity**
Spectral variant using eigenvalue entropy (normalized spread) combined with fitness rank for velocity modulation — distinct from condition-number-only approaches.
```python
def _position_update_fitness_rank(self):
    """Spectral: eigenvalue entropy + fitness-rank velocity modulation (variant_10).
    
    Uses eigenvalue spread (normalized variance of eigenspectrum) as primary
    spectral signal. Unlike condition-number approaches, this captures the
    full distribution shape. Fitness rank modulates velocity scale per-particle.
    """
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        total_var = np.sum(eigenvalues_sorted)
        
        if total_var > 1e-10:
            eigenvalues_normalized = eigenvalues_sorted / total_var
            eigenvalue_entropy = -np.sum(eigenvalues_normalized * np.log(eigenvalues_normalized + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            eigenvalue_spread = eigenvalue_entropy / (max_entropy + 1e-10)
        else:
            eigenvalue_spread = 1.0
        
        spectral_scale = 0.5 + 1.5 * eigenvalue_spread
        spectral_scale = np.clip(spectral_scale, 0.3, 2.5)
    except np.linalg.LinAlgError:
        spectral_scale = 1.0

    fitness_scale = 1.0 - 0.4 * fitness_ranks
    vel_scale = spectral_scale * fitness_scale
    vel_scale = np.clip(vel_scale, 0.1, 3.0)

    new_population = self.population + self.velocity * vel_scale[:, np.newaxis]
    self.population = self._clip_to_bounds(new_population)
```