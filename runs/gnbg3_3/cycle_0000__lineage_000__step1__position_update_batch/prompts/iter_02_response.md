**Idea: Eigenspace Velocity Scaling**

Use eigendecomposition of the population covariance matrix to scale velocity updates inversely to eigenvalue magnitudes (spectral inertia). This adapts movement based on the swarm's effective dimensionality and anisotropy — particles move more aggressively along under-explored eigendirections (small eigenvalues) and dampen movement along over-explored directions (large eigenvalues).

```python
def _position_update_batch(self):
    """Update positions using spectral velocity scaling based on population covariance."""
    # Compute population covariance matrix
    centroid = np.mean(self.population, axis=0)
    centered = self.population - centroid
    cov = (centered.T @ centered) / max(self.np - 1, 1)
    
    # Eigendecomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Clip eigenvalues to avoid numerical issues
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    
    # Compute condition number for spectral scaling intensity
    cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else 1.0
    # Scale intensity inversely with condition number (more aggressive for ill-conditioned)
    alpha = 0.5 / (1.0 + np.log1p(cond))
    
    # Spectral scaling: inverse eigenvalue weighting (small eigenvalue = more movement)
    spectral_weights = 1.0 / np.sqrt(eigenvalues + 1e-10)
    # Normalize to prevent extreme scaling
    spectral_weights = spectral_weights / np.max(spectral_weights)
    # Apply scaling (dampen high-eigenvalue directions, boost low-eigenvalue)
    scaling_factors = 1.0 + alpha * (1.0 - spectral_weights)
    
    # Project velocity to eigenspace
    vel_eigenspace = centered @ eigenvectors
    
    # Apply spectral scaling to velocity components
    vel_scaled = vel_eigenspace * scaling_factors
    
    # Reconstruct scaled velocity in original space
    velocity_contribution = vel_scaled @ eigenvectors.T
    
    # Update positions: base update + spectral correction
    new_population = self.population + self.velocity + 0.3 * velocity_contribution
    self.population = self._clip_to_bounds(new_population)
```