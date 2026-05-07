**Idea: Log-Volume Covariance Diversity**

Measures the log-volume of the population's covariance ellipsoid normalized by search space volume. Unlike mean-per-dimension std, this captures multi-dimensional spread, correlations, and detects when population collapses to low-dimensional manifolds (critical for the worst tasks stuck at ~1e+1 errors).

```python
def _compute_diversity(self):
    """Compute population diversity via normalized covariance ellipsoid volume.
    
    Measures the log-volume of the population's covariance ellipsoid,
    normalized by the expected volume under uniform distribution.
    - Detects population collapse to lower-dimensional manifolds
    - Captures inter-dimensional correlations
    - More robust than mean std for detecting stagnation
    """
    pop_centered = self.population - self.mean
    cov = np.cov(pop_centered.T)
    
    # Ensure positive definite
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 1e-10:
        cov += (1e-9 - min_eig) * np.eye(self.dim)
    
    # Normalize by expected variance of uniform distribution
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    cov_normalized = cov / expected_var
    
    # Log-determinant via SVD (numerically stable)
    sign, logdet = np.linalg.slogdet(cov_normalized)
    
    # Handle non-positive determinants (collapse to manifold)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e-30
    
    # Normalize by dimension count for scale invariance
    return np.exp(logdet / self.dim)
```