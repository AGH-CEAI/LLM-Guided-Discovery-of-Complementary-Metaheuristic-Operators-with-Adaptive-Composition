Looking at the problem, I notice the worst unsolved tasks (16-23) have errors stuck at ~1e+01, indicating the algorithm is converging to local optima but failing to escape. The `_compute_diversity` function is critical because it triggers restarts via `_restart_if_needed` — if diversity is underestimated, restarts won't trigger; if overestimated, convergence is disrupted.

The current metric uses mean per-dimension std, which is a simple average that may not capture the true "spread volume" of the population, especially in high-dimensional spaces where different dimensions may have very different scales.

**Idea: Geometric Mean of Eigenvalue Spread**
A fundamentally different approach using the geometric mean of covariance eigenvalues, which directly measures the effective volume of the search distribution. This is invariant to rotations and captures the true multi-dimensional spread rather than averaging per-dimension.

```python
def _compute_diversity(self):
    """Compute population diversity via geometric mean of covariance eigenvalues.
    
    Uses the geometric mean of eigenvalues of the covariance matrix, which:
    - Is rotation-invariant (measures true volume, not axis-aligned spread)
    - Is more sensitive to collapse than mean-per-dim std
    - Captures multi-dimensional effective sample size
    """
    # Compute population covariance with numerical stability
    pop_centered = self.population - self.mean
    cov = pop_centered.T @ pop_centered / max(self.NP - 1, 1)
    
    # Add small regularization for numerical stability
    cov += 1e-12 * np.eye(self.dim)
    
    # Eigenvalue decomposition
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-15)  # Ensure positive
    
    # Geometric mean of eigenvalues = (product)^(1/dim) = exp(mean(log(eigvals)))
    # This measures the "radius" of the equivalent spherical distribution
    log_eigvals = np.log(eigvals)
    geo_mean = np.exp(np.mean(log_eigvals))
    
    # Scale by dimension-appropriate factor for fair comparison with bounds
    expected_geo_mean = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    scaled_diversity = geo_mean / (expected_geo_mean + 1e-10)
    
    return np.clip(scaled_diversity, 0.0, 1.0)
```