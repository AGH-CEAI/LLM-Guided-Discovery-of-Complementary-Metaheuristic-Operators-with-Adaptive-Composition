**Idea: Eigenvalue-Shape-Adjusted Diversity**

Compute diversity using the effective sample size of the covariance matrix, capturing both spread and shape (elongation). This penalizes elongated, poorly-conditioned populations that waste exploration dimensions—critical for the worst tasks that appear to suffer from deceptive local optima and narrow basins.

```python
def _compute_diversity(self):
    """Compute diversity using effective sample size of the covariance matrix.
    
    Combines trace-based spread with eigenvalue-based shape penalty.
    ESS = geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues):
      ~1.0 for spherical (good) → full diversity
      ~0.0 for elongated (bad) → wasted dimensions
    
    Diversity = normalized_trace * sqrt(ESS) penalizes both:
      - Low overall spread (collapsed population)
      - High elongation (narrow ridge, CMA-ES losing rank)
    """
    # Spread: normalized trace
    diag = np.diag(self.C)
    trace = np.sum(diag)
    expected_trace = self.dim * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    trace_norm = trace / max(expected_trace, 1e-10)
    
    # Shape: effective sample size from eigenvalues
    eigvals = np.linalg.eigvalsh(self.C)
    eigvals = np.clip(eigvals, 1e-15, None)
    geo_mean = np.exp(np.mean(np.log(eigvals)))
    arith_mean = np.mean(eigvals)
    ess = geo_mean / max(arith_mean, 1e-15)
    
    # Combined metric
    diversity = trace_norm * np.sqrt(ess)
    return float(np.clip(diversity, 1e-15, None))
```