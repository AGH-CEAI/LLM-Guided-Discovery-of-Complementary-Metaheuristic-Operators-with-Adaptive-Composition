**Idea: Radius-Condition Diversity**
Measures population spread (RMS distance from centroid) normalized by expected search space radius, with a penalty for ill-conditioned covariance matrices to detect collapsed/elongated distributions.
```python
def _compute_diversity(self):
    """Compute diversity as normalized effective radius with condition number penalty."""
    pop = self.population
    n, dim = pop.shape
    
    # Compute centroid and effective radius (RMS distance from centroid)
    centroid = np.mean(pop, axis=0)
    diffs = pop - centroid
    dists_sq = np.sum(diffs ** 2, axis=1)
    effective_radius = np.sqrt(np.mean(dists_sq) + 1e-15)
    
    # Expected radius for uniform distribution over hypercube
    # For uniform in [-a, a]: variance = (2a)^2/12 = a^2/3, so std = a/sqrt(3)
    # Approximate expected diameter = sqrt(dim) * a / sqrt(3)
    bound_range = (self.ub[0] - self.lb[0]) / 2.0
    expected_radius = bound_range * np.sqrt(dim / 3.0)
    
    # Normalized radius: 0 = collapsed, 1 = well-spread
    radius_ratio = effective_radius / (expected_radius + 1e-15)
    radius_ratio = np.clip(radius_ratio, 0.0, 1.0)
    
    # Condition number penalty: reduce score for ill-conditioned covariance
    # This detects when population is trapped in narrow ridge/basin
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.maximum(np.min(eigvals), 1e-15)
    eig_max = np.maximum(np.max(eigvals), 1e-15)
    cond = eig_max / eig_min
    
    # Log-scaled penalty: cond=1e3 reduces score by ~70%, cond=1e6 by ~85%
    cond_penalty = 1.0 / (1.0 + 0.2 * np.log1p(cond))
    
    return radius_ratio * cond_penalty
```