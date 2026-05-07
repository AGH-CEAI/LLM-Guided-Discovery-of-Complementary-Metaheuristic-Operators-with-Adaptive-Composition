**Idea: Spectral Condition-Adaptive DE**
Adapt F and CR using the condition number and effective dimensionality of the population's covariance matrix eigenspectrum, targeting anisotropic population states on task 12.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR based on spectral properties of population distribution.
    
    Category B: Uses eigendecomposition of population covariance to measure
    condition number (anisotropy) and effective dimensionality. When population
    collapses into ill-conditioned subspace, reduce F to avoid overshooting.
    """
    if not hasattr(self, 'population') or self.population is None or len(self.population) < self.dim + 1:
        self.F = np.clip(self.F * (1 + 0.1 * (improvement_rate - 0.2)), 0.3, 1.5)
        self.CR = np.clip(self.CR * (1 + 0.05 * (improvement_rate - 0.2)), 0.3, 0.95)
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        return
    
    pop = self.population
    
    # Center population and compute covariance
    centroid = pop.mean(axis=0)
    centered = pop - centroid
    
    # Population covariance matrix
    cov = np.cov(centered.T)
    
    # Handle singular/degenerate covariance
    if cov.ndim == 0 or np.linalg.matrix_rank(cov) < 2:
        self.F = np.clip(self.F * 1.1, 0.3, 1.5)
        self.CR = np.clip(self.CR * 0.95, 0.3, 0.95)
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        return
    
    # Eigendecomposition: extract spectral information
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-15)
    except np.linalg.LinAlgError:
        self.F = np.clip(self.F * 1.05, 0.3, 1.5)
        self.CR = np.clip(self.CR * 0.98, 0.3, 0.95)
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        return
    
    # Condition number: ratio of max to min eigenvalue (anisotropy measure)
    cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 1e-15 else 1e6
    
    # Effective dimensionality: fraction of eigenvalues carrying significant variance
    total_var = eigenvalues.sum()
    cumsum = np.cumsum(eigenvalues[::-1])[::-1]
    n_eff = np.sum(cumsum > 0.01 * total_var)
    eff_dim_ratio = n_eff / self.dim
    
    # Spectral-based adaptation rules
    # High condition number -> ill-conditioned, elongated subspace
    # Low effective dimensionality -> population collapsed to subspace
    if cond > 50 or eff_dim_ratio < 0.3:
        # Population in poorly-conditioned subspace: reduce F to avoid overshooting
        # Increase CR to exploit best directions
        self.F = np.clip(self.F * 0.88, 0.3, 1.2)
        self.CR = np.clip(self.CR * 1.03, 0.3, 0.95)
    elif cond < 10 and eff_dim_ratio > 0.7:
        # Well-conditioned, full-dimensional: more exploration possible
        self.F = np.clip(self.F * 1.08, 0.3, 1.5)
        self.CR = np.clip(self.CR * 0.97, 0.3, 0.95)
    else:
        # Moderate conditioning: moderate adjustment
        self.F = np.clip(self.F * (1 + 0.03 * (improvement_rate - 0.2)), 0.3, 1.5)
        self.CR = np.clip(self.CR * (1 + 0.02 * (0.2 - improvement_rate)), 0.3, 0.95)
    
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```