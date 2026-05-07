**Idea: Covariance Condition Number Adaptation**

One-line description: Analyze population covariance eigenvalues to detect ill-conditioning and adapt F/Cr based on condition number and effective dimensionality (Category B: Spectral / linear-algebraic).

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using spectral analysis of population covariance matrix.
    
    Category B: Uses eigenvalues and condition number to detect:
    - Anisotropy (elongated distributions) → increase exploration
    - Collapse to low-dimensional subspace → large perturbation
    - Well-conditioned spherical → increase convergence pressure
    """
    success_rate = np.mean(improved_mask)
    
    # Initialize state on first call
    if not hasattr(self, 'cond_history'):
        self.cond_history = []
        self.eff_rank_history = []
        self.pop_spread_history = []
    
    # Compute current population spread and covariance
    centroid = np.mean(self.population, axis=0)
    centered = self.population - centroid
    
    # Covariance with population correction (ddof=1 for sample cov)
    cov = np.cov(centered, rowvar=False, ddof=1)
    
    # Fallback for edge cases (near-zero variance dimensions)
    if cov.size == 0 or np.any(np.diag(cov) <= 0):
        self.F = np.clip(self.F * (1.1 if success_rate > 0.2 else 0.9), 0.1, 1.5)
        self.Cr = np.clip(self.Cr * (1.05 if success_rate > 0.25 else 0.95), 0.1, 0.9)
        return
    
    # Eigenvalue decomposition to analyze population geometry
    try:
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-15)  # Numerical floor
        
        # Condition number: ratio of max to min eigenvalue
        # High cond → elongated/ill-conditioned → need more exploration
        cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 1e-15 else 1e10
        
        # Effective rank via entropy-based measure
        eig_sum = np.sum(eigenvalues)
        eig_norm = eigenvalues / eig_sum
        eig_norm = eig_norm[eig_norm > 1e-15]
        eff_rank = np.exp(-np.sum(eig_norm * np.log(eig_norm + 1e-15)))
        
        # Population spread: normalized by dimension
        pop_spread = np.sqrt(eigenvalues[-1]) if eigenvalues[-1] > 1e-15 else 1.0
        
    except np.linalg.LinAlgError:
        # Fallback if eigendecomposition fails
        cond = 100.0
        eff_rank = self.dim / 2
        pop_spread = 1.0
    
    # Track history for temporal smoothing
    self.cond_history.append(cond)
    self.eff_rank_history.append(eff_rank)
    self.pop_spread_history.append(pop_spread)
    
    if len(self.cond_history) > 10:
        self.cond_history.pop(0)
        self.eff_rank_history.pop(0)
        self.pop_spread_history.pop(0)
    
    # Smoothed metrics
    cond_smooth = np.mean(self.cond_history)
    eff_rank_frac = np.mean(self.eff_rank_history) / self.dim  # Normalized [0,1]
    
    # Spectral-based adaptation logic
    # High condition number → elongated landscape → need larger F for exploration
    # Low effective rank → population collapsed to subspace → need perturbation
    
    if cond_smooth < 10.0 and eff_rank_frac > 0.7:
        # Well-conditioned, diverse population: increase convergence pressure
        F_adjustment = 1.12
        Cr_adjustment = 1.10
    elif cond_smooth < 100.0:
        # Moderately ill-conditioned: balanced adjustment
        delta = np.clip((100.0 - cond_smooth) / 90.0, 0, 1)
        F_adjustment = 1.0 + 0.15 * delta
        Cr_adjustment = 1.0 - 0.08 * delta
    elif cond_smooth < 1000.0:
        # Ill-conditioned: emphasize exploration
        delta = np.clip((cond_smooth - 100.0) / 900.0, 0, 1)
        F_adjustment = 1.0 + 0.25 * delta
        Cr_adjustment = 1.0 - 0.15 * delta
    else:
        # Near-singular (cond >> 1000): population in degenerate subspace
        # Large F to escape, moderate Cr for diversity
        F_adjustment = 1.35
        Cr_adjustment = 0.85
    
    # Success rate modifier: if improving, be more aggressive; if failing, be more conservative
    if success_rate > 0.3:
        F_adjustment *= 1.08
        Cr_adjustment *= 1.05
    elif success_rate < 0.1:
        F_adjustment *= 0.92
        Cr_adjustment *= 0.95
    
    # Apply adjustments with bounds
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```