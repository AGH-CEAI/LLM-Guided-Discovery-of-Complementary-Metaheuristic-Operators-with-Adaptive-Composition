**Idea: Spectral Condition-Number-Driven Parameter Adaptation**

Adapt F and Cr based on the population covariance condition number (spectral anisotropy). High condition number indicates an elongated population in a narrow valley → increase F for exploration, decrease Cr to allow more diversity. Track EMA of condition number to detect directional changes in population geometry.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr based on population covariance condition number (spectral anisotropy)."""
    np_pop = len(improved_mask)
    
    # Compute condition number from population covariance (spectral measure)
    centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(centered.T)
    
    try:
        s = np.linalg.svd(cov, compute_uv=False)
        s = s[s > 1e-12]
        if len(s) >= 2 and s[-1] > 1e-12:
            cond = s[0] / s[-1]
        else:
            cond = 1.0
    except Exception:
        cond = 1.0
    
    # Initialize EMA state on first call
    if not hasattr(self, 'cond_ewma'):
        self.cond_ewma = cond
        self.cond_ewma_long = cond
    
    # Update EMAs
    self.cond_ewma = 0.3 * cond + 0.7 * self.cond_ewma
    self.cond_ewma_long = 0.1 * cond + 0.9 * self.cond_ewma_long
    
    # Momentum factor based on recent condition number trend
    momentum = np.clip(self.cond_ewma / (self.cond_ewma_long + 1e-10), 0.7, 1.4)
    
    # Detect regime based on condition number relative to history
    cond_trending_up = self.cond_ewma > self.cond_ewma_long
    cond_high = cond > self.cond_ewma_long
    cond_low = cond < self.cond_ewma_long
    
    # Spectral regime detection
    if cond_trending_up and cond_high:
        # Population becoming more elongated (high anisotropy) - need more exploration
        F_adjustment = 1.15 * momentum
        Cr_adjustment = 0.85
    elif not cond_trending_up and cond_low:
        # Population becoming more isotropic - can focus exploitation
        F_adjustment = 0.88 / momentum
        Cr_adjustment = 1.10
    elif cond > 100:
        # Very ill-conditioned - aggressive exploration
        F_adjustment = 1.25
        Cr_adjustment = 0.80
    elif cond < 2.0:
        # Well-conditioned - conservative exploitation
        F_adjustment = 0.92
        Cr_adjustment = 1.05
    else:
        # Neutral regime - gentle correction
        delta = self.cond_ewma - self.cond_ewma_long
        F_adjustment = 1.0 + 0.10 * delta / (np.abs(self.cond_ewma_long) + 1e-10)
        Cr_adjustment = 1.0 - 0.08 * delta / (np.abs(self.cond_ewma_long) + 1e-10)
    
    # Apply adjustments
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```