**Idea: Spectral Covariance-Adaptive Parameters**
One-line description: Adapt F and Cr by analyzing population covariance eigenvalues — detect collapse via condition number and effective rank, then scale F/Cr based on anisotropy rather than raw success rates.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr based on spectral analysis of population covariance.
    
    Category B: Uses eigenvalues to detect population collapse (ill-conditioning)
    and anisotropy (eigenvalue spread), driving parameter adaptation through
    the geometric structure of the search distribution rather than fitness history.
    """
    success_rate = np.mean(improved_mask)
    
    # Record history for potential fallback
    self.F_success_history.append((success_rate, F_used))
    self.Cr_success_history.append((success_rate, Cr_used))
    
    if len(self.F_success_history) > self.adaptation_window:
        self.F_success_history.pop(0)
        self.Cr_success_history.pop(0)
    
    # Build population covariance and extract spectral properties
    centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(centered.T)
    
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 1e-15)  # Numerical floor
        
        # Condition number: large value = elongated distribution
        cond_num = eigvals[0] / eigvals[-1] if eigvals[-1] > 1e-12 else 1e6
        
        # Effective rank via entropy-based measure
        eigvals_norm = eigvals / np.sum(eigvals)
        entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-15))
        max_entropy = np.log(len(eigvals))
        eff_rank = np.exp(entropy) if max_entropy > 0 else 1.0
        eff_rank_norm = eff_rank / self.dim
        
        # Detect population collapse
        is_collapsed = np.sum(eigvals < 1e-10) > self.dim * 0.5
    except Exception:
        cond_num = 1.0
        eff_rank_norm = 0.5
        is_collapsed = False
    
    # Spectral-driven adaptation
    if is_collapsed:
        # Population collapsed: inject maximum diversity
        self.F = np.clip(self.F * 1.5, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * 1.15, 0.1, 0.9)
    elif cond_num > 100:
        # Highly anisotropic: increase exploration
        self.F = np.clip(self.F * 1.15, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * 1.05, 0.1, 0.9)
    elif cond_num < 5 and eff_rank_norm > 0.7:
        # Near-isotropic with good diversity: fine-tune exploitation
        self.F = np.clip(self.F * 0.92, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * 0.95, 0.1, 0.9)
    else:
        # Default: moderate adjustment based on success rate
        if len(self.F_success_history) >= 5:
            recent_success = [s for s, _ in self.F_success_history[-5:]]
            avg_success = np.mean(recent_success)
            
            if avg_success > 0.3:
                self.F = np.clip(self.F * 1.05, 0.1, 1.5)
                self.Cr = np.clip(self.Cr * 0.97, 0.1, 0.9)
            elif avg_success < 0.1:
                self.F = np.clip(self.F * 0.95, 0.1, 1.5)
                self.Cr = np.clip(self.Cr * 1.02, 0.1, 0.9)
```