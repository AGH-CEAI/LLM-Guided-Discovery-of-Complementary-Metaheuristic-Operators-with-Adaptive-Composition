**Idea: Spectral Condition-Number Driven F/CR Adaptation**
Use the condition number and effective dimensionality of the population covariance matrix to drive F and CR adaptation. When the population is elongated (high condition number), reduce F and increase CR to exploit the low-dimensional subspace. When isotropic, increase F and reduce CR to explore more aggressively. Category B: Spectral / linear-algebraic, targeting task 15.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using spectral analysis of population covariance.
    
    Uses condition number and effective dimensionality of the population
    to infer anisotropy and guide parameter adaptation.
    """
    if len(self.F_history) == 0:
        return
    
    # Compute covariance matrix of population
    # Use population matrix directly (NP x dim)
    pop_centered = self.population - self.population.mean(axis=0)
    
    # Compute condition number via SVD of centered population
    # This avoids explicit covariance computation and is more numerically stable
    try:
        # Use SVD on centered population
        _, s, _ = np.linalg.svd(pop_centered, full_matrices=False)
        
        # Condition number: ratio of max to min singular value
        # Add small epsilon to avoid division by zero
        eps = 1e-12
        cond = s[0] / (s[-1] + eps) if len(s) > 0 else 1.0
        
        # Effective dimensionality: how many directions have significant variance
        # Based on ratio of cumulative variance
        total_var = np.sum(s**2)
        cumsum = np.cumsum(s**2)
        eff_dim = np.searchsorted(cumsum, 0.95 * total_var) + 1 if total_var > eps else 1
        eff_dim = np.clip(eff_dim, 1, self.dim)
        
        # Normalize condition number (typical range: 1 to ~1000)
        log_cond = np.log1p(cond)
        cond_norm = np.clip(log_cond / 7.0, 0.0, 1.0)
        
        # Effective dimensionality ratio
        dim_ratio = eff_dim / self.dim
        
        # F adaptation: high anisotropy -> smaller F, low anisotropy -> larger F
        # Base adjustment from condition number
        f_adjustment = 0.4 * (1.0 - cond_norm) + 0.2 * (1.0 - dim_ratio)
        target_F = np.clip(0.6 + f_adjustment, 0.3, 1.2)
        
        # CR adaptation: high anisotropy -> higher CR (exploit subspace), 
        # low anisotropy -> lower CR (explore)
        cr_adjustment = 0.3 * cond_norm + 0.2 * dim_ratio
        target_CR = np.clip(0.85 + cr_adjustment, 0.5, 0.98)
        
    except Exception:
        # Fallback to improvement rate only if spectral analysis fails
        target_F = 0.6 + 0.3 * improvement_rate
        target_CR = 0.85 - 0.2 * improvement_rate
    
    # Momentum-based smoothing
    momentum = 0.7
    new_F = momentum * self.F_history[-1] + (1 - momentum) * target_F
    new_CR = momentum * self.CR_history[-1] + (1 - momentum) * target_CR
    
    # Apply with bounded changes
    max_change_F = 0.15
    max_change_CR = 0.1
    
    F_diff = np.clip(new_F - self.F, -max_change_F, max_change_CR)
    CR_diff = np.clip(new_CR - self.CR, -max_change_CR, max_change_CR)
    
    self.F = np.clip(self.F + F_diff, 0.3, 1.2)
    self.CR = np.clip(self.CR + CR_diff, 0.5, 0.98)
    
    # Track history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
    
    # Limit history length
    max_history = 100
    if len(self.F_history) > max_history:
        self.F_history = self.F_history[-max_history:]
        self.CR_history = self.CR_history[-max_history:]
```