**Idea: Spectral Condition-Adaptive Parameter Control**

Leverages the condition number and effective rank of the population covariance matrix to detect anisotropy and subspace collapse, then adjusts F and Cr accordingly. When eigenvalues are highly imbalanced (high condition number), the population is stuck in a degenerate subspace — increase F for escape. When effective dimensionality is low, bias toward exploration (lower Cr). Category B: Spectral / linear-algebraic.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr based on spectral properties of population covariance."""
    success_rate = np.mean(improved_mask)
    
    self.F_success_history.append((success_rate, F_used))
    self.Cr_success_history.append((success_rate, Cr_used))
    
    if len(self.F_success_history) > self.adaptation_window:
        self.F_success_history.pop(0)
        self.Cr_success_history.pop(0)
    
    # Compute spectral properties of population for parameter guidance
    pop_centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(pop_centered.T)
    
    # Eigenvalue decomposition of covariance
    try:
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        # Condition number: ratio of max to min eigenvalue
        cond_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-12 else 1e6
        
        # Effective rank: entropy-based measure of dimensionality
        eig_total = np.sum(eigenvalues)
        eig_normalized = eigenvalues / eig_total
        eig_normalized = eig_normalized[eig_normalized > 1e-12]
        effective_rank = np.exp(-np.sum(eig_normalized * np.log(eig_normalized + 1e-12)))
        
        # Spectral spread: ratio of top eigenvalue to mean of rest
        if len(eigenvalues) > 1:
            spectral_dominance = eigenvalues[0] / np.mean(eigenvalues[1:])
        else:
            spectral_dominance = 1.0
            
    except np.linalg.LinAlgError:
        cond_number = 1e4
        effective_rank = 1.0
        spectral_dominance = 10.0
    
    # Base adaptation from success history
    if len(self.F_success_history) >= 5:
        recent_success = [s for s, _ in self.F_success_history[-5:]]
        avg_success = np.mean(recent_success)
        
        if avg_success > 0.3:
            F_adjustment = 1.1
            Cr_adjustment = 0.95
        elif avg_success < 0.1:
            F_adjustment = 0.9
            Cr_adjustment = 1.05
        else:
            F_adjustment = 1.0
            Cr_adjustment = 1.0
    else:
        F_adjustment = 1.0
        Cr_adjustment = 1.0
    
    # Spectral-based modulation: detect degenerate subspaces
    dim_ratio = effective_rank / self.dim
    
    if cond_number > 100 or dim_ratio < 0.3:
        # Population is anisotropic/collapsed — need exploration
        # Increase F to jump out, decrease Cr to recombine more dimensions
        F_adjustment *= 1.15
        Cr_adjustment *= 0.90
    elif spectral_dominance > 5.0:
        # One direction dominates — moderate increase in exploration
        F_adjustment *= 1.08
        Cr_adjustment *= 0.95
    elif dim_ratio > 0.7 and cond_number < 10:
        # Well-spread population — can be more exploitative
        F_adjustment *= 0.95
        Cr_adjustment *= 1.02
    
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```