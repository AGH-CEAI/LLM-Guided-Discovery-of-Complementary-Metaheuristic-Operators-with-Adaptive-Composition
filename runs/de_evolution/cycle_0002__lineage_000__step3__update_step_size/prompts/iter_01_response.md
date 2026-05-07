**Idea: CSA with Median-Based Damping and Fitness-Aware Sigma Adaptation**
Combines the standard CSA path-length control with a fitness-aware component that shrinks sigma more aggressively when close to optimum and uses a median-based damping factor for robustness across dimensions.

```python
def _update_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Standard CSA update
    csa_factor = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
    
    # Adaptive damping based on dimension and progress
    # When sigma is very large relative to search space, increase damping
    search_range = self.ub - self.lb
    relative_sigma = self.sigma / search_range
    
    # Extra damping when sigma is disproportionately large or small
    if relative_sigma > 0.5:
        extra_damp = 1.0 + 0.5 * np.log(relative_sigma / 0.5)
        csa_factor /= extra_damp
    
    # Fitness-aware component: when we're close to a good solution,
    # be more conservative with sigma increases but allow decreases
    if self.best_fitness < 1e-4 and self.best_fitness > 0:
        # Near optimum: reduce sigma increase rate
        if csa_factor > 0:
            # Scale down increases when close to optimum
            scale = max(0.1, min(1.0, np.log10(self.best_fitness + 1e-20) / (-4.0)))
            csa_factor *= scale
    
    # Apply the update with numerical safety
    # Clip the exponent to prevent extreme jumps
    csa_factor = np.clip(csa_factor, -0.5, 0.5)
    self.sigma *= np.exp(csa_factor)
    
    # Condition-number aware lower bound: if covariance is well-conditioned,
    # allow smaller sigma
    if np.min(self.D) > 0:
        cond = np.max(self.D) / np.min(self.D)
        if cond > 1e6:
            # Poor conditioning: don't let sigma get too small
            min_sigma = 1e-15
        else:
            min_sigma = 1e-20
    else:
        min_sigma = 1e-20
    
    self.sigma = np.clip(self.sigma, min_sigma, 1e6)
```