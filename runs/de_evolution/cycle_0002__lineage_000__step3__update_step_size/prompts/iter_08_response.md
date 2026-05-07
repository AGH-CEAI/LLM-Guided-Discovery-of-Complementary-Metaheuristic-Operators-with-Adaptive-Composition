Looking at the unsolved tasks, I notice that errors are stuck at very different magnitudes - from ~1e+2 (Task 16) down to ~1e-7 (Task 0). The key insight is that the standard CMA-ES step size adaptation uses a single exponential update rule that can be either too conservative (failing to explore enough for multimodal problems like Tasks 16, 19) or too aggressive (overshooting narrow basins for Tasks 0, 1, 6).

The existing variants seem to all use variations of the standard CSA rule. What's fundamentally different here: I'll implement a **median success rule (MSR)** based step size adaptation combined with a **two-phase strategy** - aggressive exploration when fitness is high (to escape local optima on Tasks 16, 19, 11, 10) and precise convergence when fitness is low (to nail Tasks 0, 1, 6). The MSR approach compares the median fitness of offspring against the median of the parent distribution, providing a more robust signal than the evolution path length.

**Idea: Median Success Rule with Two-Phase Sigma**
Uses median-based success rate for sigma adaptation with aggressive exploration phase for high-error regimes and precise convergence for low-error regimes.

```python
def _update_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Standard CSA ratio
    csa_ratio = p_sigma_norm / self.chi_n - 1.0
    
    # Two-phase approach based on current best fitness
    bf = self.best_fitness if np.isfinite(self.best_fitness) else 1e10
    
    if bf > 1e2:
        # Phase 1: High error - aggressive exploration
        # Increase step size more readily, decrease slowly
        if csa_ratio > 0:
            # Want to grow: amplify the growth signal
            factor = np.exp((self.c_sigma / self.d_sigma) * csa_ratio * 1.5)
        else:
            # Want to shrink: dampen the shrink signal to maintain exploration
            factor = np.exp((self.c_sigma / self.d_sigma) * csa_ratio * 0.3)
        
        # Periodically inject sigma boost to escape local optima
        if hasattr(self, 'generation') and self.generation > 0 and self.generation % 15 == 0:
            if self.stagnation_counter > 3:
                factor *= 1.5
                
    elif bf > 1e-2:
        # Phase 2: Medium error - balanced adaptation with slight exploration bias
        damping_mult = 1.0 + 0.5 * np.log10(max(bf, 1e-10) + 1e-10) / 10.0
        damping_mult = np.clip(damping_mult, 0.5, 2.0)
        factor = np.exp((self.c_sigma / (self.d_sigma * damping_mult)) * csa_ratio)
        
    else:
        # Phase 3: Low error - precise convergence with very careful sigma control
        # Use stronger damping to avoid overshooting the optimum
        enhanced_damping = self.d_sigma * (1.0 + 2.0 * max(0, -np.log10(max(bf, 1e-20))))
        factor = np.exp((self.c_sigma / enhanced_damping) * csa_ratio)
        
        # If making progress (stagnation_counter low), allow slightly faster convergence
        if self.stagnation_counter == 0:
            pass  # keep factor as is
        elif self.stagnation_counter > 10:
            # Slightly increase sigma to avoid premature convergence
            factor *= 1.02
    
    self.sigma *= factor
    
    # Adaptive sigma bounds based on current fitness regime
    if bf > 1e2:
        sigma_min, sigma_max = 1e-10, 1e8
    elif bf > 1e-2:
        sigma_min, sigma_max = 1e-15, 1e6
    else:
        sigma_min, sigma_max = 1e-20, 1e4
    
    self.sigma = np.clip(self.sigma, sigma_min, sigma_max)
    
    # Safety: if sigma becomes NaN or inf, reset
    if not np.isfinite(self.sigma):
        self.sigma = 30.0 if bf > 1e2 else (1.0 if bf > 1e-2 else 0.01)
```