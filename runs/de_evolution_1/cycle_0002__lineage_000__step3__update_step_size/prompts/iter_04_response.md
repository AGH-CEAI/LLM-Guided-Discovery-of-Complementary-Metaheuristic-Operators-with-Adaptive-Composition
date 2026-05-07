Looking at the unsolved tasks, I see errors spanning from ~1e+3 (Task 16) down to ~1e-7 (Task 0). The original and variant_01 perform best on most tasks, suggesting the standard CMA-ES step size update is reasonable but convergence is too slow or gets stuck. Variant_02 catastrophically fails on many tasks (likely too aggressive), while variant_03 also degrades on several.

**Key insight for the worst tasks:** Tasks 16, 19, 20, 23 have errors stuck at large values (1-900), suggesting the algorithm gets trapped in local optima and restarts aren't effective enough. The step size adaptation needs to be more aggressive about *increasing* sigma when far from the optimum (to enable exploration/escape), while being more precise about *decreasing* sigma when close (to enable fine convergence).

**My approach:** A fitness-aware adaptive step size rule that:
1. Uses the standard CSA update as a base
2. Applies a multiplicative correction based on the current fitness level and improvement trend
3. When fitness is large and stagnating, injects upward pressure on sigma to promote exploration
4. When fitness is small, allows tighter convergence with a reduced damping factor
5. Tracks a short-term improvement ratio to detect when sigma should be boosted

This is fundamentally different from prior variants because it couples the step size update directly to the fitness landscape feedback, not just the evolution path geometry.

**Idea: Fitness-Coupled Adaptive Sigma with Exploration Boost**
Combines standard CSA with fitness-level-dependent damping and stagnation-triggered sigma boosts for better exploration on hard multimodal tasks.
```python
def _update_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Standard CSA ratio
    csa_ratio = p_sigma_norm / self.chi_n - 1.0
    
    # Adaptive damping based on fitness level
    bf = self.best_fitness if np.isfinite(self.best_fitness) else 1e10
    
    if bf > 1e2:
        # Far from optimum: reduce damping to allow larger sigma changes
        effective_d = self.d_sigma * 0.5
    elif bf > 1e0:
        effective_d = self.d_sigma * 0.7
    elif bf > 1e-4:
        effective_d = self.d_sigma * 0.85
    else:
        # Near optimum: increase damping slightly for stability
        effective_d = self.d_sigma * 1.1
    
    # Base update
    self.sigma *= np.exp((self.c_sigma / max(effective_d, 1e-10)) * csa_ratio)
    
    # Stagnation-triggered exploration boost
    # If stagnating and fitness is still large, push sigma upward
    if hasattr(self, 'stagnation_counter') and hasattr(self, 'best_fitness'):
        stag = self.stagnation_counter
        
        if bf > 1e2 and stag > 3:
            # Strong boost when far from optimum and stagnating
            boost = 1.0 + 0.05 * min(stag, 20)
            self.sigma *= boost
        elif bf > 1e0 and stag > 5:
            boost = 1.0 + 0.02 * min(stag, 15)
            self.sigma *= boost
        elif bf > 1e-2 and stag > 10:
            boost = 1.0 + 0.01 * min(stag, 10)
            self.sigma *= boost
    
    # Prevent sigma from collapsing too early when fitness is still large
    if bf > 1e0:
        min_sigma = 1e-3 * (1.0 + np.log10(max(bf, 1.0)))
        self.sigma = max(self.sigma, min_sigma)
    elif bf > 1e-4:
        self.sigma = max(self.sigma, 1e-8)
    
    # Anti-explosion: if sigma is getting too large relative to search space
    max_sigma = min(1e6, 50.0 * (self.ub - self.lb))
    
    self.sigma = np.clip(self.sigma, 1e-20, max_sigma)
```