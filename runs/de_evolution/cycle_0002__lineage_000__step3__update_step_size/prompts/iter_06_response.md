Looking at the unsolved tasks, I see errors spanning from ~1e+3 (Task 16) down to ~1e-7 (Task 0). The existing variants seem to either use standard CSA step-size adaptation or minor tweaks. The key insight is that for high-dimensional, multimodal, or ill-conditioned problems (Tasks 16, 19, 20, 23, 10, 11), the step size often either converges prematurely (sigma shrinks too fast) or explodes. For tasks near convergence (Tasks 0, 1, 6), sigma doesn't shrink precisely enough.

My strategy: implement a **median-based step size adaptation** inspired by the Median Success Rule (MSR) combined with cumulative step-size adaptation. The idea is to use the actual fitness improvement signal to modulate sigma, not just the evolution path length. When the median offspring is better than the median parent, increase sigma; otherwise decrease it. This gives a more robust signal on multimodal/rugged landscapes where the evolution path can be misleading. I also add a fitness-based precision mode for fine-tuning near the optimum.

**Idea: Fitness-Guided Median Success Rule Step Size**
Combines CSA with median success rule: uses fitness rank improvement to adjust sigma, preventing premature convergence on multimodal tasks and enabling precise convergence near optima.

```python
def _update_step_size(self):
    # Standard CSA component
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    csa_factor = np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
    
    # Median Success Rule component
    # Track fitness to compute success rate
    if not hasattr(self, '_msr_prev_median'):
        self._msr_prev_median = None
        self._msr_s = 0.0  # smoothed success rate
        self._msr_target = 0.25  # target success rate (1/4 rule generalization)
        self._msr_c = 0.3  # learning rate for success rate
        self._msr_damping = 2.0 + self.dim / self.pop_size
    
    # Estimate current population median fitness from recent tracking
    # Use best_fitness trend as a proxy for success
    if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) >= 2:
        recent_best = self.best_fitness_history[-1]
        prev_best = self.best_fitness_history[-2] if len(self.best_fitness_history) >= 2 else recent_best
        
        # Success if improvement occurred
        success = 1.0 if recent_best < prev_best - 1e-30 * abs(prev_best) else 0.0
        self._msr_s = (1.0 - self._msr_c) * self._msr_s + self._msr_c * success
        
        # MSR factor: increase sigma if success rate > target, decrease otherwise
        msr_exponent = (self._msr_s - self._msr_target) / (1.0 - self._msr_target + 1e-30)
        msr_factor = np.exp(msr_exponent / self._msr_damping)
    else:
        msr_factor = 1.0
    
    # Blend CSA and MSR based on current error regime
    current_error = self.best_fitness if np.isfinite(self.best_fitness) else 1e10
    
    if current_error > 1e2:
        # Far from optimum: trust MSR more (better for multimodal)
        alpha = 0.3  # weight on CSA
    elif current_error > 1e-2:
        # Intermediate: balanced
        alpha = 0.6
    else:
        # Near optimum: trust CSA more (better path geometry)
        alpha = 0.85
    
    # Geometric blend of the two factors
    combined_factor = csa_factor ** alpha * msr_factor ** (1.0 - alpha)
    
    # Apply with safety clipping on the factor itself
    combined_factor = np.clip(combined_factor, 0.5, 2.0)
    
    self.sigma *= combined_factor
    
    # Adaptive sigma floor based on error magnitude
    if current_error > 1e-4:
        sigma_min = 1e-12
    elif current_error > 1e-8:
        sigma_min = 1e-16
    else:
        sigma_min = 1e-20
    
    self.sigma = np.clip(self.sigma, sigma_min, 1e6)
```