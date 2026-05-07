Looking at the unsolved tasks, I see errors spanning from ~1e+3 (Task 16) down to ~1e-7 (Task 0). The key issue is that ALL tasks except Task 5 remain unsolved, with most stuck at errors of 1-1000. This suggests the step-size adaptation is either converging too fast (premature convergence to local optima) or not adapting aggressively enough to escape.

**Analysis of failure modes:**
- Tasks 16, 19, 11, 10 (errors 5-937): Likely multimodal/ill-conditioned functions where CMA-ES gets trapped. The step size shrinks too fast, preventing exploration.
- Tasks 20, 21, 23 (errors 4-6): Stuck at local optima - need step size to remain large enough for longer.
- Tasks closer to solution (0, 1, 6, 7): Need fine-tuning precision - step size should shrink smoothly near optimum.

**Key insight:** The standard CSA rule is purely based on path length. I'll implement a **median-fitness-triggered biphasic step size** that uses both the CSA mechanism AND direct fitness feedback. When fitness is stagnating at high values, it aggressively inflates sigma to escape; when fitness is improving and small, it uses a dampened CSA for precision. This combines exploration pressure with convergence guarantee.

**Idea: Fitness-Aware Biphasic CSA with Stochastic Perturbation**
Combines standard CSA with fitness-proportional sigma inflation when stuck at high errors, plus periodic multiplicative noise to break symmetry.

```python
def _update_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Standard CSA update
    csa_factor = np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
    
    # Fitness-aware correction: if best_fitness is large and stagnating, boost sigma
    fitness_val = self.best_fitness if np.isfinite(self.best_fitness) else 1e10
    
    # Track fitness improvement rate
    if not hasattr(self, '_sigma_fitness_prev'):
        self._sigma_fitness_prev = fitness_val
        self._sigma_no_improve_count = 0
        self._sigma_gen_counter = 0
    
    self._sigma_gen_counter += 1
    
    # Check improvement every few generations
    check_interval = max(3, self.dim // 2)
    if self._sigma_gen_counter % check_interval == 0:
        rel_improv = (self._sigma_fitness_prev - fitness_val) / (abs(self._sigma_fitness_prev) + 1e-30)
        if rel_improv < 1e-8:
            self._sigma_no_improve_count += 1
        else:
            self._sigma_no_improve_count = max(0, self._sigma_no_improve_count - 2)
        self._sigma_fitness_prev = fitness_val
    
    # Biphasic behavior based on fitness level and stagnation
    if fitness_val > 1.0 and self._sigma_no_improve_count > 3:
        # Aggressive: inflate sigma to escape local optima
        boost = 1.0 + 0.3 * min(self._sigma_no_improve_count, 10) / 10.0
        csa_factor = max(csa_factor, boost)
        # Add stochastic perturbation to break symmetry
        noise = np.exp(0.1 * np.random.randn())
        csa_factor *= noise
    elif fitness_val > 1e-2 and self._sigma_no_improve_count > 5:
        # Moderate boost
        boost = 1.0 + 0.15 * min(self._sigma_no_improve_count, 10) / 10.0
        csa_factor = max(csa_factor, boost)
    elif fitness_val < 1e-4:
        # Near optimum: use dampened CSA for precision, slower decay
        dampening = 0.7  # slow down sigma decrease
        if csa_factor < 1.0:
            csa_factor = csa_factor ** dampening
    
    # Prevent sigma from collapsing too fast
    min_sigma_factor = 0.85  # never shrink by more than 15% per step
    csa_factor = max(csa_factor, min_sigma_factor)
    
    self.sigma *= csa_factor
    
    # Adaptive bounds based on fitness
    if fitness_val > 1.0:
        sigma_min, sigma_max = 1e-10, 1e8
    elif fitness_val > 1e-4:
        sigma_min, sigma_max = 1e-15, 1e6
    else:
        sigma_min, sigma_max = 1e-20, 1e4
    
    self.sigma = np.clip(self.sigma, sigma_min, sigma_max)
```