**Idea: Cumulative Step-Size Adaptation with Median Success Rule and Aggressive Sigma Boosting**
Uses a hybrid CSA/MSR approach: standard CSA update but with aggressive sigma floor based on search domain and fitness magnitude, plus periodic sigma boosting when fitness is far from target to escape local optima on hard multimodal tasks.

```python
def _update_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Standard CSA update
    csa_factor = np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
    
    # Compute a fitness-aware damping modifier
    # When fitness is large (far from optimum), be more aggressive with sigma increases
    # and less aggressive with sigma decreases
    if self.best_fitness > 1e2:
        # Very far from optimum - heavily bias toward exploration
        if csa_factor < 1.0:
            # Dampen decreases: don't let sigma shrink too fast
            csa_factor = csa_factor ** 0.3
        else:
            # Amplify increases
            csa_factor = csa_factor ** 1.5
    elif self.best_fitness > 1e0:
        if csa_factor < 1.0:
            csa_factor = csa_factor ** 0.5
        else:
            csa_factor = csa_factor ** 1.3
    elif self.best_fitness > 1e-4:
        if csa_factor < 1.0:
            csa_factor = csa_factor ** 0.7
        else:
            csa_factor = csa_factor ** 1.1
    # else: near optimum, use standard CSA factor (no modification)
    
    self.sigma *= csa_factor
    
    # Adaptive sigma floor: prevent premature convergence based on fitness level
    # The further from target, the higher the minimum sigma should be
    if self.best_fitness > 1e2:
        sigma_floor = 5.0
    elif self.best_fitness > 1e1:
        sigma_floor = 1.0
    elif self.best_fitness > 1e0:
        sigma_floor = 0.1
    elif self.best_fitness > 1e-2:
        sigma_floor = 0.01
    elif self.best_fitness > 1e-4:
        sigma_floor = 1e-4
    elif self.best_fitness > 1e-6:
        sigma_floor = 1e-8
    else:
        sigma_floor = 1e-15
    
    self.sigma = max(self.sigma, sigma_floor)
    
    # Periodic sigma boosting for hard problems - every N generations,
    # if we're still far from target, inject a sigma boost
    if hasattr(self, 'generation') and self.generation > 0:
        boost_interval = max(10, 5 + self.dim)
        if self.generation % boost_interval == 0 and self.best_fitness > 1.0:
            # Check if sigma has collapsed relative to what's needed
            max_cov_scale = self.sigma * np.max(self.D) if np.max(self.D) > 0 else self.sigma
            domain_size = self.ub - self.lb  # 200
            if max_cov_scale < domain_size * 0.001:
                # Sigma has collapsed but we're still far - boost it
                self.sigma = max(self.sigma * 10.0, 1.0)
        
        # Also boost if stagnation is building up and fitness is bad
        if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 3:
            if self.best_fitness > 10.0:
                self.sigma = max(self.sigma * 1.5, 2.0)
            elif self.best_fitness > 1.0:
                self.sigma = max(self.sigma * 1.2, 0.5)
    
    # Hard bounds on sigma
    self.sigma = np.clip(self.sigma, 1e-20, 1e6)
```