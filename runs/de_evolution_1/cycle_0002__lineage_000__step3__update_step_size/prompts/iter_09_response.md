Looking at the unsolved tasks, I see errors ranging from ~1e+2 down to ~1e-7. The worst tasks (16, 19, 11, 10) have errors stuck at large values, suggesting the algorithm gets trapped in local optima on multimodal/ill-conditioned landscapes. The near-solved tasks (0, 1, 6) need just a bit more precision.

Key observations:
1. **Variant 08** (which seems most different) does well on tasks 13, 14, 15, 20, 23 but poorly on low-dimensional/smooth tasks — suggesting an aggressive/explorative step size helps on multimodal problems.
2. **Original/variant_01** do well on smooth tasks but can't escape local optima on hard multimodal ones.
3. No variant combines: (a) aggressive sigma inflation when stuck at large errors to escape basins, (b) controlled shrinkage near the optimum, and (c) periodic sigma pulses to break out of deceptive attractors.

My strategy: Implement a **dual-regime step size with periodic perturbation pulses**. When fitness is still large, use a much more aggressive (slower-decaying) sigma with periodic multiplicative boosts. When fitness is small, use standard CSA but with a tighter damping. Additionally, track median population fitness to detect when the search distribution has collapsed prematurely, and re-inflate sigma aggressively.

**Idea: Dual-Regime Pulsed Step Size with Collapse Detection**
Combines aggressive sigma inflation pulses for escaping local optima on hard tasks with tight CSA control near the optimum, plus collapse detection based on eigenvalue spread.

```python
def _update_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    
    # Standard CSA update as baseline
    csa_factor = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
    
    current_fitness = self.best_fitness
    
    if current_fitness > 1e2:
        # Very far from optimum: heavily damped decrease, amplified increase
        if csa_factor > 0:
            # sigma wants to grow — let it grow faster
            self.sigma *= np.exp(csa_factor * 2.0)
        else:
            # sigma wants to shrink — resist shrinkage strongly
            self.sigma *= np.exp(csa_factor * 0.3)
        
        # Periodic pulse: every 10 generations, boost sigma
        if self.generation > 0 and self.generation % 10 == 0:
            self.sigma *= 1.5
            
    elif current_fitness > 1e0:
        # Moderate distance: slightly asymmetric
        if csa_factor > 0:
            self.sigma *= np.exp(csa_factor * 1.5)
        else:
            self.sigma *= np.exp(csa_factor * 0.5)
        
        # Less frequent pulse
        if self.generation > 0 and self.generation % 20 == 0:
            self.sigma *= 1.2
            
    elif current_fitness > 1e-4:
        # Getting closer: standard CSA with mild asymmetry
        self.sigma *= np.exp(csa_factor * 0.8)
    else:
        # Near optimum: standard CSA, tight control
        self.sigma *= np.exp(csa_factor)
    
    # Collapse detection: if eigenvalue spread is too narrow relative to
    # remaining error, re-inflate
    max_D = np.max(self.D)
    min_D = max(np.min(self.D), 1e-30)
    effective_radius = self.sigma * max_D
    
    if current_fitness > 1e-4 and effective_radius < 1e-8:
        # Search has collapsed but we're nowhere near the optimum
        self.sigma = max(self.sigma * 100.0, 1.0)
    elif current_fitness > 1e0 and effective_radius < 1e-3:
        self.sigma = max(self.sigma * 50.0, 5.0)
    elif current_fitness > 1e2 and effective_radius < 1.0:
        self.sigma = max(self.sigma * 20.0, 10.0)
    
    # Stagnation-aware boost: if stuck for many generations at large error
    if hasattr(self, 'stagnation_counter') and current_fitness > 1e0:
        stag_thresh = 3 + int(5 * self.dim / self.pop_size)
        if self.stagnation_counter > stag_thresh:
            boost = 1.0 + 0.1 * (self.stagnation_counter - stag_thresh)
            self.sigma *= min(boost, 5.0)
    
    self.sigma = np.clip(self.sigma, 1e-20, 1e8)
```