Looking at the unsolved tasks, I see errors ranging from ~1e+4 down to ~1e-7. The worst tasks (17, 16, 11, 19, 12) have errors stuck at 1e+1 to 1e+4, suggesting the algorithm gets trapped in local optima on highly multimodal/non-separable functions. The near-solved tasks (0, 1, 6) at ~1e-7 suggest the algorithm nearly converges but lacks fine-tuning precision.

**Key insight:** All previous variants use similar restart strategies — reinitialize with small perturbations around the best. For highly multimodal functions, we need IPOP-CMA-ES style restarts with **increasing population sizes** to escape deep local optima, combined with a **biphasic strategy**: large-population exploratory restarts for diversity AND small-sigma local refinement restarts around the best solution. This is fundamentally different from prior variants which never change population size or combine exploration/exploitation restart modes.

**Idea: IPOP Biphasic Restart**
Increase population size on each restart (IPOP strategy) and alternate between wide exploration and tight local refinement around the best solution.

```python
def _restart(self):
    """IPOP-style restart: double population on restarts + alternate between exploration and local refinement."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count via an attribute
    if not hasattr(self, 'restart_count'):
        self.restart_count = 0
    self.restart_count += 1
    
    # IPOP: increase population size every other restart (capped)
    if self.restart_count % 2 == 0:
        # Double population for exploration restarts
        new_pop_size = min(self.pop_size * 2, 256)
        if new_pop_size % 2 != 0:
            new_pop_size += 1
        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
    
    self._initialize_state()
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    if self.restart_count % 3 == 0:
        # Local refinement restart: small sigma around best solution
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 0.5, self.lb, self.ub)
        self.sigma = 2.0  # small step size for fine-tuning
    elif self.restart_count % 3 == 1:
        # Medium-range exploration restart: moderate perturbation from best
        scale = 10.0 + 20.0 * np.random.rand()
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * scale, self.lb, self.ub)
        self.sigma = 20.0
    else:
        # Full random restart with large population (IPOP exploration)
        self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
        self.sigma = 50.0
    
    # Occasionally initialize covariance with diagonal scaling to handle ill-conditioning
    if self.restart_count % 4 == 0 and self.restart_count > 2:
        # Use a random diagonal covariance to break symmetry
        diag_scales = np.exp(np.random.randn(self.dim) * 1.5)
        diag_scales = np.clip(diag_scales, 0.1, 10.0)
        self.C = np.diag(diag_scales)
        self.D = np.sqrt(diag_scales)
        self.B = np.eye(self.dim)
        inv_D = 1.0 / self.D
        self.invsqrt_C = np.diag(inv_D)
```