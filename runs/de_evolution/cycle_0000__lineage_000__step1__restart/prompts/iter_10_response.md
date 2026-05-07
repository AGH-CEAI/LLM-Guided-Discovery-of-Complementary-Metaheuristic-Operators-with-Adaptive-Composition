Looking at the results, I notice that:

1. **All tasks are 8-10+ decades above target** - the algorithm fundamentally isn't converging well enough. The errors are enormous (10s to 1000s) on high-dimensional multimodal problems.
2. **variant_09 did best on several hard tasks** (11, 12, 15, 17, 22) suggesting it had some useful mechanism, but still far from target.
3. **The core issue**: CMA-ES with standard restarts isn't enough. The algorithm needs IPOP/BIPOP-style restarts with increasing population sizes to handle multimodality, plus local refinement around the best solution found.

The key insight: on restart, we should **double the population size** (IPOP strategy) to improve global exploration on multimodal landscapes, AND alternate with **small-sigma local restarts** around the best solution to refine precision. This is fundamentally different from all previous variants which kept constant population size and only varied the mean/sigma initialization.

**Idea: IPOP-BIPOP Restart with Population Doubling and Local Refinement**
Alternates between IPOP (doubling population) for global exploration and small-population local refinement around best, with adaptive sigma scaling.
```python
def _restart(self):
    """IPOP/BIPOP-style restart: alternate between large-pop global and small-pop local restarts."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count
    if not hasattr(self, '_restart_count'):
        self._restart_count = 0
        self._base_pop_size = self.pop_size
        self._best_sigma_at_solution = 30.0
    
    self._restart_count += 1
    
    # Alternate between global (IPOP - increasing pop) and local (small pop, small sigma)
    do_local = (self._restart_count % 3 == 0) and saved_best_f < np.inf
    
    if do_local:
        # Local restart: small population, small sigma, centered on best
        self._initialize_state()
        self.pop_size = max(self._base_pop_size, 8)
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        
        # Center on best with small perturbation and small sigma
        scale = max(1e-3, min(10.0, abs(saved_best_f) * 1e-4))
        scale = min(scale, 5.0)
        perturbation = np.random.randn(self.dim) * scale
        self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
        self.sigma = max(scale * 2.0, 0.01)
    else:
        # IPOP restart: double population size each time (up to a limit)
        new_pop_size = min(self._base_pop_size * (2 ** min(self._restart_count // 2, 5)), 
                          self._base_pop_size * 32)
        if new_pop_size % 2 != 0:
            new_pop_size += 1
        
        self._initialize_state()
        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        
        # Randomize initial mean broadly
        self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
        
        # Larger initial sigma for broader search
        self.sigma = 40.0 + 10.0 * min(self._restart_count, 5)
        
        # With some probability, bias toward best region
        if np.random.rand() < 0.2 and saved_best_f < np.inf:
            perturbation = np.random.randn(self.dim) * 30.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
    
    # Always preserve best solution
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Reset stagnation counter with longer patience for larger populations
    self.stagnation_counter = 0
```