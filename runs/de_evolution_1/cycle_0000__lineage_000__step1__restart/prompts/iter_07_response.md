Looking at the unsolved tasks, I notice a clear pattern: tasks 10-23 have errors in the range 1e+0 to 1e+4, suggesting these are high-dimensional, multimodal, or ill-conditioned problems where the algorithm gets stuck in local optima. The current restart strategies all use relatively similar approaches - reinitializing with small perturbations around the best or random restarts with similar population sizes.

The key insight is that the algorithm needs **IPOP-CMA-ES style restarts** with increasing population sizes to escape deep local optima on multimodal landscapes, combined with a **bidirectional restart strategy** that alternates between exploitation (small sigma, near best) and exploration (large population, large sigma, diverse starting points). The worst tasks (17, 16, 11) likely have many local optima requiring much larger populations to explore, while near-solved tasks (0, 1, 6) need fine-tuned local search.

**Idea: IPOP Bidirectional Restart with Population Doubling and Opposition-Based Sampling**
Alternates between doubling population size for exploration and halving sigma for exploitation, using opposition-based initialization to maximize coverage of the search space.

```python
def _restart(self):
    """IPOP-style restart with population doubling and opposition-based initialization."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count via a persistent attribute
    if not hasattr(self, 'restart_count'):
        self.restart_count = 0
        self.initial_pop_size = self.pop_size
    self.restart_count += 1
    
    # Alternate between exploration (large pop) and exploitation (small pop, small sigma)
    if self.restart_count % 3 == 0:
        # Exploitation restart: small sigma near best solution
        self._initialize_state()
        self.sigma = max(1.0, 5.0 / (self.restart_count + 1))
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * self.sigma * 0.5, self.lb, self.ub)
        self.pop_size = self.initial_pop_size
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
    else:
        # IPOP exploration restart: double population each time
        factor = min(2 ** self.restart_count, 32)  # Cap at 32x initial
        self.pop_size = self.initial_pop_size * factor
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        
        # Recompute strategy parameters for new pop size
        self._initialize_strategy_params()
        self._initialize_state()
        
        # Opposition-based initialization: start from opposite side of search space
        if self.restart_count % 2 == 1:
            # Opposition point of best
            opposition = self.lb + self.ub - saved_best_x
            # Random point between opposition and random
            alpha = np.random.uniform(0.3, 0.7, size=self.dim)
            random_point = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
            self.mean = np.clip(alpha * opposition + (1 - alpha) * random_point, self.lb, self.ub)
        else:
            # Quasi-random restart in unexplored region
            # Use golden ratio-based spacing to avoid previous regions
            golden = (np.sqrt(5) - 1) / 2
            offsets = np.array([(golden * self.restart_count * (d + 1)) % 1.0 for d in range(self.dim)])
            self.mean = self.lb + offsets * (self.ub - self.lb)
        
        self.sigma = 40.0 + 10.0 * min(self.restart_count, 5)  # Larger initial sigma
    
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    self.stagnation_counter = 0
```