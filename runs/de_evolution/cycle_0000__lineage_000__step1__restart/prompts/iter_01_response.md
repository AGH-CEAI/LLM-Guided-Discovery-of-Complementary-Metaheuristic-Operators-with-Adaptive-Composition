**Idea: IPOP-CMA-ES Restart with Doubled Population**
Implements IPOP (Increasing Population) restart strategy where population size doubles on each restart, with the mean initialized using a weighted combination of the best solution and a random point.

```python
def _restart(self):
    """IPOP-style restart: double population size, keep best, initialize mean
    as weighted blend of best-known solution and random point."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # IPOP: double the population size (capped to avoid excessive memory)
    old_pop_size = self.pop_size
    self.pop_size = min(self.pop_size * 2, 512)
    # Keep pop_size even
    if self.pop_size % 2 != 0:
        self.pop_size += 1
    self.mu = self.pop_size // 2
    
    # Recompute strategy parameters for new population size
    self._initialize_strategy_params()
    
    # Re-initialize state (mean, C, paths, sigma, etc.)
    self._initialize_state()
    
    # Restore best
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Initialize mean as weighted combination of best and random
    # Use a blend that starts near the best but allows exploration
    # The larger the population, the more we can afford to explore
    random_point = np.random.uniform(self.lb, self.ub, size=self.dim)
    
    # Adaptive blending weight: more exploration with larger populations
    alpha = min(0.5, old_pop_size / float(self.pop_size))  # starts at 0.5, decreases
    
    self.mean = np.clip(
        alpha * saved_best_x + (1.0 - alpha) * random_point,
        self.lb, self.ub
    )
    
    # Adapt initial sigma based on distance from mean to best
    dist_to_best = np.linalg.norm(self.mean - saved_best_x)
    if dist_to_best > 1e-10:
        # Set sigma proportional to distance, but capped
        self.sigma = np.clip(dist_to_best / np.sqrt(self.dim), 1.0, 50.0)
    else:
        # If mean is very close to best, use moderate sigma for exploration
        self.sigma = 30.0
```