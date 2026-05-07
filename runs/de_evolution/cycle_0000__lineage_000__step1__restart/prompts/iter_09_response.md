Looking at the results, I notice that:

1. **All 23 unsolved tasks** have errors many orders of magnitude above target, suggesting the algorithm fundamentally lacks the ability to escape local optima and explore sufficiently.
2. The worst tasks (17, 16, 11, 19, 23) have errors in the hundreds to thousands range - these are likely highly multimodal, non-separable functions where CMA-ES gets trapped.
3. All previous variants seem to produce very similar results (the per-task errors barely differ across variants), suggesting the restart strategy alone isn't changing the fundamental search behavior enough.
4. variant_07 crashed on all tasks (-inf), so that approach failed entirely.

The key insight: **the restart strategy needs to fundamentally change the algorithm's parameters, not just the starting point**. The current approach always uses the same population size and strategy parameters. For hard multimodal problems, we need IPOP-CMA-ES style restarts where population size doubles each restart, combined with much more aggressive exploration (larger sigma, diverse starting regions), and occasionally a completely fresh start ignoring the best-so-far to avoid being anchored to a deceptive basin.

**Idea: IPOP Restart with Increasing Population and Aggressive Diversification**
Double population size on each restart (IPOP-CMA-ES), alternate between best-biased and fully random restarts, and use large initial sigma to escape deceptive basins.

```python
def _restart(self):
    """IPOP-style restart: double pop size, alternate random/biased starts, large sigma."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count
    if not hasattr(self, '_restart_count'):
        self._restart_count = 0
        self._base_pop_size = self.pop_size
    self._restart_count += 1
    
    # IPOP: increase population size (double each restart, cap at reasonable limit)
    new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
    new_pop_size = min(new_pop_size, 512)  # cap to avoid excessive evals
    if new_pop_size % 2 != 0:
        new_pop_size += 1
    
    self.pop_size = new_pop_size
    self.mu = self.pop_size // 2
    
    # Recompute strategy parameters for new pop size
    self._initialize_strategy_params()
    
    # Re-initialize state
    self._initialize_state()
    
    # Restore best
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Alternate between strategies based on restart count
    strategy = self._restart_count % 4
    
    if strategy == 0:
        # Fully random restart - explore completely new region
        self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
        self.sigma = 50.0  # large sigma for broad exploration
    elif strategy == 1:
        # Start near best but with very large sigma
        perturbation = np.random.randn(self.dim) * 40.0
        self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
        self.sigma = 60.0
    elif strategy == 2:
        # Opposition-based restart: mirror best through center
        center = np.zeros(self.dim)  # center of domain
        opposite = 2.0 * center - saved_best_x
        noise = np.random.randn(self.dim) * 10.0
        self.mean = np.clip(opposite + noise, self.lb, self.ub)
        self.sigma = 40.0
    else:
        # Start at best with moderate sigma for local refinement
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
        self.sigma = 15.0
    
    # Every 4th restart cycle, reset pop size to avoid getting stuck with huge pop
    if self._restart_count % 8 == 0:
        self._restart_count = 0
```