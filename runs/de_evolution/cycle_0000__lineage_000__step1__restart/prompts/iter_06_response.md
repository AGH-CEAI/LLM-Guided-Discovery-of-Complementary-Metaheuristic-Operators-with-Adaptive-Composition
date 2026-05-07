Looking at the benchmark results, I notice that ALL variants perform nearly identically across all tasks - the errors barely differ between variants. This suggests the `_restart` function changes have been too conservative or too similar in nature. The worst tasks (17, 16, 11, 19, 23) have errors in the range 10^1 to 10^4, indicating the algorithm is stuck in poor local optima and restarts aren't providing enough diversity or adaptation.

**Key insight**: The current approach always restarts with the same population size and sigma. For hard multimodal/ill-conditioned problems (tasks 10-23), we need an IPOP-CMA-ES style strategy where each restart **doubles the population size** to improve global search, combined with a much larger initial sigma to cover the full search space, and multiple diverse starting points instead of always biasing toward the previous best (which may be a deep local optimum).

**Idea: IPOP Restart with Population Doubling and Opposition-Based Diversification**
Doubles population size on each restart (IPOP strategy), uses opposition-based sampling for diversity, and alternates between exploring full space vs. refining near best.

```python
def _restart(self):
    """IPOP-style restart: double pop size each restart, opposition-based diversity."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count
    if not hasattr(self, 'restart_count'):
        self.restart_count = 0
    self.restart_count += 1
    
    # IPOP: double population size on each restart (capped)
    old_pop_size = self.pop_size
    self.pop_size = min(old_pop_size * 2, 256)
    if self.pop_size % 2 != 0:
        self.pop_size += 1
    self.mu = self.pop_size // 2
    
    # Recompute strategy parameters for new pop size
    self._initialize_strategy_params()
    
    # Reinitialize state with new dimensions
    self._initialize_state()
    
    # Restore best
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Use larger sigma to cover more of the search space
    self.sigma = min(60.0 * (1.0 + 0.5 * self.restart_count), 100.0)
    
    # Alternate restart strategies based on restart count
    strategy = self.restart_count % 4
    
    if strategy == 0:
        # Opposition-based: start from the point opposite to best
        self.mean = np.clip(-saved_best_x, self.lb, self.ub)
    elif strategy == 1:
        # Random point far from best
        direction = np.random.randn(self.dim)
        direction /= max(np.linalg.norm(direction), 1e-30)
        distance = 50.0 + 50.0 * np.random.rand()
        self.mean = np.clip(saved_best_x + direction * distance, self.lb, self.ub)
    elif strategy == 2:
        # Fully random in the domain (maximum diversity)
        self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
    else:
        # Mirrored sampling around center of domain with perturbation
        center = np.zeros(self.dim)
        perturbation = np.random.randn(self.dim) * 40.0
        self.mean = np.clip(center + perturbation, self.lb, self.ub)
    
    # Reset covariance to scaled identity to avoid inheriting bad structure
    self.C = np.eye(self.dim)
    self.B = np.eye(self.dim)
    self.D = np.ones(self.dim)
    self.invsqrt_C = np.eye(self.dim)
    self.p_sigma = np.zeros(self.dim)
    self.p_c = np.zeros(self.dim)
    self.eigen_decomp_gen = 0
```