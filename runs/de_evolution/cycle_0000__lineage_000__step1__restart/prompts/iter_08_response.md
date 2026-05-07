Looking at the results, I notice:

1. **variant_07 produced `-inf` for all tasks** — it clearly crashed/failed completely.
2. **All tasks have errors many orders of magnitude above 1e-08**, with the worst tasks (17, 16, 11) having errors ~10³-10⁴.
3. The existing variants all produce very similar results, suggesting the restart strategy barely matters — the core CMA-ES is getting stuck in local optima on multimodal/ill-conditioned landscapes.

The key insight: the current restarts just re-randomize but don't fundamentally change the search behavior. For hard multimodal problems (Tasks 10-23 with errors 1-10⁴), we need **IPOP-CMA-ES style restarts** that progressively increase population size, giving the algorithm much more exploration power. For near-solved tasks (0-9), we need restarts that **shrink sigma** and focus on local refinement around the best solution.

My strategy: **Biphasic IPOP restart** — each restart doubles the population size (IPOP strategy) to escape multimodal landscapes, while also alternating between global exploration (large sigma, random mean) and local exploitation (small sigma, mean near best). This fundamentally changes the algorithm's sampling density on each restart.

**Idea: IPOP Biphasic Restart**
Double population size each restart (IPOP-CMA-ES) and alternate between global exploration and local refinement phases.

```python
def _restart(self):
    """IPOP-style restart: double pop size each time, alternate global/local phases."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count via an attribute
    if not hasattr(self, 'restart_count'):
        self.restart_count = 0
    self.restart_count += 1
    
    # IPOP: increase population size (up to a limit)
    if not hasattr(self, 'initial_pop_size'):
        self.initial_pop_size = self.pop_size
    
    # Double pop size every restart, cap at reasonable limit
    new_pop_size = min(self.initial_pop_size * (2 ** self.restart_count), 
                       max(256, 20 * self.dim))
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
    
    # Alternate between global exploration and local refinement
    if self.restart_count % 3 == 0:
        # Local refinement: small sigma, mean near best
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 2.0, self.lb, self.ub)
        self.sigma = max(5.0, 0.1 * np.std(saved_best_x - self.mean + 1e-10))
        # Reset to smaller pop for local search
        self.pop_size = self.initial_pop_size
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
    elif self.restart_count % 3 == 1:
        # Global exploration: large sigma, random mean, large pop
        self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
        self.sigma = 50.0
    else:
        # Semi-local: moderate perturbation from best
        scale = min(30.0, 10.0 * self.restart_count)
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * scale, self.lb, self.ub)
        self.sigma = scale
    
    # Reset covariance to identity (fresh start)
    self.C = np.eye(self.dim)
    self.B = np.eye(self.dim)
    self.D = np.ones(self.dim)
    self.invsqrt_C = np.eye(self.dim)
    self.p_sigma = np.zeros(self.dim)
    self.p_c = np.zeros(self.dim)
    self.eigen_decomp_gen = 0
    self.stagnation_counter = 0
```