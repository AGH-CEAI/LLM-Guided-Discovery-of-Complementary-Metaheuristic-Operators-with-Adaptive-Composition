Looking at the unsolved tasks, the errors span from ~1e+04 (Task 17) down to ~1e-07 (Task 0). The worst tasks (17, 16, 11, 19, 12, 23) have errors in the range 10¹-10⁴, suggesting the algorithm gets trapped in local optima on multimodal/rugged landscapes. The current restart strategies are too conservative — they either restart nearly randomly or bias toward the current best with small perturbation, never escaping the basin of attraction.

The key insight: for highly multimodal problems, we need **IPOP-CMA-ES style restarts** where population size increases on each restart (doubling), combined with **diverse initialization strategies** that systematically explore different regions. Larger populations are critical for escaping local optima on multimodal functions. Additionally, we should sometimes restart with a much larger sigma to cover more of the search space, and use opposition-based learning to explore the "opposite" side of the domain from where we've been searching.

**Idea: IPOP Restart with Opposition-Based Diverse Exploration**
Increase population size on each restart (IPOP), use opposition-based initialization, larger sigma, and systematically avoid previously explored regions.
```python
def _restart(self):
    """IPOP-style restart: double pop size, opposition-based diverse init, large sigma."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count
    if not hasattr(self, 'restart_count'):
        self.restart_count = 0
        self.initial_pop_size = self.pop_size
        self.visited_means = []
    
    self.visited_means.append(self.mean.copy())
    self.restart_count += 1
    
    # IPOP: increase population size (double each restart, cap at reasonable limit)
    new_pop_size = min(self.initial_pop_size * (2 ** self.restart_count), 512)
    if new_pop_size % 2 != 0:
        new_pop_size += 1
    self.pop_size = new_pop_size
    self.mu = self.pop_size // 2
    
    # Recompute strategy parameters for new pop size
    self._initialize_strategy_params()
    
    # Reinitialize state
    self._initialize_state()
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Use larger initial sigma for broader exploration
    self.sigma = min(50.0 + 10.0 * self.restart_count, 80.0)
    
    # Choose initialization strategy based on restart count
    strategy = self.restart_count % 4
    
    if strategy == 0:
        # Opposition-based: reflect best through center of domain
        center = 0.5 * (self.lb + self.ub)
        opposite = 2.0 * center - saved_best_x
        perturbation = np.random.randn(self.dim) * 10.0
        self.mean = np.clip(opposite + perturbation, self.lb, self.ub)
    elif strategy == 1:
        # Random far from all previously visited means
        candidate = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
        if len(self.visited_means) > 0:
            # Pick candidate that maximizes min distance to visited means
            best_cand = candidate
            best_min_dist = 0
            for _ in range(20):
                cand = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
                min_dist = min(np.linalg.norm(cand - vm) for vm in self.visited_means)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_cand = cand
            candidate = best_cand
        self.mean = np.clip(candidate, self.lb, self.ub)
    elif strategy == 2:
        # Near best but with very large perturbation
        perturbation = np.random.randn(self.dim) * 40.0
        self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
    else:
        # Completely random uniform
        self.mean = np.random.uniform(self.lb * 0.9, self.ub * 0.9, size=self.dim)
```