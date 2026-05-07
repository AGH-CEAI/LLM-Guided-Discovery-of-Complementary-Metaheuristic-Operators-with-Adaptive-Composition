Looking at the results, all tasks are far from the target. The worst tasks (17, 16, 11, etc.) have errors in the thousands/hundreds, suggesting the algorithm gets trapped in local optima on complex multimodal landscapes. The current restart strategy either reinitializes randomly or biases toward the best solution with a small perturbation — both insufficient for escaping deep local optima basins.

**Key insight:** The current approach uses a fixed small population and restarts with the same population size. For hard multimodal problems, we need IPOP-CMA-ES style restarts where the population size **doubles** each restart, dramatically increasing exploration. Additionally, we should use a much larger initial sigma covering the full search space, and diversify the initial mean using opposition-based learning and Latin hypercube-like sampling to avoid re-exploring the same basins.

**Idea: IPOP Restart with Increasing Population and Opposition-Based Diversification**
Doubles population size each restart, uses opposition-based mean initialization, full-range sigma, and maintains a tabu-like memory of previous basins to force exploration of new regions.

```python
def _restart(self):
    """IPOP-style restart: double pop size, opposition-based diversification, full sigma."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track previous mean positions to avoid revisiting (store on first use)
    if not hasattr(self, '_previous_means'):
        self._previous_means = []
        self._restart_count = 0
    
    self._previous_means.append(self.mean.copy())
    self._restart_count += 1
    
    # IPOP: increase population size (double every restart, cap at reasonable size)
    old_pop_size = self.pop_size
    new_pop_size = min(old_pop_size * 2, 512)  # cap to avoid excessive evaluations
    
    # Reinitialize state
    self._initialize_state()
    
    # Apply increased population size
    self.pop_size = new_pop_size
    if self.pop_size % 2 != 0:
        self.pop_size += 1
    self.mu = self.pop_size // 2
    self._initialize_strategy_params()
    
    # Restore best
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Use larger sigma to cover more of the search space
    self.sigma = 50.0 + 20.0 * min(self._restart_count, 5)
    
    # Diversify mean initialization: try to go far from previous basins
    if self._restart_count % 3 == 1:
        # Opposition-based: reflect best through center of domain
        center = (self.lb + self.ub) / 2.0
        self.mean = np.clip(2.0 * center - saved_best_x, self.lb, self.ub)
        # Add noise
        self.mean += np.random.randn(self.dim) * 10.0
        self.mean = np.clip(self.mean, self.lb, self.ub)
    elif self._restart_count % 3 == 2 and len(self._previous_means) > 0:
        # Maximize distance from all previous means
        candidate_means = []
        for _ in range(50):
            candidate = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
            min_dist = min(np.linalg.norm(candidate - pm) for pm in self._previous_means)
            candidate_means.append((min_dist, candidate))
        candidate_means.sort(key=lambda x: -x[0])
        self.mean = np.clip(candidate_means[0][1], self.lb, self.ub)
    else:
        # Random with wide spread
        self.mean = np.random.uniform(self.lb * 0.9, self.ub * 0.9, size=self.dim)
    
    # Reset covariance to identity scaled appropriately
    self.C = np.eye(self.dim)
    self.B = np.eye(self.dim)
    self.D = np.ones(self.dim)
    self.invsqrt_C = np.eye(self.dim)
    self.p_sigma = np.zeros(self.dim)
    self.p_c = np.zeros(self.dim)
```