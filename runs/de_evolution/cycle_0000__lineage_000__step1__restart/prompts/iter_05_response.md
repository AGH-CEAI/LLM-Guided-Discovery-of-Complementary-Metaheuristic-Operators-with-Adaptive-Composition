Looking at the results, all 23 unsolved tasks have errors many orders of magnitude above target. The worst tasks (17, 16, 11, 19, 23) have errors in the thousands to tens range, suggesting the algorithm gets trapped in local optima and restarts don't provide enough diversity or search power.

**Key observations:**
1. Tasks with errors ~1e+1 to 1e+4 are likely high-dimensional multimodal/non-separable functions where CMA-ES converges to local optima
2. The current restart strategy and all variants seem to produce very similar results (errors barely differ across variants), suggesting the restart mechanism isn't fundamentally changing the search behavior
3. The algorithm needs restarts that dramatically increase population size (IPOP-CMA-ES style) to escape deep local optima, combined with a much larger initial sigma and truly random restarts to explore different basins

**My approach:** Implement an IPOP (Increasing POPulation) restart strategy that doubles the population size on each restart, recomputes all strategy parameters accordingly, and alternates between fully random restarts and restarts near the best solution. Larger populations are crucial for escaping local optima on multimodal landscapes. This is fundamentally different from prior variants which kept the same population size.

**Idea: IPOP Doubling Population Restart**
Double population size each restart (IPOP-CMA-ES), recompute all strategy parameters, alternate random/biased restarts with large sigma.

```python
def _restart(self):
    """IPOP restart: double population size each restart, recompute strategy params."""
    saved_best_x = self.best_x.copy()
    saved_best_f = self.best_fitness
    
    # Track restart count
    if not hasattr(self, 'restart_count'):
        self.restart_count = 0
    self.restart_count += 1
    
    # IPOP: double population size each restart, up to a limit
    max_pop = max(256, 20 * self.dim)
    new_pop_size = min(self.pop_size * 2, max_pop)
    self.pop_size = new_pop_size
    if self.pop_size % 2 != 0:
        self.pop_size += 1
    self.mu = self.pop_size // 2
    
    # Recompute all strategy parameters for new population size
    self._initialize_strategy_params()
    
    # Re-initialize state
    self._initialize_state()
    
    # Restore best
    self.best_x = saved_best_x
    self.best_fitness = saved_best_f
    
    # Larger initial sigma for broader exploration
    self.sigma = 50.0 + 10.0 * min(self.restart_count, 5)
    
    # Alternate between random restarts and biased restarts
    if self.restart_count % 3 == 0:
        # Fully random restart - explore new basin
        self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
    elif self.restart_count % 3 == 1:
        # Restart near best with large perturbation
        scale = 30.0 + 10.0 * min(self.restart_count, 5)
        perturbation = np.random.randn(self.dim) * scale
        self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
    else:
        # Opposition-based restart: reflect best through center of domain
        center = (self.lb + self.ub) / 2.0
        opposite = 2.0 * center - saved_best_x
        noise = np.random.randn(self.dim) * 15.0
        self.mean = np.clip(opposite + noise, self.lb, self.ub)
    
    # Reset covariance to identity with slight random perturbation
    # to break symmetry in search directions
    diag_vals = np.exp(np.random.randn(self.dim) * 0.3)
    self.C = np.diag(diag_vals)
    self.D = np.sqrt(diag_vals)
    self.B = np.eye(self.dim)
    self.invsqrt_C = np.diag(1.0 / np.sqrt(diag_vals))
```