**Idea: Hybrid Multi-Strategy Sampling with Exploration and Boundary Injection**

This approach combines three sampling strategies: (1) standard CMA-ES sampling for exploitation, (2) wide uniform exploration across bounds to escape local optima on the worst tasks (16, 19, 20, 23 which show errors ~10-400), and (3) boundary-focused sampling to probe the search space edges. The key insight is that tasks with errors stuck at ~10^1 to ~10^2 likely have deceptive basins that CMA-ES converges into but cannot escape—the algorithm needs explicit diversity injection. This hybrid approach is fundamentally different from all prior variants (which all used pure CMA-ES sampling) and is designed to specifically attack the priority targets.

```python
def _sample_population_batch(self):
    """
    Hybrid multi-strategy sampling: CMA-ES + uniform exploration + boundary injection.
    Designed to escape local optima on difficult tasks (16, 19, 20, 23, etc.).
    """
    pop_size = self.pop_size
    dim = self.dim
    
    # Determine exploration fraction based on covariance condition number
    # High condition number -> more exploration needed
    max_D = self.D[-1] if len(self.D) > 0 else 1.0
    min_D = max(self.D[0], 1e-20) if len(self.D) > 0 else 1.0
    cond = max_D / min_D
    explore_frac = 0.35 if cond > 1e5 else 0.25
    n_explore = max(2, int(pop_size * explore_frac))
    n_boundary = max(1, int(pop_size * 0.10))
    n_cmaes = pop_size - n_explore - n_boundary
    
    # 1. Standard CMA-ES sampling (main exploitation component)
    z_main = np.random.randn(n_cmaes, dim)
    scaled = z_main * self.D[np.newaxis, :]
    rotated = scaled @ self.B.T
    pop_main = self.mean[np.newaxis, :] + self.sigma * rotated
    
    # 2. Wide uniform exploration across bounds
    # This provides truly global search to escape local optima
    z_explore = np.random.randn(n_explore, dim)
    range_arr = np.array([self.ub - self.lb] * dim)
    mean_arr = np.array([(self.ub + self.lb) / 2.0] * dim)
    # Sample from uniform distribution scaled by sigma
    pop_explore = np.random.uniform(
        self.lb, self.ub, size=(n_explore, dim)
    )
    
    # 3. Boundary injection: sample near search space edges
    # This helps on tasks with optima near boundaries
    z_boundary = np.zeros((n_boundary, dim))
    pop_boundary = np.zeros((n_boundary, dim))
    for i in range(n_boundary):
        # Randomly choose corners/edges
        for d in range(dim):
            if np.random.rand() < 0.3:
                # Near boundary
                if np.random.rand() < 0.5:
                    pop_boundary[i, d] = np.random.uniform(self.lb, self.lb + 0.1 * range_arr[d])
                else:
                    pop_boundary[i, d] = np.random.uniform(self.ub - 0.1 * range_arr[d], self.ub)
            else:
                # Near current mean with perturbation
                pop_boundary[i, d] = self.mean[d] + np.random.uniform(-1.0, 1.0) * self.sigma
    
    # Clip all to bounds
    pop_explore = np.clip(pop_explore, self.lb, self.ub)
    pop_boundary = np.clip(pop_boundary, self.lb, self.ub)
    
    # Combine all components
    population = np.vstack([pop_main, pop_explore, pop_boundary])
    z_vectors = np.vstack([z_main, z_explore, z_boundary])
    
    return population, z_vectors
```