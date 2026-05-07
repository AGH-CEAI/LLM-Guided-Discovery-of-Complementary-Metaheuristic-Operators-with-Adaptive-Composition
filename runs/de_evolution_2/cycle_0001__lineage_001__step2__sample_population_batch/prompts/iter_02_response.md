**Idea: Hybrid Isotropic-Directed Sampling with Adaptive Exploration Ratio**

A fundamentally different strategy that abandons the pure covariance-matrix-based sampling. Instead, it dynamically balances between: (1) global isotropic exploration using uniform random sampling, and (2) directed perturbations around the global best solution found so far. The exploration ratio adapts based on whether recent generations improved, enabling escape from deceptive local optima that trap the covariance-only approach. This is critical for the worst tasks (16, 19, 20, 23) where errors > 1e+01 suggest the optimizer is trapped in a basin far from the global optimum.

```python
def _sample_population_batch(self):
    """
    Hybrid sampling: dynamically balance global isotropic exploration with
    directed perturbations around the global best to escape local optima.
    """
    pop_size = self.pop_size
    dim = self.dim
    lb, ub = self.lb, self.ub
    
    # Adaptive exploration ratio: decrease when improving, increase when stuck
    if self.generation == 0:
        self._exploration_ratio = 0.5
    else:
        improved = (self.last_best_f - self.best_f) > 1e-10 * (1.0 + abs(self.last_best_f))
        delta = 0.05 if improved else -0.08
        self._exploration_ratio = np.clip(self._exploration_ratio + delta, 0.1, 0.9)
    
    # Number of globally sampled candidates
    n_global = max(1, int(pop_size * self._exploration_ratio))
    n_directed = pop_size - n_global
    
    # Part 1: Global isotropic exploration (uniform sampling)
    global_pop = np.random.uniform(lb, ub, size=(n_global, dim))
    
    # Part 2: Directed perturbations around global best
    directed_pop = np.zeros((n_directed, dim))
    if self.best_x is not None:
        # Adaptive step size for directed search
        step = self.sigma * np.clip(np.log(1 + self.generation + 1) / 10.0, 0.1, 5.0)
        for i in range(n_directed):
            # Sample from Cauchy-like distribution for heavy-tailed exploration
            alpha = np.random.uniform(0, 2 * np.pi, size=dim)
            r = np.random.standard_cauchy(dim)
            r = np.clip(r, -10, 10)
            directed_pop[i] = self.best_x + step * r
    else:
        # Fallback to uniform if no best found yet
        directed_pop = np.random.uniform(lb, ub, size=(n_directed, dim))
    
    # Combine and clip to bounds
    population = np.vstack([global_pop, directed_pop])
    population = np.clip(population, lb, ub)
    
    # Generate corresponding z_vectors for compatibility (approximate)
    z = (population - self.mean[np.newaxis, :]) / max(self.sigma, 1e-20)
    
    return population, z
```