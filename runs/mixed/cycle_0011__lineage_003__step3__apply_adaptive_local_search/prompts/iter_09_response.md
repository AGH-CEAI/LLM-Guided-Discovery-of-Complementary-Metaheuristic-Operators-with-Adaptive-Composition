Looking at the unsolved tasks, I notice Tasks 21 and 20 are stuck at exactly 50.0 and 5.0 respectively—classic plateau/local optima traps. The current local search only tests 3 candidates per iteration (±random direction + center), which is too sparse to escape these flat regions or explore valleys effectively.

**Idea: CMA-ES-style Covariance-Adaptive Local Search**
Use a covariance matrix adaptation approach with multiple offspring sampling (like CMA-ES), enabling anisotropic exploration that can navigate narrow valleys and escape plateaus—fundamentally different from the single-direction search in all prior variants.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    # Initialize CMA-style state
    mean = self.best_solution.copy()
    dim = self.dim
    
    # Initialize covariance and step-size
    cov = np.eye(dim)
    step_size = getattr(self, '_ls_step_size', min(self.local_radius, 2.0))
    min_step = 1e-6
    max_step = 15.0
    
    # Population size for local search (adaptive to dim)
    lambda_ls = max(20, min(4 * dim, 60))
    
    # Track best found locally
    best_local = mean.copy()
    best_local_fit = self.best_fitness
    
    # CMA-ES control parameters
    p_c = np.zeros(dim)
    p_s = np.zeros(dim)
    c_1 = 2.0 / (dim ** 2 + 6.0)
    c_mu = min(1.0 - c_1, 2.0 * (0.25 - 1.0 / dim ** 2) / (dim ** 2 + 4.0))
    c_s = (2.0 + dim) / (2.0 * dim + 4.0)
    d_s = 1.0 + c_s
    
    for gen in range(self.local_max_iter):
        if step_size < min_step:
            break
        
        # Generate lambda offspring from multivariate normal
        try:
            L = np.linalg.cholesky(cov)
            noise = np.random.randn(lambda_ls, dim)
            offspring = mean + step_size * np.dot(noise, L.T)
        except np.linalg.LinAlgError:
            # Fallback if covariance becomes singular
            offspring = mean + step_size * np.random.randn(lambda_ls, dim)
        
        # Clip to bounds
        offspring = np.clip(offspring, self.lower, self.upper)
        
        # Evaluate all offspring
        fvals = self._eval_wrapper_batch(offspring, func)
        
        # Guard against invalid evaluations
        fvals = np.clip(fvals, -1e50, 1e50)
        
        # Find best offspring
        best_idx = np.argmin(fvals)
        best_off = offspring[best_idx]
        best_off_fit = float(fvals[best_idx])
        
        if best_off_fit < best_local_fit:
            best_local = best_off.copy()
            best_local_fit = best_off_fit
            step_size = min(step_size * 1.5, max_step)
        else:
            step_size = max(step_size * 0.7, min_step)
        
        # Weighted recombination (CMA-ES style): favor better solutions
        weights = np.zeros(lambda_ls)
        sorted_idx = np.argsort(fvals)
        for i, idx in enumerate(sorted_idx):
            weights[idx] = max(0.0, np.log((lambda_ls + 1.0) / (i + 1.0)))
        
        weights_sum = np.sum(weights)
        if weights_sum > 1e-10:
            weights = weights / weights_sum
        
        # Update mean
        old_mean = mean.copy()
        mean = np.zeros(dim)
        for i in range(lambda_ls):
            mean += weights[i] * offspring[i]
        
        # Update evolution paths and covariance
        y = (mean - old_mean) / (step_size + 1e-10)
        p_s = (1.0 - c_s) * p_s + np.sqrt(c_s * (2.0 - c_s)) * y
        
        p_c = (1.0 - c_1) * p_c + np.sqrt(c_1 * (2.0 - c_1)) * y
        
        # Rank-1 update
        cov = (1.0 - c_1 - c_mu) * cov + c_1 * np.outer(p_c, p_c)
        
        # Rank-mu update (weighted)
        for i in range(lambda_ls):
            z = (offspring[i] - old_mean) / (step_size + 1e-10)
            cov += c_mu * weights[i] * np.outer(z, z)
        
        # Ensure covariance stays positive definite
        cov = (cov + cov.T) / 2.0
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) < 1e-12:
            cov += (1e-11 - np.min(eigvals)) * np.eye(dim)
    
    # Apply best found if improved
    if best_local_fit < self.best_fitness:
        self.best_fitness = best_local_fit
        self.best_solution = best_local.copy()
        self._ls_step_size = min(step_size * 1.2, max_step)
    else:
        self._ls_step_size = max(step_size * 0.8, 0.5)
    
    # Update local radius for main loop
    self.local_radius = max(min(self._ls_step_size, 10.0), 0.1)
```