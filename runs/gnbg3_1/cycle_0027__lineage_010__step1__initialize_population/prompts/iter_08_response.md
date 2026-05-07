Looking at the unsolved tasks, the worst ones (Tasks 16-20 with errors 10-60) suggest highly multimodal or ill-conditioned landscapes where the current Sobol-based initialization may be placing samples in regions that get trapped in local optima.

**Idea: Adaptive Multi-Method Initialization with Boundary Focus**

Instead of pure quasi-random coverage, this approach uses three complementary strategies: (1) quasi-random base coverage, (2) Latin Hypercube stratified sampling for better dimension-wise independence, and (3) boundary-focused sampling with adaptive density based on problem scale. This is fundamentally different because it combines multiple sampling paradigms with adaptive weighting rather than relying on a single method.

```python
def _initialize_population(self):
    """Initialize population using adaptive multi-method sampling with boundary focus."""
    samples_list = []
    
    # Method 1: Sobol quasi-random (60% of population)
    n_sobol = int(0.6 * self.NP)
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(self.dim, scramble=True)
        sobol_samples = sampler.random(n_sobol)
        sobol_samples = qmc.scale(sobol_samples, self.lb, self.ub)
        samples_list.append(sobol_samples)
    except Exception:
        pass
    
    # Method 2: Latin Hypercube for better dimension-wise independence (25% of population)
    n_lhs = int(0.25 * self.NP)
    try:
        sampler = qmc.LatinHypercube(self.dim, scramble=True)
        lhs_samples = sampler.random(n_lhs)
        lhs_samples = qmc.scale(lhs_samples, self.lb, self.ub)
        samples_list.append(lhs_samples)
    except Exception:
        pass
    
    # Method 3: Boundary-focused sampling with adaptive density (15% of population)
    # For ill-conditioned/multimodal tasks, optima often lie near boundaries
    n_boundary = self.NP - sum(len(s) for s in samples_list)
    if n_boundary > 0:
        boundary_samples = np.zeros((n_boundary, self.dim))
        # Use power-law scaling to concentrate samples near boundaries
        # pow=3 creates strong concentration near edges
        u = np.random.random((n_boundary, self.dim))
        power = 3.0
        scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
        # Randomly choose which dimensions get boundary treatment
        dim_mask = np.random.random(self.dim) < 0.3
        boundary_samples[:, dim_mask] = scaled[:, dim_mask]
        boundary_samples[:, ~dim_mask] = u[:, ~dim_mask]
        # Scale to bounds
        for d in range(self.dim):
            boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])
        samples_list.append(boundary_samples)
    
    # Combine all samples
    if samples_list:
        samples = np.vstack(samples_list)
    else:
        # Ultimate fallback: simple random
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
    
    # Ensure exact size and numerical robustness
    if samples.shape[0] > self.NP:
        samples = samples[:self.NP]
    elif samples.shape[0] < self.NP:
        extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
        samples = np.vstack([samples, extra])
    
    self.population = samples
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()

    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)

    self.fitness = self.func(self.population)

    best_idx = np.argmin(self.fitness)
    self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
    self.x_opt = self.population[best_idx].copy()
    self.f_opt_prev = self.f_opt

    self.stagnation_counter = 0
    self.generation = 0
    self.current_operator = 0

    self._reset_operator_state()
```