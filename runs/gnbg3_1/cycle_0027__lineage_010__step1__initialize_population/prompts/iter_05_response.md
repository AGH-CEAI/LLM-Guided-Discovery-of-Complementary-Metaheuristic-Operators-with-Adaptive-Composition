**Idea: Boundary-Layer Adaptive Initialization**

A fundamentally different initialization strategy that focuses samples near problem boundaries (where many benchmark optima reside) combined with multi-seed starting points around elite candidates to escape local optima traps that cause the worst tasks (16, 17, 10, 20, 12, 18) to stagnate at errors ~10-60.

```python
def _initialize_population(self):
    """Initialize population using boundary-layer adaptive sampling with multi-seed starts."""
    # Phase 1: Generate base samples using Halton quasi-random sequence
    try:
        from scipy.stats import qmc
        # Use Halton (more robust than Sobol for varying dimensions)
        sampler = qmc.Halton(self.dim, scramble=True)
        base_samples = sampler.random(self.NP // 2)
        base_samples = qmc.scale(base_samples, self.lb, self.ub)
    except Exception:
        base_samples = np.random.uniform(self.lb, self.ub, (self.NP // 2, self.dim))
    
    # Phase 2: Generate boundary-layer samples (global optima often at boundaries)
    boundary_ratio = 0.3  # 30% of samples near boundaries
    n_boundary = max(4, int(self.NP * boundary_ratio))
    boundary_samples = np.zeros((n_boundary, self.dim))
    
    for i in range(n_boundary):
        # Sample from corners/edges with Gaussian falloff
        boundary_layer_width = 0.15 * (self.ub - self.lb)
        for d in range(self.dim):
            # Choose boundary side (lower or upper) with probability based on position
            side = np.random.choice([0, 1])
            if side == 0:
                boundary_samples[i, d] = self.lb[d] + np.random.exponential(boundary_layer_width[d])
            else:
                boundary_samples[i, d] = self.ub[d] - np.random.exponential(boundary_layer_width[d])
        boundary_samples[i] = np.clip(boundary_samples[i], self.lb, self.ub)
    
    # Phase 3: Multi-seed starting points (escape local optima traps)
    n_seeds = max(2, self.dim // 4)
    seed_samples = np.zeros((n_seeds, self.dim))
    for i in range(n_seeds):
        # Start from random boundary point and spread locally
        seed_center = boundary_samples[i % len(boundary_samples)].copy()
        spread = 0.2 * (self.ub - self.lb)
        seed_samples[i] = seed_center + np.random.randn(self.dim) * spread
        seed_samples[i] = np.clip(seed_samples[i], self.lb, self.ub)
    
    # Combine all samples
    remaining = self.NP - len(base_samples) - n_boundary - n_seeds
    if remaining > 0:
        extra_samples = np.random.uniform(self.lb, self.ub, (remaining, self.dim))
        samples = np.vstack([base_samples, boundary_samples, seed_samples, extra_samples])
    else:
        samples = np.vstack([base_samples, boundary_samples, seed_samples])
    
    # Ensure we have exactly NP samples
    if len(samples) > self.NP:
        samples = samples[:self.NP]
    elif len(samples) < self.NP:
        extra = np.random.uniform(self.lb, self.ub, (self.NP - len(samples), self.dim))
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