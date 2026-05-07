**Idea: Fitness-Adaptive Two-Phase LHS Initialization**

Replaces uniform Sobol sampling with a two-phase approach: (1) broad LHS exploration to identify promising regions via fitness evaluation, then (2) reinitialization concentrated around elite solutions with diversity-preserving perturbations. This targets the worst tasks where uniform sampling likely misses the global optimum's basin entirely.

```python
def _initialize_population(self):
    """Initialize population using fitness-adaptive two-phase LHS sampling."""
    # Phase 1: Broad exploration with Latin Hypercube Sampling
    try:
        from scipy.stats import qmc
        lhd = qmc.LatinHypercube(self.dim, scramble=True)
        broad_samples = lhd.random(self.NP)
        broad_samples = qmc.scale(broad_samples, self.lb, self.ub)
    except Exception:
        broad_samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
    
    # Evaluate fitness to identify promising regions
    broad_fitness = self.func(broad_samples)
    broad_fitness = np.asarray(broad_fitness).flatten()
    
    # Phase 2: Reinitialize concentrated around elite regions
    sorted_indices = np.argsort(broad_fitness)
    num_elites = max(1, self.NP // 8)
    elite_indices = sorted_indices[:num_elites]
    
    # Build focused population around best solutions
    focused_pop = np.zeros((self.NP, self.dim))
    
    for i in range(self.NP):
        elite_idx = elite_indices[i % num_elites]
        center = broad_samples[elite_idx]
        
        # Adaptive spread: tighter around best, wider for diversity
        spread_scale = 0.1 + 0.4 * (i / self.NP)
        spread = spread_scale * (self.ub - self.lb)
        
        # Sample with Gaussian falloff from center
        candidate = center + np.random.randn(self.dim) * spread
        focused_pop[i] = self._clip_to_bounds(candidate)
    
    # Preserve some broad exploration (20% of population)
    preserve_count = max(1, self.NP // 5)
    perm = np.random.permutation(self.NP)
    keep_indices = perm[:preserve_count]
    replace_indices = perm[preserve_count:]
    
    samples = focused_pop.copy()
    samples[replace_indices] = broad_samples[keep_indices]
    
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