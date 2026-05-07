**Idea: LHS with Bounded Random Walk Exploration**
Initialize with Latin Hypercube Sampling (better stratified coverage) plus bounded random walk perturbation to escape deceptive local optima from the start.

```python
def _initialize_population(self):
    """Initialize population using Latin Hypercube Sampling with bounded random walk exploration."""
    # Latin Hypercube Sampling for better stratified coverage across dimensions
    # vs Sobol's low-discrepancy which may cluster in certain regions
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(self.dim, scramble=True)
        samples = sampler.random(self.NP)
        samples = qmc.scale(samples, self.lb, self.ub)
    except Exception:
        # Fallback: stratified sampling with jitter
        samples = np.zeros((self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            bin_width = bins[1] - bins[0]
            samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
        # Shuffle each dimension independently
        for d in range(self.dim):
            samples[:, d] = samples[np.random.permutation(self.NP), d]

    # Bounded random walk exploration: perturb samples to escape
    # potential deceptive local optima regions in worst tasks (errors ~10-60)
    walk_scale = 0.15 * (self.ub - self.lb)
    walk_scale = np.maximum(walk_scale, 1e-6)
    num_walk = max(1, self.NP // 5)
    walk_perturbation = np.random.randn(num_walk, self.dim) * walk_scale
    samples[:num_walk] = samples[:num_walk] + walk_perturbation

    # Additional extreme exploration: uniform random samples in outer bounds
    num_extreme = max(1, self.NP // 10)
    extreme_samples = np.random.uniform(self.lb, self.ub, (num_extreme, self.dim))
    samples[num_walk:num_walk + num_extreme] = extreme_samples

    # Clip all samples to bounds
    samples = np.clip(samples, self.lb, self.ub)

    self.population = samples
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()

    self.C = np.cov(self.population.T)
    if self.C.size == 1:
        self.C = np.array([[max(self.C.item(), 1e-8)]])
    self.C += 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.sigma = max(self.sigma, 1e-8)
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