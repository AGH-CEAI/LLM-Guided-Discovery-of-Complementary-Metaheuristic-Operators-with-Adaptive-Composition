**Idea: Sobol Quasi-Random Initialization**
Use Sobol low-discrepancy sequences for space-filling initialization, which provides mathematically optimal coverage compared to Latin Hypercube sampling.

```python
def _initialize_population(self):
    """Initialize population using Sobol quasi-random sequences."""
    # Generate Sobol sequence samples (quasi-random, low-discrepancy)
    # Falls back to stratified sampling if Sobol unavailable
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(self.dim, scramble=True)
        samples = sampler.random(self.NP)
        # Map from [0,1]^dim to [lb, ub]
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