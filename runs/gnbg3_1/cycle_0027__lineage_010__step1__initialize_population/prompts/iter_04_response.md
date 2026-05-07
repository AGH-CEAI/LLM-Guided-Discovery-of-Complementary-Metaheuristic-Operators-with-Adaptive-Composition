**Idea: LHS with Elite-Targeted Resampling**

Use Latin Hypercube Sampling for better space-filling coverage, then perform fitness-based adaptive resampling concentrated around top-performing regions to escape deceptive local optima that trap the current Sobol-based initialization on high-error tasks (16, 17, 10, 20, 23).

```python
def _initialize_population(self):
    """Initialize population using LHS with elite-targeted adaptive resampling."""
    try:
        from scipy.stats import qmc
        # Latin Hypercube Sampling: better space-filling than Sobol for escaping local optima
        sampler = qmc.LatinHypercube(self.dim, scramble=True)
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
        for d in range(self.dim):
            samples[:, d] = samples[np.random.permutation(self.NP), d]

    self.population = samples.copy()
    
    # Evaluate initial LHS population
    init_fitness = self.func(self.population)
    init_fitness = np.asarray(init_fitness).flatten()
    
    # For high-error tasks: perform adaptive resampling around elite regions
    # This helps escape deceptive local optima that trap Sobol-based initialization
    if np.min(init_fitness) > 1.0:  # High error detected
        n_elite = max(3, self.NP // 16)
        elite_indices = np.argsort(init_fitness)[:n_elite]
        elite_samples = self.population[elite_indices]
        
        # Generate additional samples around each elite region
        n_resample = self.NP // 4
        resample_per_elite = max(1, n_resample // n_elite)
        
        resampled = []
        for elite in elite_samples:
            # Scale spread by distance to bounds for bounded optima
            spread = 0.15 * (self.ub - self.lb)
            spread = np.clip(spread, 1.0, 50.0)
            
            for _ in range(resample_per_elite):
                candidate = elite + np.random.randn(self.dim) * spread
                candidate = np.clip(candidate, self.lb, self.ub)
                resampled.append(candidate)
        
        resampled = np.array(resampled[:n_resample])
        
        # Evaluate resampled candidates
        if len(resampled) > 0:
            resample_fitness = np.asarray(self.func(resampled)).flatten()
            
            # Keep all original samples plus best resampled candidates
            combined_pop = np.vstack([self.population, resampled])
            combined_fit = np.concatenate([init_fitness, resample_fitness])
            
            # Select best NP individuals (elitist selection)
            keep_indices = np.argsort(combined_fit)[:self.NP]
            self.population = combined_pop[keep_indices]
            init_fitness = combined_fit[keep_indices]

    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()

    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)

    self.fitness = init_fitness[:self.NP]

    best_idx = np.argmin(self.fitness)
    self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
    self.x_opt = self.population[best_idx].copy()
    self.f_opt_prev = self.f_opt

    self.stagnation_counter = 0
    self.generation = 0
    self.current_operator = 0

    self._reset_operator_state()
```