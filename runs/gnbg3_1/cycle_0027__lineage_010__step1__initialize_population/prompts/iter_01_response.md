**Idea: Latin Hypercube Sampling with Gaussian Perturbation**
A well-established space-filling initialization using Latin Hypercube Sampling for uniform coverage with optional Gaussian perturbation, providing diverse exploration while maintaining numerical stability across all dimensions.

```python
def _initialize_population(self):
    """Initialize population using Latin Hypercube Sampling with Gaussian perturbation."""
    # Ensure bounds are numpy arrays
    lb = np.asarray(self.lb, dtype=np.float64)
    ub = np.asarray(self.ub, dtype=np.float64)
    
    # Handle edge case of zero range
    range_vec = ub - lb
    range_vec = np.where(range_vec < 1e-15, 1.0, range_vec)
    
    # Latin Hypercube Sampling for space-filling coverage
    try:
        from scipy.stats import qmc
        # Use Halton sequence as base for LHS (more robust than random LHS)
        sampler = qmc.Halton(d=self.dim, scramble=True)
        samples = sampler.random(self.NP)
        
        # Apply inverse CDF transform for better uniformity
        from scipy import stats
        samples = stats.norm.ppf(np.clip(samples, 1e-10, 1 - 1e-10))
        
        # Scale to bounds using center and range
        center = (ub + lb) / 2.0
        half_range = range_vec / 2.0
        
        # Map from N(0,1) to [lb, ub] - use moderate sigma for exploration
        sigma_init = 0.5  # 50% of half-range as initial step size
        samples = center + samples * half_range * sigma_init
        
    except Exception:
        # Fallback: stratified grid with jitter
        samples = np.zeros((self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(lb[d], ub[d], self.NP + 1)
            bin_width = bins[1] - bins[0]
            # Place points at bin centers with uniform jitter
            samples[:, d] = bins[:-1] + bin_width * (0.5 + (np.random.random(self.NP) - 0.5) * 0.8)
    
    # Clip to bounds ensuring numerical stability
    samples = np.clip(samples, lb, ub)
    
    # Ensure no duplicate individuals (replace duplicates with perturbed versions)
    unique_mask = np.ones(self.NP, dtype=bool)
    for i in range(self.NP):
        if not unique_mask[i]:
            continue
        for j in range(i + 1, self.NP):
            if unique_mask[j]:
                dist = np.linalg.norm(samples[i] - samples[j])
                if dist < 1e-8 * np.mean(range_vec):
                    # Add small perturbation to avoid duplicate
                    samples[j] += np.random.randn(self.dim) * range_vec * 0.01
                    samples[j] = np.clip(samples[j], lb, ub)
                    unique_mask[j] = False
    
    self.population = samples
    
    # Initialize CMA-ES state variables
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    # Covariance matrix with numerical safety
    cov_matrix = np.cov(self.population.T)
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        cov_matrix = np.eye(self.dim) * (np.mean(range_vec) / 6.0) ** 2
    min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
    if min_eig < 1e-10:
        cov_matrix += (1e-8 - min_eig) * np.eye(self.dim)
    self.C = cov_matrix
    
    # Step size initialization based on problem scale
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.sigma = np.clip(self.sigma, 1e-10, np.mean(range_vec) / 3.0)
    
    # Path vectors
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    
    # Evaluate initial population
    self.fitness = self.func(self.population)
    
    # Track best solution
    best_idx = np.argmin(self.fitness)
    self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
    self.x_opt = self.population[best_idx].copy()
    self.f_opt_prev = self.f_opt
    
    # Reset counters
    self.stagnation_counter = 0
    self.generation = 0
    self.current_operator = 0
    
    self._reset_operator_state()
```