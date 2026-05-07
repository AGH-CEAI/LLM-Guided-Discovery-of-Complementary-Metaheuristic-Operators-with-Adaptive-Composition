Looking at the worst unsolved tasks (17, 11, 16, 12, 10, 20, 23, 18, 19 with errors 10-2000), they're stuck at very high errors indicating likely local optima trapping from the start. The current Latin Hypercube sampling is purely space-filling but doesn't use any fitness feedback or adapt to problem scale. The key insight: **adaptive sampling that uses fitness-informed perturbations and multi-scale exploration should help escape these massive local optima from initialization**.

**Idea: Adaptive Multi-Scale Sampling with Meritocratic Perturbation**
Initialize using a hybrid of uniform, log-uniform, and adaptive normal sampling. Use fitness feedback to selectively perturb top candidates and adapt variance based on problem dimensionality and elite performance.
```python
def _initialize_population(self):
    """Initialize population using adaptive multi-scale sampling with meritocratic perturbation."""
    candidates = []
    
    # Standard uniform sampling
    uniform_samples = np.random.uniform(self.lb, self.ub, (self.NP * 3, self.dim))
    candidates.append(uniform_samples)
    
    # Log-uniform sampling for bounded problems (multi-scale exploration)
    if self.dim > 1:
        for d in range(self.dim):
            if np.isfinite(self.lb[d]) and np.isfinite(self.ub[d]) and self.lb[d] > 0 and self.ub[d] > 0:
                log_lb, log_ub = np.log(self.lb[d] + 1e-10), np.log(self.ub[d] + 1e-10)
                log_samples = np.exp(np.random.uniform(log_lb, log_ub, (self.NP, self.dim)))
                candidates.append(log_samples)
            elif self.lb[d] < 0 and self.ub[d] > 0:
                # Symmetric log for zero-crossing bounds
                for sign in [1, -1]:
                    log_mag = np.random.uniform(0, np.log(self.ub[d] + 1e-10), (self.NP // 2, self.dim))
                    samples = sign * np.exp(log_mag)
                    candidates.append(samples)
    
    # Adaptive normal sampling centered at origin (controlled exploration)
    sigma_init = (self.ub[0] - self.lb[0]) / (6.0 * np.sqrt(self.dim))
    sigma_init = np.clip(sigma_init, 1e-3, 10.0)
    normal_samples = np.random.randn(self.NP * 2, self.dim) * sigma_init
    normal_samples = np.clip(normal_samples, self.lb, self.ub)
    candidates.append(normal_samples)
    
    # Combine all candidates
    all_candidates = np.vstack(candidates)
    
    # Evaluate all candidates
    all_fitness = self.func(all_candidates)
    all_fitness = np.asarray(all_fitness).flatten()
    
    # Meritocratic selection: keep top NP candidates
    sorted_indices = np.argsort(all_fitness)
    top_indices = sorted_indices[:min(self.NP, len(all_candidates))]
    self.population = all_candidates[top_indices].copy()
    
    # Adaptive perturbation: perturb top candidates for diversity
    top_count = max(1, self.NP // 10)
    for i in range(top_count):
        if i < len(self.population):
            # Compute adaptive sigma based on performance
            fit_range = max(all_fitness[top_indices[0]], 1e-10)
            adaptive_sigma = sigma_init * np.clip(1.0 / (1.0 + np.log1p(fit_range)), 0.01, 1.0)
            
            # Perturb along random directions
            direction = np.random.randn(self.dim)
            direction /= (np.linalg.norm(direction) + 1e-10)
            magnitude = np.random.uniform(0.1, 1.0) * adaptive_sigma * (self.ub[0] - self.lb[0])
            self.population[i] += direction * magnitude
            self.population[i] = np.clip(self.population[i], self.lb, self.ub)
    
    # Ensure population size
    if len(self.population) < self.NP:
        additional = np.random.uniform(self.lb, self.ub, (self.NP - len(self.population), self.dim))
        self.population = np.vstack([self.population, additional])
    
    # Re-evaluate final population
    self.fitness = self.func(self.population)
    self.fitness = np.asarray(self.fitness).flatten()
    
    # Update statistics
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.sigma = np.clip(self.sigma, 1e-6, (self.ub[0] - self.lb[0]) / 3.0)
    
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    
    best_idx = np.argmin(self.fitness)
    self.f_opt = float(self.fitness[best_idx])
    self.x_opt = self.population[best_idx].copy()
    self.f_opt_prev = self.f_opt
    
    self.stagnation_counter = 0
    self.generation = 0
    self.current_operator = 0
    
    self._reset_operator_state()
```