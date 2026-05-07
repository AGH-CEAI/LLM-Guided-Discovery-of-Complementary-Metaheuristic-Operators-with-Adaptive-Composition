**Idea: Restart-Aware Archive Seeding with Covariance Boost**
One-line description: Initialize population by seeding 30% of individuals near diverse historical best solutions from a persistent archive, boosting exploration covariance on restart to escape local optima.
```python
def _initialize_population(self):
    """Initialize population with archive-guided seeding and restart-aware exploration boost."""
    # Generate base population using Latin Hypercube sampling
    samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
    for d in range(self.dim):
        bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
        perm = np.random.permutation(self.NP)
        samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
    
    # Initialize persistent global archive for tracking diverse best solutions across restarts
    if not hasattr(self, 'global_best_archive'):
        self.global_best_archive = []
    
    # Add current best to archive if sufficiently distinct from existing entries
    if hasattr(self, 'x_opt') and self.f_opt is not None:
        min_separation = 0.05 * (self.ub[0] - self.lb[0])
        is_novel = True
        for archived in self.global_best_archive:
            if np.linalg.norm(self.x_opt - archived) < min_separation:
                is_novel = False
                break
        if is_novel and len(self.global_best_archive) < 25:
            self.global_best_archive.append(self.x_opt.copy())
    
    # Seed 30% of population near archived solutions to escape local optima
    n_seeds = max(1, int(0.3 * self.NP))
    if self.global_best_archive:
        spread_scale = 0.15 * (self.ub[0] - self.lb[0])
        for i in range(n_seeds):
            center = self.global_best_archive[i % len(self.global_best_archive)]
            spread = spread_scale * (0.5 + 0.5 * np.random.random())
            samples[i] = center + spread * np.random.randn(self.dim)
            samples[i] = np.clip(samples[i], self.lb, self.ub)
    
    self.population = samples
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    
    # Boost exploration on restart to escape local optima (critical for worst tasks)
    is_restart = hasattr(self, 'generation') and self.generation > 0
    if is_restart and hasattr(self, 'f_opt') and self.f_opt is not None:
        self.sigma = min(self.sigma * 2.5, (self.ub[0] - self.lb[0]) * 0.3)
        self.C *= 2.0
    
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