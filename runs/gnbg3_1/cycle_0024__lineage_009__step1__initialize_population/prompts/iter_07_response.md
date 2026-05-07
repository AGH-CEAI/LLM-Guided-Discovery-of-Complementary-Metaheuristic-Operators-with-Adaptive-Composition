**Idea: Fitness-Weighted Quasi-Random Initialization**
Sample 5x NP candidates using low-discrepancy Sobol sequences, evaluate them, and select the top NP based on actual fitness. This combines space-filling coverage with problem-specific guidance, directly targeting the worst tasks where CMA-ES gets trapped in distant local optima due to poor initial seeding.

```python
def _initialize_population(self):
    """Initialize population using Sobol quasi-random sampling with fitness-based selection."""
    num_candidates = self.NP * 5
    
    # Generate quasi-random Sobol samples for better space-filling
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        candidates = qmc.scale(
            sampler.random(num_candidates),
            self.lb, self.ub
        )
    except Exception:
        # Fallback to uniform random if Sobol unavailable
        candidates = np.random.uniform(self.lb, self.ub, (num_candidates, self.dim))
    
    # Evaluate all candidates to find promising regions
    candidate_fitness = self.func(candidates)
    candidate_fitness = np.asarray(candidate_fitness).flatten()
    
    # Select top NP individuals based on actual fitness
    sorted_indices = np.argsort(candidate_fitness)
    samples = candidates[sorted_indices[:self.NP]]
    
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