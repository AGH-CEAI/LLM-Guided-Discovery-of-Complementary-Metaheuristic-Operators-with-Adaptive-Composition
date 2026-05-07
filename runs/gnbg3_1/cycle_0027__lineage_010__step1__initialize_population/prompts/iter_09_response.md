**Idea: Latin Hypercube + Fitness-Guided Niching with Cauchy Perturbations**
Replaces Sobol quasi-random sampling with LHS space-filling + elite-centered niches using heavy-tailed perturbations to escape local optima on multimodal/ill-conditioned tasks.
```python
def _initialize_population(self):
    """Initialize population using Latin Hypercube + fitness-guided niching with Cauchy perturbations."""
    import numpy as np
    
    # Use Latin Hypercube for better space-filling than Sobol
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(self.dim, scramble=True)
        samples = sampler.random(self.NP)
        samples = qmc.scale(samples, self.lb, self.ub)
    except Exception:
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
    
    # Evaluate initial samples
    init_fitness = self.func(samples)
    
    # Identify top solutions as niche centers
    sorted_indices = np.argsort(init_fitness)
    num_elites = max(1, self.NP // 10)
    elite_indices = sorted_indices[:num_elites]
    
    # Create niches around elite solutions
    num_niches = min(num_elites, 4)
    samples_per_niche = self.NP // num_niches
    
    population = []
    
    for niche_id in range(num_niches):
        elite_idx = elite_indices[niche_id % len(elite_indices)]
        center = samples[elite_idx]
        
        # Add elite solution
        population.append(center)
        
        # Generate Cauchy perturbations around niche center (heavy-tailed)
        remaining = samples_per_niche - 1
        for _ in range(remaining):
            scale = 0.2 * (self.ub - self.lb)
            perturbation = np.random.standard_cauchy(self.dim) * scale
            candidate = center + perturbation
            candidate = np.clip(candidate, self.lb, self.ub)
            population.append(candidate)
    
    # Fill remaining slots with LHS samples
    remaining_slots = self.NP - len(population)
    if remaining_slots > 0:
        extra = sampler.random(remaining_slots)
        extra = qmc.scale(extra, self.lb, self.ub)
        population.extend(extra)
    
    self.population = np.array(population[:self.NP])
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