**Idea: Two-Phase LHS with Adaptive Focusing**

This strategy uses Latin Hypercube Sampling (LHS) for better space-filling properties in Phase 1, then adaptively concentrates additional samples around the most promising initial region discovered in Phase 2. Unlike Sobol's uniform coverage, this focuses initialization on regions that show initial promise—critical for the multimodal tasks (16, 17, 10, 20, 23) where the algorithm gets trapped in local optima with errors ~10-60.

```python
def _initialize_population(self):
    """Two-phase initialization: LHS for coverage, then adaptive focusing around best region."""
    # Phase 1: Generate base samples using Latin Hypercube (better space-filling than Sobol)
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(self.dim, scramble=True)
        samples = sampler.random(self.NP)
        samples = qmc.scale(samples, self.lb, self.ub)
    except Exception:
        samples = self.lb + np.random.random((self.NP, self.dim)) * (self.ub - self.lb)

    # Phase 2: Adaptive focusing - concentrate samples around best initial region
    phase2_budget = max(10, self.NP // 5)
    phase2_samples = samples[:phase2_budget]
    phase2_fitness = self.func(phase2_samples)
    
    best_phase2_idx = np.argmin(phase2_fitness)
    best_phase2_fit = phase2_fitness[best_phase2_idx]
    best_phase2_point = phase2_samples[best_phase2_idx]
    
    # Determine focusing radius based on search space size and dimension
    space_diameter = np.linalg.norm(self.ub - self.lb)
    focus_radius = 0.15 * space_diameter / np.sqrt(self.dim)
    focus_radius = np.clip(focus_radius, 1e-6, space_diameter * 0.3)
    
    # Generate focused samples around best point with dimension-wise scaling
    dim_ranges = self.ub - self.lb
    dim_scales = np.maximum(dim_ranges, 1e-8)
    
    focus_count = self.NP - phase2_budget
    if focus_count > 0:
        # Use t-distribution for heavier tails (more exploration in focused region)
        focus_z = np.random.standard_t(df=4, size=(focus_count, self.dim))
        focus_samples = best_phase2_point + focus_z * (focus_scales * focus_radius / 2)
        focus_samples = self._clip_to_bounds(focus_samples)
        samples = np.vstack([phase2_samples, focus_samples])
    
    # Assign to population
    self.population = samples
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()

    # Covariance from LHS + focused samples (should show initial anisotropy)
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