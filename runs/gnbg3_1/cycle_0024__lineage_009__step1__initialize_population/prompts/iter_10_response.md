**Idea: Boundary-Focused LHS with Adaptive Diversity Injection**

Proposes a hybrid initialization that combines boundary-heavy Latin Hypercube sampling with adaptive diversity injection based on early fitness evaluation, designed to break through local optima traps on the hardest multi-modal tasks (errors ~1e+01 to 2e+03).

```python
def _initialize_population(self):
    """Initialize population using boundary-focused LHS with adaptive diversity injection."""
    # Phase 1: Generate base population with boundary-focused Latin Hypercube
    # Sample more densely near boundaries where global optima often hide
    samples = np.zeros((self.NP, self.dim))
    
    for d in range(self.dim):
        bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
        bin_width = bins[1] - bins[0]
        
        # 60% boundary-biased samples, 40% uniform within bins
        n_boundary = int(0.6 * self.NP)
        n_inner = self.NP - n_boundary
        
        perm = np.random.permutation(self.NP)
        
        # Boundary-biased: place samples near bin edges
        boundary_positions = np.concatenate([
            bins[:n_boundary//2],  # near lower bounds
            bins[-(n_boundary - n_boundary//2):]  # near upper bounds
        ])
        if len(boundary_positions) > n_boundary:
            boundary_positions = boundary_positions[:n_boundary]
        
        samples[perm[:n_boundary], d] = boundary_positions + np.random.uniform(0, 0.1 * bin_width, n_boundary)
        
        # Inner samples: scatter within bins
        samples[perm[n_boundary:], d] = bins[perm[n_boundary:]] + bin_width * np.random.random(n_inner)
    
    # Phase 2: Diversity injection - add samples along coordinate axes
    n_axis = max(2, self.NP // 20)
    axis_indices = np.random.choice(self.NP, min(n_axis, self.NP), replace=False)
    for i, idx in enumerate(axis_indices[:min(n_axis, self.NP)]):
        dim_choice = i % self.dim
        pos = np.random.choice([self.lb[dim_choice], self.ub[dim_choice]])
        samples[idx, dim_choice] = pos
        for other_d in range(self.dim):
            if other_d != dim_choice:
                samples[idx, other_d] = np.random.uniform(self.lb[other_d], self.ub[other_d])
    
    # Phase 3: Adaptive diversity based on early fitness (if func available)
    self.population = samples
    self.population = self._clip_to_bounds(self.population)
    
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    
    # Initial fitness evaluation
    self.fitness = self.func(self.population)
    
    # Phase 4: Diversity-guided repositioning based on fitness clustering
    fit_arr = np.asarray(self.fitness).flatten()
    fit_range = np.max(fit_arr) - np.min(fit_arr) + 1e-10
    
    # Identify potential local optima (low-fitness cluster centers)
    threshold = np.percentile(fit_arr, 25)
    elite_mask = fit_arr <= threshold
    n_elite = np.sum(elite_mask)
    
    if n_elite > 0:
        elite_positions = self.population[elite_mask]
        elite_mean = np.mean(elite_positions, axis=0)
        
        # Inject diverse samples away from elite cluster
        n_diverse = max(2, self.NP // 10)
        diverse_indices = np.random.choice(self.NP, n_diverse, replace=False)
        
        for idx in diverse_indices:
            # Sample in direction opposite to elite cluster
            to_elite = elite_mean - self.population[idx]
            dist = np.linalg.norm(to_elite)
            if dist > 1e-6:
                direction = -to_elite / dist
                # Move away from elite cluster
                move_dist = np.random.uniform(0.2, 0.5) * (self.ub[0] - self.lb[0])
                self.population[idx] = self._clip_to_bounds(self.population[idx] + direction * move_dist)
    
    # Re-evaluate repositioned population
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