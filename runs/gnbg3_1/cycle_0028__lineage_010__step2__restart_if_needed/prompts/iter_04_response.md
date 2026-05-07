**Idea: Covariance-Directed Multi-Point Restart with Archive Injection**

Instead of random reinitialization, this strategy uses eigendecomposition of the current covariance matrix to generate restart points along diverse eigendirections. This preserves learned geometry while injecting diversity along principal axes where the optimizer was exploring when trapped.

```python
def _restart_if_needed(self):
    """Restart population using covariance-directed multi-point sampling with archive injection."""
    diversity = self._compute_diversity()
    
    if (self.stagnation_counter > self.max_stagnation or 
        diversity < self.min_diversity or 
        np.any(np.isnan(self.C))):
        
        # Preserve elite
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]
        
        # Get eigendecomposition of current covariance for directional restart
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
        except np.linalg.LinAlgError:
            eigvals = np.ones(self.dim)
            eigvecs = np.eye(self.dim)
        
        # Generate restart points along principal axes
        num_restart_points = min(5, self.NP // 10)
        restart_points = []
        
        # Point 1: Elite (already saved)
        restart_points.append(elite.copy())
        
        # Points along top eigendirections (exploit known geometry)
        for i in range(num_restart_points - 1):
            direction = eigvecs[:, -(i + 1)]
            scale = np.sqrt(eigvals[-(i + 1)]) * 2.0
            # Alternate between positive and negative directions
            sign = 1.0 if i % 2 == 0 else -1.0
            offset = sign * scale * direction
            restart_point = self._clip_to_bounds(elite + offset * (0.3 + 0.7 * np.random.random()))
            restart_points.append(restart_point)
        
        # Build new population: inject restart points, fill rest with quasi-random
        new_population = []
        for pt in restart_points:
            new_population.append(pt)
        
        # Fill remaining with boundary-focused Latin Hypercube for diversity
        remaining = self.NP - len(new_population)
        if remaining > 0:
            try:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(self.dim, scramble=True)
                lhs_samples = qmc.scale(sampler.random(remaining), self.lb, self.ub)
                new_population.extend([lhs_samples[i] for i in range(remaining)])
            except Exception:
                # Fallback to uniform
                for _ in range(remaining):
                    new_population.append(np.random.uniform(self.lb, self.ub))
        
        # Reinitialize population and inject restart points
        self._initialize_population()
        
        # Overwrite first positions with our curated restart points
        for i, pt in enumerate(restart_points):
            if i < self.NP:
                self.population[i] = pt
        
        # Re-evaluate fitness for all individuals
        self.fitness = self.func(self.population)
        
        # Update best with elite
        self.population[0] = elite
        self.fitness[0] = elite_fit
        
        best_new_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
        self.x_opt = self.population[best_new_idx].copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
```