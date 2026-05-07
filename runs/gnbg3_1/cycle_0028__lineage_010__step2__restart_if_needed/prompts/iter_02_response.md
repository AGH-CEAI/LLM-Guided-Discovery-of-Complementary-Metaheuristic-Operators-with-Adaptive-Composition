**Idea: Archive-Guided Restart with Adaptive Intensity**
When restarting, preserve multiple archive solutions and generate new samples around them proportionally, with restart intensity scaling based on stagnation severity and error magnitude. This targets the multimodal/deceptive tasks (17, 16, 10, etc.) where the algorithm gets stuck far from global optimum because the current restart (single elite + uniform random) cannot escape local basins.

```python
def _restart_if_needed(self):
    """Restart population using archive-guided diversity with adaptive intensity.
    
    For high-error tasks (10-75 range), the current single-elite restart is
    insufficient because it discards all learned information. This version:
    1. Uses multiple archive solutions (if available) as seeds
    2. Scales restart intensity with stagnation severity and error magnitude
    3. Samples around archive solutions proportionally to their fitness
    4. Adds boundary-focused exploration for severe trapping
    """
    diversity = self._compute_diversity()
    
    # Determine restart intensity: 0=normal restart, 1=aggressive escape
    stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
    error_magnitude = np.log1p(max(abs(self.f_opt), 1.0))
    intensity = np.clip(0.3 * stagnation_ratio + 0.1 * error_magnitude / 10.0, 0.0, 1.0)
    
    should_restart = (self.stagnation_counter > self.max_stagnation or 
                      diversity < self.min_diversity or 
                      np.any(np.isnan(self.C)))
    
    if not should_restart:
        return
    
    # Collect archive solutions if available
    archive_solutions = []
    if hasattr(self, 'archive_positions') and len(self.archive_positions) >= 2:
        # Sort archive by fitness
        sorted_indices = np.argsort(self.archive_fitness)
        for idx in sorted_indices[:min(5, len(self.archive_positions))]:
            archive_solutions.append(self.archive_positions[idx].copy())
    
    # Get current best
    best_idx = np.argmin(self.fitness)
    elite = self.population[best_idx].copy()
    elite_fit = self.fitness[best_idx]
    
    # Reinitialize base population
    self._initialize_population()
    
    # Determine how many slots to fill with archive-guided vs random samples
    n_archive = int(np.floor(self.NP * intensity * 0.5))
    n_archive = min(n_archive, len(archive_solutions), self.NP - 1)
    
    # Fill with archive-guided samples (proportional to fitness rank)
    if n_archive > 0 and archive_solutions:
        for i in range(n_archive):
            if i >= len(archive_solutions):
                break
            # Sample around archive solution with adaptive spread
            spread = self.sigma * (2.0 + 3.0 * intensity)
            sample = archive_solutions[i] + np.random.randn(self.dim) * spread
            sample = self._clip_to_bounds(sample)
            self.population[i + 1] = sample
    
    # Fill remaining with boundary-focused exploration
    n_boundary = self.NP - 1 - n_archive
    if n_boundary > 0:
        # Power-law sampling for strong boundary focus on high-intensity restarts
        power = 2.0 + 3.0 * intensity
        u = np.random.random((n_boundary, self.dim))
        boundary_samples = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
        
        # Apply to random subset of dimensions (more for aggressive restarts)
        dim_fraction = 0.2 + 0.4 * intensity
        dim_mask = np.random.random(self.dim) < dim_fraction
        for d in range(self.dim):
            if not dim_mask[d]:
                boundary_samples[:, d] = np.random.uniform(0, 1, n_boundary)
        
        # Scale to bounds
        for d in range(self.dim):
            boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])
        
        start_idx = 1 + n_archive
        self.population[start_idx:start_idx + n_boundary] = boundary_samples
    
    # Restore elite as first individual
    self.population[0] = elite
    self.fitness[0] = elite_fit
    
    # Update state
    self.f_opt = float(np.asarray(elite_fit).flatten()[0])
    self.x_opt = elite.copy()
    self.f_opt_prev = self.f_opt
    self.stagnation_counter = 0
    
    # Reset sigma and covariance for aggressive restarts
    if intensity > 0.5:
        self.sigma = max(self.sigma, 0.5 * (self.ub[0] - self.lb[0]) / 6.0)
        self.C = np.eye(self.dim) * (self.sigma ** 2)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
```