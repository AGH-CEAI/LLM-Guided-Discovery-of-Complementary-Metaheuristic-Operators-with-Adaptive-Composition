**Idea: Basin-Jumping Restart with Archive-Driven Diversification**

For the worst tasks (17, 16, 20, 18, 10, 23, 12, 19) with errors 10-100, the algorithm is trapped in a completely wrong basin. The current restart reinitializes uniformly at random, which is blind to the landscape structure. This variant uses the fitness gap to diagnose basin-hopping failure and performs archive-driven diversification that explicitly searches in orthogonal basins.

```python
def _restart_if_needed(self):
    """Restart with basin-jumping strategy: diagnose basin entrapment and diversify."""
    diversity = self._compute_diversity()
    
    if (self.stagnation_counter > self.max_stagnation or 
        diversity < self.min_diversity or 
        np.any(np.isnan(self.C))):
        
        # Initialize archive tracking for basin detection
        if not hasattr(self, 'restart_archive'):
            self.restart_archive = []
        if not hasattr(self, 'restart_archive_fitness'):
            self.restart_archive_fitness = []
        
        # Store current best in archive for basin tracking
        if len(self.restart_archive) < 20:
            self.restart_archive.append(self.x_opt.copy())
            self.restart_archive_fitness.append(self.f_opt)
        else:
            # Evict worst basin if we found a better one
            worst_idx = np.argmax(self.restart_archive_fitness)
            if self.f_opt < self.restart_archive_fitness[worst_idx]:
                self.restart_archive[worst_idx] = self.x_opt.copy()
                self.restart_archive_fitness[worst_idx] = self.f_opt
        
        # Diagnose basin entrapment: large error = wrong basin
        # Small error = right basin but slow convergence
        abs_error = abs(self.f_opt)
        is_wrong_basin = abs_error > 10.0
        
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]
        
        self._initialize_population()
        
        self.population[0] = elite
        self.fitness[0] = elite_fit
        self.f_opt = float(np.asarray(elite_fit).flatten()[0])
        self.x_opt = elite.copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
        
        # Basin-jumping: for wrong-basin tasks, inject archive-based diversity
        if is_wrong_basin and len(self.restart_archive) >= 3:
            # Compute centroid of known basins (avoid re-exploring them)
            archive_array = np.array(self.restart_archive)
            archive_centroid = np.mean(archive_array, axis=0)
            
            # Direction away from centroid = toward unexplored regions
            escape_dir = self.mean - archive_centroid
            escape_dist = np.linalg.norm(escape_dir)
            
            if escape_dist > 1e-6:
                escape_dir /= escape_dist
                # Inject 30% of population along escape direction
                n_escape = max(1, int(0.3 * self.NP))
                for j in range(n_escape):
                    if j < len(self.population):
                        # Generate points along escape direction with noise
                        scale = np.random.uniform(0.5, 2.0) * escape_dist
                        new_point = self.mean + escape_dir * scale + \
                                    np.random.randn(self.dim) * 0.1 * scale
                        self.population[j] = self._clip_to_bounds(new_point)
            
            # Also inject points near best archive members (diversify basin search)
            n_archive_pts = max(1, int(0.15 * self.NP))
            sorted_arch_indices = np.argsort(self.restart_archive_fitness)
            for j in range(min(n_archive_pts, len(sorted_arch_indices))):
                idx = sorted_arch_indices[j]
                if j + n_escape < len(self.population):
                    arch_pt = self.restart_archive[idx]
                    # Perturb around archive member
                    pert = arch_pt + np.random.randn(self.dim) * 0.5 * abs(self.restart_archive_fitness[idx] + 1)
                    self.population[j + n_escape] = self._clip_to_bounds(pert)
        
        # Re-evaluate injected points
        if is_wrong_basin and len(self.restart_archive) >= 3:
            eval_slice = slice(0, min(n_escape + n_archive_pts, self.NP))
            self.fitness[eval_slice] = self.func(self.population[eval_slice])
            
            # Update best if any injected point improved
            injected_best_idx = np.argmin(self.fitness[eval_slice])
            if self.fitness[eval_slice][injected_best_idx] < self.f_opt:
                self.f_opt = self.fitness[eval_slice][injected_best_idx]
                self.x_opt = self.population[eval_slice][injected_best_idx].copy()
                self.f_opt_prev = self.f_opt
```