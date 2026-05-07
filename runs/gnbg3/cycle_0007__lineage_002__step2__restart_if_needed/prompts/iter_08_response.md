**Idea: Fitness-Spatial Hybrid Restart with Data-Driven Weighting**

Combine two distinct restart triggers (fitness distribution analysis + spatial clustering analysis) with weighting based on the relative severity of each stagnation signal. Restart intensity blends archive-based re-seeding (exploitation) with random reinitialization (exploration) based on which mechanism dominates.

```python
def _restart_if_needed(self, population, fitness):
    """Hybrid restart: combine fitness-variance and spatial-clustering signals with data-driven weighting."""
    if len(population) < 4:
        return population, np.nanmin(fitness), population[np.nanargmin(fitness)]
    
    # Initialize temporal tracking
    if not hasattr(self, '_fitness_cv_history'):
        self._fitness_cv_history = []
        self._spatial_cv_history = []
    
    # --- Mechanism 1: Fitness Distribution Analysis ---
    valid_fitness = fitness[~np.isnan(fitness)]
    if len(valid_fitness) < 2:
        fitness_diversity = 1.0
    else:
        cv = np.std(valid_fitness) / (np.abs(np.mean(valid_fitness)) + 1e-10)
        fitness_cv = np.clip(cv, 0.0, 10.0)
        self._fitness_cv_history.append(fitness_cv)
        if len(self._fitness_cv_history) > 15:
            self._fitness_cv_history.pop(0)
        if len(self._fitness_cv_history) >= 3:
            baseline_cv = np.mean(self._fitness_cv_history[:3])
            current_cv = np.mean(self._fitness_cv_history[-3:])
            fitness_diversity = current_cv / (baseline_cv + 1e-10)
        else:
            fitness_diversity = 1.0
    
    # --- Mechanism 2: Spatial Clustering Analysis ---
    centroid = population.mean(axis=0)
    dists = np.linalg.norm(population - centroid, axis=1)
    spatial_spread = np.mean(dists)
    self._spatial_cv_history.append(spatial_spread)
    if len(self._spatial_cv_history) > 15:
        self._spatial_cv_history.pop(0)
    if len(self._spatial_cv_history) >= 3:
        baseline_spread = np.mean(self._spatial_cv_history[:3])
        current_spread = np.mean(self._spatial_cv_history[-3:])
        spatial_diversity = current_spread / (baseline_spread + 1e-10)
    else:
        spatial_diversity = 1.0
    
    # --- Data-Driven Weighting: Severity-Based ---
    fitness_severity = 1.0 / (fitness_diversity + 1e-10)
    spatial_severity = 1.0 / (spatial_diversity + 1e-10)
    total_severity = fitness_severity + spatial_severity + 1e-10
    w_fitness = fitness_severity / total_severity
    w_spatial = spatial_severity / total_severity
    
    # Combined stagnation signal
    stagnation_signal = w_fitness * fitness_severity + w_spatial * spatial_severity
    
    # Restart if either signal indicates convergence
    if stagnation_signal < 0.5 or fitness_diversity < 0.3 or spatial_diversity < 0.3:
        # Determine restart intensity from combined signal
        intensity = np.clip(1.0 - stagnation_signal, 0.2, 0.8)
        
        n_replace = max(int(self.NP * intensity), 2)
        worst_idx = np.argsort(fitness)[-n_replace:]
        
        # Archive blend: high intensity -> favor archive (exploit), low -> favor random (explore)
        archive_blend = np.clip(intensity, 0.3, 0.7)
        
        for idx in worst_idx:
            if np.random.rand() < archive_blend and len(self.archive) > 0:
                arch = np.vstack(self.archive)
                donor = arch[np.random.randint(len(arch))]
                noise_scale = 15.0 * (1.0 - intensity)
                population[idx] = np.clip(donor + np.random.randn(self.dim) * noise_scale, -100.0, 100.0)
            else:
                population[idx] = np.random.uniform(-100.0, 100.0, self.dim)
        
        # Reset temporal tracking to prevent immediate re-trigger
        self._fitness_cv_history = self._fitness_cv_history[-3:]
        self._spatial_cv_history = self._spatial_cv_history[-3:]
    
    best_idx = np.nanargmin(fitness)
    return population, fitness[best_idx], population[best_idx].copy()
```