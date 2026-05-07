**Idea: Restart-Guided Anti-Centralized Initialization**
Use restart history to detect convergence patterns and bias new initializations away from previously explored regions, actively diversifying the search for stuck tasks.

```python
def _initialize_population(self):
    """Initialize population with restart-guided exploration bias."""
    # Standard quasi-random base (same as original)
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(self.dim, scramble=True)
        samples = sampler.random(self.NP)
        samples = qmc.scale(samples, self.lb, self.ub)
    except Exception:
        samples = np.zeros((self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            bin_width = bins[1] - bins[0]
            samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
        for d in range(self.dim):
            samples[:, d] = samples[np.random.permutation(self.NP), d]

    # Restart-guided anti-centralization: bias away from previously explored regions
    if hasattr(self, 'restart_archive') and len(self.restart_archive) >= 3:
        archive_array = np.array(self.restart_archive)
        archive_centroid = np.mean(archive_array, axis=0)
        
        # Compute typical spread scale of archive
        distances = np.linalg.norm(archive_array - archive_centroid, axis=1)
        escape_scale = max(np.median(distances), 1.0)
        
        # Bias samples away from archive centroid
        for i in range(self.NP):
            to_archive = samples[i] - archive_centroid
            dist = np.linalg.norm(to_archive)
            if dist < escape_scale * 0.8:
                # Push away from archive with Gaussian perturbation
                direction = to_archive / (dist + 1e-10)
                push_dist = escape_scale * (0.5 + 0.5 * np.random.random())
                samples[i] = samples[i] + direction * push_dist
                samples[i] += np.random.randn(self.dim) * escape_scale * 0.3
        
        # Clip to bounds
        samples = np.clip(samples, self.lb, self.ub)

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