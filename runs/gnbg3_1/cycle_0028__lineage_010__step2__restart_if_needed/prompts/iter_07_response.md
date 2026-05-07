**Idea: Covariance-Guided Reinitialization with Multi-Elite Preservation**

Instead of uniform reinitialization, this approach uses eigendecomposition of the covariance matrix to generate new population along unexplored eigendirections while preserving multiple top solutions. This targets the worst tasks (error ~10-100) that are likely stuck in deceptive local optima far from the global optimum.

```python
def _restart_if_needed(self):
    """Restart with covariance-guided exploration and multi-elite preservation."""
    diversity = self._compute_diversity()
    
    # Detect stagnation using adaptive threshold based on current error scale
    base_scale = max(abs(self.f_opt), 1.0)
    rel_threshold = 1e-6 * base_scale
    threshold = max(rel_threshold, 1e-10)
    improvement = self.f_opt_prev - self.f_opt
    
    is_stagnant = (self.stagnation_counter > self.max_stagnation or 
                   improvement <= threshold)
    
    should_restart = (is_stagnant or 
                      diversity < self.min_diversity or 
                      np.any(np.isnan(self.C)))
    
    if not should_restart:
        return
    
    # Preserve top solutions (more than just the single best)
    n_elites = min(5, self.NP // 10)
    sorted_indices = np.argsort(self.fitness)
    elites = [self.population[sorted_indices[i]].copy() for i in range(n_elites)]
    elite_fits = [self.fitness[sorted_indices[i]] for i in range(n_elites)]
    
    # Use eigendecomposition to guide reinitialization
    # Sample along principal axes but in unexplored directions
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        
        # Compute cumulative variance explained to identify principal directions
        cumvar = np.cumsum(eigvals) / np.sum(eigvals)
        n_principal = np.searchsorted(cumvar, 0.95) + 1
        n_principal = min(n_principal, self.dim)
        
        # Generate new population: 50% along principal directions, 50% orthogonal
        n_new = self.NP - n_elites
        
        # Part 1: Sample along principal eigendirections (explore promising regions)
        n_principal_samples = n_new // 2
        for i in range(n_principal_samples):
            # Random direction along principal axes
            direction = np.zeros(self.dim)
            for j in range(n_principal):
                direction += eigvecs[:, j] * np.random.randn()
            direction /= (np.linalg.norm(direction) + 1e-10)
            
            # Scale by eigenvalue spread - larger steps along high-variance axes
            scale = np.sqrt(np.max(eigvals)) * np.random.uniform(0.5, 3.0)
            sample = self.x_opt + direction * scale
            
            # Clip to bounds
            sample = self._clip_to_bounds(sample)
            elites.append(sample)
        
        # Part 2: Latin Hypercube for orthogonal exploration (cover search space)
        n_orthogonal = n_new - n_principal_samples
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            ortho_samples = sampler.random(n_orthogonal)
            ortho_samples = qmc.scale(ortho_samples, self.lb, self.ub)
            for i in range(n_orthogonal):
                elites.append(ortho_samples[i])
        except Exception:
            # Fallback: uniform random
            for i in range(n_orthogonal):
                elites.append(np.random.uniform(self.lb, self.ub))
    except np.linalg.LinAlgError:
        # Fallback to standard initialization if eigendecomposition fails
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            new_samples = sampler.random(self.NP - n_elites)
            new_samples = qmc.scale(new_samples, self.lb, self.ub)
            for i in range(len(new_samples)):
                elites.append(new_samples[i])
        except Exception:
            for _ in range(self.NP - n_elites):
                elites.append(np.random.uniform(self.lb, self.ub))
    
    # Build new population: elites first, then new samples
    all_samples = np.vstack(elites[:self.NP]) if len(elites) >= self.NP else np.vstack(elites)
    if all_samples.shape[0] < self.NP:
        extra = np.random.uniform(self.lb, self.ub, (self.NP - all_samples.shape[0], self.dim))
        all_samples = np.vstack([all_samples, extra])
    elif all_samples.shape[0] > self.NP:
        all_samples = all_samples[:self.NP]
    
    self.population = all_samples
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    self.L = None
    
    self.fitness = self.func(self.population)
    
    best_idx = np.argmin(self.fitness)
    self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
    self.x_opt = self.population[best_idx].copy()
    self.f_opt_prev = self.f_opt
    self.stagnation_counter = 0
```