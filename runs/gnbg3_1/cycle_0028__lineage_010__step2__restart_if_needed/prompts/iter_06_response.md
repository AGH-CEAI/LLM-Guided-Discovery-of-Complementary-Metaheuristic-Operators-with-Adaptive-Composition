Looking at the unsolved tasks, the worst ones (Tasks 17, 16, 10, 20, 18, 23, 12, 19) all have errors in the 10-74 range, suggesting severely ill-conditioned or deceptive landscapes where the algorithm gets trapped far from the global optimum.

**Analysis:**
- Current restart: Saves elite, reinitializes population (Sobol+LHS+boundary), resets covariance to identity-like
- Problem: For high-error tasks, this discards the *learned covariance structure* that encodes the problem's geometry. Reinitializing loses all progress on understanding the landscape.

**Key Insight:**
For ill-conditioned tasks with error ~10-74, the covariance matrix has learned *something* about the search landscape even if suboptimal. The restart should preserve and leverage this structure rather than discarding it.

**Idea: Covariance-Preserving Targeted Restart**
Instead of full reinitialization, re-center the population around the elite using the current covariance but with increased variance. This maintains learned search geometry while injecting diversity toward unexplored regions.

```python
def _restart_if_needed(self):
    """Restart with covariance-preserving re-seeding around elite for ill-conditioned tasks."""
    diversity = self._compute_diversity()
    
    if (self.stagnation_counter > self.max_stagnation or 
        diversity < self.min_diversity or 
        np.any(np.isnan(self.C))):
        
        # Save elite and current state
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]
        
        # Preserve covariance structure but increase variance for exploration
        # Get eigenvalues and eigenvectors of current covariance
        eigvals, eigvecs = np.linalg.eigh(self.C)
        
        # Increase variance along principal axes (1.5x for mild, 3x for severe stagnation)
        stagnation_severity = min(self.stagnation_counter / (2.0 * self.max_stagnation), 1.0)
        variance_scale = 1.5 + 1.5 * stagnation_severity
        
        # Reconstruct covariance with scaled eigenvalues
        scaled_eigvals = eigvals * variance_scale
        C_scaled = eigvecs @ np.diag(scaled_eigvals) @ eigvecs.T
        C_scaled = self._ensure_positive_definite(C_scaled)
        
        # Generate new population centered on elite using scaled covariance
        n_elite_preserve = min(5, self.NP // 10)
        n_scaled = self.NP - n_elite_preserve
        
        # Ensure positive definiteness for Cholesky
        min_eig = np.min(np.linalg.eigvalsh(C_scaled))
        if min_eig < 1e-8:
            C_scaled += (1e-7 - min_eig) * np.eye(self.dim)
        
        try:
            L = np.linalg.cholesky(C_scaled)
        except np.linalg.LinAlgError:
            diag_C = np.diag(C_scaled)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))
        
        # Sample around elite
        z = np.random.randn(n_scaled, self.dim)
        samples = elite + self.sigma * (z @ L.T)
        samples = self._clip_to_bounds(samples)
        
        # Add diverse variants of elite (perturb along eigenvectors)
        for i in range(n_elite_preserve):
            # Generate perturbation along principal components
            pert_z = np.random.randn(self.dim)
            pert = self.sigma * 2.0 * (pert_z @ L.T)
            sample = elite + pert
            samples = np.vstack([samples, self._clip_to_bounds(sample)])
        
        # Fill remaining with boundary-focused diversity
        n_boundary = self.NP - len(samples)
        if n_boundary > 0:
            u = np.random.random((n_boundary, self.dim))
            power = 3.0
            boundary = np.where(
                np.random.random((n_boundary, self.dim)) > 0.5,
                np.power(u, power),
                1.0 - np.power(u, power)
            )
            dim_mask = np.random.random((n_boundary, self.dim)) < 0.3
            boundary = np.where(dim_mask, boundary, u)
            for d in range(self.dim):
                boundary[:, d] = self.lb[d] + boundary[:, d] * (self.ub[d] - self.lb[d])
            samples = np.vstack([samples, boundary])
        
        # Ensure exact size
        if samples.shape[0] > self.NP:
            samples = samples[:self.NP]
        elif samples.shape[0] < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
            samples = np.vstack([samples, extra])
        
        # Apply new population and state
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Update covariance with scaled version (preserves structure, increases spread)
        self.C = C_scaled
        self.sigma = np.clip(self.sigma * 1.5, 1e-10, 10.0)
        
        self.fitness = self.func(self.population)
        
        # Restore elite
        self.population[0] = elite
        self.fitness[0] = elite_fit
        
        # Update best
        best_new_idx = np.argmin(self.fitness)
        if self.fitness[best_new_idx] < elite_fit:
            self.f_opt = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
            self.x_opt = self.population[best_new_idx].copy()
        else:
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
        
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
        self.generation = 0
        self.pc = np.zeros(self.dim)
        self._reset_operator_state()
```