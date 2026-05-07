Looking at the priority targets, the worst unsolved tasks (Tasks 17, 16, 10, 20, 18, 23, 12, 19) all have errors in the 10-74 range, indicating severe multimodality with deceptive local optima. The current restart strategy completely reinitializes the population, losing all covariance learning—this causes the algorithm to repeatedly fail to escape local optima basins.

**Idea: Focused Reinitialization with Elite-Directed Perturbation**
Instead of full reinitialization, inject diversity by sampling from an anisotropic distribution centered on the elite, with spread proportional to the current covariance matrix scaled toward the best solution. This preserves learned correlation structure while directing exploration toward promising regions.

```python
def _restart_if_needed(self):
    """Restart population with elite-directed focused reinitialization."""
    diversity = self._compute_diversity()
    
    if (self.stagnation_counter > self.max_stagnation or 
        diversity < self.min_diversity or 
        np.any(np.isnan(self.C))):
        
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]
        
        # Compute condition number to detect ill-conditioned state
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        
        # Determine restart strategy based on condition and stagnation severity
        is_ill_conditioned = cond > 1e6
        is_deeply_stagnant = self.stagnation_counter > 2 * self.max_stagnation
        
        # Compute current step size scale
        sigma_scale = max(self.sigma, 1e-6)
        
        # Strategy 1: Elite-directed restart (default for multimodal)
        # Sample from anisotropic distribution around elite using current covariance
        if not is_ill_conditioned:
            # Use Cholesky of current covariance for correlated sampling
            try:
                L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                diag_C = np.diag(self.C)
                diag_C = np.maximum(diag_C, 1e-10)
                L = np.diag(np.sqrt(diag_C))
            
            # Scale factor: larger for deeply stagnant, smaller for moderate
            if is_deeply_stagnant:
                exploration_radius = 5.0 * sigma_scale
            else:
                exploration_radius = 2.0 * sigma_scale
            
            # Sample new population around elite
            n_new = self.NP - 1
            z = np.random.randn(n_new, self.dim)
            new_pop = elite + exploration_radius * (z @ L.T)
            new_pop = self._clip_to_bounds(new_pop)
            
            # Strategy 2: Hyper-sphere restart (for ill-conditioned CMA state)
        else:
            # Reset covariance to spherical for numerical stability
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
            
            # Sample uniformly in hyper-sphere around elite
            n_new = self.NP - 1
            exploration_radius = max(10.0 * sigma_scale, 0.1 * (self.ub[0] - self.lb[0]))
            u = np.random.randn(n_new, self.dim)
            norms = np.linalg.norm(u, axis=1, keepdims=True)
            # Random radii for uniform hyper-sphere sampling
            radii = np.random.random((n_new, 1)) ** (1.0 / self.dim)
            u = u / (norms + 1e-10) * radii * exploration_radius
            new_pop = elite + u
            new_pop = self._clip_to_bounds(new_pop)
        
        # Reinitialize rest of population with boundary-focused sampling
        n_remain = self.NP - 1 - len(new_pop)
        if n_remain > 0:
            # Use boundary-focused sampling for multimodal landscapes
            boundary_ratio = 0.4 if is_deeply_stagnant else 0.25
            n_boundary = int(boundary_ratio * n_remain)
            n_random = n_remain - n_boundary
            
            remainder = []
            if n_random > 0:
                random_samples = np.random.uniform(self.lb, self.ub, (n_random, self.dim))
                remainder.append(random_samples)
            
            if n_boundary > 0:
                u = np.random.random((n_boundary, self.dim))
                power = 3.0
                scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
                dim_mask = np.random.random(self.dim) < 0.3
                boundary = np.zeros((n_boundary, self.dim))
                boundary[:, dim_mask] = scaled[:, dim_mask]
                boundary[:, ~dim_mask] = u[:, ~dim_mask]
                for d in range(self.dim):
                    boundary[:, d] = self.lb[d] + boundary[:, d] * (self.ub[d] - self.lb[d])
                remainder.append(boundary)
            
            if remainder:
                new_pop = np.vstack([new_pop] + remainder)
        
        # Ensure exact population size
        if len(new_pop) < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - len(new_pop), self.dim))
            new_pop = np.vstack([new_pop, extra])
        elif len(new_pop) > self.NP:
            new_pop = new_pop[:self.NP]
        
        self.population = new_pop
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Keep current covariance structure (don't reset C for non-ill-conditioned cases)
        if not is_ill_conditioned:
            # Dampen covariance slightly to allow re-exploration
            self.C = 0.9 * self.C + 0.1 * np.eye(self.dim) * np.trace(self.C) / self.dim
        # For ill-conditioned case, C was already reset to spherical above
        
        self.sigma = np.clip(self.sigma * 1.5, 1e-6, 10.0)
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        
        # Insert elite at best position
        best_new_idx = np.argmin(self.fitness)
        if self.fitness[best_new_idx] > elite_fit:
            # Replace worst individual with elite
            worst_idx = np.argmax(self.fitness)
            self.population[worst_idx] = elite
            self.fitness[worst_idx] = elite_fit
        
        # Update best
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
```