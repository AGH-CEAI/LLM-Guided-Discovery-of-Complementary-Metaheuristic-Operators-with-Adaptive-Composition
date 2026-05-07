Looking at the worst unsolved tasks (errors ~10-74), they represent extreme optimization challenges. The current restart logic is too simplistic—it blindly reinitializes without analyzing WHY the optimization failed or adapting the restart strategy to the failure mode.

**Idea: Failure-Mode-Aware Adaptive Restart**
Instead of a one-size-fits-all restart, this approach:
1. Classifies the failure mode (stagnation vs. diversity collapse vs. ill-conditioning)
2. Adapts restart population sampling to specifically counter that failure mode
3. Uses anisotropic scaling of restart samples along eigenvectors for ill-conditioned failures
4. Uses boundary-focused sampling for deceptive local optima failures
5. Uses covariance-guided "tunneling" samples to escape trapped regions

```python
def _restart_if_needed(self):
    """Adaptive restart with failure-mode diagnosis and targeted resampling."""
    # Diagnose failure mode
    stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
    diversity = self._compute_diversity()
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min, eig_max = np.min(eigvals), np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-15)
    
    pop_diffs = self.population - self.mean
    max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
    spread_ratio = np.sqrt(max_sq_dist) / max(1.0, np.sqrt(self.dim) * self.sigma)
    
    # Classify primary failure mode
    is_ill_conditioned = cond > 1e6 or (eig_min / (eig_max + 1e-15)) < 1e-8
    is_collapsed = spread_ratio < 0.1 or diversity < 0.01
    is_stagnant = stagnation_ratio > 0.7
    
    # Save elite
    best_idx = np.argmin(self.fitness)
    elite = self.population[best_idx].copy()
    elite_fit = self.fitness[best_idx]
    
    # Determine restart strategy based on failure mode
    if is_ill_conditioned:
        # For ill-conditioned: restart along eigenvectors with anisotropic scaling
        eigvecs = np.linalg.eigh(self.C)[1]
        base_samples = []
        
        # 40% of population: anisotropic restart (stretched along eigenvectors)
        n_aniso = int(0.4 * self.NP)
        for _ in range(n_aniso):
            z = np.random.randn(self.dim)
            # Scale each direction by inverse condition weight (favor exploration along weak directions)
            weights = 1.0 / np.sqrt(eigvals + 1e-10)
            weights /= np.max(weights)
            sample = self.mean + self.sigma * 2.0 * (eigvecs @ (weights * z))
            base_samples.append(sample)
        
        # 40% of population: random restart (explore new regions)
        n_random = int(0.4 * self.NP)
        for _ in range(n_random):
            sample = np.random.uniform(self.lb, self.ub, self.dim)
            base_samples.append(sample)
        
        # 20% of population: elite-focused (refine best)
        n_elite = self.NP - n_aniso - n_random
        for _ in range(n_elite):
            sample = elite + np.random.randn(self.dim) * self.sigma * 0.5
            base_samples.append(sample)
            
    elif is_collapsed:
        # For collapsed: restart with aggressive boundary exploration
        base_samples = []
        
        # 50% of population: boundary-focused (power-law scaling)
        n_boundary = int(0.5 * self.NP)
        for _ in range(n_boundary):
            u = np.random.random(self.dim)
            power = 2.0 + np.random.random() * 3.0
            scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
            sample = self.lb + scaled * (self.ub - self.lb)
            base_samples.append(sample)
        
        # 30% of population: wide random (explore far regions)
        n_wide = int(0.3 * self.NP)
        for _ in range(n_wide):
            sample = np.random.uniform(self.lb * 0.8, self.ub * 0.8, self.dim)
            base_samples.append(sample)
        
        # 20% of population: elite-focused
        n_elite = self.NP - n_boundary - n_wide
        for _ in range(n_elite):
            sample = elite + np.random.randn(self.dim) * self.sigma * 1.5
            base_samples.append(sample)
            
    elif is_stagnant:
        # For stagnant (deceptive local optima): restart with tunneling strategy
        base_samples = []
        
        # 30% of population: directional restart toward unexplored regions
        n_directional = int(0.3 * self.NP)
        if hasattr(self, 'archive_positions') and len(self.archive_positions) >= 2:
            centroid = np.mean(self.archive_positions, axis=0)
            direction = self.mean - centroid
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction /= norm_dir
            for _ in range(n_directional):
                # Jump in opposite direction from centroid
                jump = np.random.uniform(0.5, 3.0) * (self.ub[0] - self.lb[0])
                sample = self.mean - direction * jump + np.random.randn(self.dim) * self.sigma
                base_samples.append(sample)
        else:
            for _ in range(n_directional):
                sample = np.random.uniform(self.lb, self.ub, self.dim)
                base_samples.append(sample)
        
        # 40% of population: quasi-random (Sobol/LHS for better coverage)
        n_quasi = int(0.4 * self.NP)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            quasi_samples = sampler.random(n_quasi)
            quasi_samples = qmc.scale(quasi_samples, self.lb, self.ub)
            base_samples.extend(quasi_samples.tolist())
        except Exception:
            for _ in range(n_quasi):
                base_samples.append(np.random.uniform(self.lb, self.ub, self.dim))
        
        # 30% of population: elite perturbation
        n_elite = self.NP - n_directional - n_quasi
        for _ in range(n_elite):
            sample = elite + np.random.randn(self.dim) * self.sigma * 2.0
            base_samples.append(sample)
    else:
        # Default: balanced restart (similar to original but with diversity boost)
        base_samples = []
        
        # 50% Sobol/LHS
        n_quasi = int(0.5 * self.NP)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            quasi_samples = sampler.random(n_quasi)
            quasi_samples = qmc.scale(quasi_samples, self.lb, self.ub)
            base_samples.extend(quasi_samples.tolist())
        except Exception:
            for _ in range(n_quasi):
                base_samples.append(np.random.uniform(self.lb, self.ub, self.dim))
        
        # 30% random
        n_random = int(0.3 * self.NP)
        for _ in range(n_random):
            base_samples.append(np.random.uniform(self.lb, self.ub, self.dim))
        
        # 20% elite
        n_elite = self.NP - n_quasi - n_random
        for _ in range(n_elite):
            sample = elite + np.random.randn(self.dim) * self.sigma
            base_samples.append(sample)
    
    # Stack and clip samples
    base_samples = np.vstack(base_samples)
    base_samples = self._clip_to_bounds(base_samples)
    
    # Reset population
    self.population = base_samples[:self.NP]
    
    # Reset CMA-ES state
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    # Reset covariance: preserve structure for ill-conditioned, reset for others
    if is_ill_conditioned:
        # Keep scaled identity (neutral restart)
        self.C = np.eye(self.dim) * np.mean(np.var(self.population, axis=0))
    else:
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    self.L = None
    
    # Evaluate and update
    self.fitness = self.func(self.population)
    
    best_idx = np.argmin(self.fitness)
    self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
    self.x_opt = self.population[best_idx].copy()
    self.f_opt_prev = self.f_opt
    
    # Reinsert saved elite if better
    elite_fit_val = float(np.asarray(elite_fit).flatten()[0])
    if elite_fit_val < self.f_opt:
        self.f_opt = elite_fit_val
        self.x_opt = elite.copy()
        self.population[0] = elite
        self.fitness[0] = elite_fit
    
    self.stagnation_counter = 0
    self.generation = 0
    self._reset_operator_state()
```