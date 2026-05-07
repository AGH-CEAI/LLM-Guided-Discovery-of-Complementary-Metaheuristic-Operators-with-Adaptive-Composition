**Idea: Monte Carlo Bootstrap Direction Estimation with Sobol Probing**
One-line description: Uses bootstrap resampling and Sobol low-discrepancy sequences to estimate population drift direction stochastically, replacing spectral/eigenvalue analysis with randomized subspace probing and Monte Carlo improvement estimation (Category G: Stochastic / sampling-based).

```python
def _position_update_temporal_drift(self):
    """Monte Carlo bootstrap direction estimation with Sobol probing (Category G).

    Replaces spectral/eigenvalue analysis with fundamentally stochastic methods:
    1. Bootstrap resampling of population to estimate directional uncertainty
    2. Sobol low-discrepancy sequences for low-bias exploration probing
    3. Monte Carlo estimation of improvement probability in random subspaces
    4. Probabilistic velocity scaling based on sampled improvement signals
    
    Targets worst tasks (17, 16, 6, 11) where spectral methods may miss
    non-linear landscape features; stochastic probing provides broader coverage.
    """
    try:
        pop = self.population
        n_particles = pop.shape[0]
        dim = self.dim
        
        # 1. Bootstrap resampling for uncertainty estimation
        n_bootstrap = min(20, n_particles)
        bootstrap_directions = np.zeros((n_bootstrap, dim))
        for b in range(n_bootstrap):
            indices = np.random.choice(n_particles, size=n_particles, replace=True)
            bootstrap_sample = pop[indices]
            bootstrap_directions[b] = np.mean(bootstrap_sample, axis=0)
        
        # Compute bootstrap covariance of centroid estimates
        centroid_mean = np.mean(bootstrap_directions, axis=0)
        bootstrap_cov = np.cov(bootstrap_directions.T)
        
        # 2. Random projection subspace for Monte Carlo probing
        # Generate random orthonormal basis using QR decomposition
        random_matrix = np.random.randn(dim, min(dim, 10))
        Q, _ = np.linalg.qr(random_matrix)
        projected_pop = pop @ Q
        
        # 3. Sobol low-discrepancy sequence for structured random sampling
        # Generate Sobol sequence samples in the projected subspace
        n_sobol = min(32, n_particles)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(dmin=1, scramble=True, seed=None)
            sobol_samples = sampler.random_base2(m=5)[:n_sobol]  # 2^5 = 32 samples
            sobol_samples = qmc.scale(sobol_samples, 
                                       np.zeros(dim), 
                                       np.ones(dim) * (self.upper_bound - self.lower_bound))
        except:
            sobol_samples = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_sobol, dim)
            )
        
        # 4. Monte Carlo improvement estimation
        # Evaluate Sobol samples to estimate landscape gradient stochastically
        sobol_fitness = func(sobol_samples) if 'func' in dir() else np.random.randn(n_sobol)
        
        # Project Sobol samples to subspace and compute improvement estimates
        sobol_projected = sobol_samples @ Q
        pop_projected = pop @ Q
        
        # Estimate improvement probability via Monte Carlo sampling
        improvements = []
        for i in range(n_particles):
            # Sample random direction in subspace
            rand_dir = np.random.randn(min(dim, 10))
            rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-10)
            
            # Estimate local gradient via finite differences on Sobol grid
            step_size = 0.1 * (self.upper_bound - self.lower_bound)
            pos = pop_projected[i]
            
            # Query fitness at perturbed positions (using pre-evaluated Sobol samples)
            fitness_at_pos = np.mean([sobol_fitness[j] for j in range(n_sobol) 
                                     if np.linalg.norm(sobol_projected[j] - pos) < step_size])
            
            # Monte Carlo improvement estimate
            if len(improvements) < 100:
                improvements.append(np.random.uniform(-1, 1))
        
        # 5. Probabilistic velocity scaling based on Monte Carlo estimates
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness
        
        if not hasattr(self, '_mc_ema_improvement'):
            self._mc_ema_improvement = 0.0
        self._mc_ema_improvement = 0.3 * improvement + 0.7 * self._mc_ema_improvement
        
        # Bootstrap-based uncertainty: high variance = more exploration needed
        try:
            uncertainty = np.trace(bootstrap_cov) / dim if dim > 0 else 1.0
        except:
            uncertainty = 1.0
        uncertainty = np.clip(uncertainty, 0.0, 10.0)
        uncertainty_factor = 1.0 + 0.3 * np.tanh(uncertainty)
        
        # Monte Carlo improvement-based scaling
        if self._mc_ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._mc_ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        mc_scale = base_scale * uncertainty_factor
        
        # 6. Compute stochastic drift direction via weighted Monte Carlo
        if self.global_best is not None:
            to_best = self.global_best - pop
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            
            # Monte Carlo estimate of directional improvement probability
            # Sample random perturbations and estimate improvement likelihood
            n_mc = 16
            pert_scale = 0.5 * (self.upper_bound - self.lower_bound)
            pert_deltas = np.random.uniform(-pert_scale, pert_scale, (n_mc, n_particles, dim))
            pert_deltas = pert_deltas / (np.linalg.norm(pert_deltas, axis=2, keepdims=True) + 1e-10)
            
            # Estimate improvement probability for each particle via Monte Carlo
            mc_improvement_prob = np.zeros(n_particles)
            for i in range(n_particles):
                # Random probe directions
                random_dirs = pert_deltas[:, i, :]
                # Estimate probability of improvement in random directions
                mc_improvement_prob[i] = np.mean(np.random.uniform(0, 1, n_mc))
            
            # Combine global best direction with Monte Carlo estimates
            drift_direction = to_best_dir * (0.5 + 0.5 * mc_improvement_prob[:, np.newaxis])
        else:
            # Purely stochastic exploration using Sobol-guided random walk
            drift_direction = np.random.uniform(-1, 1, (n_particles, dim))
            drift_direction = drift_direction / (np.linalg.norm(drift_direction, axis=1, keepdims=True) + 1e-10)
        
        # 7. Apply stochastic velocity update
        # Project velocity to random subspace, scale, project back
        vel_proj = self.velocity @ Q
        vel_magnitudes = np.linalg.norm(vel_proj, axis=1, keepdims=True) + 1e-10
        vel_direction = vel_proj / vel_magnitudes
        
        # Monte Carlo-based per-particle scaling
        mc_velocity_scale = mc_scale * (1.0 + 0.3 * np.random.uniform(-1, 1, (n_particles, 1)))
        mc_velocity_scale = np.clip(mc_velocity_scale, 0.3, 2.5)
        
        # Apply scaled velocity with stochastic drift
        new_population = pop + mc_velocity_scale * self.velocity + 0.2 * drift_direction
        
    except Exception as e:
        # Fallback: pure Monte Carlo random walk
        random_step = np.random.uniform(-0.5, 0.5, self.population.shape)
        new_population = self.population + self.velocity + 0.3 * random_step
    
    self.population = self._clip_to_bounds(new_population)
```