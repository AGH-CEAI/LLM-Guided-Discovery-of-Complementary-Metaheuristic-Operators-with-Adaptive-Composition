**Idea: Monte Carlo Bootstrap Velocity Estimation**
Category G: Uses LHS sampling to generate candidate velocity directions and bootstrap resampling to estimate which directions most reliably lead toward personal best improvements, with Sobol quasi-random perturbation for low-discrepancy exploration.

```python
def _velocity_update_base(self):
    """Stochastic velocity update via Monte Carlo bootstrap estimation (Category G).
    
    Key insight: Instead of deterministic cognitive/social coefficients, use
    Latin hypercube sampling to generate candidate velocity directions, then
    bootstrap resampling to estimate which directions reliably lead toward
    personal best improvements. This provides statistically robust exploration
    that adapts to the landscape without gradient information.
    """
    cognitive, social = self._adaptive_coefficients()
    
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
    mutation_active = mutation_mask < mutation_threshold
    
    mutation_vectors = np.zeros((self.np, self.dim))
    for i in range(self.np):
        indices = np.random.choice(
            [j for j in range(self.np) if j != i], 3, replace=False
        )
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    mutation_component = np.where(
        mutation_active,
        0.3 * (mutation_vectors - self.population),
        0.0
    )
    
    # === CATEGORY G: MONTE CARLO BOOTSTRAP VELOCITY ESTIMATION ===
    # Generate candidate velocity directions using Latin hypercube sampling
    n_candidates = min(32, max(8, self.dim))
    n_bootstrap = min(30, self.np)
    
    # Latin hypercube sampling for candidate directions
    lhs_samples = np.zeros((n_candidates, self.dim))
    for d in range(self.dim):
        intervals = np.linspace(0, 1, n_candidates + 1)
        sample_pos = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(sample_pos)
        lhs_samples[:, d] = sample_pos
    
    # Scale to search space
    candidate_directions = lhs_samples * (self.upper_bound - self.lower_bound) + self.lower_bound
    
    # Normalize directions
    dir_norms = np.linalg.norm(candidate_directions, axis=1, keepdims=True) + 1e-10
    candidate_directions = candidate_directions / dir_norms
    
    # Bootstrap estimation: which directions lead toward personal best?
    # For each candidate, compute bootstrap distribution of "improvement score"
    bootstrap_scores = np.zeros((n_bootstrap, n_candidates))
    
    for b in range(n_bootstrap):
        # Bootstrap sample of particles (sampling with replacement)
        boot_idx = np.random.choice(self.np, size=self.np, replace=True)
        boot_pb = self.personal_best[boot_idx]
        boot_pop = self.population[boot_idx]
        
        # Score each candidate direction: positive = moving toward personal best
        for j in range(n_candidates):
            direction = candidate_directions[j]
            # Project onto direction
            proj_pb = np.sum(boot_pb * direction, axis=1)
            proj_pop = np.sum(boot_pop * direction, axis=1)
            # Score = mean improvement potential
            bootstrap_scores[b, j] = np.mean(proj_pb - proj_pop)
    
    # Compute bootstrap statistics
    mean_scores = np.mean(bootstrap_scores, axis=0)
    std_scores = np.std(bootstrap_scores, axis=0) + 1e-10
    
    # Select directions with positive expected improvement
    # Weight by t-statistic-like score: mean / std (higher = more reliable)
    t_scores = mean_scores / std_scores
    
    # Top candidates by bootstrap-estimated improvement
    top_k = min(5, n_candidates)
    top_indices = np.argsort(t_scores)[-top_k:]
    
    # Compute stochastic velocity component from selected directions
    stochastic_component = np.zeros((self.np, self.dim))
    for idx in top_indices:
        direction = candidate_directions[idx]
        # Weight by normalized t-score (positive only)
        weight = max(0.0, t_scores[idx]) / (np.max(np.abs(t_scores)) + 1e-10)
        
        # Per-particle projection onto direction
        proj_pop = np.sum(self.population * direction, axis=1, keepdims=True)
        proj_pb = np.sum(self.personal_best * direction, axis=1, keepdims=True)
        
        # Distance to personal best along this direction
        distance = proj_pb - proj_pop
        
        # Add weighted direction component
        stochastic_component += weight * direction * np.abs(distance) / top_k
    
    # Sobol quasi-random perturbation for low-discrepancy exploration
    sobol_perturbation = np.zeros((self.np, self.dim))
    for d in range(self.dim):
        base = [2, 3, 5, 7, 11, 13, 17][d % 7]
        h = 0.5
        for k in range(min(self.np, 64)):
            h = h / base
            sobol_perturbation[k % self.np, d] += (2 * (k * h - int(k * h)) - 1)
    
    # Scale Sobol perturbation
    sobol_perturbation = sobol_perturbation * 0.1 * (self.upper_bound - self.lower_bound) / self.np
    
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        mutation_component +
        0.3 * stochastic_component +
        sobol_perturbation
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```