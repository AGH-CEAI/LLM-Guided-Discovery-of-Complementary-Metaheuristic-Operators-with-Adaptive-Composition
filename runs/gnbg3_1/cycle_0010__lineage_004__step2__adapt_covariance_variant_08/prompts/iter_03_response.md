**Idea: Stagnation-Triggered Archive Exploration**

A fundamentally different approach: detect when the mean is stagnating (small mean displacement across generations), then inject archive-guided directional perturbations into the covariance matrix. This directly attacks the failure mode where the evolution path gets trapped in deceptive local optima — something the dual-path approach of the current variant_08 fails to do. The key difference is *explicit stagnation detection* that triggers a discrete strategy change, rather than relying on passive history weighting.

```python
def _adapt_covariance_variant_08(self):
    """Stagnation-triggered archive exploration with adaptive learning rates."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Exponential decay learning rates (different from current fixed alphas)
    if not hasattr(self, 'gen_count'):
        self.gen_count = 0
    self.gen_count += 1
    cc_adapt = min(0.3, 2.0 / (self.dim + 5.0))
    ccov_adapt = min(0.3, 1.5 / (self.dim + 3.0))
    
    # Track mean displacement for stagnation detection
    if not hasattr(self, 'prev_mean_displacement'):
        self.prev_mean_displacement = np.zeros(self.dim)
        self.stagnation_gens = 0
    mean_displacement = self.mean - self.old_mean
    displacement_norm = np.linalg.norm(mean_displacement)
    prev_norm = np.linalg.norm(self.prev_mean_displacement)
    
    # Stagnation detection: mean barely moving in consistent direction
    if displacement_norm < 1e-8 * self.sigma and prev_norm < 1e-8 * self.sigma:
        self.stagnation_gens += 1
    else:
        self.stagnation_gens = 0
    self.prev_mean_displacement = mean_displacement.copy()
    
    # Evolution path update
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Initialize archive
    if not hasattr(self, 'archive_positions'):
        self.archive_positions = []
        self.archive_ages = []
    
    # Add current best to archive if distinct
    min_dist = 1e-3 * (self.ub[0] - self.lb[0])
    is_distinct = True
    for arch_pos in self.archive_positions:
        if np.linalg.norm(self.x_opt - arch_pos) < min_dist:
            is_distinct = False
            break
    
    if is_distinct:
        self.archive_positions.append(self.x_opt.copy())
        self.archive_ages.append(0)
        if len(self.archive_positions) > 10:
            oldest_idx = np.argmax(self.archive_ages)
            self.archive_positions.pop(oldest_idx)
            self.archive_ages.pop(oldest_idx)
    
    for i in range(len(self.archive_ages)):
        self.archive_ages[i] += 1
    
    # Stagnation-triggered archive exploration
    archive_perturb = np.zeros((self.dim, self.dim))
    if self.stagnation_gens > 3 and len(self.archive_positions) >= 2:
        centroid = np.mean(self.archive_positions, axis=0)
        toward_centroid = centroid - self.mean
        norm_toward = np.linalg.norm(toward_centroid)
        if norm_toward > 1e-10:
            toward_unit = toward_centroid / norm_toward
            archive_perturb = 0.5 * np.outer(toward_unit, toward_unit)
        # Also add axis-aligned exploration
        archive_perturb += 0.1 * np.eye(self.dim)
    
    # Rank-one update from primary evolution path
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine updates
    self.C = ((1.0 - ccov_adapt) * self.C +
              ccov_adapt * rank_one +
              (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
    
    # Inject archive perturbation on stagnation
    if self.stagnation_gens > 3:
        self.C = self.C + archive_perturb
        self.stagnation_gens = 0
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Condition number control
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)
    if cond > 1e7:
        self.C *= 0.5
```