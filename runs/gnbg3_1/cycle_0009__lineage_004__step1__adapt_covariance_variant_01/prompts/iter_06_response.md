Looking at the data, the worst tasks (16-23 with errors 10-80) are likely **deceptive multimodal landscapes** where CMA-ES-style covariance adaptation gets trapped in local optima. The current variant_01 scales learning rates based on condition number but doesn't fundamentally escape local basins.

**Key insight:** When trapped, we need to **actively perturb the covariance matrix** away from the current search direction using an archive of past best solutions to generate diverse escape directions.

**Idea: Archive-Guided Covariance Perturbation**
Maintain an archive of distinct best solutions; when stagnation is detected (low fitness variance, high condition number, or eigenspread collapse), inject a perturbation to the covariance matrix along the direction away from the archive centroid toward diverse archive members. This forces exploration in orthogonal directions.

```python
def _adapt_covariance_variant_01(self):
    """Archive-guided covariance perturbation for escaping deceptive local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma

    # Initialize archive for tracking distinct best solutions
    if not hasattr(self, 'archive_positions'):
        self.archive_positions = []
        self.archive_fitness = []
        self.archive_generations = []

    # Add current best to archive if distinct enough
    min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
    is_distinct = True
    for arch_pos in self.archive_positions:
        dist = np.linalg.norm(self.x_opt - arch_pos)
        if dist < min_dist_threshold:
            is_distinct = False
            break

    if is_distinct and len(self.archive_positions) < 10:
        self.archive_positions.append(self.x_opt.copy())
        self.archive_fitness.append(self.f_opt)
        self.archive_generations.append(self.generation)
    elif is_distinct and len(self.archive_positions) >= 10:
        # Replace worst entry
        worst_idx = np.argmax(self.archive_fitness)
        self.archive_positions[worst_idx] = self.x_opt.copy()
        self.archive_fitness[worst_idx] = self.f_opt
        self.archive_generations[worst_idx] = self.generation

    # Compute diversity and condition metrics
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)

    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    eig_spread = eig_min / (eig_max + 1e-10)
    cond = eig_max / (max(eig_min, 1e-10))

    # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
    is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

    # Compute archive-based escape direction
    escape_perturb = np.zeros((self.dim, self.dim))
    if is_trapped and len(self.archive_positions) >= 3:
        # Direction from mean toward archive centroid, weighted by diversity
        centroid = np.mean(self.archive_positions, axis=0)
        toward_centroid = centroid - self.mean
        norm_toward = np.linalg.norm(toward_centroid)
        if norm_toward > 1e-10:
            toward_centroid /= norm_toward
            # Also consider directions to individual archive members
            for arch_pos in self.archive_positions:
                dir_to_arch = arch_pos - self.mean
                norm_dir = np.linalg.norm(dir_to_arch)
                if norm_dir > 1e-10:
                    dir_to_arch /= norm_dir
                    # Perturb along diverse directions
                    escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
        # Add global exploration along principal axes
        escape_perturb += 0.1 * np.eye(self.dim)

    # Adaptive learning rates with exploration boost when trapped
    if is_trapped:
        ccov_scale = 0.8
        cc_scale = 0.8
    else:
        ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
        cc_scale = 1.0

    ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
    ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

    # Evolution path update
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)

    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)

    # Combine updates with archive-based perturbation
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu)

    # Inject escape perturbation when trapped
    if is_trapped and len(self.archive_positions) >= 3:
        self.C = self.C + escape_perturb

    self.C = self._ensure_positive_definite(self.C)

    # Full restart on extreme ill-conditioning
    if cond > 1e8 or eig_spread < 1e-8:
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.L = None
```