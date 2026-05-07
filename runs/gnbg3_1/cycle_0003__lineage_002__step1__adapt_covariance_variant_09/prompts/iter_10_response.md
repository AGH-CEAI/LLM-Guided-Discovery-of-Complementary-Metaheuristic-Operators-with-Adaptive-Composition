**Idea: Elite Archive with Adaptive Conditioning**
Archive-driven exploration that maintains diverse elite solutions across generations and uses eigenvalue conditioning to prevent covariance collapse, targeting the hardest multimodal/ill-conditioned tasks (16-23) where current variants fail catastrophically.
```python
def _adapt_covariance_variant_09(self):
    """Elite archive exploration with adaptive conditioning control."""
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.05, 1.0)
    
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.improvement_ema = 1.0
    
    improvement = max(1e-10, self.prev_f_opt - self.f_opt)
    self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
    self.prev_f_opt = self.f_opt
    
    stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
    
    # Maintain archive of diverse elite solutions
    if not hasattr(self, 'elite_archive'):
        self.elite_archive = []
    
    best_idx = np.argmin(self.fitness)
    best_point = self.population[best_idx].copy()
    best_fit = self.fitness[best_idx]
    
    if len(self.elite_archive) == 0 or best_fit < self.elite_archive[-1][1]:
        self.elite_archive.append((best_point, best_fit))
        if len(self.elite_archive) > self.NP // 2:
            self.elite_archive.pop(0)
    
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    
    # Adaptive learning rate: conservative when improving, aggressive when stuck
    if stagnation:
        ccov_adaptive = base_ccov * 2.0
    elif self.improvement_ema > 1e-6 * max(1.0, abs(self.f_opt)):
        ccov_adaptive = base_ccov * 0.3
    else:
        ccov_adaptive = base_ccov
    
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Compute archive covariance for diversity injection
    C_archive = np.zeros((self.dim, self.dim))
    if len(self.elite_archive) > 3:
        archive_points = np.array([p for p, _ in self.elite_archive])
        if len(archive_points) > 1:
            C_archive = np.cov(archive_points.T) + 1e-8 * np.eye(self.dim)
    
    archive_weight = 0.15 if len(self.elite_archive) > 5 else 0.0
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              ccov_adaptive * (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu +
              archive_weight * C_archive * ccov_adaptive)
    
    # Eigenvalue conditioning control
    eigvals = np.linalg.eigvalsh(self.C)
    if len(eigvals) > 0 and eigvals[0] > 0:
        cond = eigvals[-1] / eigvals[0]
        if cond > 1e6:
            self.C *= 0.5
    
    self.C = self._ensure_positive_definite(self.C)
```