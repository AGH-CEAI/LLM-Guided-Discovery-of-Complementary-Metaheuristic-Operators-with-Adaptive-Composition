Looking at the unsolved tasks, the worst ones (Tasks 16, 17, 18, 20, 23, etc.) have errors in the 10-100 range, suggesting they're highly multi-modal or ill-conditioned functions where the optimizer is getting stuck in local optima. The current variant_09 uses diversity and stagnation detection, but the errors remain very large.

**Analysis of failure mode**: Errors stuck at ~1e+1 to ~1e+02 indicate the optimizer is converging to local optima without escaping. The current approach scales learning rates but doesn't address the fundamental issue of premature convergence in deceptive landscapes.

**Idea: Fitness Landscape Curvature-Adaptive Covariance with Active Direction Boosting**
This approach monitors the eigenvalue spread of the covariance matrix (condition number) and the fitness gradient along eigendirections. When the condition number is extreme OR when the algorithm is stuck, it boosts exploration along the smallest eigendirections (which are being neglected) and adds a momentum term that remembers successful search directions across restarts.

```python
def _adapt_covariance_variant_09(self):
    """Fitness landscape curvature-adaptive covariance with active direction boosting."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Eigenvalue analysis of covariance matrix
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    cond = np.max(eigvals) / np.min(eigvals)
    
    # Fitness gradient along eigendirections
    fitness_along_axes = []
    for i in range(min(self.dim, 10)):
        v = eigvecs[:, i]
        test_points = self.mean + self.sigma * v * np.array([-1, 1])
        fitness_along_axes.append(np.mean([self.func(p.reshape(1, -1)) for p in test_points]))
    avg_fitness_axes = np.mean(fitness_along_axes)
    
    # Stagnation detection
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.improvement_ema = 1.0
        self.global_improvement_ema = 1.0
    
    improvement = max(1e-10, self.prev_f_opt - self.f_opt)
    self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
    self.global_improvement_ema = 0.99 * self.global_improvement_ema + 0.01 * improvement
    self.prev_f_opt = self.f_opt
    
    local_stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
    global_stagnation = self.global_improvement_ema < 1e-12 * max(1.0, abs(self.f_opt))
    stuck_in_local_opt = local_stagnation and (avg_fitness_axes < self.f_opt * 1.5)
    
    # Build momentum from successful directions
    if not hasattr(self, 'successful_dirs'):
        self.successful_dirs = []
    if improvement > 1e-10:
        self.successful_dirs.append(y_mean.copy())
    if len(self.successful_dirs) > 50:
        self.successful_dirs.pop(0)
    
    momentum = np.zeros(self.dim)
    if len(self.successful_dirs) >= 5:
        for d in self.successful_dirs[-20:]:
            momentum += d
        momentum /= len(self.successful_dirs[-20:])
    
    rank_one = np.outer(self.pc, self.pc)
    rank_momentum = np.outer(momentum, momentum)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rates based on condition and stagnation
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    
    if stuck_in_local_opt or global_stagnation:
        ccov_adaptive = base_ccov * 3.0
        ccov_momentum = 0.3
    elif cond > 1e6:
        ccov_adaptive = base_ccov * 2.0
        ccov_momentum = 0.1
    else:
        ccov_adaptive = base_ccov
        ccov_momentum = 0.05
    
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
    
    # Active direction boosting: add exploration along smallest eigendirections
    boosting = 0.0
    if cond > 1e4 or stuck_in_local_opt:
        boosting = 0.1 * min(1.0, self.improvement_ema / max(1e-10, abs(self.f_opt)))
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              ccov_momentum * rank_momentum +
              (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
    # Boost smallest eigendirections
    if boosting > 0:
        self.C += boosting * np.outer(eigvecs[:, 0], eigvecs[:, 0])
    
    self.C = self._ensure_positive_definite(self.C)
```