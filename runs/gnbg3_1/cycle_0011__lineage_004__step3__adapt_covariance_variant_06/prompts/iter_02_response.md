Looking at the priority tasks (16, 17, 11, 23, 18, 20, 19, 12) with errors 10-60, these represent severe optimization failures. The common thread is that all variants use a single evolution path direction (`pc`), which can become misleading when trapped in deceptive basins. I need a fundamentally different approach that doesn't rely on a single potentially-stuck direction.

**Idea: Multi-Direction Ensemble Covariance**

Instead of trusting a single `pc` direction, compute multiple candidate covariance updates from diverse population members, evaluate which directions actually predict improvement, and blend them proportionally. This creates a more robust covariance estimate that can't be dominated by a single misleading direction.

```python
def _adapt_covariance_variant_06(self):
    """Multi-direction ensemble covariance: blend diverse search directions by predicted quality."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute candidate updates from top-k diverse population members
    k_directions = min(self.mu, max(3, self.dim // 2))
    candidate_dirs = []
    candidate_scores = []
    
    for i in range(k_directions):
        diff = (self.population[i] - self.old_mean) / self.sigma
        dir_norm = np.linalg.norm(diff)
        if dir_norm > 1e-10:
            # Score = alignment with fitness improvement potential
            # Higher score = more promising direction
            pop_dist = np.linalg.norm(self.population[i] - self.mean)
            fitness_rank = (self.mu - i) / max(self.mu, 1)
            score = fitness_rank * np.log1p(pop_dist / max(dir_norm, 1e-10))
            candidate_dirs.append(diff)
            candidate_scores.append(score)
    
    if len(candidate_dirs) >= 2:
        # Normalize scores to weights
        max_score = max(abs(s) for s in candidate_scores)
        if max_score > 1e-10:
            weights_dir = [s / max_score for s in candidate_scores]
        else:
            weights_dir = [1.0 / len(candidate_dirs)] * len(candidate_dirs)
        
        # Blend weighted ensemble direction for pc
        ensemble_pc = np.zeros(self.dim)
        for d, w in zip(candidate_dirs, weights_dir):
            ensemble_pc += w * d
        ensemble_pc /= sum(weights_dir)
        
        # Blend rank-one matrices from each candidate direction
        ensemble_rank_one = np.zeros((self.dim, self.dim))
        for d, w in zip(candidate_dirs, weights_dir):
            d_normalized = d / max(np.linalg.norm(d), 1e-10)
            ensemble_rank_one += w * np.outer(d_normalized, d_normalized)
        ensemble_rank_one /= sum(weights_dir)
    else:
        # Fallback to standard pc update
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        ensemble_rank_one = np.outer(self.pc, self.pc)
    
    # Compute rank-mu from diverse population sampling
    rank_mu = np.zeros((self.dim, self.dim))
    total_w = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_w += abs(w)
    
    if total_w > 0:
        rank_mu /= total_w
    
    # Adaptive learning rates based on condition and diversity
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)
    
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Detect problematic states
    is_ill_cond = cond > 1e5
    is_low_var = rel_var < 1e-3
    is_narrow = eig_spread < 1e-5
    
    if is_ill_cond or is_low_var or is_narrow:
        # Aggressive exploration mode
        ccov_eff = min(self.ccov * 5.0, 0.5)
        cc_eff = min(self.cc * 2.0, 0.3)
        # Inject diagonal exploration
        rank_mu += 0.2 * np.eye(self.dim)
        ensemble_rank_one += 0.1 * np.eye(self.dim)
    else:
        ccov_eff = self.ccov
        cc_eff = self.cc
    
    # Final covariance update with ensemble direction
    self.C = ((1.0 - ccov_eff) * self.C + 
              ccov_eff * ensemble_rank_one +
              (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu)
    
    # Emergency repair for extreme ill-conditioning
    if cond > 1e8:
        self.C = 0.5 * self.C + 0.5 * np.eye(self.dim) * np.mean(eigvals)
        self.pc *= 0.1
    
    self.C = self._ensure_positive_definite(self.C)
```