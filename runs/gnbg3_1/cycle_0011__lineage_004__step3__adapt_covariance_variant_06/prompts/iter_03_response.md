**Idea: Proactive Stagnation-Guided Escaping**

A fundamentally different approach: instead of reactively detecting convergence (low variance + tight eigenspectrum), this variant proactively monitors stagnation counter and condition number to detect when the algorithm is *failing to improve* — the true symptom of deceptive local optima. It then injects directional perturbations toward historically unexplored regions, with perturbation strength that scales with stagnation severity.

```python
def _adapt_covariance_variant_06(self):
    """Proactive stagnation-guided escaping with directional perturbations."""
    stagnation_threshold = max(40, self.dim * 3)
    is_stagnant = self.stagnation_counter > stagnation_threshold
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    
    if not hasattr(self, 'best_history'):
        self.best_history = []
    
    min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
    is_distinct = True
    for hist_pos in self.best_history:
        if np.linalg.norm(self.x_opt - hist_pos) < min_dist_threshold:
            is_distinct = False
            break
    
    if is_distinct and len(self.best_history) < 15:
        self.best_history.append(self.x_opt.copy())
    elif is_distinct and len(self.best_history) >= 15:
        worst_idx = np.argmax([np.linalg.norm(self.x_opt - p) for p in self.best_history])
        self.best_history[worst_idx] = self.x_opt.copy()
    
    if is_stagnant or cond > 1e6:
        escape_dir = np.zeros(self.dim)
        if len(self.best_history) >= 3:
            centroid = np.mean(self.best_history, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                escape_dir = toward_centroid
            else:
                escape_dir = self.mean - centroid
                norm_dir = np.linalg.norm(escape_dir)
                if norm_dir > 1e-10:
                    escape_dir /= norm_dir
        else:
            escape_dir = np.random.randn(self.dim)
            norm_dir = np.linalg.norm(escape_dir)
            if norm_dir > 1e-10:
                escape_dir /= norm_dir
        
        escape_strength = 0.5 * (1.0 + self.stagnation_counter / max(stagnation_threshold, 1))
        escape_perturb = escape_strength * np.outer(escape_dir, escape_dir)
        self.C += escape_perturb
        self.C += 0.3 * np.eye(self.dim)
        self.C = self._ensure_positive_definite(self.C)
        self.pc *= 0.1
        self.stagnation_counter = 0
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    if is_stagnant:
        cc_scale = 1.5
        ccov_scale = 2.0
    else:
        cc_scale = 1.0
        ccov_scale = min(1.0 + 3.0 * rel_var, 3.0)
    
    cc_adapt = np.clip(self.cc * cc_scale, 0.001, 0.5)
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.6)
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```