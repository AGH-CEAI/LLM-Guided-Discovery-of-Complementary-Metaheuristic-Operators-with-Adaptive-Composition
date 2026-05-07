Looking at the priority unsolved tasks (errors 10-70), the key failure mode appears to be **premature covariance matrix collapse** — the algorithm converges to a narrow region and loses the ability to explore. The original, archive-guided, and active CMA-ES variants all use fixed or mildly adaptive learning rates that don't sufficiently respond to stagnation.

My strategy: **Stagnation-triggered covariance expansion with condition-based damping** — a fundamentally different approach that:
1. Monitors recent fitness improvement to detect stagnation
2. Computes the condition number of C to detect covariance collapse
3. Aggressively boosts learning rates AND injects diagonal variance proportional to σ when stuck
4. Uses condition-based damping to prevent runaway elongation

This is distinct from archive-guided (adds archive perturbations to C) and active CMA-ES (changes negative weight handling) — this directly expands C when stagnation is detected, forcing re-exploration.

```python
def _adapt_covariance_original(self):
    """Stagnation-triggered covariance expansion with condition-based damping."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Monitor recent fitness improvement to detect stagnation
    if not hasattr(self, 'recent_improvements'):
        self.recent_improvements = []
    self.recent_improvements.append(self.f_opt)
    if len(self.recent_improvements) > 10:
        self.recent_improvements.pop(0)
    
    # Detect stagnation: little improvement over recent generations
    if len(self.recent_improvements) >= 5:
        recent_change = abs(self.recent_improvements[-1] - self.recent_improvements[0])
        is_stagnant = recent_change < 1e-8 * max(abs(self.f_opt), 1.0)
    else:
        is_stagnant = False
    
    # Compute condition number of C to detect covariance collapse
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = max(np.min(eigvals), 1e-10)
    eig_max = np.max(eigvals)
    cond_C = eig_max / eig_min
    
    # Condition-based damping: reduce learning when C is ill-conditioned
    # to prevent runaway elongation
    cond_damping = np.sqrt(np.log1p(cond_C)) / np.sqrt(np.log1p(self.dim))
    cond_damping = np.clip(cond_damping, 0.3, 3.0)
    
    # Aggressive stagnation detection for high-error tasks
    if cond_C > 1e6:
        is_stagnant = True
    
    # Stagnation-triggered expansion: boost learning rates and inject variance
    if is_stagnant:
        # Strong boost to escape local optima
        stagnation_boost = 5.0
        # Inject diagonal variance proportional to sigma
        diag_injection = 0.2 * self.sigma * np.eye(self.dim)
    else:
        stagnation_boost = 1.0
        diag_injection = np.zeros((self.dim, self.dim))
    
    # Effective learning rates with stagnation boost and condition damping
    cc_eff = min(self.cc * stagnation_boost, 0.5)
    ccov_eff = min(self.ccov * stagnation_boost / cond_damping, 0.5)
    ccov_eff = max(ccov_eff, 1e-8)
    
    # Evolution path update with effective rate
    self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update (unchanged from original)
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Covariance update with stagnation-triggered expansion
    self.C = ((1.0 - ccov_eff) * self.C + 
              ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))
    
    # Inject diagonal variance when stagnant to force re-exploration
    self.C = self.C + diag_injection
    
    self.C = self._ensure_positive_definite(self.C)
```