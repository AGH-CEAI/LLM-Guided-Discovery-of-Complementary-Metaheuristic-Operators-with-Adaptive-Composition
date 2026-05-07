**Idea: Aggressive Noise Injection for Stagnation Escape**

One-line description: Detect stagnation via fitness plateau and inject isotropic noise proportional to error magnitude to escape local optima basins on the worst tasks.

```python
def _adapt_covariance_variant_06(self):
    """Stagnation-driven noise injection with error-proportional exploration."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    # Detect stagnation: no improvement over recent generations
    if not hasattr(self.stagnation_counter):
        self.stagnation_counter = 0
    
    # Track fitness trend
    if not hasattr(self._prev_fitnesses):
        self._prev_fitnesses = []
    self._prev_fitnesses.append(self.f_opt)
    if len(self._prev_fitnesses) > max(10, self.dim // 2):
        self._prev_fitnesses.pop(0)
    
    # Compute stagnation metrics
    if len(self._prev_fitnesses) >= 5:
        fitness_improvement = self._prev_fitnesses[0] - self._prev_fitnesses[-1]
        stagnation_ratio = max(0.0, fitness_improvement) / (abs(self.f_opt) + 1.0)
        is_stagnant = (stagnation_ratio < 1e-6) and (len(self._prev_fitnesses) >= max(10, self.dim // 2))
    else:
        is_stagnant = False
    
    # Error-magnitude-based exploration boost
    error_scale = np.log1p(max(abs(self.f_opt), 1e-10))
    exploration_boost = np.clip(error_scale / 10.0, 0.0, 3.0)
    
    # Adaptive learning rates
    if is_stagnant:
        ccov_scale = 1.0 + exploration_boost
        cc_scale = 1.0 + 0.5 * exploration_boost
    else:
        ccov_scale = 1.0
        cc_scale = 1.0
    
    ccov_adaptive = min(self.ccov * ccov_scale, 0.5)
    cc_adaptive = min(self.cc * cc_scale, 0.3)
    
    # Re-compute pc with adaptive rate
    self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu))
    
    # Inject isotropic noise when stagnant — break out of local optima
    if is_stagnant:
        noise_magnitude = max(abs(self.f_opt), 1.0) * 0.01 * exploration_boost
        noise = np.random.randn(self.dim, self.dim) * noise_magnitude
        noise = 0.5 * (noise + noise.T)
        self.C = self.C + noise
    
    self.C = self._ensure_positive_definite(self.C)
```