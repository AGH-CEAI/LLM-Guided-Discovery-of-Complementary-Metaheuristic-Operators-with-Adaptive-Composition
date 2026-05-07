**Idea: Stagnation-Triggered Covariance Explosion with Directional Kick**

When the optimizer is stuck at local optima (as shown by high errors on tasks 16, 17, 20, etc.), this variant triggers a dramatic covariance explosion proportional to stagnation depth, followed by a directional "kick" in the search path direction to escape the basin. Unlike previous variants that use static thresholds or condition-number-based scaling, this approach directly measures how long and how severely the algorithm is stuck and responds with exponentially stronger exploration that decays once progress resumes.

```python
def _adapt_covariance_variant_06(self):
    """Stagnation-depth-driven covariance explosion with directional kick."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Stagnation depth drives exploration intensity
    stagnation_ratio = min(self.stagnation_counter / max(self.max_stagnation, 1), 1.0)
    
    # Exponential scaling: stagnation_ratio=0.5 → ~2.7x, stagnation_ratio=0.8 → ~10x
    explosion_mult = 1.0 + 10.0 * np.tanh(3.0 * stagnation_ratio)
    
    # Decay explosion strength when improving
    if not hasattr(self, 'explosion_strength'):
        self.explosion_strength = 1.0
    if stagnation_ratio < 0.1:
        self.explosion_strength *= 0.95
        self.explosion_strength = max(self.explosion_strength, 1.0)
    else:
        self.explosion_strength = explosion_mult
    
    # Base adaptive ccov
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    ccov_base = min(self.ccov * (1.0 + 5.0 * min(rel_var, 0.1)), 0.5)
    ccov_scaled = ccov_base * self.explosion_strength
    ccov_scaled = min(ccov_scaled, 0.8)
    
    # Directional kick: push covariance along the current search path
    eigvals, eigvecs = np.linalg.eigh(self.C)
    max_eig = np.max(eigvals)
    
    if np.linalg.norm(self.pc) > 1e-12:
        kick_dir = self.pc / np.linalg.norm(self.pc)
    else:
        kick_dir = eigvecs[:, -1]
        kick_dir /= (np.linalg.norm(kick_dir) + 1e-12)
    
    kick_magnitude = 0.5 * max_eig * self.explosion_strength
    kick_matrix = kick_magnitude * np.outer(kick_dir, kick_dir)
    
    self.C = ((1.0 - ccov_scaled) * self.C + 
              ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu) +
              0.1 * ccov_scaled * kick_matrix)
    
    self.C = self._ensure_positive_definite(self.C)
```