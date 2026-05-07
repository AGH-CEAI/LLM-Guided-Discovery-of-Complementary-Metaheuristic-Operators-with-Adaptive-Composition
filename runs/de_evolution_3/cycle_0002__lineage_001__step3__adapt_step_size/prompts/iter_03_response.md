**Idea: Multi-Signal Adaptive Sigma with Diversity Detection**

The current step-size adaptation relies solely on the evolution path norm, which becomes ineffective when the optimizer is stuck at large errors (~1e+0 to 5e+0) on tasks 19, 21, 17, 18, etc. These tasks likely suffer from premature convergence to local optima or poor exploration. This variant adds two additional feedback signals — fitness improvement rate and population diversity — to detect stagnation and apply corrective sigma boosts, making it fundamentally different from the single-signal original.

```python
def _adapt_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    # Primary signal: evolution path normalized deviation (original)
    sig1 = p_sigma_norm / self.chi_n - 1.0
    
    # Secondary signal: fitness improvement rate
    if hasattr(self, 'prev_fitness_for_sigma') and self.prev_fitness_for_sigma > 0:
        rel_improve = max(self.prev_fitness_for_sigma - self.best_f, 0.0) / (abs(self.prev_fitness_for_sigma) + 1e-20)
        sig2 = np.tanh(5.0 * rel_improve)  # Map improvement to [-1, 1]
    else:
        sig2 = 0.0
    
    # Tertiary signal: population diversity (rank-based spread)
    if hasattr(self, '_cached_diversity') and self._cached_diversity > 0:
        sig3 = 2.0 * (self._cached_diversity / (self.sigma + 1e-20) - 1.0)
        sig3 = np.clip(sig3, -3.0, 3.0)
    else:
        sig3 = 0.0
    
    self.prev_fitness_for_sigma = self.best_f if np.isfinite(self.best_f) else self.prev_fitness_for_sigma
    
    # Adaptive weights: emphasize primary when healthy, secondary/tertiary when stuck
    w1 = 0.7
    w2 = 0.15
    w3 = 0.15
    combined = w1 * sig1 + w2 * sig2 + w3 * sig3
    
    self.sigma *= np.exp((self.c_sigma / self.d_sigma) * combined)
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```