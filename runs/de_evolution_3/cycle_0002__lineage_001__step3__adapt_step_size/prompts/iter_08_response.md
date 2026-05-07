**Idea: Stagnation-Triggered Aggressive Step Size Scaling**

Instead of relying solely on evolution path norm (which can be slow to respond when stuck in local optima), this approach monitors the stagnation counter and applies multiplicative step size boosts when progress stalls. This directly targets the worst unsolved tasks (errors 1-4) that are stuck at local optima — the standard path-based adaptation is too gradual to escape these basins.

```python
def _adapt_step_size(self):
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    base_factor = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
    
    # Stagnation-triggered aggressive scaling: when stuck, multiply step size
    if self.stagnation_counter > 5:
        boost = min(2.0, 1.0 + 0.15 * self.stagnation_counter)
        base_factor = max(base_factor, np.log(boost))
    
    self.sigma *= np.exp(base_factor)
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```