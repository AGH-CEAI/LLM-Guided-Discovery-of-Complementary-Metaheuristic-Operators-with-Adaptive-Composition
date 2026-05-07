Looking at the priority targets (Tasks 21, 17, 18, 19 with errors 2.5-4.1), the optimizer appears to be stuck in local optima. The standard CMA-ES step size adaptation based on evolution path norm is too slow to escape these basins. The key insight is that the current `_adapt_step_size` only adapts sigma based on the evolution path, which can get stuck when the search converges to a local optimum (path norm stays close to expected, so sigma shrinks or stays flat).

**Idea: Stagnation-Triggered Exponential Sigma Boost**
When fitness improvements stall, exponentially increase sigma to escape local optima. This directly addresses the failure mode where the worst tasks (error > 1) are stuck in basins—the standard adaptation has no mechanism to break out when the evolution path looks "normal" but fitness isn't improving.

```python
def _adapt_step_size(self):
    # Standard CMA-ES adaptation
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    self.sigma *= np.exp(
        (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
    )
    
    # Stagnation-triggered exponential sigma boost
    # When fitness isn't improving, standard adaptation is insufficient
    # We need aggressive sigma inflation to escape local optima
    stagnation_depth = self.stagnation_counter
    
    if stagnation_depth > 0:
        # Exponential boost: deeper stagnation -> bigger exploration
        # Factor grows as 1.5^depth: depth 2 -> 2.25x, depth 5 -> 7.6x, depth 10 -> 57x
        boost_factor = 1.5 ** stagnation_depth
        # Cap to prevent numerical overflow
        boost_factor = min(boost_factor, 1e10)
        self.sigma *= boost_factor
    
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```