Looking at the data, ALL 24 tasks remain unsolved with errors ranging from ~0.4 to ~2628. The two previous variants (02, 03) produced `-inf` (crashed), while original and variant_01 are stuck at various error levels. The key insight: the worst tasks (12, 21, 13, 8, 16, 2) have errors 10+ decades above target, suggesting the algorithm gets trapped in local optima and the quantum tunneling mechanism isn't aggressive enough to escape.

The `_eval_wrapper` is called during quantum tunneling to evaluate candidates. Currently it's a simple passthrough. I need a fundamentally different evaluation strategy that can escape local basins.

**Idea: MultiScale Perturbation with Tabu-Like Memory**
Use a coarse-to-fine perturbation search inside the wrapper itself - try multiple perturbation scales (large to small), keeping the best improvement. This gives the tunnel mechanism "intelligent exploration" that can escape local optima at multiple scales, combining global search with local refinement.

```python
def _eval_wrapper(self, x):
    """Multi-scale perturbation wrapper for escape from local optima."""
    x = np.clip(x, self.lower, self.upper)
    current_f = self._user_func(x.reshape(1, -1))[0]
    if not np.isfinite(current_f):
        return current_f
    
    best_x, best_f = x.copy(), current_f
    search_range = self.upper - self.lower
    
    # Multi-scale perturbation: coarse to fine
    for scale_exp in range(8, -4, -1):
        scale = search_range * (0.1 ** scale_exp)
        for _ in range(8):
            noise = np.random.randn(self.dim) * scale
            candidate = np.clip(best_x + noise, self.lower, self.upper)
            f = self._user_func(candidate.reshape(1, -1))[0]
            if np.isfinite(f) and f < best_f:
                best_x, best_f = candidate.copy(), f
    
    return best_f
```