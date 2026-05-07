**Idea: Multi-Scale Restart Injection**

A single wrapper evaluation that generates multiple progressively refined samples around the trial point, escaping poor local optima by injecting fresh restart-like perturbations at multiple scales.

```python
def _eval_wrapper(self, x):
    """Multi-scale restart injection: evaluate best of progressively refined samples."""
    best_x = x.copy()
    best_f = self._user_func(x.reshape(1, -1))[0]
    
    # Progressive multi-scale perturbation: from wide exploration to fine refinement
    scales = [1.0, 0.5, 0.2, 0.1]
    
    for scale in scales:
        # Generate candidate samples at this scale
        n_candidates = max(3, self.dim // 3)
        candidates = np.zeros((n_candidates, self.dim))
        
        for j in range(n_candidates):
            # Random walk from current best
            r = np.random.uniform(-1, 1, size=self.dim)
            candidates[j] = best_x + r * scale * (self.upper - self.lower) * 0.1
        
        # Clip and evaluate
        candidates = self._clip_to_bounds_batch(candidates)
        fitness = self._user_func(candidates)
        
        # Track best from this scale
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_f:
            best_f = fitness[best_idx]
            best_x = candidates[best_idx]
    
    return best_f
```