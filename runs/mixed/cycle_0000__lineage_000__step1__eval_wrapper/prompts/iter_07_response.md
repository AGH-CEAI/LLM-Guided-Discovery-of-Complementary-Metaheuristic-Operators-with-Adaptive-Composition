Looking at the data, I notice that **variants 02-06 all crashed with `-inf` errors**, leaving only `original.py` and `variant_01_idea_0.py` with non-crashing results. The worst tasks (12, 21, 13, 8, 16, 2) have errors in the hundreds to thousands, suggesting the optimizer is getting trapped in local minima with no escape mechanism.

Since `_eval_wrapper` is called during the quantum tunneling phase, I'll make it perform **adaptive multi-scale basin hopping** — a restart-based local search that explores at multiple scales (coarse to fine) to escape local minima. This is fundamentally different from the simple perturbation in the current implementation.

**Idea: Adaptive Multi-Scale Basin Hopping**
Performs multi-scale exploration during tunnel evaluation: coarse random search, medium-scale local search, fine-scale refinement. Returns the best fitness found, helping stagnant particles escape local basins.
```python
def _eval_wrapper(self, x):
    """Wrapper with adaptive multi-scale basin hopping to escape local minima."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, self.lower, self.upper)
    
    f_best = self._user_func(x.reshape(1, -1))[0]
    x_best = x.copy()
    
    # Multi-scale search: coarse -> medium -> fine
    scales = [0.3, 0.1, 0.03]
    
    for scale in scales:
        x_search = x_best.copy()
        step = (self.upper - self.lower) * scale
        n_iter = max(5, int(20 * scale))
        
        for _ in range(n_iter):
            # Random restart within current best's neighborhood
            x_restart = x_best + np.random.uniform(-step, step, size=self.dim)
            x_restart = np.clip(x_restart, self.lower, self.upper)
            
            # Hill climbing from restart
            x_local = x_restart.copy()
            local_step = step * 0.5
            n_local = max(3, int(10 * scale))
            
            for _ in range(n_local):
                dims = np.random.choice(self.dim, max(1, self.dim // 10), replace=False)
                x_candidate = x_local.copy()
                x_candidate[dims] += np.random.uniform(-local_step, local_step, size=len(dims))
                x_candidate = np.clip(x_candidate, self.lower, self.upper)
                
                f_candidate = self._user_func(x_candidate.reshape(1, -1))[0]
                if f_candidate < f_local:
                    x_local = x_candidate
                    f_local = f_candidate
            
            if f_local < f_best:
                x_best = x_local
                f_best = f_local
    
    return f_best
```