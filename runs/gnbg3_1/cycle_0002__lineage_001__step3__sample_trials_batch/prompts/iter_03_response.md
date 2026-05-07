**Idea: Restart-Triggered Hybrid Sampling**

Inject diversity when stagnation is detected by switching from pure local Gaussian sampling to a hybrid of global uniform + local Gaussian exploration. The worst tasks (errors ~10^2-10^1) are likely stuck in local optima because the current sampling is too narrowly focused around the mean. This fundamentally different strategy breaks convergence traps by periodically mixing in broad exploration.

```python
def _sample_trials_batch(self):
    """Sample trials with restart-triggered hybrid global+local exploration."""
    # Use wider sampling when stagnating to escape local optima
    is_stagnating = self.stagnation_counter > self.max_stagnation // 3
    
    if is_stagnating:
        # Hybrid: 40% global uniform + 60% local Gaussian with inflated sigma
        n_global = int(0.4 * self.NP)
        n_local = self.NP - n_global
        
        # Global exploration: uniform across entire bounds
        global_samples = np.random.uniform(self.lb, self.ub, (n_global, self.dim))
        
        # Local search: Gaussian around mean with wider spread
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        z = np.random.randn(n_local, self.dim)
        local_samples = self.mean + 2.5 * self.sigma * (z @ L.T)
        
        self.trials = np.vstack([global_samples, local_samples])
    else:
        # Standard sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
    
    self.trials = self._clip_to_bounds(self.trials)
```