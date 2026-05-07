**Idea: Stagnation-Triggered Heavy-Tailed Exploration**

The worst unsolved tasks (17, 16, 20, 18, 12, etc.) are stuck at errors 10^1–10^2, suggesting the algorithm converges to local basins but lacks the ability to escape. All existing variants use Gaussian sampling (possibly with different covariance adaptation), which has light tails and rarely produces the large jumps needed to break out. This variant directly modifies the sampling distribution to use a heavier-tailed Student's t-distribution when stagnation is detected.

```python
def _sample_trials_batch(self):
    """Sample with stagnation-triggered heavy-tailed exploration to escape local optima."""
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    # Track generations since best improvement (independent of main stagnation counter)
    if not hasattr(self, '_best_at_last_improvement'):
        self._best_at_last_improvement = self.f_opt
        self._gen_at_last_improvement = self.generation
    
    gens_since_best = self.generation - self._gen_at_last_improvement
    
    # Update tracking if improved
    if self.f_opt < self._best_at_last_improvement - 1e-14:
        self._best_at_last_improvement = self.f_opt
        self._gen_at_last_improvement = self.generation
        gens_since_best = 0
    
    # Stagnation threshold scales with dimension
    stagnation_threshold = max(8, self.dim // 2)
    
    if gens_since_best >= stagnation_threshold:
        # Exploration mode: t-distribution with heavier tails for escaping local optima
        # Degrees of freedom decreases with dim for heavier tails in high dimensions
        df = max(2.0, self.dim / 5.0)
        z_raw = np.random.standard_t(df, size=(self.NP, self.dim))
        
        # Normalize each row, then scale by random factor in [1.5, 4.0] for varied jump sizes
        row_norms = np.linalg.norm(z_raw, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-8)
        z_normalized = z_raw / row_norms
        jump_factors = np.random.uniform(1.5, 4.0, size=(self.NP, 1))
        z_explore = z_normalized * jump_factors
        
        # Use larger sigma during exploration
        sigma_explore = self.sigma * 3.0
        self.trials = self.mean + sigma_explore * (z_explore @ L.T)
    else:
        # Normal mode: standard multivariate normal sampling
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
    
    self.trials = self._clip_to_bounds(self.trials)
```