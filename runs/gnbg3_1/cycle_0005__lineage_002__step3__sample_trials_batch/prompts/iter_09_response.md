**Idea: Adaptive Radius Hybrid Sampling with Orthogonal Exploration**

This variant replaces Cholesky-based multivariate sampling with a hybrid approach combining: (1) adaptive-radius bounded sampling scaled by population diversity, (2) orthogonal direction injection to escape correlated regions, and (3) boundary-triggered perturbation for constraint-aware exploration. This fundamentally differs from all prior variants which use Cholesky decomposition of the covariance matrix, and directly targets the stuck-at-large-error behavior by forcing broader exploration when the population loses diversity.

```python
def _sample_trials_batch(self):
    """Hybrid sampling with adaptive radius, orthogonal exploration, and boundary perturbation."""
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Adaptive radius: larger when diversity is low (stuck), smaller when converging
    base_radius = self.sigma * max(1.0, np.sqrt(self.dim) * (0.5 + 0.5 * diversity))
    
    # Phase 1: Standard normal sampling with adaptive radius
    z = np.random.randn(self.NP, self.dim)
    trials = self.mean + base_radius * z
    
    # Phase 2: Orthogonal direction injection (1/4 of population)
    n_ortho = max(1, self.NP // 4)
    if n_ortho > 0:
        # Generate orthogonal search directions
        for k in range(n_ortho):
            idx = (k * 7 + self.generation * 3) % self.NP
            # Random orthogonal direction via Householder
            v = np.random.randn(self.dim)
            v = v - np.dot(v, np.ones(self.dim)) / self.dim * np.ones(self.dim)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v = v / v_norm
            # Alternate between shrinking and expanding
            if k % 2 == 0:
                scale = base_radius * (0.5 + 0.5 * np.random.rand())
            else:
                scale = base_radius * (1.5 + 1.0 * np.random.rand())
            trials[idx] = self.mean + scale * v
    
    # Phase 3: Boundary perturbation (1/3 of population)
    n_boundary = max(1, self.NP // 3)
    for k in range(n_boundary):
        idx = (k * 13 + self.generation * 5 + 1) % self.NP
        # Random point on boundary with small perturbation toward interior
        r = np.random.rand(self.dim)
        boundary_point = self.lb + r * (self.ub - self.lb)
        # Blend with mean for interior guidance
        trials[idx] = 0.7 * boundary_point + 0.3 * self.mean + 0.1 * base_radius * np.random.randn(self.dim)
    
    # Phase 4: Elite-guided exploration (1/5 of population)
    n_elite = max(1, self.NP // 5)
    for k in range(n_elite):
        idx = (k * 11 + self.generation * 7 + 2) % self.NP
        if len(self.population) > 0:
            # Sample around elite individuals
            elite_idx = k % min(self.mu, len(self.population))
            elite = self.population[elite_idx]
            trials[idx] = elite + 0.5 * base_radius * np.random.randn(self.dim)
    
    self.trials = self._clip_to_bounds(trials)
```