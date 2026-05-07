**Idea: Restart-Driven Diversity Sampling**

A fundamentally different sampling strategy that explicitly handles the stuck-at-large-error failure mode by injecting restart-mode samples when diversity collapses. When the optimizer has high error (>1e+0), it has typically lost diversity and is exploring the wrong region — this approach samples from a much wider distribution to escape local optima.

```python
def _sample_population_batch(self):
    """
    Sample a batch combining standard CMA-ES sampling with restart-mode sampling.
    When diversity collapses (large condition number or small sigma), inject
    restart candidates sampled from a wide distribution to escape local optima.
    """
    # Standard CMA-ES sampling
    z = np.random.randn(self.pop_size, self.dim)
    scaled = z * self.D[np.newaxis, :]
    rotated = scaled @ self.B.T
    cmaes_population = self.mean[np.newaxis, :] + self.sigma * rotated
    
    # Determine if restart-mode sampling should be triggered
    # Use condition number of covariance as diversity metric
    if self.D is not None and len(self.D) > 0:
        cond_C = np.max(self.D) / np.maximum(np.min(self.D), 1e-30)
        trigger_restart = cond_C > 1e5 or self.sigma < 0.1
    else:
        trigger_restart = False
    
    if trigger_restart:
        # Restart-mode: sample from a MUCH wider distribution
        # Scale is 10x larger than current sigma to escape local optima
        restart_scale = max(self.sigma * 10.0, 10.0)
        restart_z = np.random.randn(self.pop_size, self.dim)
        # Sample from uniform cube of radius restart_scale in each eigen-direction
        restart_population = self.mean[np.newaxis, :] + restart_scale * restart_z * self.D[np.newaxis, :]
        restart_population = np.clip(restart_population, self.lb, self.ub)
        
        # Blend: 50% CMA-ES, 50% restart-mode for diversity
        blend_mask = np.random.rand(self.pop_size) < 0.5
        population = np.where(blend_mask[:, np.newaxis], restart_population, cmaes_population)
        # Recompute z for restart candidates (for path tracking)
        z = (population - self.mean[np.newaxis, :]) / max(self.sigma, 1e-30)
        # Project back to z-space using inverse transform
        z = z @ self.B / self.D[np.newaxis, :]
    else:
        population = cmaes_population
    
    return population, z
```