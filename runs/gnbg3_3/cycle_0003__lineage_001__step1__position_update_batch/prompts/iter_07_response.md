**Idea: Monte Carlo Bootstrap Velocity Estimation**

Use bootstrap resampling of the population to estimate the uncertainty in the velocity direction, then modulate the update stochastically. This provides statistically grounded exploration that adapts to the population's confidence in its movement direction — fundamentally different from the deterministic spectral approach.

```python
def _position_update_batch(self):
    """Update positions using Monte Carlo bootstrap velocity estimation."""
    n_samples = min(50, max(10, self.np // 2))
    
    sampled_velocities = []
    for _ in range(n_samples):
        indices = np.random.choice(self.np, size=self.np, replace=True)
        sampled_velocities.append(self.velocity[indices])
    
    sampled_velocities = np.array(sampled_velocities)
    
    vel_mean = np.mean(sampled_velocities, axis=0)
    vel_std = np.std(sampled_velocities, axis=0)
    
    global_dir = np.zeros_like(self.population)
    if self.global_best is not None:
        global_dir = self.global_best - self.population
        global_dist = np.linalg.norm(global_dir, axis=1, keepdims=True) + 1e-10
        global_dir = global_dir / global_dist
    
    r_boost = np.random.uniform(0, 1, (self.np, 1))
    boost_strength = r_boost * np.mean(vel_std) / (np.mean(vel_std) + 1e-6)
    stochastic_velocity = vel_mean + boost_strength * global_dir
    
    confidence = 1.0 / (1.0 + np.mean(vel_std) / (np.mean(np.abs(vel_mean)) + 1e-10))
    confidence = np.clip(confidence, 0.1, 0.9)
    
    final_velocity = (
        confidence[:, np.newaxis] * self.velocity +
        (1.0 - confidence[:, np.newaxis]) * stochastic_velocity
    )
    
    r_mc = np.random.uniform(0, 1, (self.np, self.dim))
    mc_mask = r_mc < 0.3
    mc_perturbation = np.random.randn(self.np, self.dim) * np.mean(vel_std) * 0.5
    final_velocity = np.where(mc_mask, final_velocity + mc_perturbation, final_velocity)
    
    new_population = self.population + final_velocity
    self.population = self._clip_to_bounds(new_population)
```