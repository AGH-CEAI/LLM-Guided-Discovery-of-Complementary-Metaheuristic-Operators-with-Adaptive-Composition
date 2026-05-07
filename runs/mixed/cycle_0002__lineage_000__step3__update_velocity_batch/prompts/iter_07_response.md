**Idea: Success-History Adaptive Velocity with Component Performance Tracking**

This approach tracks the historical success rate of each velocity component (cognitive, social, neighborhood) by comparing position changes against recent performance. Unlike the current random-coefficient approach, this creates an adaptive mechanism that amplifies components that consistently produce position improvements and dampens ineffective ones. This directly addresses the failure mode on the worst tasks (12, 8, 16, 13) where particles get stuck in local optima because all components use fixed random weights regardless of their actual effectiveness.

```python
def _update_velocity_batch(self, inertia):
    """Update velocity using success-history adaptive components."""
    # Track position deltas for each component's effectiveness
    if not hasattr(self, '_prev_pos_for_success'):
        self._prev_pos_for_success = self.population.copy()
        self._success_cognitive = np.ones(self.pop_size)
        self._success_social = np.ones(self.pop_size)
        self._success_neighborhood = np.ones(self.pop_size)
    
    # Compute how much each component actually moved the particle
    delta_pos = self.population - self._prev_pos_for_success
    self._prev_pos_for_success = self.population.copy()
    
    # Success rate = proportion of dimensions where position changed meaningfully
    pos_mag = np.linalg.norm(delta_pos, axis=1, keepdims=True) + 1e-10
    delta_normalized = np.abs(delta_pos) / (pos_mag + 1e-10)
    
    # Update success rates with exponential moving average
    alpha = 0.3
    self._success_cognitive = (1 - alpha) * self._success_cognitive + alpha * np.mean(delta_normalized, axis=1)
    self._success_social = (1 - alpha) * self._success_social + alpha * np.mean(delta_normalized, axis=1)
    self._success_neighborhood = (1 - alpha) * self._success_neighborhood + alpha * np.mean(delta_normalized, axis=1)
    
    # Normalize success rates and compute adaptive weights
    total_success = self._success_cognitive + self._success_social + self._success_neighborhood + 1e-10
    w_c = self._success_cognitive / total_success
    w_s = self._success_social / total_success
    w_n = self._success_neighborhood / total_success
    
    # Apply minimum exploration weight to prevent premature convergence
    min_weight = 0.05
    w_c = np.clip(w_c, min_weight, None)
    w_s = np.clip(w_s, min_weight, None)
    w_n = np.clip(w_n, min_weight, None)
    
    # Recompute after clipping
    total = w_c + w_s + w_n
    w_c, w_s, w_n = w_c / total, w_s / total, w_n / total
    
    # Compute velocity components with success-adaptive coefficients
    cognitive = self.c1 * w_c[:, np.newaxis] * (self.personal_best - self.population)
    social = self.c2 * w_s[:, np.newaxis] * (self.global_best - self.population)
    neighborhood = self.c3 * w_n[:, np.newaxis] * (self.neighborhood_best - self.population)
    
    self.velocity = inertia * self.velocity + cognitive + social + neighborhood
    
    # Velocity clamping
    max_vel = (self.upper - self.lower) * 0.2 * self.step_size
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity /= scale
```