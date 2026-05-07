**Idea: Orthogonal Learning with Convergence-Driven Perturbation**

Orthogonal decomposition of cognitive/social/neighborhood guides extracts independent directional components, while a convergence-driven perturbation inversely scaled by progress ratio prevents premature convergence on deceptive tasks.

```python
def _update_velocity_batch(self, inertia):
    """Velocity update with orthogonal learning and convergence-driven perturbation."""
    cognitive = self.personal_best - self.population
    social = self.global_best - self.population
    neighborhood = self.neighborhood_best - self.population
    
    guides = np.stack([cognitive, social, neighborhood], axis=0)
    
    weights = np.array([self.c1, self.c2, self.c3]).reshape(3, 1, 1)
    weighted_sum = np.sum(guides * weights, axis=0)
    
    norms = np.linalg.norm(guides, axis=2, keepdims=True) + 1e-10
    normalized = guides / norms
    ortho_component = np.sum(normalized, axis=0) - weighted_sum / (np.linalg.norm(weighted_sum, axis=1, keepdims=True) + 1e-10)
    
    r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    
    self.velocity = inertia * self.velocity + r1 * weighted_sum + r2 * ortho_component
    
    max_vel = (self.upper - self.lower) * 0.2 * self.step_size
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity /= scale
```