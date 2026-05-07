**Idea: Orthogonal Chaos Descent**
Combines chaos-based exploration with orthogonal perturbations to the global best direction and adaptive strategy selection based on stagnation level.

```python
def _update_velocity_batch(self, inertia):
    """Update velocity using orthogonal chaos descent with stagnation-adaptive strategy."""
    if self.global_best is None:
        return
    
    # Chaos component: logistic map perturbation for escaping local optima
    chaos = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    chaos = chaos * 2 - 1  # Map to [-1, 1]
    
    # Direction to global best (normalized)
    direction = self.global_best - self.population
    dir_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    dir_norm = np.clip(dir_norm, 1e-10, None)
    dir_normalized = direction / dir_norm
    
    # Orthogonal perturbation: project random vector perpendicular to direction
    random_vec = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
    random_vec = random_vec - np.sum(random_vec * dir_normalized, axis=1, keepdims=True) * dir_normalized
    ortho_norm = np.linalg.norm(random_vec, axis=1, keepdims=True)
    ortho_norm = np.clip(ortho_norm, 1e-10, None)
    orthogonal = random_vec / ortho_norm
    
    # Stagnation-adaptive strategy selection (sigmoid on stagnation level)
    max_stag = np.max(self.stagnation_counter) + 1e-10
    prob_explore = 1.0 / (1.0 + np.exp(-(max_stag - self.stagnation_limit / 2) * 0.3))
    
    # Strategy weights per particle
    explore_weights = np.random.uniform(0, 1, size=(self.pop_size, 1))
    explore_mask = explore_weights < prob_explore[:, np.newaxis]
    
    # Compute velocity components
    range_scale = (self.upper - self.lower) * self.step_size
    
    # Chaos velocity: strong exploration when stagnant
    chaos_vel = chaos * range_scale * 0.5 * np.clip(max_stag / 20, 0.1, 2.0)
    
    # Orthogonal velocity: spread search perpendicular to best direction
    ortho_vel = orthogonal * range_scale * 0.3
    
    # Standard best-direction velocity for exploitation
    r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    best_vel = self.c2 * r1 * direction * 0.5
    
    # Combine strategies per particle
    new_velocity = np.where(explore_mask, chaos_vel + ortho_vel, best_vel + ortho_vel * 0.5)
    
    # Add inertia component
    self.velocity = inertia * self.velocity * 0.3 + new_velocity
    
    # Velocity clamping with bounds awareness
    max_vel = range_scale * 0.5
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 0.0, 1.0)
    scale = np.maximum(scale, 1e-10)  # Prevent division by zero
    self.velocity = self.velocity / scale
```