**Idea: Stagnation-Driven Exploration Velocity Update**
A fundamentally different velocity update that explicitly handles the premature convergence problem. Stagnant particles receive exploration velocity directed toward the opposite of the population centroid (implicit niching), while improving particles use standard exploitation velocity. This dual-mode approach addresses the diversity collapse that's causing the worst tasks to stall at 1e+2-1e+3 error.

```python
def _update_velocity_batch(self, inertia):
    """Dual-mode velocity update: stagnation-driven exploration vs exploitation."""
    # Compute population centroid for exploration direction
    centroid = np.mean(self.population, axis=0)
    
    # Determine stagnant vs improving particles
    stagnant_mask = self.stagnation_counter >= self.stagnation_limit // 2
    
    # Exploration mode: move toward OPPOSITE of centroid (implicit niching)
    exploration_dir = centroid - self.population  # Direction away from centroid
    exploration_mag = np.linalg.norm(exploration_dir, axis=1, keepdims=True)
    safe_mag = np.where(exploration_mag < 1e-10, 1.0, exploration_mag)
    exploration_dir_normalized = exploration_dir / safe_mag
    
    # Scale exploration by stagnation level and search space
    stagnation_factor = np.minimum(self.stagnation_counter[:, np.newaxis] / max(1, self.stagnation_limit), 3.0)
    exploration_scale = stagnation_factor * (self.upper - self.lower) * 0.1
    exploration_velocity = exploration_dir_normalized * exploration_scale
    
    # Exploitation mode: standard PSO velocity components
    r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    
    cognitive = self.c1 * r1 * (self.personal_best - self.population)
    social = self.c2 * r2 * (self.global_best - self.population)
    neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)
    
    # Dual-mode selection: stagnant = exploration, improving = exploitation
    exploitation_velocity = inertia * self.velocity + cognitive + social + neighborhood
    self.velocity = np.where(stagnant_mask[:, np.newaxis], exploration_velocity, exploitation_velocity)
    
    # Velocity clamping
    max_vel = (self.upper - self.lower) * 0.2 * self.step_size
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity /= scale
```