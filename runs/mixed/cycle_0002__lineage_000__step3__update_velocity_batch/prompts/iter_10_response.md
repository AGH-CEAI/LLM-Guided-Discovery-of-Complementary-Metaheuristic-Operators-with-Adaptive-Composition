**Idea: Fitness-Gradient & Stagnation-Aware Velocity Update**

A fundamentally different approach: instead of random coefficients (r1, r2, r3), use the *actual fitness improvement direction* as a gradient signal, combined with stagnation-triggered exploration bursts and individual particle success tracking. This targets the worst tasks by breaking the premature convergence pattern that causes errors stuck at 1e+01 to 1e+03.

```python
def _update_velocity_batch(self, inertia):
    """Update velocity using fitness-gradient signals and stagnation-aware exploration."""
    # Compute per-particle fitness gradient (improvement direction)
    current_fitness = self.personal_best_fit.copy()
    # Normalize fitness gradient to [-1, 1] per dimension
    pos_diff = self.personal_best - self.population
    fit_range = np.ptp(current_fitness) + 1e-10
    fit_gradient = (current_fitness[:, np.newaxis] - current_fitness.mean()) / fit_range
    
    # Individual particle success rate for adaptive learning
    if not hasattr(self, 'particle_success_rate'):
        self.particle_success_rate = np.ones(self.pop_size) * 0.5
    success_decay = 0.9
    improvement_mask = current_fitness < self.prev_personal_best_fit if hasattr(self, 'prev_personal_best_fit') else np.zeros(self.pop_size, dtype=bool)
    self.particle_success_rate = self.particle_success_rate * success_decay + improvement_mask.astype(float) * (1 - success_decay)
    self.prev_personal_best_fit = current_fitness.copy()
    
    # Adaptive cognitive component using fitness gradient direction
    adaptive_c1 = self.c1 * np.clip(self.particle_success_rate[:, np.newaxis], 0.1, 2.0)
    cognitive = adaptive_c1 * fit_gradient[:, np.newaxis] * pos_diff
    
    # Social component with global best
    r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    social = self.c2 * r2 * (self.global_best - self.population)
    
    # Neighborhood component
    r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)
    
    # Stagnation-triggered exploration burst
    stagnant_mask = self.stagnation_counter >= self.stagnation_limit
    exploration_scale = np.where(stagnant_mask, 2.0, 1.0)
    exploration = np.random.randn(self.pop_size, self.dim) * exploration_scale[:, np.newaxis]
    
    # Combine all components with adaptive inertia
    self.velocity = inertia * self.velocity + cognitive + social + neighborhood + exploration
    
    # Velocity clamping with adaptive bounds
    max_vel = (self.upper - self.lower) * 0.2 * self.step_size
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity = np.where(vel_mag > max_vel * 1e-6, self.velocity / scale, np.zeros_like(self.velocity))
```