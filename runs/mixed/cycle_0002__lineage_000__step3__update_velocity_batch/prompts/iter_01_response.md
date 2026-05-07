**Idea: Comprehensive Learning PSO (CLPSO) Velocity Update**
Each dimension learns from a different exemplar (own best or another particle's best) rather than all dimensions following the same global best. This prevents premature convergence and is known to excel on complex, multi-modal landscapes.

```python
def _update_velocity_batch(self, inertia):
    """CLPSO-style velocity update with per-dimension exemplar selection."""
    if self.velocity is None:
        self.velocity = np.zeros((self.pop_size, self.dim))
    
    # Learning probability for CLPSO (determines how often to learn from own best)
    pc = 1.0 / (1.0 + self.c1)
    
    # Select exemplar for each particle and each dimension
    exemplars = np.zeros((self.pop_size, self.dim), dtype=np.int32)
    
    for i in range(self.pop_size):
        # Decide exemplar for each dimension independently
        use_own_best = np.random.uniform(0, 1, size=self.dim) < pc
        
        # For dimensions learning from others, pick random particle (not self)
        other_best_indices = np.concatenate([
            np.arange(i), np.arange(i + 1, self.pop_size)
        ])
        other_choices = np.random.choice(other_best_indices, size=self.dim)
        
        exemplars[i] = np.where(use_own_best, i, other_choices)
    
    # Build exemplar positions matrix
    exemplar_positions = self.personal_best[exemplars]  # (pop_size, dim)
    
    # Random coefficients for velocity update
    r = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    
    # Velocity update: inertia + exemplar-guided component
    cognitive = self.c1 * r * (exemplar_positions - self.population)
    self.velocity = inertia * self.velocity + cognitive
    
    # Velocity clamping using magnitude-based scaling
    max_vel = (self.upper - self.lower) * 0.2 * self.step_size
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    # Avoid division by zero
    vel_mag = np.maximum(vel_mag, 1e-12)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity /= scale
```