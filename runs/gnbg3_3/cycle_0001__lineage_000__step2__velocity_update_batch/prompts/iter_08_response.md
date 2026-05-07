Looking at the priority targets (Tasks 17, 16, 6 with errors 10^4-10^2), these represent severe convergence failures where particles get trapped in local optima. The existing variants each address different aspects but none combine complementary mechanisms with principled switching.

For Category H, I'll create a hybrid combining:
1. **Fitness-gradient direction** - Uses historical fitness differences to estimate local gradient direction for escaping local optima
2. **Adaptive momentum** - Tracks improvement rate and modulates velocity scaling

The switching is **stagnation-driven**: when stagnating, favor gradient-based escape; when improving, favor momentum-driven exploitation.

**Idea: Fitness-Gradient Escape with Adaptive Momentum**
One-line description (category H: Hybrid / multi-strategy) combining fitness-gradient direction estimation with adaptive momentum switching based on stagnation detection.

```python
def _velocity_update_batch(self):
    """Hybrid velocity update: fitness-gradient escape + adaptive momentum."""
    # --- Component 1: Fitness-gradient direction estimation ---
    # Compute gradient direction from personal-best improvements across dim
    fitness_diff = self.personal_best_fitness - self.current_fitness
    fitness_diff = np.clip(fitness_diff, -1e10, 1e10)
    
    # Directional gradient: how much personal best changed per dimension
    position_diff = self.personal_best - self.population
    
    # Gradient magnitude estimate per particle
    grad_mag = np.mean(np.abs(position_diff), axis=1, keepdims=True) + 1e-10
    grad_mag = np.clip(grad_mag, 1e-10, 1e10)
    
    # Normalized gradient direction (avoid division issues)
    grad_dir = position_diff / grad_mag
    
    # Weight gradient by how much personal best improved
    improvement_weight = np.clip(fitness_diff / (np.abs(self.current_fitness) + 1e-10), -5, 5)
    improvement_weight = np.maximum(improvement_weight, 0.0)  # Only positive contributions
    
    # Gradient component: move along estimated gradient direction
    grad_component = 0.2 * improvement_weight.reshape(-1, 1) * grad_dir
    
    # --- Component 2: Adaptive momentum with stagnation detection ---
    # Track improvement rate for principled switching
    if self.stagnation_counter > 10:
        # Stagnating: increase exploration via momentum boost
        momentum_scale = 1.0 + 0.5 * min(self.stagnation_counter / 50.0, 1.0)
    else:
        # Normal: use base momentum
        momentum_scale = 0.9
    
    # --- Component 3: Cognitive + Social with adaptive coefficients ---
    cognitive, social = self._adaptive_coefficients()
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # --- Principled switching between gradient and cognitive/social ---
    # Use stagnation level as switching signal (data-driven)
    stagnation_ratio = min(self.stagnation_counter / 30.0, 1.0)
    
    # When stagnating: emphasize gradient escape, reduce cognitive/social pull
    gradient_weight = stagnation_ratio * 0.4
    traditional_weight = 1.0 - stagnation_ratio * 0.3
    
    # Velocity update with principled hybrid weighting
    new_velocity = (
        momentum_scale * self.inertia_weight * self.velocity +
        traditional_weight * (cognitive_component + social_component) +
        gradient_weight * grad_component
    )
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```