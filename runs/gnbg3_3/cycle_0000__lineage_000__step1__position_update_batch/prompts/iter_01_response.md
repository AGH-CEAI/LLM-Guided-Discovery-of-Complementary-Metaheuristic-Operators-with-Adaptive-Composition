**Idea: Spread-Normalized Position Update with Centroid Correction**

Category A: Geometry / spatial — Uses axis-aligned spread (population range per dimension) to normalize velocity updates, then applies a small geometric correction toward the centroid to balance exploration/exploitation based on swarm spread.

```python
def _position_update_batch(self):
    """Update positions using geometric normalization and centroid guidance."""
    # Compute population centroid
    centroid = np.mean(self.population, axis=0)
    
    # Compute axis-aligned spread (range per dimension)
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    spread = pop_max - pop_min
    
    # Normalize velocity by spread (prevent extreme steps in narrow dimensions)
    epsilon = 1e-10
    spread_safe = np.where(spread > epsilon, spread, 1.0)
    normalized_velocity = self.velocity / spread_safe
    
    # Compute centroid-relative distance for each particle
    centroid_dist = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
    global_dist = np.linalg.norm(centroid) + epsilon
    
    # Direction toward centroid (negative of position offset)
    direction_to_centroid = centroid - self.population
    
    # Geometric correction strength: stronger when spread is large (encourage convergence)
    correction_strength = 0.02 * np.clip(spread.mean() / (global_dist + epsilon), 0.01, 0.1)
    
    # Apply geometric correction toward centroid
    correction = correction_strength * direction_to_centroid
    
    # Update positions with normalized velocity and geometric correction
    new_population = self.population + normalized_velocity + correction
    self.population = self._clip_to_bounds(new_population)
```