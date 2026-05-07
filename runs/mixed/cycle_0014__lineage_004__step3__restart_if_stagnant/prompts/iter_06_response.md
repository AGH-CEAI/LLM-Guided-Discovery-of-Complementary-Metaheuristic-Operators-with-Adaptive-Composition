**Idea: Hyperbolic Tangent Boundary Explosion**
Uses tanh() to map random directions to extreme boundary regions, plus stagnation-adjusted distance scaling for aggressive escape from local optima. This targets tasks stuck at specific error plateaus (Tasks 20, 21 at exactly 5.0) where the current small perturbation restart fails to explore new regions.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        # Compute stagnation factor: increases with stagnation count
        stagnation_gen = getattr(self, 'stagnation', 0)
        distance_scale = 1.0 + min(stagnation_gen / 20.0, 5.0)  # Cap at 6x
        
        # Replace ALL individuals (not just worst 1/3) for full diversity
        for idx in range(NP):
            # Generate random direction vector
            direction = np.random.randn(self.dim)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-15:
                direction = direction / direction_norm
            
            # Use tanh to map to [0,1] then scale to bounds - pushes to BOUNDARIES
            boundary_t = np.tanh(np.random.uniform(-3, 3, self.dim))  # ~[-1,1]
            
            # Blend: some near-best, some at boundaries, some in between
            blend = np.random.random()
            if blend < 0.3:
                # 30%: Near best with hyperbolic expansion
                offset = distance_scale * 50.0 * boundary_t
                new_pop[idx] = x_best + offset
            elif blend < 0.7:
                # 40%: Fully boundary-focused exploration
                range_val = self.upper - self.lower
                boundary_pos = (self.lower + self.upper) / 2 + boundary_t * range_val / 2
                new_pop[idx] = boundary_pos
            else:
                # 30%: Random in full space
                new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```