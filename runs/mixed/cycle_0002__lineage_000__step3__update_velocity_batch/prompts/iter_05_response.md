Looking at the data, the worst tasks (12, 16, 8, 13 with errors >100) are likely multimodal/deceptive functions where the optimizer gets trapped in local optima. The current velocity update uses random coefficients r1,r2,r3 ~ U(0,1) which provide undirected exploration.

**Idea: Orthogonal Learning Velocity Update**
Use orthogonal experimental design to identify promising dimension subgroups, then apply targeted coordinated updates to break out of local optima traps.

```python
def _update_velocity_batch(self, inertia):
    """Update velocity using orthogonal learning to escape local optima."""
    pop_size, dim = self.pop_size, self.dim
    
    # Standard PSO components
    r1 = np.random.uniform(0, 1, size=(pop_size, dim))
    r2 = np.random.uniform(0, 1, size=(pop_size, dim))
    r3 = np.random.uniform(0, 1, size=(pop_size, dim))
    
    cognitive = self.c1 * r1 * (self.personal_best - self.population)
    social = self.c2 * r2 * (self.global_best - self.population)
    neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)
    
    base_velocity = inertia * self.velocity + cognitive + social + neighborhood
    
    # Orthogonal learning: construct candidate solutions via dimension grouping
    n_groups = max(2, dim // 10)
    group_size = dim // n_groups
    
    # Select particles with above-median fitness for orthogonal perturbation
    median_fit = np.median(self.personal_best_fit)
    elite_mask = self.personal_best_fit <= median_fit
    n_elite = np.sum(elite_mask)
    
    if n_elite > 0 and dim >= 4:
        # Create orthogonal test matrix using Hadamard-like grouping
        ort_velocity = np.zeros_like(base_velocity)
        
        for i in np.where(elite_mask)[0]:
            # Generate 2 candidate perturbations per group
            for g in range(n_groups):
                start = g * group_size
                end = start + group_size if g < n_groups - 1 else dim
                group_dims = end - start
                
                if group_dims < 2:
                    continue
                    
                # Two orthogonal perturbation directions for this group
                p1 = np.random.uniform(-1, 1, group_dims)
                p2 = np.random.uniform(-1, 1, group_dims)
                # Orthogonalize
                p2 = p2 - np.dot(p1, p2) / (np.dot(p1, p1) + 1e-10) * p1
                
                # Test both and use better direction
                delta1 = self.tunnel_scale * (self.upper - self.lower) * p1
                delta2 = self.tunnel_scale * (self.upper - self.lower) * p2
                
                cand1 = self.population[i].copy()
                cand2 = self.population[i].copy()
                cand1[start:end] += delta1
                cand2[start:end] += delta2
                
                # Quick fitness comparison (clip to bounds)
                cand1 = np.clip(cand1, self.lower, self.upper)
                cand2 = np.clip(cand2, self.lower, self.upper)
                
                # Use the better direction for velocity contribution
                dist1 = np.linalg.norm(cand1 - self.global_best)
                dist2 = np.linalg.norm(cand2 - self.global_best)
                
                if dist1 < dist2:
                    ort_velocity[i, start:end] = delta1 * 0.5
                else:
                    ort_velocity[i, start:end] = delta2 * 0.5
        
        # Blend base velocity with orthogonal learning
        blend = np.where(elite_mask, 0.3, 0.0)
        blend = blend.reshape(-1, 1)
        self.velocity = (1 - blend) * base_velocity + blend * (base_velocity + ort_velocity)
    else:
        self.velocity = base_velocity
    
    # Velocity clamping
    max_vel = (self.upper - self.lower) * 0.2 * self.step_size
    vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity /= scale
```