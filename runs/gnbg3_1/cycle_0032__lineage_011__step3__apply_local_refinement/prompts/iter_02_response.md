Looking at the unsolved tasks, the worst performers (Tasks 16, 19, 23, 17 with errors 10^2-10^3) suggest the algorithm is stuck in poor local optima or hasn't discovered the correct basin. The current Nelder-Mead approach:
1. Only performs ONE iteration
2. Uses a fixed large step size (0.5 of bounds)
3. Only applies to top 3 individuals
4. Has no mechanism to escape local optima

**Idea: Adaptive Coordinate Perturbation with Random Jumps**

This fundamentally different approach uses multi-scale coordinate-wise perturbation with adaptive step sizes, multiple random direction probes, and random restarts when stuck. Unlike simplex-based Nelder-Mead, it systematically explores along axes and random directions at multiple scales, providing a much higher chance of discovering the global basin on multimodal landscapes.

```python
def _apply_local_refinement(self, population, fitness, func):
    """Apply adaptive coordinate perturbation with multi-scale search and random jumps."""
    if len(population) < 2:
        return population, fitness
    
    n_elite = min(2, len(population))
    range_size = self.upper_bound - self.lower_bound
    
    for i in range(n_elite):
        x_best = population[i].copy()
        f_best = fitness[i]
        
        # Multi-scale step schedule: start coarse, refine
        step_scales = [0.2, 0.05, 0.01, 0.001]
        
        for base_step in step_scales:
            improved_this_scale = False
            
            # Coordinate-wise perturbation (both directions)
            for d in range(self.dim):
                for sign in [1, -1]:
                    x_try = x_best.copy()
                    x_try[d] += sign * base_step * range_size
                    x_try = np.clip(x_try, self.lower_bound, self.upper_bound)
                    
                    f_try = func(x_try.reshape(1, -1))[0]
                    
                    if f_try < f_best:
                        x_best = x_try
                        f_best = f_try
                        improved_this_scale = True
            
            # Random direction probes if no coordinate improvement
            if not improved_this_scale:
                for _ in range(3):
                    direction = np.random.randn(self.dim)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction = direction / norm
                    
                    x_try = x_best + base_step * range_size * direction
                    x_try = np.clip(x_try, self.lower_bound, self.upper_bound)
                    
                    f_try = func(x_try.reshape(1, -1))[0]
                    
                    if f_try < f_best:
                        x_best = x_try
                        f_best = f_try
                        improved_this_scale = True
                        break
            
            # Early exit if significant improvement found
            if f_best < 1e-10:
                break
        
        # Random jump to escape local optima (last resort)
        if f_best > 1e-6:
            # Jump to random location, but bias toward promising regions
            jump_scale = 0.3
            x_jump = x_best + np.random.randn(self.dim) * jump_scale * range_size
            x_jump = np.clip(x_jump, self.lower_bound, self.upper_bound)
            
            f_jump = func(x_jump.reshape(1, -1))[0]
            
            if f_jump < f_best:
                x_best = x_jump
                f_best = f_jump
        
        population[i] = x_best
        fitness[i] = f_best
    
    return population, fitness
```