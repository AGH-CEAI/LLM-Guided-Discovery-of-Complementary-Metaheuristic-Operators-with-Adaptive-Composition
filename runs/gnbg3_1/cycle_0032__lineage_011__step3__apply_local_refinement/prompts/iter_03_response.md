**Idea: Powell's Method with Random Restarts**
Multi-start Powell's direction-set optimization with random restarts when stuck, escaping local optima traps that paralyze Nelder-Mead.

```python
def _apply_local_refinement(self, population, fitness, func):
    """Powell's method with multi-start restarts to escape local optima traps."""
    if len(population) < 2:
        return population, fitness
    
    # Apply to top individual only — more iterations for better convergence
    x_best = population[0].copy()
    f_best = fitness[0]
    
    n_restarts = 3  # Multiple restarts to escape local optima
    max_iter_per_restart = 15  # Powell iterations per restart
    
    best_overall = x_best.copy()
    f_best_overall = f_best
    
    for restart in range(n_restarts):
        x_current = x_best.copy() if restart == 0 else best_overall.copy()
        f_current = f_best if restart == 0 else f_best_overall
        
        # Initialize direction set (coordinate directions)
        if restart == 0:
            directions = np.eye(self.dim)
        else:
            # Perturb directions for diversity on restart
            directions = np.eye(self.dim) + 0.1 * np.random.randn(self.dim, self.dim)
            # Orthonormalize
            for i in range(self.dim):
                for j in range(i):
                    directions[i] -= np.dot(directions[i], directions[j]) * directions[j]
                norm = np.linalg.norm(directions[i])
                if norm > 1e-10:
                    directions[i] /= norm
        
        # Powell's method iterations
        for iteration in range(max_iter_per_restart):
            improved = False
            
            # Try each direction
            for d in range(self.dim):
                dir_vec = directions[d]
                if np.linalg.norm(dir_vec) < 1e-10:
                    continue
                
                # Initial step size based on distance to bounds
                bounds_range = self.upper_bound - self.lower_bound
                step = 0.1 * bounds_range
                
                # Golden section line search along direction
                phi = (1 + np.sqrt(5)) / 2
                resphi = 2 - phi
                
                # Bracket phase
                a = 0.0
                x_a = x_current
                f_a = f_current
                
                x_b = x_current + step * dir_vec
                x_b = np.clip(x_b, self.lower_bound, self.upper_bound)
                f_b = func(x_b.reshape(1, -1))[0]
                
                if f_b < f_current:
                    # Move further in same direction
                    step = step * phi
                    x_c = x_current + step * dir_vec
                    x_c = np.clip(x_c, self.lower_bound, self.upper_bound)
                    f_c = func(x_c.reshape(1, -1))[0]
                else:
                    # Shrink back
                    step = -step * resphi
                    x_c = x_current + step * dir_vec
                    x_c = np.clip(x_c, self.lower_bound, self.upper_bound)
                    f_c = func(x_c.reshape(1, -1))[0]
                
                # Ensure ordering: a=0, b=step, c=phi*step (or reversed)
                if f_c < f_b:
                    x_a, f_a = x_b, f_b
                    x_b, f_b = x_c, f_c
                    step = step * phi
                else:
                    # Keep a and swap b,c if needed
                    if f_b > f_a:
                        x_a, f_a, x_b, f_b = x_b, f_b, x_a, f_a
                        step = -step
                
                # Refine bracket
                for refine in range(10):
                    if abs(step) < 1e-8:
                        break
                    x_c = x_b + step * resphi * dir_vec
                    x_c = np.clip(x_c, self.lower_bound, self.upper_bound)
                    f_c = func(x_c.reshape(1, -1))[0]
                    
                    if f_c < f_b:
                        if f_c < f_a:
                            x_a, f_a, x_b, f_b = x_b, f_b, x_c, f_c
                            step = step * phi
                        else:
                            x_a, f_a = x_b, f_b
                            x_b, f_b = x_c, f_c
                            step = step * resphi
                    else:
                        step = step * resphi
                
                # Update direction with best found point
                if f_b < f_current:
                    x_current = x_b
                    f_current = f_b
                    improved = True
            
            # Update direction set (Powell's update)
            if improved:
                # Direction from start to end of iteration
                total_dir = x_current - (best_overall.copy() if restart == 0 and iteration == 0 else x_best)
                if np.linalg.norm(total_dir) > 1e-10:
                    # Shift directions (drop first, add new)
                    directions = np.roll(directions, -1, axis=0)
                    directions[-1] = total_dir / np.linalg.norm(total_dir)
            
            # Check for improvement
            if f_current < f_best_overall:
                f_best_overall = f_current
                best_overall = x_current.copy()
        
        # Random restart if stuck
        if f_best_overall >= f_best - 1e-6:
            # Generate random perturbation
            perturbation = np.random.uniform(-0.5, 0.5, self.dim)
            perturbation = perturbation * (self.upper_bound - self.lower_bound)
            x_perturbed = best_overall + perturbation
            x_perturbed = np.clip(x_perturbed, self.lower_bound, self.upper_bound)
            f_perturbed = func(x_perturbed.reshape(1, -1))[0]
            
            if f_perturbed < f_best_overall:
                best_overall = x_perturbed
                f_best_overall = f_perturbed
    
    # Update population if improvement found
    if f_best_overall < f_best:
        population[0] = best_overall
        fitness[0] = f_best_overall
    
    return population, fitness
```