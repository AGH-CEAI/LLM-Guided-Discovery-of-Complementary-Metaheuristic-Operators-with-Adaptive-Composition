**Idea: Hooke-Jeeves Pattern Search**
A direct search method that uses a fundamentally different pattern-based directional exploration strategy vs. Nelder-Mead's simplex transformations. Known for strong convergence properties on smooth and irregular landscapes.

```python
def _apply_local_refinement(self, population, fitness, func):
    """Apply Hooke-Jeeves pattern search to top individuals."""
    if len(population) < 1:
        return population, fitness
    
    n_elite = min(2, len(population))
    delta = 0.5 * (self.upper_bound - self.lower_bound)
    delta_min = 1e-8
    
    for i in range(n_elite):
        x_curr = population[i].copy()
        f_curr = fitness[i]
        
        while delta >= delta_min:
            improved = False
            
            # Search in all positive and negative coordinate directions
            for d in range(self.dim):
                for sign in [1.0, -1.0]:
                    x_try = x_curr.copy()
                    # Scale step by delta fraction of search range
                    step = sign * delta * 0.1 * (self.upper_bound - self.lower_bound)
                    x_try[d] += step
                    x_try = np.clip(x_try, self.lower_bound, self.upper_bound)
                    
                    f_try = func(x_try.reshape(1, -1))
                    if len(f_try) < 1:
                        break
                    f_try = f_try[0]
                    
                    if f_try < f_curr:
                        x_curr = x_try
                        f_curr = f_try
                        improved = True
                        break
                
                if improved:
                    break
            
            # If no improvement, shrink step size
            if not improved:
                delta *= 0.5
            
            # Pattern move: after successful exploration, try moving in same direction
            if improved and delta > delta_min:
                pattern_dir = x_curr - population[i]
                x_pattern = x_curr + 0.5 * pattern_dir
                x_pattern = np.clip(x_pattern, self.lower_bound, self.upper_bound)
                
                f_pattern = func(x_pattern.reshape(1, -1))
                if len(f_pattern) >= 1 and f_pattern[0] < f_curr:
                    x_curr = x_pattern
                    f_curr = f_pattern[0]
            
            # Update if improved
            if f_curr < fitness[i]:
                population[i] = x_curr
                fitness[i] = f_curr
        
        # Ensure we return the best found
        if f_curr < fitness[i]:
            population[i] = x_curr
            fitness[i] = f_curr
    
    return population, fitness
```