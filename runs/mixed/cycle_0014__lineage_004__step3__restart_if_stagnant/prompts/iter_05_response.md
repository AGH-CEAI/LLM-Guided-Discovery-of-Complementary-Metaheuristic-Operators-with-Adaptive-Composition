**Idea: Orthogonal Line Sampling with Boundary Explosion**

Replaces the entire worst half with solutions generated along orthogonal lines from x_best and near-boundary extreme points. This fundamentally differs from replacing just 1/3 with uniform perturbations—it's designed to escape the massive local optima traps causing errors of 10^1–10^3 by aggressively sampling the search space's extremities.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        # Sort fitness to identify worst half
        sorted_indices = np.argsort(fitness)[::-1]  # worst first
        half_count = NP // 2
        worst_indices = sorted_indices[:half_count]
        
        # Generate orthogonal basis directions from current population
        centered = population - x_best
        U, _, _ = np.linalg.svd(centered, full_matrices=False)
        if U.shape[0] < self.dim:
            # Fallback to identity if SVD fails
            directions = np.eye(self.dim)
        else:
            directions = U[:min(self.dim, 20)].T  # Top principal directions
        
        # Generate replacement solutions for each worst individual
        for i, idx in enumerate(worst_indices):
            strategy = i % 5
            
            if strategy == 0:
                # Strategy 0: Extreme boundary exploration
                # Find worst fitness to determine exploration scale
                worst_f = fitness[sorted_indices[0]]
                scale = min(100.0, max(20.0, abs(worst_f) ** 0.5))
                new_pop[idx] = x_best + np.random.uniform(-scale, scale, self.dim)
                
            elif strategy == 1:
                # Strategy 1: Orthogonal line sampling
                if len(directions) > 0:
                    dir_idx = i % len(directions)
                    direction = directions[dir_idx]
                    t = np.random.uniform(-50, 50)
                    new_pop[idx] = x_best + t * direction
                else:
                    new_pop[idx] = x_best + np.random.uniform(-50, 50, self.dim)
                    
            elif strategy == 2:
                # Strategy 2: Corner/edge exploration
                corner = np.random.choice([self.lower, self.upper], self.dim)
                new_pop[idx] = x_best * 0.3 + corner * 0.7 + np.random.uniform(-5, 5, self.dim)
                
            elif strategy == 3:
                # Strategy 3: Gaussian scatter around x_best with large std
                std = max(20.0, (self.upper - self.lower) * 0.15)
                new_pop[idx] = x_best + np.random.normal(0, std, self.dim)
                
            else:
                # Strategy 4: Random restart from scratch
                new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```