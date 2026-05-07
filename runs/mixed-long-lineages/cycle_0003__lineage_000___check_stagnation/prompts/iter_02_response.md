**Idea: Diversity-Aware Stagnation Detection**

A fundamentally different stagnation detection that considers both best fitness AND population diversity. Tasks failing on difficult problems often get trapped in low-diversity states where the counter-based approach is too aggressive. This monitors population spread and adapts the threshold based on problem scale.

```python
def _check_stagnation(self, fitness):
    current_best = float(np.min(fitness))
    
    # Check for improvement
    if current_best < self.last_best_fitness - 1e-10:
        self.generation_without_improvement = 0
        self.last_best_fitness = current_best
        # Track diversity when making progress
        if hasattr(self, 'population') and len(self.population) > 0:
            div = self._compute_diversity(self.population)
            self.diversity_history.append(div)
        return False
    else:
        self.generation_without_improvement += 1
        
        # Compute current diversity
        if hasattr(self, 'population') and len(self.population) > 0:
            current_diversity = self._compute_diversity(self.population)
        else:
            current_diversity = 1.0
        
        # Adaptive threshold based on problem scale
        search_range = self.upper - self.lower
        dim_scale = np.sqrt(self.dim) * search_range
        diversity_threshold = dim_scale * 1e-4
        
        # Stagnation: no improvement AND low diversity (converged to local optima)
        diversity_stagnant = (current_diversity < max(diversity_threshold, 1e-6))
        
        # Use adaptive generation threshold: harder problems need more time
        base_threshold = 50
        if hasattr(self, 'diversity_history') and len(self.diversity_history) >= 10:
            recent_div = np.mean(self.diversity_history[-10:])
            if recent_div < diversity_threshold * 2:
                # Problem appears hard, extend stagnation tolerance
                base_threshold = 100
        
        return (self.generation_without_improvement > base_threshold) and diversity_stagnant
```