**Idea: Diversity-Aware Stagnation Detection**

Monitors population diversity (centroid movement and fitness variance) alongside best fitness improvement. Triggers reinitialization only when BOTH fitness stagnates AND population loses diversity, preventing premature reinitialization on multi-modal landscapes where best may not improve but population still explores.

```python
def _check_stagnation(self, fitness):
        current_best = float(np.min(fitness))
        improved = current_best < self.last_best_fitness - 1e-10
        
        if improved:
            self.generation_without_improvement = 0
            self.last_best_fitness = current_best
            return False
        
        self.generation_without_improvement += 1
        
        # Compute population diversity metrics
        if hasattr(self, 'population_fitness') and len(self.population_fitness) > 1:
            fit_variance = float(np.var(self.population_fitness))
            prev_variance = getattr(self, 'prev_fitness_variance', fit_variance)
            self.prev_fitness_variance = fit_variance
            
            # Diversity is low if variance collapsed or centroid didn't move much
            diversity_low = fit_variance < prev_variance * 0.8 and fit_variance < 1e-6
        else:
            diversity_low = False
        
        # Trigger only if stagnated AND diversity is low
        return self.generation_without_improvement > 50 and diversity_low
```