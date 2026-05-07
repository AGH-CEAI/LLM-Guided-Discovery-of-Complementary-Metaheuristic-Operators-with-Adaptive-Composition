**Idea: Diversity-Weighted Stagnation Detection**
Uses combined fitness improvement AND population diversity metrics to detect stagnation, resetting when either metric indicates the search has stalled.

```python
def _check_stagnation(self, fitness):
    current_best = float(np.min(fitness))
    
    # Track fitness improvement
    if current_best < self.last_best_fitness - 1e-10:
        self.generation_without_improvement = 0
        self.last_best_fitness = current_best
        fitness_stagnant = False
    else:
        self.generation_without_improvement += 1
        fitness_stagnant = self.generation_without_improvement > 50
    
    # Track population diversity (complementary signal)
    pop_std = float(np.std(fitness))
    diversity_threshold = 1e-8 * (abs(self.last_best_fitness) + 1e-10)
    diversity_stagnant = pop_std < diversity_threshold
    
    # Combined decision: stagnation if BOTH signals indicate stall
    # or if fitness stagnation is severe (>100 gen without improvement)
    severe_stagnation = self.generation_without_improvement > 100
    
    return (fitness_stagnant and diversity_stagnant) or severe_stagnation
```