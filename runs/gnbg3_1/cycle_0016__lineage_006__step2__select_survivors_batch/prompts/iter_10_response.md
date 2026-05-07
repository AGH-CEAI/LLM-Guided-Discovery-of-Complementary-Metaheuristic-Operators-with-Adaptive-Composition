Looking at the problem, all 24 tasks remain unsolved with errors ranging from 4.046e+01 (Task 17) down to 9.914e-05 (Task 5). The current survivor selection uses pure elitist (mu, lambda)-selection, which keeps the best NP individuals regardless of diversity. This causes premature convergence, especially on the worst tasks where errors are stuck at ~1e+01.

**Analysis of failure modes:**
- Tasks 17, 16, 23, 20, 11, 12, 19, 18 (errors ~1e+01): Classic premature convergence — the algorithm finds a local optimum but loses diversity needed to escape
- The current selection provides ZERO diversity pressure — it's purely fitness-based
- All 9 covariance adaptation variants failed because survivor selection collapses diversity before they can act

**Key insight:** A fundamentally different survivor selection needs to actively maintain diversity by considering spatial distribution, not just fitness. This directly addresses the root cause of getting stuck at 1e+01 error.

**Idea: Fitness-Diversity Balanced Survivor Selection**
Uses temperature-controlled softmax selection combining fitness rank with distance-from-mean rank. When population diversity is low, temperature increases to boost diversity pressure and prevent premature convergence.

```python
def _select_survivors_batch(self):
    """Select survivors via fitness-diversity balanced selection with adaptive temperature."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    n_combined = len(combined_fit)
    
    # Compute diversity metrics
    pop_std = np.std(combined_pop, axis=0)
    mean_std = np.mean(pop_std)
    expected_std = (self.ub[0] - self.lb[0]) / 6.0
    diversity = mean_std / (expected_std + 1e-10)
    diversity = np.clip(diversity, 0.0, 1.0)
    
    # Adaptive temperature: higher when diversity is low
    # Low diversity (converging) -> high temp (force diversity pressure)
    # High diversity (exploring) -> low temp (allow elitism)
    if not hasattr(self, 'selection_temp'):
        self.selection_temp = 1.0
    target_temp = 0.1 + 2.0 * (1.0 - diversity)
    self.selection_temp = 0.9 * self.selection_temp + 0.1 * target_temp
    self.selection_temp = np.clip(self.selection_temp, 0.05, 5.0)
    
    # Fitness rank (lower fitness = lower rank number = better)
    fitness_ranks = np.argsort(np.argsort(combined_fit))
    
    # Distance from centroid rank (more distant = lower rank = preferred for diversity)
    centroid = np.mean(combined_pop, axis=0)
    distances = np.linalg.norm(combined_pop - centroid, axis=1)
    dist_ranks = np.argsort(np.argsort(-distances))  # Negative so far=low rank
    
    # Combined score: weighted average of ranks (both normalized to [0,1])
    fitness_norm = fitness_ranks / max(n_combined - 1, 1)
    dist_norm = dist_ranks / max(n_combined - 1, 1)
    combined_score = 0.6 * fitness_norm + 0.4 * dist_norm
    
    # Softmax with temperature for probabilistic selection
    exp_scores = np.exp(-combined_score / (self.selection_temp + 1e-10))
    probs = exp_scores / (np.sum(exp_scores) + 1e-10)
    
    # Select NP survivors probabilistically
    selected_indices = np.random.choice(n_combined, size=self.NP, replace=False, p=probs)
    
    self.population = combined_pop[selected_indices]
    self.fitness = combined_fit[selected_indices]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```