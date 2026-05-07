**Idea: Rank-Success + Convergence-Hybrid Adaptation**

Category H: Combine **rank-based success tracking** (how are top-ranked individuals performing?) with **population convergence detection** (how clustered is the population?). Switch between exploration/exploitation modes based on BOTH signals, using a principled decision tree.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Hybrid adaptation: rank-based success + convergence detection."""
    np_pop = len(improved_mask)
    
    # --- Mechanism 1: Rank-weighted success rate ---
    # Fitness ranks within generation (lower rank = better fitness)
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    
    # Get current fitness from population (stored during optimization)
    current_fitness = getattr(self, '_current_fitness', np.ones(np_pop) * np.inf)
    self._fitness_history.append(current_fitness.copy())
    if len(self._fitness_history) > self.adaptation_window:
        self._fitness_history.pop(0)
    
    # Compute rank-weighted improvement
    if len(self._fitness_history) >= 2:
        prev_fitness = self._fitness_history[-2]
        curr_fitness = self._fitness_history[-1]
        rank_improved = np.zeros(np_pop, dtype=bool)
        for i in range(np_pop):
            if prev_fitness[i] < curr_fitness[i]:  # Lower fitness = better
                rank_improved[i] = True
        
        # Weight by fitness rank (top performers matter more)
        sorted_indices = np.argsort(curr_fitness)
        rank_weights = np.zeros(np_pop)
        for rank, idx in enumerate(sorted_indices):
            rank_weights[idx] = 1.0 / (rank + 1)
        rank_weights = rank_weights / np.sum(rank_weights)
        
        rank_success_rate = np.sum(rank_weights[rank_improved])
    else:
        rank_success_rate = 0.5
    
    # --- Mechanism 2: Population convergence detection ---
    # Use spread of fitness values (not raw distances - stays in category D)
    if len(current_fitness) > 1 and np.std(current_fitness) > 1e-10:
        fitness_spread = np.log1p(np.ptp(current_fitness) / (np.mean(current_fitness) + 1e-10))
        fitness_spread = np.clip(fitness_spread, -5, 5)
        converged = fitness_spread < 1.0  # Population clustered in fitness space
    else:
        converged = True
    
    # --- Principled switching logic ---
    if converged:
        # Population clustered: prioritize exploration
        if rank_success_rate < 0.1:
            # Stagnant + converged = aggressive exploration
            self.F = np.clip(self.F * 1.3, 0.1, 2.0)
            self.Cr = np.clip(self.Cr * 0.8, 0.1, 0.9)
        else:
            # Converged but improving: moderate exploration
            self.F = np.clip(self.F * 1.15, 0.1, 2.0)
            self.Cr = np.clip(self.Cr * 0.9, 0.1, 0.9)
    else:
        # Population spread: can afford exploitation
        if rank_success_rate > 0.3:
            # Good top-rank improvement: intensify exploitation
            self.F = np.clip(self.F * 0.85, 0.1, 2.0)
            self.Cr = np.clip(self.Cr * 1.1, 0.1, 0.9)
        elif rank_success_rate < 0.1:
            # Poor improvement despite diversity: shift strategy
            self.F = np.clip(self.F * 1.1, 0.1, 2.0)
            self.Cr = np.clip(self.Cr * 1.05, 0.1, 0.9)
        # else: maintain current settings
```