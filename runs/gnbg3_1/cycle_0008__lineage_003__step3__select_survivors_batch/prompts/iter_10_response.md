**Idea: Hybrid Elitism with Fitness-Diversity Balancing**

A hybrid selection that deterministically keeps the top 10% of combined population (exploitation) and stochastically selects the remaining 90% using exponential ranking (exploration). This directly addresses the convergence failure on the worst tasks by preventing premature convergence while still maintaining selective pressure.
```python
def _select_survivors_batch(self):
    """Hybrid elitist selection: deterministic top-k + stochastic rank-based remainder."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    # Handle edge cases
    total_size = len(combined_fit)
    if total_size == 0:
        return
    if total_size <= self.NP:
        self.population = combined_pop
        self.fitness = combined_fit
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
        return
    
    # Sort combined population by fitness
    sorted_indices = np.argsort(combined_fit)
    
    # Deterministic elite: top 10% (minimum 1)
    num_elite = max(1, int(np.ceil(0.1 * self.NP)))
    elite_indices = sorted_indices[:num_elite]
    
    # Stochastic selection for remaining slots using exponential ranking
    remaining_slots = self.NP - num_elite
    if remaining_slots > 0 and total_size > num_elite:
        # Compute ranks (1 = best, larger = worse)
        ranks = np.zeros(total_size)
        ranks[sorted_indices] = np.arange(1, total_size + 1)
        
        # Exponential ranking probability: P(i) ∝ exp(-rank(i) / T)
        # Temperature T controls selection pressure
        pop_variance = np.mean(np.var(combined_pop, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # High diversity → lower pressure (higher T), Low diversity → higher pressure (lower T)
        T = 0.05 + 0.5 * diversity
        T = np.clip(T, 0.01, 1.0)
        
        # Compute selection probabilities for non-elite individuals
        non_elite_mask = np.ones(total_size, dtype=bool)
        non_elite_mask[elite_indices] = False
        non_elite_ranks = ranks[non_elite_mask]
        
        if np.any(non_elite_mask):
            probs = np.exp(-non_elite_ranks / T)
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.ones(np.sum(non_elite_mask)) / np.sum(non_elite_mask)
            
            # Sample with replacement for remaining slots
            non_elite_indices = np.where(non_elite_mask)[0]
            sampled_positions = np.random.choice(
                len(non_elite_indices), 
                size=remaining_slots, 
                replace=True, 
                p=probs
            )
            stochastic_indices = non_elite_indices[sampled_positions]
            
            # Combine elite and stochastic selections
            selected_indices = np.concatenate([elite_indices, stochastic_indices])
        else:
            selected_indices = elite_indices
    else:
        selected_indices = sorted_indices[:self.NP]
    
    # Update population and fitness
    self.population = combined_pop[selected_indices]
    self.fitness = combined_fit[selected_indices]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```