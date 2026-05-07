**Idea: Boltzmann Rank-Scaled Selection**
Uses temperature-scaled fitness ranking with deterministic truncation, providing adaptive selection pressure that starts explorative and becomes more selective as optimization progresses.

```python
def _select_survivors_batch(self):
    """Select survivors via Boltzmann temperature-scaled rank selection."""
    # Combine parents and offspring
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    # Get ranks (lower is better)
    num_inds = len(combined_fit)
    ranks = np.argsort(np.argsort(combined_fit))
    
    # Adaptive temperature based on generation and stagnation
    if not hasattr(self):
        base_temp = max(1.0, (self.ub[0] - self.lb[0]) * 0.01)
    else:
        base_temp = max(0.1, (self.ub[0] - self.lb[0]) * 0.01 * np.exp(-self.generation / (self.dim * 10)))
        base_temp = max(base_temp, 0.1)
    
    # Scale fitness by temperature and convert to selection probabilities
    fit_min = np.min(combined_fit)
    fit_range = np.max(combined_fit) - fit_min + 1e-10
    
    # Normalized fitness differences
    norm_fit = (combined_fit - fit_min) / fit_range
    
    # Boltzmann probabilities with temperature
    boltz_weights = np.exp(-norm_fit / base_temp)
    boltz_weights /= np.sum(boltz_weights)
    
    # Stochastic selection based on Boltzmann weights
    cumsum = np.cumsum(boltz_weights)
    rand_vals = np.random.random(self.NP)
    selected_indices = np.searchsorted(cumsum, rand_vals)
    selected_indices = np.clip(selected_indices, 0, num_inds - 1)
    
    # If we need exactly NP (handle edge cases)
    if len(selected_indices) < self.NP:
        # Fallback to deterministic rank-based selection
        sorted_indices = np.argsort(combined_fit)
        selected_indices = sorted_indices[:self.NP]
    else:
        # Ensure no duplicates and exactly NP unique selections
        unique_selected = np.unique(selected_indices)
        if len(unique_selected) < self.NP:
            # Add best unselected individuals
            remaining = self.NP - len(unique_selected)
            all_selected = set(unique_selected)
            sorted_indices = np.argsort(combined_fit)
            for idx in sorted_indices:
                if idx not in all_selected:
                    unique_selected = np.append(unique_selected, idx)
                    remaining -= 1
                    if remaining <= 0:
                        break
            selected_indices = unique_selected
    
    self.population = combined_pop[selected_indices]
    self.fitness = combined_fit[selected_indices]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```