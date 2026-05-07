Looking at the situation:
- **Worst tasks**: 17 (6.4e+04), 16 (4.9e+03), 11 (3.5e+02), 19 (1.0e+02), 18 (5.8e+01)
- **Pattern**: Errors in 10^1 to 10^4 range suggest the greedy selection is getting stuck in local optima, unable to escape
- **Category G requirement**: Must be fundamentally stochastic in computation

**Analysis**: The current greedy selection (`trial_fitness < fitness`) is purely deterministic. For the hardest tasks with large errors, this likely causes premature convergence to local optima. The key insight for Category G is to use **probabilistic acceptance** with **Monte Carlo sampling** — this allows occasional acceptance of worse solutions (controlled exploration) while still preferring improvements.

**Idea: Metropolis-Hastings Selection with Exponential Ranking**

Use two stochastic mechanisms: (1) temperature-scaled acceptance probability for worse solutions (Metropolis criterion), and (2) exponential ranking to convert fitness to selection probabilities for sampling from a mixed pool. This fundamentally stochastic approach can escape local optima that trap greedy selection.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Stochastic selection via temperature-scaled acceptance + exponential ranking sampling."""
    np_pop = len(population)
    
    # Track temperature state for adaptive exploration
    if not hasattr(self, 'selection_temperature'):
        self.selection_temperature = 1.0
    
    # Temperature schedule: start high (exploration), decay toward exploitation
    self.selection_temperature = max(0.01, self.selection_temperature * 0.999)
    
    # Always accept improvements
    improved_mask = trial_fitness < fitness
    
    # For non-improvers: accept with probability based on temperature
    accept_prob = np.ones(np_pop)
    non_improved = ~improved_mask
    if np.any(non_improved):
        # Temperature-scaled acceptance for worse solutions (Metropolis-style)
        delta = (trial_fitness[non_improved] - fitness[non_improved]) / (np.abs(fitness[non_improved]) + 1e-10)
        delta = np.clip(delta, 0, 100)
        accept_prob[non_improved] = np.exp(-delta / (self.selection_temperature + 1e-10))
    
    # Probabilistic acceptance via Monte Carlo sampling
    random_draw = np.random.rand(np_pop)
    stochastic_accept = improved_mask | (random_draw < accept_prob)
    
    # Build mixed pool for exponential ranking sampling
    mixed_pop = np.vstack([population, trials])
    mixed_fit = np.concatenate([fitness, trial_fitness])
    
    # Exponential ranking: convert fitness to selection probabilities
    ranks = np.argsort(np.argsort(mixed_fit))
    probs = np.exp(-0.1 * ranks)
    probs = probs / np.sum(probs)
    
    # Sample survivors from mixed pool using computed probabilities
    selected_indices = np.random.choice(2 * np_pop, size=np_pop, replace=True, p=probs)
    
    new_population = mixed_pop[selected_indices].copy()
    new_fitness = mixed_fit[selected_indices].copy()
    
    return new_population, new_fitness, improved_mask
```