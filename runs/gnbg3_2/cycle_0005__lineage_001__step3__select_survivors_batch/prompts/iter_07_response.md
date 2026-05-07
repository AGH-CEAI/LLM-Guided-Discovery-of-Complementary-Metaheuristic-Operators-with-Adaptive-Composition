**Idea: Bootstrap Probabilistic Selection with Diversity Sampling**

For the worst unsolved tasks (17, 16, 11, 6), errors are catastrophically large (~1e+02 to 7e+04), indicating the algorithm gets trapped in local optima with insufficient diversity. The current greedy survivor selection is purely deterministic — it immediately replaces all worse performers, accelerating convergence but collapsing diversity. Category G requires fundamentally stochastic computation, so I'll replace deterministic greedy selection with a Monte-Carlo sampling approach: use bootstrap resampling of fitness values to estimate confidence intervals, then make probabilistic survival decisions weighted by both expected improvement and population diversity. This provides a different angle from the temporal dynamics of Category F by using stochastic sampling within each generation rather than tracking across generations.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Bootstrap probabilistic selection with diversity-weighted sampling."""
    np_pop = len(population)
    
    # Bootstrap resampling to estimate uncertainty in fitness estimates
    n_bootstrap = 20
    bootstrap_std = np.std(fitness) + 1e-10
    
    # Monte Carlo sampling: draw bootstrap replications
    fitness_bootstrap = fitness + bootstrap_std * np.random.randn(n_bootstrap, np_pop)
    trial_bootstrap = trial_fitness + bootstrap_std * np.random.randn(n_bootstrap, np_pop)
    
    # Estimate survival probability via Monte Carlo comparison
    trial_better = trial_bootstrap < fitness_bootstrap
    survival_prob = np.mean(trial_better, axis=0)  # Probability trial beats parent
    
    # Adaptive diversity weight: high when population is clustered (low diversity)
    pop_spread = np.std(fitness) / (np.mean(np.abs(fitness)) + 1e-10)
    diversity_weight = np.clip(0.3 * (1.0 - pop_spread), 0.05, 0.5)
    
    # Probabilistic selection: accept with probability based on bootstrap confidence
    # High-probability improvements always accepted; uncertain cases sampled
    accept_prob = survival_prob + diversity_weight * (1.0 - survival_prob)
    accept_rolls = np.random.rand(np_pop)
    improved_mask = accept_rolls < accept_prob
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    return new_population, new_fitness, improved_mask
```