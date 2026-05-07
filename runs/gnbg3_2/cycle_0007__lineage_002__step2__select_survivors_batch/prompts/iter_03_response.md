**Idea: Entropy-KL Dual Metric Selection**
Uses histogram entropy of fitness distribution to detect convergence state, combined with KL divergence between trial and population fitness distributions to identify exploratory trials. Category C: Information-theoretic / distributional.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Entropy-based selection with KL divergence for exploratory acceptance."""
    improved_mask = trial_fitness < fitness
    improvement = fitness - trial_fitness

    # === ENTROPY OF CURRENT FITNESS DISTRIBUTION ===
    n_bins = min(20, max(5, len(fitness) // 3))
    hist, bin_edges = np.histogram(fitness, bins=n_bins)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    fit_entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(n_bins)
    entropy_ratio = np.clip(fit_entropy / (max_entropy + 1e-12), 0.02, 0.98)

    # === KL DIVERGENCE: trial fitness vs population fitness ===
    # Fit both to histogram and compute KL(P_trial || P_pop)
    trial_hist, _ = np.histogram(trial_fitness, bins=bin_edges)
    trial_probs = trial_hist / (np.sum(trial_hist) + 1e-12)
    trial_probs = np.where(trial_probs > 1e-10, trial_probs, 1e-10)
    pop_probs = np.where(probs > 1e-10, probs, 1e-10)
    
    # Align bins: probs is over original bins, trial_probs over same bins
    # KL divergence measures how much trial distribution diverges from population
    kl_div = np.sum(trial_probs * np.log(trial_probs / pop_probs + 1e-12))
    kl_div = np.clip(kl_div, 0.0, 5.0)

    # === COMBINED ACCEPTANCE CRITERION ===
    fit_range = np.ptp(fitness)
    fit_range = max(fit_range, 1e-10)
    scaled_improvement = improvement / fit_range

    # Low entropy → population converged → reduce selection pressure
    # High KL → trial explores new fitness region → bonus for exploration
    selection_pressure = 0.02 * entropy_ratio
    exploration_bonus = 0.01 * (kl_div / (1.0 + kl_div))
    
    accept_threshold = selection_pressure + exploration_bonus
    accept_mask = improved_mask | (scaled_improvement > accept_threshold)

    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]

    return new_population, new_fitness, accept_mask
```