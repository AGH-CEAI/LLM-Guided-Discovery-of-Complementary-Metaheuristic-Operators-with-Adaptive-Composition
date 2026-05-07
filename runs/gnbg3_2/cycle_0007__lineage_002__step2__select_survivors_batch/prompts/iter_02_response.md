**Idea: Effective-Rank Spectral Selection**
Use eigenvalue-based effective rank (spectral entropy) to modulate selection pressure differently than the condition-number approach — measuring population spread across the full eigenspace rather than just anisotropy ratio.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Survivor selection via spectral effective rank — measures population spread across eigenspace."""
    improved_mask = trial_fitness < fitness
    improvement = fitness - trial_fitness

    # --- Spectral analysis of population ---
    centered = population - np.mean(population, axis=0)
    cov = np.cov(centered.T)

    # Guard against degenerate or invalid covariance
    if cov.shape[0] == 0 or np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        new_population = population.copy()
        new_fitness = fitness.copy()
        accept_mask = trial_fitness < fitness
        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]
        return new_population, new_fitness, accept_mask

    # Eigenvalue decomposition
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-15)  # Ensure positivity for log

    # Normalize eigenvalues to probability distribution over eigenspace
    p = eigvals / np.sum(eigvals)

    # Spectral entropy: measures how uniformly spread the population is across eigendirections
    # Uniform spread (isotropic) → high entropy → high effective_rank
    # Concentrated (anisotropic) → low entropy → low effective_rank
    entropy = -np.sum(p * np.log(p + 1e-15))
    max_entropy = np.log(len(eigvals))
    effective_rank = np.exp(entropy) / len(eigvals)  # Normalized [0, 1]

    # Selection pressure inversely proportional to effective rank
    # High effective_rank (diverse) → low pressure → accept more exploration
    # Low effective_rank (collapsed) → high pressure → accept only best improvements
    selection_pressure = 1.0 - effective_rank  # Maps [0,1] → [1,0]

    # Scaled improvement with spectral-based pressure modulation
    fit_range = np.ptp(fitness)
    fit_range = max(fit_range, 1e-10)
    scaled_improvement = improvement / fit_range

    # Accept threshold: looser when diverse, stricter when collapsed
    base_threshold = 0.01
    accept_threshold = base_threshold * (0.5 + 0.5 * selection_pressure)

    # Accept if improved OR scaled improvement exceeds threshold
    accept_mask = improved_mask | (scaled_improvement > accept_threshold)

    new_population = population.copy()
    new_fitness = fitness.copy()

    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]

    return new_population, new_fitness, accept_mask
```