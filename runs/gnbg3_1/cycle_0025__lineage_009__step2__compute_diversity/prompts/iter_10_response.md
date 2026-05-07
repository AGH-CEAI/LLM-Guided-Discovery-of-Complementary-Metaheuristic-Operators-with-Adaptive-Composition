**Idea: Condition-Number Penalized Fitness-Aware Diversity**

A fundamentally different diversity metric that combines condition number penalization (for elongated/ill-conditioned covariances), fitness spread, and a logarithmic scale for better sensitivity across multiple orders of magnitude. This targets the worst tasks where the optimizer gets stuck in narrow ridges or deceptive local optima.

```python
def _compute_diversity(self):
    """Compute diversity using condition-number penalization and fitness-aware scaling.

    Combines four signals:
    1. Spread ratio: population variance normalized by expected variance
    2. Condition penalty: exp(-sqrt(cond-1)/dim) penalizes ill-conditioned C
    3. Eigenvalue concentration: measures how concentrated eigenvalues are
    4. Fitness diversity: relative variance of fitness values

    Uses logarithmic scaling for better sensitivity across orders of magnitude.
    Designed to trigger restarts more aggressively on deceptive/rugged landscapes.
    """
    # Signal 1: Spread ratio
    diag = np.diag(self.C)
    trace = np.sum(diag)
    expected_trace = self.dim * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    trace_norm = trace / max(expected_trace, 1e-10)
    spread_ratio = np.clip(trace_norm, 1e-10, 1e10)

    # Signal 2: Condition number penalty
    eigvals = np.linalg.eigvalsh(self.C)
    eigvals = np.clip(eigvals, 1e-15, None)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-15)
    # Exponential decay penalty: cond=1 → 1.0, cond=100 → ~0.04, cond=1000 → ~0.0006
    cond_penalty = np.exp(-np.sqrt(max(cond - 1.0, 0.0)) / max(np.sqrt(self.dim), 1.0))

    # Signal 3: Eigenvalue concentration (normalized)
    arith_mean = np.mean(eigvals)
    geo_mean = np.exp(np.mean(np.log(eigvals)))
    ess = geo_mean / max(arith_mean, 1e-15)
    # Concentration: ess near 1 is good (spherical), ess near 0 is bad (elongated)
    concentration_factor = np.clip(ess, 1e-10, 1.0)

    # Signal 4: Fitness diversity
    fit_variance = np.var(self.fitness)
    fit_range = np.max(self.fitness) - np.min(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0, np.mean(np.abs(self.fitness)))
    rel_variance = fit_variance / (fit_scale ** 2 + 1e-10)
    rel_range = fit_range / (fit_scale + 1e-10)
    fitness_diversity = np.clip(np.sqrt(rel_variance) + 0.1 * np.log1p(rel_range), 1e-10, 1e10)

    # Combine with logarithmic scaling for multi-order sensitivity
    log_spread = np.log1p(spread_ratio) / 20.0  # Normalize log scale
    log_fitness = np.log1p(fitness_diversity) / 10.0

    # Final diversity: product of spread × condition_penalty × concentration × fitness_diversity
    diversity = (log_spread + 1e-10) * cond_penalty * concentration_factor * (log_fitness + 1e-10)

    # Apply more aggressive minimum threshold for harder tasks
    min_threshold = 1e-10 * (1.0 + np.log1p(self.dim))
    return float(np.clip(diversity, min_threshold, None))
```