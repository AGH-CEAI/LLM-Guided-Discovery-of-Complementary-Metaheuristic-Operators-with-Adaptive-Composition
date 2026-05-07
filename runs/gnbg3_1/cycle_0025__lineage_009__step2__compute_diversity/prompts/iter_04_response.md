**Idea: Hybrid Fitness-Variance + Condition-Number Diversity**

A fundamentally different approach that combines fitness variance (detecting local optima trapping), covariance condition number (detecting elongated ridges), and actual population spread — all weighted by their relative importance for escaping the worst-performing tasks where the algorithm gets stuck at high error values.

```python
def _compute_diversity(self):
    """Compute diversity using hybrid fitness variance, condition number, and spread.

    Unlike the eigenvalue-based ESS approach, this combines:
    1. Fitness variance (relative) — detects population collapse into local optima
    2. Condition number penalty — detects elongated covariance (narrow ridges)
    3. Actual population spread — measures true geometric extent

    This directly addresses the failure mode seen in tasks 17, 16, 23, etc.
    where error > 1e+01 implies the population is trapped in a local optimum
    with low fitness diversity and/or highly elongated covariance.
    """
    # 1. Fitness variance component: detects local optima trapping
    # Low fitness variance = everyone stuck at same fitness = bad
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    rel_var = float(np.clip(rel_var, 1e-10, 1.0))
    fitness_component = np.sqrt(rel_var)

    # 2. Condition number penalty: detects elongated covariance
    # High condition number = narrow ridge = CMA losing rank = bad
    eigvals = np.linalg.eigvalsh(self.C)
    eigvals = np.clip(eigvals, 1e-15, None)
    eig_min = float(np.min(eigvals))
    eig_max = float(np.max(eigvals))
    cond = eig_max / max(eig_min, 1e-15)
    # Penalize high condition number; log-scale for numerical stability
    cond_penalty = 1.0 / np.log1p(cond)
    cond_penalty = float(np.clip(cond_penalty, 1e-10, 1.0))

    # 3. Population spread: actual geometric extent
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    spread = pop_variance / max(expected_var, 1e-10)
    spread = float(np.clip(spread, 1e-10, 1.0))

    # Combine with geometric mean (all factors must be decent)
    diversity = (fitness_component * cond_penalty * spread) ** (1.0 / 3.0)
    return float(np.clip(diversity, 1e-15, None))
```