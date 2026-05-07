Looking at the data, I need to design a diversity metric that specifically helps the adaptive selector choose operators capable of tackling the worst unsolved tasks (Tasks 16, 17, 20, 23 with errors ~1e+1 to 5e+1). These catastrophic failures suggest the optimizer is getting trapped in local optima or losing exploration capability entirely.

**Key insight**: The current ESS-based metric is smooth and continuous but doesn't capture extreme population states (complete collapse, severe elongation). I need a more robust diversity metric that:
1. Uses MAX-normalized spread to catch catastrophic collapse (vs. mean-based)
2. Uses percentile-based fitness diversity to detect trapping in local optima
3. Uses condition number directly to detect rank deficiency
4. Uses population spread ratio to detect elongation

**Idea: Percentile-Robust Diversity with Condition Tracking**
This uses max-normalized metrics and percentile-based fitness diversity instead of geometric-mean-based metrics, making it more sensitive to extreme states that cause the optimizer to fail on multimodal tasks.

```python
def _compute_diversity(self):
    """Compute diversity using percentile-robust and condition-aware metrics.

    Key differences from ESS-based approach:
    - Uses MAX-normalized spread (catches catastrophic collapse better)
    - Uses fitness percentile range (detects local optima trapping)
    - Uses condition number directly (detects rank deficiency)
    - Uses population spread ratio (detects severe elongation)

    This makes the metric more sensitive to extreme states that cause
    catastrophic failure on multimodal/ill-conditioned tasks.
    """
    # Max-normalized spread: captures catastrophic collapse better than trace
    pop_diffs = self.population - self.mean
    max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
    expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
    spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
    spread_norm = np.clip(spread_norm, 0.0, 2.0)

    # Fitness percentile range: detects local optima trapping
    sorted_fit = np.sort(self.fitness)
    fit_range = sorted_fit[-1] - sorted_fit[0]
    fit_median = sorted_fit[len(sorted_fit) // 2]
    fit_p10 = sorted_fit[max(0, len(sorted_fit) // 10 - 1)]
    fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
    fit_percentile_range = np.clip(fit_percentile_range, 0.0, 1.0)

    # Condition number: directly detects rank deficiency
    eigvals = np.linalg.eigvalsh(self.C)
    eigvals = np.clip(eigvals, 1e-15, None)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-15)
    cond_norm = 1.0 / np.log1p(cond)
    cond_norm = np.clip(cond_norm, 0.0, 1.0)

    # Spread ratio: detects severe elongation (ratio of max to min axis lengths)
    diag = np.diag(self.C)
    diag = np.maximum(diag, 1e-15)
    max_var = np.max(diag)
    min_var = np.min(diag)
    spread_ratio = min_var / max_var
    spread_ratio = np.clip(spread_ratio, 0.0, 1.0)

    # Weighted combination: emphasize condition and spread ratio for ill-conditioned tasks
    diversity = (
        0.25 * spread_norm +
        0.35 * fit_percentile_range +
        0.25 * cond_norm +
        0.15 * spread_ratio
    )

    return float(np.clip(diversity, 1e-15, None))
```