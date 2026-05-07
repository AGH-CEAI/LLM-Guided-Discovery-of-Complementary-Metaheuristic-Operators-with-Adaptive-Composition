**Idea: Pairwise Distance Diversity**

Replace eigenvalue-based diversity with direct population metrics: mean pairwise Euclidean distance, fitness variance, and solution-space coverage ratio. This captures actual population spread rather than relying on the covariance matrix's shape, which becomes unreliable when the algorithm struggles on the worst tasks.

```python
def _compute_diversity(self):
        """Compute diversity using direct population metrics.

        Combines three complementary measures:
        1. Normalized mean pairwise distance (actual spatial spread)
        2. Relative fitness variance (selection pressure indicator)
        3. Solution-space coverage ratio (bound utilization)

        This avoids reliance on covariance matrix eigenvalues which become
        unreliable when the algorithm struggles on difficult tasks.
        """
        # 1. Spatial spread: mean pairwise Euclidean distance
        pop = self.population[:len(self.fitness)]
        n = len(pop)
        if n < 2:
            return 1e-15

        # Sample subset for efficiency (O(n^2) is too slow for large populations)
        sample_size = min(50, n)
        indices = np.random.choice(n, sample_size, replace=False)
        pop_sample = pop[indices]

        pairwise_dists = []
        for i in range(sample_size):
            diffs = pop_sample - pop_sample[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            pairwise_dists.extend(dists[i + 1:])

        mean_pairwise_dist = np.mean(pairwise_dists) if pairwise_dists else 0.0
        max_possible_dist = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        spatial_spread = mean_pairwise_dist / max(max_possible_dist, 1e-10)
        spatial_spread = np.clip(spatial_spread, 0.0, 1.0)

        # 2. Fitness diversity: relative variance
        fit_arr = np.asarray(self.fitness).flatten()
        fit_var = np.var(fit_arr)
        fit_range = np.ptp(fit_arr) + 1e-10
        fit_scale = max(abs(self.f_opt), 1.0, fit_range)
        fitness_diversity = np.clip(fit_var / (fit_scale ** 2 + 1e-10), 0.0, 1.0)

        # 3. Coverage: ratio of population span to search space
        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        span = pop_max - pop_min
        search_span = self.ub - self.lb
        coverage = np.mean(span / np.maximum(search_span, 1e-10))
        coverage = np.clip(coverage, 0.0, 1.0)

        # Combine with adaptive weighting
        diversity = (0.5 * spatial_spread + 0.3 * fitness_diversity + 0.2 * coverage)
        return float(np.clip(diversity, 1e-15, None))
```