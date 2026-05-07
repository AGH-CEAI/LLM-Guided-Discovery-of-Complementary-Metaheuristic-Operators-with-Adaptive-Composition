**Idea: Inter-Individual Distance Diversity**

Replaces eigenvalue-based diversity with a geometric-mean combination of pairwise population distances and covariance condition number. This is fundamentally different from the eigenvalue ESS approach because it directly measures how far apart individuals are in solution space, which is the actual information needed for restart detection.
```python
def _compute_diversity(self):
        """Compute diversity using inter-individual distances and condition number.

        Uses actual pairwise distances between population members (not eigenvalues)
        combined with condition number to detect:
          - Low spread: population clustered in small region
          - High conditioning: elongated covariance (narrow ridge)

        Diversity = geometric_mean(avg_pairwise_dist, min_pairwise_dist) * sqrt(condition_penalty):
          - avg_pairwise_dist: normalized by expected search space diameter
          - min_pairwise_dist: normalized minimum inter-individual distance
          - condition_penalty: 1/sqrt(cond) caps at 1e-7 to prevent vanishing

        This directly measures population dispersion rather than matrix properties,
        making it more sensitive to premature convergence on deceptive landscapes.
        """
        # Pairwise distances: O(NP^2) but NP is bounded (80-240) and this is only
        # called on stagnation checks, so cost is acceptable
        NP = self.population.shape[0]
        if NP < 2:
            return 1e-15

        # Compute all pairwise Euclidean distances
        # Using broadcasting for efficiency: (NP, dim) - (NP, dim) -> (NP, NP, dim)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))

        # Extract upper triangle (avoid diagonal zeros)
        upper_tri_indices = np.triu_indices(NP, k=1)
        dists = pairwise_dists[upper_tri_indices]

        if len(dists) == 0:
            return 1e-15

        avg_dist = float(np.mean(dists))
        min_dist = float(np.min(dists))

        # Normalize by expected search space diameter
        search_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        expected_dist = search_diameter / 2.0  # Population should span ~half the space

        avg_dist_norm = avg_dist / max(expected_dist, 1e-10)
        min_dist_norm = min_dist / max(expected_dist * 0.1, 1e-10)  # Min should be at least 10% of expected

        # Clip both to [0, 1] range for stable geometric mean
        avg_dist_norm = float(np.clip(avg_dist_norm, 0.0, 1.0))
        min_dist_norm = float(np.clip(min_dist_norm, 0.0, 1.0))

        # Condition number penalty: prevents high condition from giving false high diversity
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        cond = eig_max / max(eig_min, 1e-15)
        cond_penalty = 1.0 / np.sqrt(max(cond, 1.0))
        cond_penalty = float(np.clip(cond_penalty, 1e-7, 1.0))

        # Geometric mean of distance metrics, then combine with condition penalty
        # geometric_mean(a, b) = sqrt(a * b) provides balanced weighting
        dist_geo_mean = np.sqrt(max(avg_dist_norm * min_dist_norm, 1e-30))
        diversity = dist_geo_mean * cond_penalty

        return float(np.clip(diversity, 1e-15, None))
```