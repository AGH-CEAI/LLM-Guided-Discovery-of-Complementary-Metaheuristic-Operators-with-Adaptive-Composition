**Idea: Nearest-Neighbor Average Distance**
Uses mean distance-to-nearest-neighbor as the primary diversity metric, which directly measures population clustering and is more sensitive to premature convergence than eigenvalue-based metrics.

```python
def _compute_diversity(self):
        """Compute diversity using nearest-neighbor average distance.

        Primary metric: mean distance from each individual to its nearest neighbor,
        normalized by expected search-space diameter. This directly measures population
        clustering (collapsed = low diversity) without eigenvalue computation overhead.

        Secondary metric: condition number of covariance matrix captures elongation.
        Combined metric = NN_diversity * log_robust(condition) balances:
          - Population spread (NN distance - sensitive to local optima trapping)
          - Covariance shape (condition number - detects narrow ridges)
        """
        # Primary: nearest-neighbor distance (O(N*k) with k=5 neighbors)
        k_neighbors = min(5, self.NP - 1)
        if k_neighbors < 1:
            return float(np.clip(np.mean(np.std(self.population, axis=0)), 1e-15, None))
        
        nn_distances = np.zeros(self.NP)
        for i in range(self.NP):
            diffs = self.population - self.population[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf  # Exclude self
            nn_distances[i] = np.partition(dists, k_neighbors - 1)[k_neighbors - 1]
        
        mean_nn_dist = np.mean(nn_distances)
        expected_diameter = np.linalg.norm(self.ub - self.lb)
        nn_normalized = mean_nn_dist / max(expected_diameter, 1e-10)
        
        # Secondary: condition number from covariance eigenvalues
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_max = np.max(eigvals)
        eig_min = np.min(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        
        # Combine: NN spread * log-scaled condition penalty
        # log(cond) -> 0 when cond=1 (spherical), grows with elongation
        cond_penalty = 1.0 / np.log1p(max(cond, 1.0))
        diversity = nn_normalized * cond_penalty
        
        return float(np.clip(diversity, 1e-15, None))
```