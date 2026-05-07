**Idea: Entropy-Modulated Velocity Scaling**
Category C: Information-theoretic / distributional approach using non-parametric entropy estimation from k-NN distances to modulate velocity scaling. Particles in low-entropy (clustered) regions get exploration boosts; high-entropy (dispersed) regions get exploitation-focused scaling. This is fundamentally different from spectral approaches (Category B) which operate on matrix decompositions rather than distributional entropy signals.

```python
def _position_update_svd_whitening(self):
    """Entropy-modulated velocity scaling (Category C: Information-theoretic).
    
    Uses non-parametric entropy estimation from k-NN distances to characterize
    population distribution. Low entropy = clustered = more exploration needed.
    High entropy = dispersed = exploitation-focused movement.
    """
    if self.np < 3:
        self.population = self._clip_to_bounds(self.population + self.velocity)
        return
    
    # Compute pairwise squared distances
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    distances = np.sqrt(sq_dists)
    
    k = min(5, self.np - 1)
    sorted_dists = np.sort(distances, axis=1)
    knn_dists = sorted_dists[:, :k]  # k-nearest neighbor distances
    
    # Entropy estimation via distance ratios (non-parametric)
    # Using ratio of successive NN distances as probability proxy
    entropy_signal = np.zeros(self.np)
    for i in range(self.np):
        d = knn_dists[i]
        if d[0] > 1e-15 and d[-1] > d[0]:
            ratios = d[1:k] / (d[0] + 1e-15)
            p = ratios / (np.sum(ratios) + 1e-15)
            p = np.clip(p, 1e-15, 1.0)
            entropy_signal[i] = -np.sum(p * np.log(p))
    
    # Normalize entropy to [0, 1] range
    max_entropy = np.log(k)
    entropy_norm = entropy_signal / (max_entropy + 1e-15)
    entropy_norm = np.clip(entropy_norm, 0.0, 1.0)
    
    # Velocity modulation: inverse relationship with entropy
    # Low entropy (clustered) -> high exploration scaling
    # High entropy (dispersed) -> low exploration scaling
    explore_scale = 1.0 + 1.5 * (1.0 - entropy_norm)
    explore_scale = np.clip(explore_scale, 0.5, 2.5)
    
    # Apply scaled velocity update
    scaled_velocity = self.velocity * explore_scale[:, np.newaxis]
    new_population = self.population + scaled_velocity
    
    self.population = self._clip_to_bounds(new_population)
```