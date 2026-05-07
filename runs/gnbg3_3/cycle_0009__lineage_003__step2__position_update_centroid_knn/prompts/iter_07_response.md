**Idea: Bootstrap Centroid with Uncertainty-Weighted Attraction**

Monte Carlo centroid estimation via bootstrap resampling. Instead of a single deterministic centroid, I sample multiple bootstrap centroids from the population, compute their empirical covariance to quantify uncertainty, and modulate attraction strength inversely with that uncertainty. Particles in stable regions get precise centroid pull; particles in unstable regions get noisy exploration. This is fundamentally stochastic (bootstrap is inherently random) and can escape the collapse patterns plaguing the worst tasks (17, 16, 6, 11).

```python
def _position_update_centroid_knn(self):
    """Bootstrap centroid + uncertainty-weighted attraction (Category G).
    
    Monte Carlo estimation: sample bootstrap centroids from the population,
    compute their empirical covariance to measure centroid uncertainty, and
    modulate attraction inversely with that uncertainty. Stable regions get
    precise pull; unstable regions get exploratory noise.
    """
    # Number of bootstrap samples for centroid estimation
    n_bootstrap = max(20, self.np // 2)
    
    # Bootstrap resampling: sample population indices with replacement
    bootstrap_centroids = np.zeros((n_bootstrap, self.dim))
    for b in range(n_bootstrap):
        indices = np.random.randint(0, self.np, size=self.np)
        bootstrap_centroids[b] = np.mean(self.population[indices], axis=0)
    
    # Monte Carlo estimate of centroid and its covariance
    mc_centroid = np.mean(bootstrap_centroids, axis=0)
    mc_cov = np.cov(bootstrap_centroids.T) if n_bootstrap > 1 else np.eye(self.dim)
    
    # Uncertainty: trace of bootstrap covariance (higher = more uncertain)
    uncertainty = np.trace(mc_cov) / self.dim + 1e-10
    
    # Clip covariance eigenvalues for numerical stability
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(mc_cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        mc_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    except np.linalg.LinAlgError:
        mc_cov = np.eye(self.dim) * uncertainty
    
    # Direction to MC centroid
    to_centroid_dir = mc_centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    
    # Uncertainty-weighted attraction strength
    # High uncertainty → weaker attraction (more exploration)
    # Low uncertainty → stronger attraction (more exploitation)
    global_spread = np.mean(np.linalg.norm(self.population - mc_centroid, axis=1)) + 1e-10
    dist_to_centroid = np.linalg.norm(self.population - mc_centroid, axis=1, keepdims=True)
    base_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
    
    # Uncertainty modulation: inverse relationship
    uncertainty_factor = np.clip(1.0 / (uncertainty * 10.0 + 1e-10), 0.1, 2.0)
    attraction = base_attraction * uncertainty_factor
    
    # Per-particle Monte Carlo noise from bootstrap covariance
    # Cholesky decomposition for correlated sampling
    try:
        L = np.linalg.cholesky(mc_cov + 1e-6 * np.eye(self.dim))
        mc_noise = np.random.randn(self.np, self.dim) @ L.T
    except np.linalg.LinAlgError:
        mc_noise = np.random.randn(self.np, self.dim) * np.sqrt(uncertainty)
    
    # Fitness-based weighting: better particles get more precise guidance
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    fitness_weight = 1.0 - 0.5 * fitness_ranks[:, np.newaxis]  # [0.5, 1.0]
    
    # Combine: deterministic attraction + MC noise (scaled by uncertainty)
    correction = (
        attraction * to_centroid_dir +
        0.3 * mc_noise * (1.0 - fitness_weight) * np.sqrt(uncertainty)
    )
    
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
    
    self.population = self._clip_to_bounds(new_population)
```