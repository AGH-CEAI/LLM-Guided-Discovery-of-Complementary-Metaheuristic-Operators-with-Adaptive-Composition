**Idea: Random Projection Bootstrap Inertia**

Uses random orthogonal projections to sample population spread in multiple random directions, then bootstrap-resamples the directional spreads to estimate a confidence interval. The bootstrap confidence interval width and mean drive stochastic inertia adaptation — high uncertainty (wide CI) calls for exploration (higher inertia), while tight consensus (narrow CI) signals reliable convergence (lower inertia). This is fundamentally stochastic: random projection matrices and bootstrap resampling are the core estimation mechanisms.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight via random projection bootstrap of directional spreads."""
    n_proj = min(30, self.dim)
    n_bootstrap = 50
    
    # Generate random projection directions using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(self.dim, n_proj))
    projections = Q / (np.linalg.norm(Q, axis=0) + 1e-10)  # normalized
    
    # Project population onto each direction and compute spread
    proj_spreads = np.zeros(n_proj)
    for d in range(n_proj):
        projected = self.population @ projections[:, d]
        proj_spreads[d] = np.max(projected) - np.min(projected) + 1e-10
    
    # Bootstrap resampling of directional spreads
    bootstrap_means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        indices = np.random.randint(0, n_proj, size=n_proj)
        bootstrap_means[b] = np.mean(proj_spreads[indices])
    
    # Bootstrap statistics
    spread_mean = np.mean(bootstrap_means)
    spread_std = np.std(bootstrap_means)
    spread_ci_width = np.percentile(bootstrap_means, 97.5) - np.percentile(bootstrap_means, 2.5)
    
    # Normalize using running statistics
    if not hasattr(self, '_spread_stats'):
        self._spread_stats = {'mean': spread_mean, 'std': 0.1 * spread_mean}
    alpha = 0.1
    self._spread_stats['mean'] = (1 - alpha) * self._spread_stats['mean'] + alpha * spread_mean
    self._spread_stats['std'] = (1 - alpha) * self._spread_stats['std'] + alpha * spread_std
    
    # Z-score normalized spread
    z_spread = (spread_mean - self._spread_stats['mean']) / (self._spread_stats['std'] + 1e-10)
    z_spread_clipped = np.clip(z_spread, -2.0, 2.0)
    
    # CI width as uncertainty signal (landscape complexity)
    ci_normalized = np.clip(spread_ci_width / (spread_mean + 1e-10), 0.0, 3.0)
    
    # Target inertia: exploration when uncertain, exploitation when certain
    uncertainty_bonus = 0.2 * ci_normalized
    spread_penalty = 0.1 * z_spread_clipped
    
    target_inertia = 0.729 + uncertainty_bonus - spread_penalty
    decay = -0.01 * (self.generation / 1000)
    target_inertia = np.clip(target_inertia + decay, 0.4, 0.95)
    
    # EMA update
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```