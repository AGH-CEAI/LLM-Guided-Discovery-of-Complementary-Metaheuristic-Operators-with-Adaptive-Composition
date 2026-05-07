Looking at the worst unsolved tasks (17, 16, 6, 19, etc.) with errors 10+ decades above target, the optimizer is failing to explore effectively. The current `_adapt_inertia_weight` uses a single point estimate of diversity, which is brittle to outliers and provides no uncertainty information.

**Idea: Bootstrap Confidence Interval Inertia Weight**
Use bootstrap resampling to estimate confidence intervals on swarm diversity, then make stochastic decisions about inertia weight based on interval overlap and width. This provides robust uncertainty quantification that can drive more aggressive exploration on deceptive landscapes.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using bootstrap confidence intervals on diversity.
    
    Category G: Stochastic / sampling-based
    Uses bootstrap resampling to estimate diversity uncertainty, then makes
    probabilistic decisions about exploration vs exploitation.
    """
    pop = self.population
    np_pop = self.np
    
    # Bootstrap resampling for diversity confidence interval
    n_bootstrap = 50
    bootstrap_diversities = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Sample with replacement (stratified: sample indices, use those rows)
        indices = np.random.randint(0, np_pop, size=np_pop)
        sample = pop[indices]
        centroid = np.mean(sample, axis=0)
        distances = np.linalg.norm(sample - centroid, axis=1)
        bootstrap_diversities[b] = np.mean(distances)
    
    # Compute confidence interval statistics
    ci_low = np.percentile(bootstrap_diversities, 10)
    ci_high = np.percentile(bootstrap_diversities, 90)
    ci_width = ci_high - ci_low
    median_diversity = np.median(bootstrap_diversities)
    
    # Use Latin hypercube sampling to probe inertia-diversity relationship
    # Sample 20 random inertia candidates and estimate their effect
    n_probes = 20
    probe_inertias = np.linspace(0.4, 0.95, n_probes)
    # Shuffle to ensure coverage (Latin hypercube property)
    np.random.shuffle(probe_inertias)
    
    # Monte Carlo estimate: expected diversity change per inertia setting
    # Model: higher inertia -> higher diversity (for fixed velocities)
    expected_diversity = np.zeros(n_probes)
    for i, w in enumerate(probe_inertias):
        # Simulate: diversity_next ≈ current_diversity + w * velocity_effect
        # Sample velocity magnitudes to get stochastic estimate
        vel_mags = np.linalg.norm(self.velocity, axis=1)
        vel_effect = np.mean(vel_mags) * w
        # Add noise from bootstrap uncertainty
        noise_std = ci_width / 2
        expected_diversity[i] = median_diversity + vel_effect + np.random.normal(0, noise_std)
    
    # Select inertia that maximizes expected diversity within safe bounds
    # (Monte Carlo selection based on sampled outcomes)
    best_idx = np.argmax(expected_diversity)
    candidate_inertia = probe_inertias[best_idx]
    
    # Stochastic decision based on CI overlap with thresholds
    rng = np.random.random()
    
    if ci_high < self.diversity_threshold_low:
        # Low diversity regime (high confidence): increase inertia
        self.inertia_weight = min(0.95, self.inertia_weight * 1.1)
    elif ci_low > self.diversity_threshold_high:
        # High diversity regime (high confidence): decrease inertia
        self.inertia_weight = max(0.4, self.inertia_weight * 0.9)
    elif ci_width > 5.0:
        # High uncertainty (wide CI): favor exploration
        if rng < 0.7:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        else:
            self.inertia_weight = candidate_inertia
    else:
        # Normal regime: use Monte Carlo-selected candidate with decay
        decay = 0.1 * (self.generation / 1000)
        self.inertia_weight = candidate_inertia - decay
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```