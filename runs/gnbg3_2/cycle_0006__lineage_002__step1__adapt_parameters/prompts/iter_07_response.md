**Idea: Bootstrap Uncertainty-Driven Parameter Adaptation**
One-line description: Uses bootstrap resampling of the success mask to estimate confidence intervals and injects stochastic perturbations proportional to estimation uncertainty, driving exploration on ill-conditioned landscapes.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using bootstrap confidence estimation of success rate."""
    n = len(improved_mask)
    success_rate = np.mean(improved_mask)
    n_success = int(success_rate * n)
    
    # Initialize bootstrap state
    if not hasattr(self, 'bootstrap_means'):
        self.bootstrap_means = []
        self.bootstrap_ci_width = []
        self.sample_buffer = improved_mask.copy().astype(float)
        self.sample_buffer[improved_mask] = 1.0
        self.sample_buffer[~improved_mask] = 0.0
    
    # Bootstrap resampling to estimate uncertainty in success rate
    n_bootstrap = max(50, 3 * n)  # More bootstrap samples for stability
    bootstrap_success_rates = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Resample with replacement (Monte Carlo sampling)
        indices = np.random.randint(0, n, size=n)
        bootstrap_success_rates[b] = np.mean(self.sample_buffer[indices])
    
    # Compute statistics from bootstrap distribution
    bootstrap_mean = np.mean(bootstrap_success_rates)
    bootstrap_std = np.std(bootstrap_success_rates)
    
    # Confidence interval width (proxy for estimation uncertainty)
    ci_lower = np.percentile(bootstrap_success_rates, 5)
    ci_upper = np.percentile(bootstrap_success_rates, 95)
    ci_width = ci_upper - ci_lower
    
    # Store for adaptive tuning
    self.bootstrap_means.append(bootstrap_mean)
    self.bootstrap_ci_width.append(ci_width)
    if len(self.bootstrap_means) > 20:
        self.bootstrap_means.pop(0)
        self.bootstrap_ci_width.pop(0)
    
    # Compute uncertainty trend (rising uncertainty = more exploration needed)
    if len(self.bootstrap_means) >= 5:
        recent_uncertainty = np.mean(self.bootstrap_ci_width[-5:])
        long_term_uncertainty = np.mean(self.bootstrap_ci_width)
        uncertainty_trend = recent_uncertainty / (long_term_uncertainty + 1e-10)
    else:
        uncertainty_trend = 1.0
    
    # Base success rate from bootstrap (more robust than raw)
    robust_success = bootstrap_mean
    
    # Stochastic perturbation magnitude proportional to uncertainty
    # High CI width → high uncertainty → large perturbations for exploration
    exploration_noise_scale = np.clip(ci_width * 2.0, 0.05, 0.4)
    
    # Regime detection via bootstrap statistics
    high_success = robust_success > 0.3
    low_success = robust_success < 0.15
    high_uncertainty = ci_width > 0.25
    rising_uncertainty = uncertainty_trend > 1.2
    
    # Compute stochastic adjustment with uncertainty-driven scaling
    if high_success and not high_uncertainty:
        # Exploitation: high success, low uncertainty → fine-tune toward convergence
        F_adjustment = 1.08
        Cr_adjustment = 1.05
        # Add small stochastic nudge (Monte Carlo perturbation)
        F_noise = np.random.normal(0, 0.02)
        Cr_noise = np.random.normal(0, 0.02)
    elif low_success or high_uncertainty or rising_uncertainty:
        # Exploration: low success OR high uncertainty → inject diversity stochastically
        # Perturbation magnitude scales with bootstrap uncertainty
        F_adjustment = 0.82 - 0.1 * bootstrap_std
        Cr_adjustment = 0.88
        # Large stochastic perturbations proportional to estimation uncertainty
        F_noise = np.random.normal(0, exploration_noise_scale)
        Cr_noise = np.random.normal(0, exploration_noise_scale * 0.5)
    else:
        # Neutral regime: gentle correction with moderate stochasticity
        delta = robust_success - 0.2
        F_adjustment = 1.0 + 0.1 * delta
        Cr_adjustment = 1.0 - 0.05 * delta
        F_noise = np.random.normal(0, exploration_noise_scale * 0.5)
        Cr_noise = np.random.normal(0, exploration_noise_scale * 0.3)
    
    # Apply adjustments with stochastic perturbation
    self.F = np.clip(self.F * F_adjustment + F_noise, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment + Cr_noise, 0.1, 0.9)
    
    # Update sample buffer for next iteration
    self.sample_buffer = improved_mask.copy().astype(float)
    self.sample_buffer[improved_mask] = 1.0
    self.sample_buffer[~improved_mask] = 0.0
```