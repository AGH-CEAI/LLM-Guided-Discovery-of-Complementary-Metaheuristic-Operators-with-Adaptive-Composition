**Idea: Adaptive Restart-Guided Hybrid Sampling**

This variant replaces the static Cholesky sampling with a dynamically adaptive hybrid strategy. It monitors fitness improvement rate and stagnation to decide when to inject diversity via restart-triggered uniform sampling, elite-guided mutations, and adaptive sigma scaling. The key mechanism is using the global best solution (x_opt) as a mutation guide for a fraction of trials, combined with non-elite guided perturbations to escape local optima. This fundamentally differs from all prior variants by making the *sampling distribution itself* adaptive rather than just the covariance adaptation.

```python
def _sample_trials_batch(self):
    """Sample using adaptive hybrid strategy with restart-triggered diversity injection."""
    # Track improvement rate for adaptive strategy selection
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.improvement_sum = 0.0
        self.stagnation_sum = 0.0
    
    recent_improvement = max(0.0, self.prev_f_opt - self.f_opt)
    self.prev_f_opt = self.f_opt
    self.improvement_sum = 0.9 * self.improvement_sum + 0.1 * recent_improvement
    self.stagnation_sum = 0.9 * self.stagnation_sum + 0.1 * (1.0 if recent_improvement < 1e-12 else 0.0)
    
    improvement_rate = self.improvement_sum / (self.stagnation_sum + 1e-10)
    stagnation_detected = self.stagnation_sum > 0.7
    
    # Compute adaptive sigma
    fitness_scale = max(abs(self.f_opt), 1.0)
    grad_magnitude = np.log1p(fitness_scale * 1e5)
    adaptive_sigma = self.sigma * (0.5 + 0.5 * np.tanh(grad_magnitude - 5.0))
    adaptive_sigma = np.clip(adaptive_sigma, 1e-10, 10.0)
    
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-8:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
    # Standard CMA-ES sampling with adaptive sigma
    try:
        L = np.linalg.cholesky(self.C)
        z = np.random.randn(self.NP, self.dim)
        trials_std = adaptive_sigma * (z @ L.T)
    except np.linalg.LinAlgError:
        diag_C = np.diag(self.C)
        diag_C = np.maximum(diag_C, 1e-10)
        L = np.diag(np.sqrt(diag_C))
        z = np.random.randn(self.NP, self.dim)
        trials_std = adaptive_sigma * (z @ L.T)
    
    # Strategy selection based on improvement rate
    if stagnation_detected and improvement_rate < 1e-8:
        # Strategy 1: Restart-triggered uniform sampling for diversity
        num_uniform = max(4, self.NP // 6)
        uniform_samples = np.random.uniform(self.lb, self.ub, (num_uniform, self.dim))
        
        # Strategy 2: Elite-guided mutations using x_opt
        num_guided = max(2, self.NP // 8)
        perp_scale = 2.0 * adaptive_sigma * np.sqrt(self.dim)
        z_guided = np.random.randn(num_guided, self.dim)
        perp_vectors = z_guided - np.mean(z_guided, axis=0)
        perp_vectors /= (np.linalg.norm(perp_vectors, axis=1, keepdims=True) + 1e-10)
        guided_samples = self.x_opt + perp_scale * perp_vectors
        
        # Strategy 3: Non-elite guided mutations
        num_nonelite = max(2, self.NP // 10)
        nonelite_idx = np.random.choice(range(min(20, self.NP)), num_nonelite, replace=False)
        nonelite_samples = self.population[nonelite_idx] + adaptive_sigma * np.random.randn(num_nonelite, self.dim) * 3.0
        
        # Combine all strategies
        combined = [self.mean + trials_std, uniform_samples, guided_samples, nonelite_samples]
        self.trials = np.vstack(combined)
    else:
        # Standard sampling with adaptive sigma
        self.trials = self.mean + trials_std
    
    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)
```