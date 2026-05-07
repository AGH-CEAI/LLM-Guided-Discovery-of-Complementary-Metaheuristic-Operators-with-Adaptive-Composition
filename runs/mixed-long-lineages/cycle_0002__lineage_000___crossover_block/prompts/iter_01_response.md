**Idea: Power Mean Crossover**
A geometric-mean inspired crossover that adaptively blends trial vectors with parents based on per-individual success history, providing rotation-invariance and smooth exploration.
```python
def _crossover_block(self, population, trial_population):
    """Op 3: Power-mean crossover with success-guided blending."""
    n_trials = self.NP
    dim = self.dim
    offspring = np.empty((n_trials, dim))
    
    # Track per-individual blending history
    if not hasattr(self, 'cx_blend_history'):
        self.cx_blend_history = np.full(n_trials, 0.5)
    
    # Success-guided blending: individuals that improved recently get more trial influence
    success_weight = np.clip(self.cx_blend_history, 0.1, 0.9)
    
    # Add small per-individual perturbation to avoid stagnation
    blend_perturb = self.rng.uniform(-0.05, 0.05, size=n_trials)
    blend = np.clip(success_weight + blend_perturb, 0.05, 0.95)
    
    # Power mean parameter (p=0 gives geometric mean)
    p = -0.5  # Sub-geometric mean for more exploration
    
    # Compute power mean: (a^p + b^p) / 2)^(1/p)
    # For p < 0, this emphasizes smaller values (trial) when beneficial
    parent_slice = population[:n_trials]
    trial_slice = trial_population
    
    # Add small epsilon for numerical stability
    eps = 1e-10
    parent_safe = np.clip(parent_slice, eps, None)
    trial_safe = np.clip(trial_slice, eps, None)
    
    # Power transformation
    parent_pow = np.power(parent_safe, p)
    trial_pow = np.power(trial_safe, p)
    
    # Blend in power space, then inverse transform
    blended_pow = blend[:, np.newaxis] * trial_pow + (1.0 - blend[:, np.newaxis]) * parent_pow
    power_mean = np.power(blended_pow, 1.0 / p)
    
    # Ensure at least one dimension comes from trial (j_rand guarantee)
    j_rand = self.rng.integers(0, dim, size=n_trials)
    power_mean[np.arange(n_trials), j_rand] = trial_slice[np.arange(n_trials), j_rand]
    
    # Fallback: if power mean produces NaN/Inf, use linear blend
    valid = np.isfinite(power_mean)
    offspring = np.where(valid, power_mean, 
                         blend[:, np.newaxis] * trial_slice + (1.0 - blend[:, np.newaxis]) * parent_slice)
    
    # Update blend history based on improvement
    trial_better = (trial_population < population[:n_trials]).astype(float)
    improve_rate = np.mean(trial_better, axis=1)
    self.cx_blend_history = 0.7 * self.cx_blend_history + 0.3 * improve_rate
    
    # Clip to bounds
    offspring = np.clip(offspring, self.lower, self.upper)
    return offspring
```