**Idea: Momentum-Corrected Step Size with Diversity Injection**

This approach replaces the simple success-rate count with a momentum-based adaptation that tracks improvement trends over generations, uses exponential moving averages of relative improvement, and injects step-size boosts when diversity collapses or improvement stalls—directly targeting the 10-100 error range on the worst tasks that indicate premature convergence to local optima.

```python
def _adapt_step_size(self):
    """Adapt step-size using momentum-tracked improvement with diversity injection."""
    if len(self.trial_fitness) < self.NP or len(self.fitness) < self.NP:
        self.sigma = np.clip(self.sigma * 1.01, 1e-10, 10.0)
        return

    # Compute per-individual improvements
    improvements = np.array([
        self.fitness[i] - self.trial_fitness[i]
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness)))
    ])

    # Relative improvement: how much better vs current fitness scale
    fit_scale = max(abs(self.f_opt), 1.0, np.std(self.fitness))
    rel_improvements = improvements / fit_scale

    # Exponential moving average of positive relative improvements
    alpha = 0.3
    if not hasattr(self, 'imp_ema'):
        self.imp_ema = 0.0
    self.imp_ema = (1.0 - alpha) * self.imp_ema + alpha * np.mean(rel_improvements)

    # Long-term baseline for comparison
    if not hasattr(self, 'imp_baseline'):
        self.imp_baseline = self.imp_ema
    tau = 0.02
    self.imp_baseline = (1.0 - tau) * self.imp_baseline + tau * self.imp_ema

    # Momentum: positive when improving faster than baseline
    momentum = self.imp_ema - self.imp_baseline

    # Population diversity for escape detection
    pop_std = np.mean(np.std(self.population, axis=0))
    expected_std = (self.ub[0] - self.lb[0]) / 6.0
    diversity_ratio = pop_std / (expected_std + 1e-10)

    # Diversity-based sigma adjustment
    div_weight = np.clip(1.0 - diversity_ratio, 0.0, 1.0)

    # Combined adaptation signal: momentum + diversity pressure
    # Momentum < 0 means slowing improvement -> increase exploration
    adaptation = -momentum * 2.0 + div_weight * 0.5
    adaptation = np.clip(adaptation, -1.0, 1.0)

    # Adaptive damping based on dimension and momentum magnitude
    damping_base = self.damping * (0.5 + 0.5 * np.log1p(self.dim))
    momentum_penalty = 1.0 + abs(momentum) * 5.0
    damping = np.clip(damping_base * momentum_penalty, 0.05, 200.0)

    # Update sigma with momentum-corrected exponential scaling
    self.sigma *= np.exp(adaptation * self.cs / damping)

    # Diversity injection: if population collapsed, boost sigma
    if diversity_ratio < 0.05:
        self.sigma *= 3.0
    elif diversity_ratio < 0.1:
        self.sigma *= 1.5

    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```