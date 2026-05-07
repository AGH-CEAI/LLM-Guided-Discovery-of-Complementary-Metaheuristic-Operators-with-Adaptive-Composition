**Idea: Improvement-Magnitude-Weighted Step Size Adaptation**
Uses exponential moving average of per-individual improvement ratios instead of binary success counts, with stagnation-triggered exploration bursts.

```python
def _adapt_step_size(self):
    """Adapt step-size using improvement-magnitude-weighted EMA with stagnation detection."""
    if not hasattr(self.improvement_ema):
        self.improvement_ema = 0.0
    
    improvement_ratios = []
    for i in range(self.NP):
        if i < len(self.trial_fitness) and i < len(self.fitness):
            trial_fit = float(np.asarray(self.trial_fitness[i]).flatten()[0])
            pop_fit = float(np.asarray(self.fitness[i]).flatten()[0])
            
            if pop_fit != 0:
                ratio = (pop_fit - trial_fit) / max(abs(pop_fit), 1e-10)
                improvement_ratios.append(max(-1.0, min(1.0, ratio)))
            elif trial_fit < pop_fit:
                improvement_ratios.append(1.0)
    
    if improvement_ratios:
        current_rate = np.mean(improvement_ratios)
    else:
        current_rate = 0.0
    
    alpha = 0.3 * (1.0 - np.exp(-self.generation / (self.dim + 1)))
    alpha = np.clip(alpha, 0.05, 0.5)
    self.improvement_ema = (1.0 - alpha) * self.improvement_ema + alpha * current_rate
    self.improvement_ema = float(np.clip(self.improvement_ema, -0.5, 1.0))
    
    target_rate = 0.15
    adaptation = (self.improvement_ema - target_rate) / max(target_rate, 1e-10)
    adaptation = np.clip(adaptation, -0.8, 0.8)
    
    damping_adaptive = self.damping * (1.0 + 0.5 * np.log1p(max(1, self.generation)))
    damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)
    
    self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
    self.sigma = np.clip(self.sigma, 1e-12, 50.0)
    
    if abs(self.improvement_ema) < 0.01 and self.generation > 5:
        burst = np.random.uniform(1.5, 4.0)
        self.sigma = np.clip(self.sigma * burst, 1e-10, 50.0)
```

**Why this targets the worst tasks:**
- Tasks 17, 16, 19 (errors ~30-420) show the optimizer is stuck in local optima — the stagnation-triggered sigma bursts directly attack this failure mode
- The improvement-magnitude weighting (vs. binary success) provides finer signal when fitness differences are large, enabling faster convergence
- Generation-dependent alpha allows early aggressive exploration that can escape the local optima traps affecting these high-error tasks