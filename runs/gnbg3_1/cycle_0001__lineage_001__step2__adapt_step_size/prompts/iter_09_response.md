**Idea: Multi-Signal Adaptive Step Size with Fitness Landscape Awareness**

This approach replaces the single evolution-path signal with a weighted combination of three signals: (1) the standard evolution path for long-term direction, (2) a diversity-adjusted signal that responds when population collapses, and (3) a fitness-gradient signal that scales step size proportionally to recent improvement rate. It also uses adaptive damping that increases when fitness variance is low (convergence) and decreases when variance is high (exploration needed). This fundamentally differs from the single-signal CSA by explicitly handling multi-modal landscapes where the evolution path alone gets stuck.

```python
def _adapt_step_size(self):
    """Multi-signal adaptive step-size with fitness landscape awareness."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Signal 1: Evolution path (standard CSA)
    self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
    
    # Signal 2: Diversity-adjusted direction
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Signal 3: Fitness gradient
    fit_improvement = max(0.0, self.f_opt_prev - self.f_opt)
    fit_scale = max(abs(self.f_opt), 1.0)
    fit_gradient = fit_improvement / (fit_scale + 1e-10)
    
    # Weighted combination of signals
    w1, w2, w3 = 0.6, 0.25, 0.15
    combined_signal = w1 * np.linalg.norm(self.ps) / np.sqrt(self.dim)
    combined_signal += w2 * (1.0 - diversity) * np.linalg.norm(y_mean) / (np.linalg.norm(y_mean) + 1e-10)
    combined_signal += w3 * min(fit_gradient * 100.0, 2.0)
    
    # Adaptive damping based on fitness variance
    fit_variance = np.var(self.fitness)
    fit_var_normalized = fit_variance / (fit_scale ** 2 + 1e-10)
    adaptive_damping = self.damping * (1.0 + 2.0 * np.exp(-fit_var_normalized * 10.0))
    
    # Step size update with combined signal
    sigma_change = np.exp((combined_signal - 1.0) * self.cs / adaptive_damping)
    self.sigma *= sigma_change
    
    # Adaptive clipping based on convergence stage
    if fit_var_normalized < 0.01:
        sigma_min, sigma_max = 1e-14, 0.1
    elif fit_var_normalized > 1.0:
        sigma_min, sigma_max = 1e-8, 50.0
    else:
        sigma_min, sigma_max = 1e-10, 10.0
    
    self.sigma = np.clip(self.sigma, sigma_min, sigma_max)
```