Looking at the benchmark results, all 24 tasks remain unsolved with errors ranging from ~1e-04 to ~2.5e+02. The worst tasks (17, 16, 20, 18, 12, 10, 23, 11) have errors stuck at 1e+0 to 2.5e+02, indicating the optimizer is making progress but getting stuck—likely due to step-size adaptation that's either too conservative or oscillating.

## Analysis of the Problem

The current CSA (Cumulative Step-size Adaptation) approach relies on a single evolution path that can be noisy and slow to respond to sudden changes in the fitness landscape. Tasks with large errors likely have:
- **Multi-modal landscapes** with deceptive local optima
- **Narrow basins** requiring fine-grained exploration
- **Plateaus** where standard evolution path signals become unreliable

## Fundamental Strategy Change

Instead of relying on the evolution path length to indirectly infer step-size quality, I'll use a **success-rate and improvement-magnitude based adaptation** that:
1. Directly tracks whether recent steps improved fitness
2. Weighs improvement by magnitude, not just success/failure
3. Uses a sliding window for faster adaptation to landscape changes
4. Employs power dampening to prevent oscillations

This is fundamentally different from CSA's evolution path approach—it's direct, responsive, and explicitly tuned for aggressive exploration when far from optimum.

**Idea: Success-Rate Step-Size Adaptation**
Direct success-rate and improvement-magnitude based step-size control with sliding window history.
```python
def _adapt_step_size(self):
    """Adapt step-size using success-rate and improvement magnitude (replaces CSA)."""
    if not hasattr(self, 'step_history'):
        self.step_history = {'success': [], 'improvement': []}
    
    improvement = self.f_opt_prev - self.f_opt
    success = improvement > 0
    
    self.step_history['success'].append(success)
    self.step_history['improvement'].append(max(0.0, improvement))
    
    if len(self.step_history['success']) > self.window_size:
        self.step_history['success'].pop(0)
        self.step_history['improvement'].pop(0)
    
    n = len(self.step_history['success'])
    if n >= 3:
        success_rate = sum(self.step_history['success']) / n
        avg_improvement = np.mean(self.step_history['improvement'])
        
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_improvement = avg_improvement / fit_scale
        
        target_rate = 0.25
        rate_ratio = success_rate / target_rate
        rate_ratio = np.clip(rate_ratio, 0.1, 4.0)
        
        adjustment = (rate_ratio ** 0.7) * (1.0 + 2.0 * min(rel_improvement, 0.1))
        
        self.sigma *= adjustment
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    self.f_opt_prev = self.f_opt
```