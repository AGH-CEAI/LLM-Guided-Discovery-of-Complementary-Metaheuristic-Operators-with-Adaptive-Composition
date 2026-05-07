Looking at the unsolved tasks, I see a clear pattern: the WORST tasks (21, 17, 18, 19 with errors 2.5-4.1) are completely stuck, while tasks near the target (16, 0, 1 with errors ~4e-8 to 8.5e-8) are barely improving. The current step size adaptation relies on the evolution path `p_sigma`, which becomes unreliable when the algorithm is stuck in local optima — it shrinks sigma aggressively, making escape impossible.

**Idea: Fitness-Improvement-Gated Sigma Adaptation**

Instead of using the evolution path (which fails when stuck), directly measure fitness improvement rate. When improvement stalls, increase sigma to escape; when improving, decrease sigma for precision. This directly addresses the "stuck in local optima" failure mode on tasks 21/17/18/19 while enabling finer convergence on near-target tasks.

```python
def _adapt_step_size(self):
    # Track fitness improvement over a sliding window
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    
    self._fitness_history.append(self.best_f)
    max_history = max(5, int(self.pop_size * 0.5))
    if len(self._fitness_history) > max_history:
        self._fitness_history.pop(0)
    
    # Compute improvement rate from history
    history_len = len(self._fitness_history)
    if history_len >= 3:
        recent = self._fitness_history[-min(3, history_len):]
        oldest = self._fitness_history[:min(3, history_len)]
        
        avg_recent = np.mean(recent)
        avg_old = np.mean(oldest)
        
        # Improvement rate: positive = getting better
        if abs(avg_old) > 1e-100:
            improvement_rate = (avg_old - avg_recent) / (abs(avg_old) + 1e-20)
        else:
            improvement_rate = 0.0
        
        # Also check stagnation counter
        is_stuck = self.stagnation_counter > 5
        
        # Gate sigma adaptation based on improvement rate
        if is_stuck or improvement_rate < 1e-6:
            # Algorithm is stuck: increase sigma for exploration
            # Use stagnation severity to scale the increase
            stagnation_factor = min(self.stagnation_counter / 20.0, 2.0)
            sigma_change = 1.0 + 0.3 * stagnation_factor
            self.sigma *= sigma_change
        else:
            # Algorithm is improving: standard CMA-ES adaptation
            p_sigma_norm = np.linalg.norm(self.p_sigma)
            target_norm = self.chi_n
            # Scale adaptation strength by improvement quality
            quality_factor = min(improvement_rate * 10.0, 1.5)
            adaptation = (self.c_sigma / self.d_sigma) * (p_sigma_norm / target_norm - 1.0) * quality_factor
            self.sigma *= np.exp(adaptation)
    else:
        # Not enough history: use standard CMA-ES
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
    
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```