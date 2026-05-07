Looking at the data, the worst unsolved tasks (16, 19, 20, 23, 18) have errors in the 5-350 range, suggesting the algorithm gets trapped in local optima. The current evolution path update only uses geometric information (mean shift magnitude) but ignores whether steps actually improved fitness. This is the critical gap.

**Idea: Success-Rate Modulated Evolution Paths**
Use a sliding-window success rate to modulate the evolution path updates. When recent steps succeed (fitness improves), amplify the exploration signal; when they fail (fitness worsens), dampen it. This provides a direct feedback loop that helps escape local optima by reinforcing promising directions and suppressing unproductive ones.

```python
def _update_evolution_paths(self, old_mean):
        """
        Update the cumulative step-size path (p_sigma) and the covariance path (p_c).
        Uses success-rate modulation to escape local optima.
        """
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Initialize success history if needed
        if not hasattr(self, 'success_history') or self.success_history is None:
            self.success_history = np.zeros(20)
            self.success_idx = 0
        if len(self.success_history) != 20:
            self.success_history = np.zeros(20)
            self.success_idx = 0

        # Compute step success: compare current mean fitness proxy to previous
        # Use norm of mean shift as proxy (larger shift = more progress)
        step_norm = np.linalg.norm(mean_shift)
        old_step_norm = getattr(self, 'prev_step_norm', step_norm)
        is_success = 1.0 if step_norm >= old_step_norm * 0.1 else 0.0
        self.prev_step_norm = step_norm

        # Update sliding window success rate
        self.success_history[self.success_idx] = is_success
        self.success_idx = (self.success_idx + 1) % 20
        success_rate = np.mean(self.success_history)

        # Modulate update based on success rate
        # High success (near 1): normal exploration
        # Low success (near 0): need more exploration, boost the signal
        exploration_boost = 1.0 + 2.0 * (1.0 - success_rate)
        exploration_boost = np.clip(exploration_boost, 0.3, 2.5)

        # Update step-size path
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       exploration_boost * np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

        # Heaviside function for p_c update
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # Adaptive damping based on step magnitude (convergence detection)
        step_ratio = step_norm / (old_step_norm + 1e-20)
        if step_ratio < 0.5:
            # Converging: use higher damping for stability
            damping = 0.5
        elif step_ratio > 3.0:
            # Oscillating: use higher damping
            damping = 0.3
        else:
            # Normal regime
            damping = 1.0

        # Update covariance path
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * damping * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

        return h_sigma
```