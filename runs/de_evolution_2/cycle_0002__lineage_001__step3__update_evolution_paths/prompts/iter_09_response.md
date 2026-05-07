**Idea: Stagnation-Triggered Evolution Path Reset**

This approach adds explicit stagnation detection inside the evolution path update. When the algorithm gets stuck (stagnation_counter exceeds threshold), BOTH evolution paths are reset using the current mean shift direction while preserving the covariance matrix. This gives the algorithm an autonomous "escape local optima" mechanism that standard CMA-ES lacks—standard CMA-ES only resets evolution paths during full restarts, but the paths can become stale mid-run when stuck in local optima (the primary failure mode for the worst unsolved tasks with errors ~1e+0 to ~1e+1).

```python
def _update_evolution_paths(self, old_mean):
    """
    Update the cumulative step-size path (p_sigma) and the covariance path (p_c).
    Includes stagnation-triggered path reset to escape local optima.
    """
    # Detect stagnation and trigger evolution path reset
    if self.stagnation_counter > 20 + self.dim:
        # Reset evolution paths using current mean shift direction
        mean_shift = (self.mean - old_mean) / self.sigma
        # Preserve covariance matrix C but reset paths for fresh direction
        self.p_sigma = mean_shift * 0.1
        self.p_c = mean_shift * 0.1
        self.stagnation_counter = 0  # Reset counter after path reset

    mean_shift = (self.mean - old_mean) / self.sigma
    transformed = self.invsqrtC @ mean_shift

    # Update step-size path
    c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
    self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                   np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

    # Heaviside function for p_c update
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
    h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

    # Update covariance path
    self.p_c = (1.0 - self.c_c) * self.p_c + \
               h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

    return h_sigma
```