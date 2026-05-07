Looking at the error patterns, the worst tasks (16-23 with errors 10-600) are failing because CMA-ES is over-adapting its covariance matrix to local optima on these deceptive/multimodal landscapes. The condition number of the covariance matrix grows unboundedly as the algorithm commits to a basin, making it nearly impossible to escape.

**Idea: Condition-Number Dampened Evolution Paths**

Instead of using fixed learning rates, adapt them based on the covariance matrix condition number K = λ_max/λ_min. When K is large (ill-conditioned), the algorithm has committed to a direction and we should explore more broadly by reducing adaptation rates. This prevents premature convergence on deceptive local optima.

```python
def _update_evolution_paths(self, old_mean):
    mean_shift = (self.mean - old_mean) / self.sigma
    transformed = self.invsqrtC @ mean_shift

    c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))

    # Compute condition number of covariance matrix
    if self.D is not None and len(self.D) > 0:
        cond_D = np.max(self.D) / np.maximum(np.min(self.D), 1e-20)
    else:
        cond_D = 1.0

    # Adaptive sigma-path learning rate: lower for ill-conditioned covariance
    c_sigma_adapt = self.c_sigma / (1.0 + np.log1p(np.maximum(cond_D - 1.0, 0.0)))
    c_sigma_adapt = np.clip(c_sigma_adapt, 1e-10, self.c_sigma)

    self.p_sigma = (1.0 - c_sigma_adapt) * self.p_sigma + \
                   np.sqrt(c_sigma_adapt * (2.0 - c_sigma_adapt) * self.mu_eff) * transformed

    p_sigma_norm = np.linalg.norm(self.p_sigma)
    threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
    h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

    # Adaptive covariance-path learning rate
    c_c_adapt = self.c_c / (1.0 + np.log1p(np.maximum(cond_D - 1.0, 0.0)))
    c_c_adapt = np.clip(c_c_adapt, 1e-10, self.c_c)

    self.p_c = (1.0 - c_c_adapt) * self.p_c + \
               h_sigma * np.sqrt(c_c_adapt * (2.0 - c_c_adapt) * self.mu_eff) * mean_shift

    return h_sigma
```