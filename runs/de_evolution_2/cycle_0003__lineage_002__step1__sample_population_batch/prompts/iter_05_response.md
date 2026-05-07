**Idea: Restart-Mode Diversity Sampling with Condition-Number Trigger**

The worst unsolved tasks (16, 19, 20, 23, 18) all have errors in the 5–500 range, meaning the algorithm gets trapped at poor local optima. The key insight is that `np.max(self.D) / np.min(self.D) > 1e7` already triggers a restart condition, but by then the algorithm has already wasted generations converging to the wrong basin. This variant proactively detects covariance matrix degeneracy (high condition number) and injects uniform diversity BEFORE getting completely stuck, combined with heavy-tailed (t-distribution) sampling during stagnation to escape local basins.

```python
def _sample_population_batch(self):
    """
    Adaptive sampling with three modes:
    1. Restart-mode: 60% uniform + 40% CMA-ES when covariance matrix degenerates
    2. Heavy-tailed: t-distribution sampling when stagnation detected (escape local optima)
    3. Standard CMA-ES sampling otherwise
    """
    # Check for covariance matrix degeneracy
    min_D = np.min(self.D)
    max_D = np.max(self.D)
    condition_number = max_D / min_D if min_D > 1e-20 else np.inf
    
    # Restart-mode: triggered by degenerate covariance OR very small eigenvalues
    # This injects uniform sampling to escape the current basin
    restart_mode = condition_number > 1e5 or min_D < 1e-10
    
    if restart_mode:
        # 60% uniform sampling for diversity, 40% CMA-ES for exploitation
        alpha = 0.4
        pop_uniform = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        pop_cmaes = self.mean[np.newaxis, :] + self.sigma * rotated
        population = alpha * pop_cmaes + (1.0 - alpha) * pop_uniform
    elif self.stagnation_counter > 5:
        # Heavy-tailed sampling: multivariate t-distribution (nu=4) for escaping local optima
        # Generates occasional large jumps by scaling normal samples with sqrt(chi2/nu)
        z = np.random.randn(self.pop_size, self.dim)
        chi2_samples = np.random.chisquare(4, size=self.pop_size)  # nu = 4 degrees of freedom
        t_scales = np.sqrt(4.0 / np.maximum(chi2_samples, 1e-10))  # prevent division by zero
        z_t = z * t_scales[:, np.newaxis]
        scaled = z_t * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
    else:
        # Standard CMA-ES sampling: x = mean + sigma * B * D * z
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
    
    # Numerical robustness: clip to bounds
    population = np.clip(population, self.lb, self.ub)
    
    return population, z
```