**Idea: Heavy-Tailed t-Distribution Sampling with Diversity Injection**

The worst unsolved tasks (16: 32.6, 19: 11.9, 23: 5.1, etc.) are stuck at multi-decade errors, indicating CMA-ES converges to distant local optima. Standard Gaussian sampling in the current method has light tails, limiting escape from poor basins. This variant uses heavy-tailed t-distribution perturbations and explicit diversity injection on stagnation to break through this barrier.

```python
def _sample_population_batch(self):
    """Sample using t-distribution mutations with diversity injection on stagnation."""
    dim = self.dim
    lam = self.pop_size

    # Base sampling: standard normal from eigendecomposed distribution
    z_base = np.random.randn(lam, dim)
    sqrt_eig = np.sqrt(self.eigenvalues)
    y_base = self.eigenvectors @ (sqrt_eig[:, None] * z_base.T)
    population_base = self.mean[None, :] + self.sigma * y_base.T

    # Heavy-tailed perturbation using t-distribution
    # t-distribution has heavier tails than Gaussian, enabling escape from local optima
    df = 4.0  # degrees of freedom — heavier tails than Gaussian (df→∞ approaches Gaussian)
    z_heavy = np.random.standard_t(df, size=(lam, dim))

    # Scale perturbation by step size and a fraction of problem scale
    problem_scale = (self.ub - self.lb) * 0.1
    perturbation = z_heavy * self.sigma * problem_scale * 0.05

    # Combine base sample with heavy-tailed perturbation
    population = population_base + perturbation

    # Diversity injection on stagnation: replace subset with diverse samples
    if self.stagnation_counter > 0:
        # Fraction to replace increases with stagnation depth
        frac_replace = min(0.5, self.stagnation_counter * 0.05)
        n_replace = max(1, int(frac_replace * lam))

        # Generate diverse samples: spread across search space
        diverse_samples = np.random.uniform(
            self.lb * 0.5, self.ub * 0.5, size=(n_replace, dim)
        )

        # Bias some diverse samples toward known good regions
        if self.best_x is not None:
            n_bias = n_replace // 2
            bias_samples = self.best_x + np.random.randn(n_bias, dim) * self.sigma * 2.0
            bias_samples = np.clip(bias_samples, self.lb, self.ub)
            diverse_samples[:n_bias] = bias_samples

        # Inject diverse samples (replace worst positions in batch)
        replace_indices = np.random.choice(lam, n_replace, replace=False)
        population[replace_indices] = diverse_samples

    # Clip to bounds
    population = np.clip(population, self.lb, self.ub)

    # Return base z for covariance update (not the heavy-tailed variant)
    return population, z_base
```