**Idea: Stagnation-Triggered Uniform Diversity Injection**

The worst unsolved tasks (16, 19, 20, 23, 18, 21, 10, 14) have errors stuck at 10^1–10^2, suggesting CMA-ES is getting trapped in local optima with insufficient diversity. The current restart mechanism only triggers after prolonged stagnation, but the sampling itself never injects fresh genetic material during normal operation. This proposal dynamically injects uniform random samples proportional to stagnation severity, breaking out of local basins without waiting for a full restart.

```python
def _sample_population_batch(self):
        """
        Sample a batch of candidates using stagnation-triggered diversity injection.
        When stuck in local optima, replace a fraction of CMA-ES samples with fresh
        uniform samples to escape basins. The injection fraction scales with stagnation.
        """
        pop_size = self.pop_size
        dim = self.dim
        
        # Compute injection rate: proportional to stagnation, capped at 50%
        stagnation_fraction = min(self.stagnation_counter / max(1, 20 + dim), 1.0)
        injection_rate = 0.05 + 0.45 * stagnation_fraction  # 5% baseline, up to 50% when stuck
        
        n_uniform = int(np.round(pop_size * injection_rate))
        n_cmaes = pop_size - n_uniform
        
        # Generate CMA-ES samples from N(mean, sigma^2 * C)
        z_cmaes = np.random.randn(n_cmaes, dim)
        scaled = z_cmaes * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        cmaes_pop = self.mean[np.newaxis, :] + self.sigma * rotated
        
        # Generate uniform injection samples across bounds
        if n_uniform > 0:
            # Sample uniformly in [-100, 100]^dim
            uniform_pop = np.random.uniform(self.lb, self.ub, size=(n_uniform, dim))
            # Bias a few toward the best-known solution if available
            if self.best_x is not None and n_uniform >= 2:
                bias_count = max(1, n_uniform // 3)
                uniform_pop[:bias_count] = self.best_x + np.random.randn(bias_count, dim) * self.sigma * 0.5
                uniform_pop[:bias_count] = np.clip(uniform_pop[:bias_count], self.lb, self.ub)
            population = np.vstack([cmaes_pop, uniform_pop])
            z_vectors = np.vstack([z_cmaes, np.zeros((n_uniform, dim))])
        else:
            population = cmaes_pop
            z_vectors = z_cmaes
        
        return population, z_vectors
```