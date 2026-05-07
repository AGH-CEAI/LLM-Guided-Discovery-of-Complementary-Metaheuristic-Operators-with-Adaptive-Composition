**Idea: Hybrid Multi-Origin Sampling with Best-Solution Perturbation**

A hybrid sampling strategy that mixes three origins: (1) CMA-ES local sampling around current mean for exploitation on easy tasks, (2) perturbation around the global best solution with larger step size to escape local basins on tasks 16-23, and (3) uniform random sampling for global exploration. This directly addresses the failure mode where all variants get stuck at identical errors (e.g., Task 17 at 3.1201e+00) — the algorithm is trapped in a basin and needs injection of diversity from the best-known solution to break out.

```python
def _sample_population_batch(self):
    """Sample using hybrid multi-origin strategy: CMA-ES, best-perturb, and random."""
    dim = self.dim
    lam = self.pop_size
    
    # Compute eigendecomposition-based sampling matrix once
    sqrt_eig = np.sqrt(self.eigenvalues + 1e-20)
    sampling_matrix = self.eigenvectors @ (sqrt_eig[:, None] * np.eye(dim))  # B @ D
    
    # Allocate population and z_vectors
    population = np.empty((lam, dim))
    z_vectors = np.empty((lam, dim))
    
    # Portion 1: Standard CMA-ES sampling around current mean (exploitation)
    n_cma = lam // 3
    if n_cma > 0:
        z_cma = np.random.randn(n_cma, dim)
        y_cma = sampling_matrix @ z_cma.T  # (dim, n_cma)
        population[:n_cma] = self.mean[None, :] + self.sigma * y_cma.T
        z_vectors[:n_cma] = z_cma
    
    # Portion 2: Perturbation around global best solution (escape local basins)
    n_best = lam // 3
    if n_best > 0:
        if self.best_x is not None:
            # Larger step for escaping: 3x sigma or 10% of range, whichever is larger
            best_step = max(self.sigma * 3.0, 0.1 * (self.ub - self.lb))
            z_best = np.random.randn(n_best, dim)
            # Rotate by eigenvectors for anisotropic exploration around best
            y_best = sampling_matrix @ z_best.T
            population[n_cma:n_cma + n_best] = self.best_x[None, :] + best_step * y_best.T
            z_vectors[n_cma:n_cma + n_best] = z_best
        else:
            # Fallback to CMA sampling if no best known
            z_fallback = np.random.randn(n_best, dim)
            y_fallback = sampling_matrix @ z_fallback.T
            population[n_cma:n_cma + n_best] = self.mean[None, :] + self.sigma * y_fallback.T
            z_vectors[n_cma:n_cma + n_best] = z_fallback
    
    # Portion 3: Uniform random sampling (global exploration)
    n_random = lam - n_cma - n_best
    if n_random > 0:
        z_random = np.random.randn(n_random, dim)
        population[n_cma + n_best:] = np.random.uniform(self.lb, self.ub, (n_random, dim))
        z_vectors[n_cma + n_best:] = z_random
    
    return population, z_vectors
```