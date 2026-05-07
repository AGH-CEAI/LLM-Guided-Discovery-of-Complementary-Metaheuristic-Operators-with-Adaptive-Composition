**Idea: Eigenvalue-Weighted Sampling with Stagnation-Triggered Covariance Mixing**

The worst unsolved tasks (16-23) have errors ~10-100+, indicating the optimizer gets trapped in local optima. Current Cholesky sampling produces samples from the current covariance structure, but when the covariance becomes elongated/narrow (high condition number), sampling is confined to a thin subspace and cannot escape. This variant addresses this by: (1) sampling preferentially along smaller eigenvalue directions for diversity, (2) detecting stagnation via eigenvalue spread (cond > 1000), and (3) mixing C with identity to isotropically expand exploration when trapped.

```python
def _sample_trials_batch(self):
    """Sample with eigenvalue-weighted diversity and stagnation-triggered mixing."""
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-8:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)

    # Eigendecomposition for eigenvalue-weighted sampling
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)

    # Detect stagnation: high condition number means elongated covariance
    cond = np.max(eigvals) / np.min(eigvals)
    if cond > 1000:
        # Mix with identity to isotropically expand exploration
        mix = min(0.5, (cond - 1000) / 10000)
        C_sampled = (1 - mix) * self.C + mix * np.mean(eigvals) * np.eye(self.dim)
        eigvals_s, eigvecs = np.linalg.eigh(C_sampled)
        eigvals_s = np.maximum(eigvals_s, 1e-10)
    else:
        eigvals_s, eigvecs = eigvals, eigvecs

    # Eigenvalue-weighted sampling: sample more in smaller eigenvalue directions
    # Use power-law weights: smaller eigvals get higher probability
    weights = 1.0 / (eigvals_s ** 0.25 + 1e-10)
    weights /= np.sum(weights)

    # Sample eigenvalue indices for each individual
    n_sample = max(1, int(0.1 * self.NP))
    sampled_eigen_indices = np.random.choice(self.dim, size=n_sample, p=weights, replace=True)

    # Standard normal samples
    z = np.random.randn(self.NP, self.dim)

    # Apply eigenvalue scaling: scale each direction by sqrt(eigenvalue)
    sqrt_eigvals = np.sqrt(eigvals_s)
    scales = np.ones(self.dim)
    for idx in sampled_eigen_indices:
        scales[idx] *= 1.5
    scales = np.clip(scales, 0.5, 2.0)

    # Transform samples: z * sqrt(eigvals) * scales
    transformed = z * (sqrt_eigvals * scales)

    # Apply to mean
    self.trials = self.mean + self.sigma * (transformed @ eigvecs.T)

    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)

    # Stagnation detection for restarts
    if hasattr(self, 'stagnation_counter'):
        if self.stagnation_counter > self.max_stagnation // 2:
            # Increase sigma to escape local optima
            self.sigma = min(self.sigma * 2.0, 10.0)
            # Reinitialize poorly-performing individuals
            n_bad = min(self.NP // 2, len(self.population))
            if n_bad > 0:
                bad_indices = np.argsort(self.fitness)[-n_bad:]
                for idx in bad_indices:
                    if idx > 0:  # Keep best individual
                        self.population[idx] = np.random.uniform(self.lb, self.ub)
                        self.fitness[idx] = float('inf')
```