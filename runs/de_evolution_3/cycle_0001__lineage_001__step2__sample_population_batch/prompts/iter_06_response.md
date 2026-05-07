**Idea: Stagnation-Triggered Archive-Guided Sampling**
Sample from an archive of elite solutions when stuck in local optima; otherwise use standard CMA-ES sampling with heavy-tailed perturbations for escape.

```python
def _sample_population_batch(self):
    """Sample from CMA-ES distribution with archive-guided diversity for escaping local optima."""
    dim = self.dim
    lam = self.pop_size

    # Maintain archive of recent good solutions
    if not hasattr(self, 'sampling_archive'):
        self.sampling_archive = []
    if len(self.sampling_archive) > 0 and len(self.sampling_archive[0]) != dim:
        self.sampling_archive = []

    # Check stagnation to trigger archive-based sampling
    is_stagnant = self.stagnation_counter > max(3, int(5 * self.dim / self.pop_size))

    # Determine how many to draw from archive
    n_archive = 0
    if is_stagnant and len(self.sampling_archive) >= 5:
        # Draw 40-70% from archive when stagnant
        n_archive = int(lam * np.random.uniform(0.4, 0.7))
        n_archive = min(n_archive, len(self.sampling_archive))

    n_cmaes = lam - n_archive

    population = np.empty((lam, dim))
    z = np.empty((lam, dim))

    # Standard CMA-ES sampling for regular portion
    if n_cmaes > 0:
        z[:n_cmaes] = np.random.randn(n_cmaes, dim)
        # Add heavy-tailed perturbations (t-distribution, df=2) for 20% of samples
        heavy_tail_mask = np.random.rand(n_cmaes) < 0.2
        if np.any(heavy_tail_mask):
            z[:n_cmaes][heavy_tail_mask] = np.random.standard_t(2.0, size=(np.sum(heavy_tail_mask), dim))

        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z[:n_cmaes].T)
        population[:n_cmaes] = self.mean[None, :] + self.sigma * y.T

    # Archive-based sampling for diversity
    if n_archive > 0:
        archive_array = np.array(self.sampling_archive)
        # Sample with replacement from archive
        archive_indices = np.random.randint(0, len(archive_array), size=n_archive)
        sampled_from_archive = archive_array[archive_indices]

        # Add exploration noise scaled to current sigma
        exploration_noise = np.random.randn(n_archive, dim)
        # Also add heavy-tailed component
        heavy_tail_pert = np.random.standard_t(2.0, size=(n_archive, dim))
        combined_noise = 0.7 * exploration_noise + 0.3 * heavy_tail_pert

        # Scale noise by sigma and eigenvalue structure
        sqrt_eig = np.sqrt(self.eigenvalues)
        noise = self.eigenvectors @ (sqrt_eig[:, None] * combined_noise.T)
        population[n_cmaes:] = sampled_from_archive + self.sigma * 2.0 * noise.T

        # Record z vectors for covariance update (use the combined noise)
        z[n_cmaes:] = combined_noise

    return population, z
```