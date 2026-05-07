**Idea: Monte Carlo Adaptive Sampling with Diversity-Guided Radius**

This approach uses Monte Carlo sampling to generate multiple candidate positions per particle, with an adaptive sampling radius that responds to velocity magnitude and swarm diversity. Unlike deterministic velocity-based updates, this samples stochastically around each particle's trajectory, introducing exploration noise that can escape local optima—the primary failure mode on the massive-error tasks (5, 17, 16). The radius grows with velocity (exploration phase) and shrinks when diversity is low (convergence phase), and candidates are selected based on proximity to personal best as a heuristic proxy.

```python
def _position_update_batch(self):
    """Update positions using Monte Carlo sampling with adaptive radius."""
    n_samples = 5  # Monte Carlo samples per particle
    
    # Adaptive sampling radius based on velocity magnitude and diversity
    avg_velocity = np.mean(np.abs(self.velocity))
    diversity = self._compute_diversity()
    
    # Scale radius: larger when exploring (high velocity or low diversity)
    base_radius = avg_velocity * 2.0
    diversity_factor = max(1.0, 10.0 / (diversity + 1e-6))
    sample_radius = base_radius * diversity_factor
    sample_radius = np.clip(sample_radius, 0.01, 50.0)
    
    # Generate Monte Carlo samples for all particles
    samples = np.zeros((self.np * n_samples, self.dim))
    for i in range(self.np):
        current_pos = self.population[i]
        perturbations = np.random.normal(0, sample_radius, (n_samples, self.dim))
        samples[i * n_samples:(i + 1) * n_samples] = np.clip(
            current_pos + perturbations,
            self.lower_bound, self.upper_bound
        )
    
    # Select best sample per particle using heuristic (proximity to personal best)
    # This is the stochastic selection step - randomness from Monte Carlo sampling
    best_indices = np.zeros(self.np, dtype=int)
    for i in range(self.np):
        start_idx = i * n_samples
        end_idx = start_idx + n_samples
        pbest = self.personal_best[i]
        
        # Euclidean distance to personal best for each sample
        sample_block = samples[start_idx:end_idx]
        distances = np.linalg.norm(sample_block - pbest, axis=1)
        
        # Select sample closest to personal best (stochastic due to sampling)
        best_indices[i] = start_idx + np.argmin(distances)
    
    self.population = samples[best_indices]
```