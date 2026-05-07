Looking at the current `_position_update_fitness_rank`, it's actually Category E (graph-based with k-NN and Laplacian eigenvalues). I need to completely redesign it as Category G: Stochastic / sampling-based.

**Analysis of worst unsolved tasks:**
- Tasks 17, 16, 6 have errors >10^2, meaning the algorithm is severely stuck
- This suggests premature convergence to local optima or failure to explore
- Stochastic/sampling approaches can help by injecting controlled randomness and estimating landscape structure statistically

**Category G strategy:**
I'll use Monte-Carlo bootstrap resampling to estimate uncertainty in the gradient direction, combined with Latin hypercube probing to escape local optima. This is fundamentally different from the deterministic spectral/graph approaches.

**Idea: Monte-Carlo Bootstrap Gradient Estimation with LHS Probing**

```python
def _position_update_fitness_rank(self):
    """Monte-Carlo Bootstrap Gradient Estimation with LHS Probing (Category G).

    Key insight: Use bootstrap resampling to estimate uncertainty in the
    gradient direction, and use Latin hypercube sampling (LHS) to generate
    structured random exploration directions. This provides statistically
    robust velocity modulation that adapts to fitness landscape uncertainty.
    """
    # --- Bootstrap Gradient Estimation ---
    n_bootstrap = min(20, self.np)
    bootstrap_gradients = np.zeros((n_bootstrap, self.dim))
    
    for b in range(n_bootstrap):
        # Sample with replacement (bootstrap)
        indices = np.random.choice(self.np, size=self.np, replace=True)
        boot_pop = self.population[indices]
        boot_fit = self.current_fitness[indices]
        
        # Estimate local gradient via weighted least squares
        if self.global_best is not None:
            to_best = self.global_best - boot_pop
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            direction = to_best / to_best_norm
            # Weighted by inverse fitness rank
            weights = 1.0 / (np.argsort(np.argsort(boot_fit)) + 1.0)
            weights = weights / (np.sum(weights) + 1e-10)
            # Gradient estimate: weighted correlation between direction and fitness change
            fitness_change = boot_fit - np.min(boot_fit) + 1e-10
            grad_estimate = np.sum(weights[:, np.newaxis] * direction * fitness_change[:, np.newaxis], axis=0)
            bootstrap_gradients[b] = grad_estimate
        else:
            bootstrap_gradients[b] = np.random.randn(self.dim) * 0.1
    
    # --- Uncertainty Quantification ---
    grad_mean = np.mean(bootstrap_gradients, axis=0)
    grad_std = np.std(bootstrap_gradients, axis=0) + 1e-10
    
    # Coefficient of variation: high uncertainty → more exploration
    cv = grad_std / (np.linalg.norm(grad_mean) + 1e-10)
    uncertainty_factor = np.clip(cv, 0.0, 2.0)
    
    # --- Latin Hypercube Sampling for Exploration ---
    dim = self.dim
    n_lhs_samples = min(10, self.np)
    try:
        # Generate LHS samples in unit hypercube
        samples = np.zeros((n_lhs_samples, dim))
        for d in range(dim):
            samples[:, d] = np.random.uniform(0, 1, n_lhs_samples)
            samples[:, d] = (samples[:, d] + np.arange(n_lhs_samples)) / n_lhs_samples
            np.random.shuffle(samples[:, d])
        
        # Scale to search bounds
        lhs_directions = (samples - 0.5) * 2.0 * (self.upper_bound - self.lower_bound) * 0.1
        
        # Evaluate LHS directions (estimate only, no actual func call)
        # Use distance-weighted combination with current velocity
        random_perturb = np.mean(lhs_directions, axis=0)
    except:
        random_perturb = np.random.randn(dim) * 0.5
    
    # --- Monte-Carlo Velocity Modulation ---
    # Fitness rank signal (from Category D)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Success history
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)
    
    # Bootstrap-based velocity scaling
    # High uncertainty → increase exploration; low uncertainty → intensify exploitation
    explore_scale = 1.0 + 0.5 * uncertainty_factor
    exploit_scale = 1.0 - 0.3 * (1.0 - fitness_ranks)
    
    # Combine with success signal
    vel_scale = explore_scale * (1.0 + 0.3 * success_norm) * exploit_scale
    vel_scale = np.clip(vel_scale, 0.3, 2.5)
    
    # Directional component from bootstrap gradient mean
    grad_dir = grad_mean / (np.linalg.norm(grad_mean) + 1e-10)
    directional = 0.3 * (1.0 - fitness_ranks[:, np.newaxis]) * grad_dir
    
    # Bootstrap confidence-weighted random perturbation
    # High uncertainty → larger perturbations
    confidence = 1.0 / (1.0 + uncertainty_factor)
    random_perturb_scaled = random_perturb * confidence * vel_scale[:, np.newaxis]
    
    # Final position update
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb_scaled
    self.population = self._clip_to_bounds(new_population)
```