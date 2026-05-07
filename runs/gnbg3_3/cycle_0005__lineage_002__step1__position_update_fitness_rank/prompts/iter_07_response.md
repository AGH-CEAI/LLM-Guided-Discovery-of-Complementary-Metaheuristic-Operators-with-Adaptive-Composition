**Idea: Monte Carlo Gradient Estimation**
Category G: Stochastic / sampling-based — Uses Monte Carlo sampling to estimate local fitness gradients at each particle, then modulates velocity using these stochastic gradient estimates and bootstrap-selected best samples.

```python
def _position_update_fitness_rank(self):
    """Monte Carlo gradient estimation with bootstrap sampling (variant_07, Category G).
    
    8 wins needed. Uses MC sampling to probe local fitness landscape around
    each particle, estimates gradient direction, and selects best sample via
    bootstrap. Fundamentally stochastic — different every call.
    """
    n_samples = 8
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    if self.global_best is not None:
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            fdc_corr = 0.0
    else:
        fdc_corr = 0.0
    
    exploration_factor = np.clip(1.0 - fdc_corr, 0.3, 2.0)
    
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)
    
    vel_scale = exploration_factor * (1.0 + 0.3 * success_norm)
    vel_scale = np.clip(vel_scale, 0.5, 2.0)
    
    new_population = np.empty_like(self.population)
    
    for i in range(self.np):
        scale = 0.2 * (self.upper_bound - self.lower_bound) / 100.0
        samples = np.random.randn(n_samples, self.dim)
        samples_norm = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-10
        samples = self.population[i] + scale * (samples / samples_norm)
        samples = np.clip(samples, self.lower_bound, self.upper_bound)
        
        sample_fitness = np.array([self.current_fitness[i]] + [
            self._single_eval(samples[j]) for j in range(n_samples)
        ])
        
        all_fitness = np.concatenate([[self.current_fitness[i]], sample_fitness[1:]])
        bootstrap_best_idx = np.argmin(all_fitness)
        bootstrap_best_fitness = all_fitness[bootstrap_best_idx]
        
        gradients = np.zeros((n_samples, self.dim))
        for j in range(n_samples):
            delta = samples[j] - self.population[i]
            delta_norm = np.linalg.norm(delta) + 1e-10
            f_diff = sample_fitness[j] - self.current_fitness[i]
            gradients[j] = -f_diff * (delta / delta_norm)
        
        gradient_estimate = np.mean(gradients, axis=0)
        grad_mag = np.linalg.norm(gradient_estimate)
        if grad_mag > 1e-10:
            gradient_estimate = gradient_estimate / grad_mag
        else:
            gradient_estimate = np.random.randn(self.dim)
            gradient_estimate = gradient_estimate / (np.linalg.norm(gradient_estimate) + 1e-10)
        
        if bootstrap_best_idx == 0:
            best_direction = gradient_estimate
        else:
            to_best = samples[bootstrap_best_idx - 1] - self.population[i]
            to_best_norm = np.linalg.norm(to_best) + 1e-10
            best_direction = to_best / to_best_norm
        
        alpha = np.random.uniform(0.3, 0.7)
        combined_direction = alpha * gradient_estimate + (1.0 - alpha) * best_direction
        combined_norm = np.linalg.norm(combined_direction) + 1e-10
        combined_direction = combined_direction / combined_norm
        
        move_strength = 0.5 * (1.0 - fitness_ranks[i]) + 0.1
        random_perturb = 0.3 * (1.0 - fitness_ranks[i]) * np.random.randn(self.dim)
        
        new_population[i] = (
            self.population[i] +
            exploration_factor * self.velocity[i] * vel_scale[i] +
            move_strength * combined_direction +
            random_perturb
        )
    
    self.population = self._clip_to_bounds(new_population)
```