**Idea: Spectral-Fitness Hybrid with Condition-Driven Switching**

Combine Category D (fitness-rank gradient) with Category B (spectral subspace detection) using a principled condition-number-driven switch. For ill-conditioned tasks (17, 16, 6, 11) where population collapses to subspaces, spectral mode activates to escape; for well-conditioned tasks, fitness-gradient mode dominates.

```python
def _position_update_fitness_rank(self):
    """Hybrid: Fitness-rank gradient + spectral-subspace escape with condition-driven switching (Category H).

    Mechanism 1 (Category D): Fitness-rank gradient - particles follow weighted
    attraction toward global best based on their rank. Stronger pull for worse particles.
    
    Mechanism 2 (Category B): Spectral-subspace escape - when condition number
    exceeds threshold, project velocity onto minor eigenvectors to escape collapsed
    subspace. Uses eigenvalue distribution of population covariance.
    
    Switching signal: Condition number of population covariance. High cond >
    threshold → spectral escape mode. Principled because condition number is a
    measurable population property, not a magic constant.
    """
    # --- Compute condition number (principled switching signal) ---
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_log = np.log1p(cond)
        
        # Effective dimensionality for additional signal
        total_var = np.sum(eigenvalues) + 1e-10
        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim
        
        # Spectral entropy as convergence detector
        eigenvalues_norm = eigenvalues / total_var
        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
    except:
        cond = 1.0
        cond_log = 0.0
        eff_dim_ratio = 1.0
        entropy_ratio = 0.5
        eigenvalues = np.ones(self.dim)
        eigenvectors = np.eye(self.dim)
    
    # --- Principled switching: condition-based weight ---
    # Threshold at log(100) ≈ 4.6 - below this, fitness gradient dominates
    # Above this, spectral escape activates progressively
    cond_threshold = 100.0
    spectral_weight = np.clip((cond - cond_threshold) / (cond_threshold * 10), 0.0, 0.8)
    fitness_weight = 1.0 - spectral_weight
    
    # Additional signal: if effective dim is low, population is collapsed
    collapse_signal = 1.0 - eff_dim_ratio
    spectral_weight = np.clip(spectral_weight + 0.2 * collapse_signal, 0.0, 0.8)
    fitness_weight = 1.0 - spectral_weight
    
    # --- MECHANISM 1: Fitness-rank gradient (Category D) ---
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Success history signal
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1e-10)
    
    # Fitness-rank-based attraction to global best
    rank_boost = 1.0 + 0.5 * (1.0 - fitness_ranks) + 0.3 * success_norm
    rank_boost = np.clip(rank_boost, 0.8, 2.0)
    
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        
        # Stronger directional pull for worse-ranked particles
        pull_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        fitness_gradient = pull_strength * to_best_dir * rank_boost[:, np.newaxis]
    else:
        # Random exploration when no global best established
        fitness_gradient = np.random.uniform(-0.5, 0.5, (self.np, self.dim))
    
    # --- MECHANISM 2: Spectral-subspace escape (Category B) ---
    try:
        _, eigenvectors = np.linalg.eigh(cov)
        eigenvectors = eigenvectors[:, np.argsort(np.linalg.eigvalsh(cov))[::-1]]
        
        # Project velocity onto eigenvector basis
        vel_proj = self.velocity @ eigenvectors
        
        # Normalize eigenvalues for scaling
        sv_scale = np.sqrt(eigenvalues + 1e-10)
        sv_norm = sv_scale / (sv_scale[0] + 1e-10)
        
        # Inverse scaling: amplify minor directions, dampen major directions
        # This helps escape collapsed subspaces
        inverse_scale = 1.0 / (sv_norm + 0.1)
        inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)
        
        # Blend with uniform scaling based on condition number
        uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
        per_comp_scale = uniform_scale * (1.0 - spectral_weight) + inverse_scale * spectral_weight
        per_comp_scale = np.clip(per_comp_scale, 0.1, 2.5)
        
        vel_scaled = vel_proj * per_comp_scale
        
        # Spectral escape perturbation: add velocity along minor eigenvectors
        # when condition is high (population collapsed)
        if cond > cond_threshold:
            escape_strength = np.clip(cond_log / 10.0, 0.0, 0.5)
            minor_perturb = np.zeros_like(self.velocity)
            # Add perturbation along bottom 30% of eigenvectors
            n_minor = max(1, int(0.3 * self.dim))
            for i in range(n_minor):
                minor_dir = eigenvectors[:, -(i + 1)]
                minor_perturb += escape_strength * np.outer(
                    np.random.uniform(-1, 1, self.np), minor_dir
                )
            spectral_component = vel_scaled @ eigenvectors.T + minor_perturb
        else:
            spectral_component = vel_scaled @ eigenvectors.T
        
    except np.linalg.LinAlgError:
        spectral_component = self.velocity
        per_comp_scale = np.ones(self.np)
    
    # --- HYBRID COMBINATION with principled weighting ---
    # Combine spectral and fitness-gradient components
    velocity_contribution = (
        fitness_weight * self.velocity * rank_boost[:, np.newaxis] +
        spectral_weight * spectral_component
    )
    
    # Add fitness gradient perturbation
    gradient_perturbation = fitness_weight * fitness_gradient
    
    # Add random exploration scaled by uncertainty (high condition → more exploration)
    explore_scale = 0.5 + 0.3 * np.clip(cond_log / 5.0, 0.0, 1.0)
    random_perturb = explore_scale * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
    
    # Combine all components
    new_population = self.population + velocity_contribution + gradient_perturbation + random_perturb
    self.population = self._clip_to_bounds(new_population)
```