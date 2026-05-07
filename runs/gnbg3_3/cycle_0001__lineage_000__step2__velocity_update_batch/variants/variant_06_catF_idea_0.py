import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Adaptive Topology Particle Swarm Optimizer with DE Mutation (ATPSO-DM).
    
    Combines:
    - Ring topology PSO with adaptive neighborhood size
    - Adaptive inertia weight based on swarm diversity
    - DE/rand/1 mutation injection for exploration
    - Adaptive cognitive/social coefficients
    - Diversity-guided restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        range_width = self.upper_bound - self.lower_bound
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
        
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        # Handle budget exhaustion gracefully
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            # Ring neighborhood indices
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            # Include self in neighborhood comparison
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on swarm diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Low diversity: increase exploration
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            # High diversity: increase exploitation
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            # Normal range: gradual convergence
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Increase neighborhood size for more diversity
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            # Decrease neighborhood size for faster convergence
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        # Adapt cognitive coefficient based on personal best improvement
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        
        # Adapt social coefficient inversely to cognitive
        social = self.social_base * (2.0 - improvement_ratio)
        
        return cognitive, social
    
    def _velocity_update_batch(self):
        """Update velocities using temporal momentum and diversity-drift detection."""
        # Initialize temporal state if not present
        if not hasattr(self, 'velocity_ema'):
            self.velocity_ema = np.zeros((self.np, self.dim))
            self.diversity_history = []
            self.velocity_magnitude_ema = 1.0
            self.prev_diversity = None
            self.ema_alpha = 0.3  # Smoothing factor for temporal tracking

        # Track current velocity magnitude for momentum analysis
        current_vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True) + 1e-10

        # Update exponential moving average of velocity magnitude
        self.velocity_magnitude_ema = (
            self.ema_alpha * np.mean(current_vel_mag) +
            (1 - self.ema_alpha) * self.velocity_magnitude_ema
        )

        # Update EMA of velocity vectors (temporal smoothed velocity)
        self.velocity_ema = (
            self.ema_alpha * self.velocity +
            (1 - self.ema_alpha) * self.velocity_ema
        )

        # Compute diversity and track its rate of change (temporal signal)
        current_diversity = self._compute_diversity()
        self.diversity_history.append(current_diversity)
        if len(self.diversity_history) > 20:
            self.diversity_history.pop(0)

        # Compute diversity rate of change (diversity drift)
        diversity_drift = 0.0
        if len(self.diversity_history) >= 5:
            recent_window = self.diversity_history[-5:]
            diversity_drift = (recent_window[-1] - recent_window[0]) / (len(recent_window) + 1e-10)

        # Detect velocity momentum (autocorrelation of velocity direction)
        vel_dot_ema = np.sum(self.velocity * self.velocity_ema, axis=1, keepdims=True)
        vel_mag_product = current_vel_mag * (np.linalg.norm(self.velocity_ema, axis=1, keepdims=True) + 1e-10)
        momentum_signal = np.clip(vel_dot_ema / (vel_mag_product + 1e-10), -1, 1)
        avg_momentum = np.mean(momentum_signal)

        # Compute adaptive inertia weight using temporal signals
        base_inertia = self.inertia_weight

        # Diversity-drift correction: positive drift = converging, need more exploration
        if diversity_drift < -1e-4:  # Diversity decreasing rapidly
            inertia_adjustment = 1.15  # Boost exploration
        elif diversity_drift > 1e-4:  # Diversity increasing
            inertia_adjustment = 0.85  # Boost exploitation
        else:
            inertia_adjustment = 1.0

        # Momentum correction: low momentum = oscillating or stuck, need boost
        if avg_momentum < 0.3:
            momentum_correction = 1.2  # Increase exploration when stuck
        elif avg_momentum > 0.8:
            momentum_correction = 0.9  # Reduce when moving consistently
        else:
            momentum_correction = 1.0

        # Apply temporal corrections to inertia
        temporal_inertia = base_inertia * inertia_adjustment * momentum_correction
        temporal_inertia = np.clip(temporal_inertia, 0.2, 1.1)

        # Adaptive cognitive/social coefficients with temporal boost
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)

        # Temporal momentum term: dampen oscillation, maintain direction
        momentum_factor = 0.15 * (1.0 - avg_momentum)
        temporal_momentum = momentum_factor * self.velocity_ema

        # Generate random matrices
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # Cognitive component
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # Social component
        social_component = social * r2 * (self.local_best - self.population)

        # DE mutation with temporal threshold adjustment
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        # Increase mutation when stuck or oscillating
        mutation_boost = 1.0 + max(0, 0.5 - avg_momentum) + min(0.5, self.stagnation_counter / 200)
        mutation_threshold = 0.1 * mutation_boost * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold

        # Generate mutation vectors
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )

        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )

        # Velocity update with temporal momentum
        new_velocity = (
            temporal_inertia * self.velocity +
            cognitive_component +
            social_component +
            temporal_momentum +
            mutation_component
        )

        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
        # Compute population covariance for spectral analysis
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)

        # Eigendecomposition for spectral scaling
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute condition-number-based scaling (stuck populations have high condition number)
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)

            # Project velocity onto principal axes and scale
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale

            # Transform back to original space
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            # Fallback to standard update on decomposition failure
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            # Reinitialize worst 30% of particles
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            # Generate new random particles
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            # Reset fitness for replaced particles
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            # Perturb global best slightly
            if self.global_best is not None:
                perturbation = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + perturbation)
                self.personal_best[0] = self.population[0]
                self.personal_best_fitness[0] = np.inf
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        # Handle truncated initial evaluation
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        # Check initial stopping condition
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Main optimization loop
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            # Check for budget exhaustion
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Update bests
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            
            # Restart check
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
