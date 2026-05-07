import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    ADAPTIVE PSO with PROBE-AND-COMMIT position-update dispatch.
    
    Mechanism: #4 (Contextual Bandit / Probe-and-commit)
    The per-task gap table shows 4 tasks with winner/runner-up gap >= 10x
    and Task 5 has a 192,910x gap. Ensemble blending is mathematically
    EXCLUDED — a 50/50 blend of 1 and 100 is 50.5, not 1. The probe-and-commit
    pattern probes each candidate variant on the actual landscape and commits
    to the winner before spending the bulk of the budget. This preserves the
    per-task winner's advantage because the committed variant gets weight 1.0.
    
    Candidates: variant_01 (2 wins), variant_08 (3 wins), variant_10 (18 wins).
    The original.py variant is not included because it never wins by >1x.
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
        
        # Probe configuration
        self._probe_variants = ['variant_01', 'variant_08', 'variant_10']
        self._probe_gens = 3  # Generations per variant in probe phase
        self._committed_variant = None
        self._probe_complete = False
        
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
        """Update velocities with adaptive components and DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)
        
        # DE/rand/1 mutation injection for enhanced exploration
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)  # Decreases over time
        mutation_active = mutation_mask < mutation_threshold
        
        # Generate mutation vectors from random distinct indices
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        # Apply mutation where active
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _position_update_batch(self, variant_id=None):
        """Update positions using the specified or committed variant."""
        if variant_id is None:
            variant_id = self._committed_variant
        
        if variant_id == 'variant_01':
            self._position_update_variant_01()
        elif variant_id == 'variant_08':
            self._position_update_variant_08()
        elif variant_id == 'variant_10':
            self._position_update_variant_10()
        else:
            # Fallback to standard update
            new_population = self.population + self.velocity
            self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_01(self):
        """Variant 01: geometric normalization and centroid guidance."""
        centroid = np.mean(self.population, axis=0)
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        spread = pop_max - pop_min
        
        epsilon = 1e-10
        spread_safe = np.where(spread > epsilon, spread, 1.0)
        normalized_velocity = self.velocity / spread_safe
        
        centroid_dist = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        global_dist = np.linalg.norm(centroid) + epsilon
        
        direction_to_centroid = centroid - self.population
        correction_strength = 0.02 * np.clip(spread.mean() / (global_dist + epsilon), 0.01, 0.1)
        correction = correction_strength * direction_to_centroid
        
        new_population = self.population + normalized_velocity + correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_08(self):
        """Variant 08: stagnation-weighted global-best attraction."""
        stagnation_threshold = 50 + self.dim // 2
        stagnation_factor = min(1.0, self.stagnation_counter / stagnation_threshold)
        
        velocity_component = self.velocity
        
        if self.global_best is not None and stagnation_factor > 0.1:
            direction_to_best = self.global_best - self.population
            best_pull = stagnation_factor * 0.5 * direction_to_best
        else:
            best_pull = 0.0
        
        new_population = self.population + velocity_component + best_pull
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_10(self):
        """Variant 10: eigenvalue-anisotropic velocity modulation."""
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
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
    
    def _trim_arrays(self, new_size):
        """Trim all population arrays to new_size after budget exhaustion."""
        self.np = new_size
        self.population = self.population[:new_size]
        self.velocity = self.velocity[:new_size]
        self.personal_best = self.personal_best[:new_size]
        self.personal_best_fitness = self.personal_best_fitness[:new_size]
        self.local_best = self.local_best[:new_size]
        self.local_best_fitness = self.local_best_fitness[:new_size]
        self.current_fitness = self.current_fitness[:new_size]
    
    def _probe_and_commit(self, func, stopping_condition):
        """
        Phase A (PROBE): Test each variant for a few generations.
        Phase B (COMMIT): Continue with the winner for the rest of the run.
        """
        if self.np < len(self._probe_variants):
            # Too few particles for probing — commit to variant_10 (most frequent winner)
            self._committed_variant = 'variant_10'
            self._probe_complete = True
            return
        
        probe_results = {}
        
        for variant_id in self._probe_variants:
            # Save state before probing this variant
            state_snapshot = {
                'population': self.population.copy(),
                'velocity': self.velocity.copy(),
                'personal_best': self.personal_best.copy(),
                'personal_best_fitness': self.personal_best_fitness.copy(),
                'local_best': self.local_best.copy(),
                'local_best_fitness': self.local_best_fitness.copy(),
                'global_best': self.global_best.copy() if self.global_best is not None else None,
                'global_best_fitness': self.global_best_fitness,
                'stagnation_counter': self.stagnation_counter,
                'generation': self.generation,
                'inertia_weight': self.inertia_weight,
            }
            
            # Run probe for this variant
            best_fitness = self.global_best_fitness if self.global_best is not None else np.min(self.current_fitness)
            
            for _ in range(self._probe_gens):
                if stopping_condition():
                    break
                
                self.generation += 1
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                self._velocity_update_batch()
                self._position_update_batch(variant_id)
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                
                if actual_n < self.np:
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                
                # Track best fitness for this variant
                current_best = self.global_best_fitness if self.global_best is not None else np.min(self.current_fitness)
                if np.isfinite(current_best):
                    best_fitness = min(best_fitness, current_best)
            
            probe_results[variant_id] = best_fitness
            
            # Restore state for next variant
            self.population = state_snapshot['population'].copy()
            self.velocity = state_snapshot['velocity'].copy()
            self.personal_best = state_snapshot['personal_best'].copy()
            self.personal_best_fitness = state_snapshot['personal_best_fitness'].copy()
            self.local_best = state_snapshot['local_best'].copy()
            self.local_best_fitness = state_snapshot['local_best_fitness'].copy()
            if state_snapshot['global_best'] is not None:
                self.global_best = state_snapshot['global_best'].copy()
            self.global_best_fitness = state_snapshot['global_best_fitness']
            self.stagnation_counter = state_snapshot['stagnation_counter']
            self.generation = state_snapshot['generation']
            self.inertia_weight = state_snapshot['inertia_weight']
            
            # Re-evaluate from restored state to get consistent current_fitness
            self.current_fitness, _ = self._evaluate_batch(self.population, func)
        
        # Commit to the best variant
        best_variant = min(probe_results.keys(), key=lambda v: probe_results[v])
        self._committed_variant = best_variant
        self._probe_complete = True
        
        # Re-initialize for main run with committed variant
        self._initialize_population()
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self._trim_arrays(actual_n)
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
    
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
            self._trim_arrays(actual_n)
        
        # Check initial stopping condition
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # PROBE PHASE: Test each variant and commit to the best
        self._probe_and_commit(func, stopping_condition)
        
        # MAIN RUN: Continue with committed variant
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update with committed variant
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            # Check for budget exhaustion
            if actual_n < self.np:
                self._trim_arrays(actual_n)
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
        
        # Final update and restart check
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
