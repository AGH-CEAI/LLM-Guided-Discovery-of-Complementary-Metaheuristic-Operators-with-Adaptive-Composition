```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    ADAPTIVE PSO with PROBE-AND-COMMIT strategy selection (mechanism #4 / probe-and-commit).
    
    WHY THIS MECHANISM FITS THE GAP TABLE:
    The per-task gap table is BLEND-HOSTILE: 4 tasks have winner/runner-up gaps >= 10×
    (tasks 1, 3, 5, 7), and task 5 has a 192,910× gap. Ensemble blending is mathematically
    excluded — a 50/50 blend of 1 and 100 is 50.5, not 1. The correct response is a
    dispatch-style mechanism that preserves per-task advantages.
    
    APPROACH:
    - PHASE A (PROBE, ~10% of budget): Run each of 3 candidate position-update strategies
      on a mini-swarm for a few generations. Compute rank-based convergence rates.
    - PHASE B (COMMIT): Use the best-performing strategy for the remainder of the run.
      Include a stagnation-aware fallback to variant_10 (18-win dominant variant) to
      handle cases where probe budget was too small to detect the true best.
    
    Three strategies:
      original:  velocity-based (wins task 21, tied on 15)
      variant_08: stagnation-weighted global-best attraction (wins tasks 0, 2, 15)
      variant_10: eigenvalue-anisotropic velocity modulation (wins 18 tasks)
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
        
        # --- ADAPTIVE MECHANISM STATE ---
        # Strategy selection: 0=original, 8=variant_08, 10=variant_10
        self.selected_strategy = 10
        self.strategy_commit_gen = 0
        
        # Strategy registry for probe phase
        self._strategy_funcs = {
            0: self._position_update_original,
            8: self._position_update_variant_08,
            10: self._position_update_variant_10,
        }
        self._strategy_names = {
            0: 'original', 8: 'variant_08', 10: 'variant_10'
        }
    
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
    
    # =====================================================================
    # POSITION UPDATE STRATEGIES (the three candidates from benchmark table)
    # =====================================================================
    
    def _position_update_original(self):
        """Standard velocity-based position update (original.py)."""
        new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_08(self):
        """Hybrid: velocity + stagnation-weighted global-best attraction (variant_08)."""
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
        """Eigenvalue-anisotropic velocity modulation (variant_10, 18-win dominant)."""
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
    
    def _execute_selected_strategy(self):
        """Execute the currently selected position update strategy."""
        strategy_func = self._strategy_funcs[self.selected_strategy]
        strategy_func()
    
    # =====================================================================
    # PROBE-AND-COMMIT MECHANISM
    # =====================================================================
    
    def _run_probe_phase(self, func, stopping_condition, probe_gens, probe_np):
        """Run probe with all strategies and return results dict {strategy_id: best_fitness}."""
        probe_results = {}
        
        for strategy_id in [0, 8, 10]:
            # Save state
            saved_pop = self.population.copy()
            saved_vel = self.velocity.copy()
            saved_personal_best = self.personal_best.copy()
            saved_personal_fitness = self.personal_best_fitness.copy()
            saved_local_best = self.local_best.copy()
            saved_local_fitness = self.local_best_fitness.copy()
            saved_global_best = self.global_best.copy() if self.global_best is not None else None
            saved_global_fitness = self.global_best_fitness
            saved_gen = self.generation
            saved_stag = self.stagnation_counter
            saved_inertia = self.inertia_weight
            
            # Use smaller probe population
            if probe_np < self.np:
                probe_pop = saved_pop[:probe_np].copy()
                probe_vel = saved_vel[:probe_np].copy()
                probe_pbest = saved_personal_best[:probe_np].copy()
                probe_pfit = saved_personal_fitness[:probe_np].copy()
                probe_lbest = saved_local_best[:probe_np].copy()
                probe_lfit = saved_local_fitness[:probe_np].copy()
                probe_gbest = saved_global_best.copy() if saved_global_best is not None else None
                probe_gfit = saved_global_fitness
            else:
                probe_pop = saved_pop.copy()
                probe_vel = saved_vel.copy()
                probe_pbest = saved_personal_best.copy()
                probe_pfit = saved_personal_fitness.copy()
                probe_lbest = saved_local_best.copy()
                probe_lfit = saved_local_fitness.copy()
                probe_gbest = saved_global_best.copy() if saved_global_best is not None else None
                probe_gfit = saved_global_fitness
            
            probe_best_fitness = probe_gfit if np.isfinite(probe_gfit) else np.inf
            
            # Run mini-optimization with this strategy
            for g in range(probe_gens):
                if stopping_condition():
                    break
                
                self.population = probe_pop.copy()
                self.velocity = probe_vel.copy()
                self.personal_best = probe_pbest.copy()
                self.personal_best_fitness = probe_pfit.copy()
                self.local_best = probe_lbest.copy()
                self.local_best_fitness = probe_lfit.copy()
                self.global_best = probe_gbest.copy() if probe_gbest is not None else None
                self.global_best_fitness = probe_gfit
                self.generation = saved_gen + g
                self.stagnation_counter = 0
                self.inertia_weight = 0.729
                
                # One step
                self._velocity_update_batch()
                strategy_func = self._strategy_funcs[strategy_id]
                strategy_func()
                
                probe_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < len(self.population):
                    break
                
                self.current_fitness = probe_fitness
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                
                if np.isfinite(self.global_best_fitness):
                    probe_best_fitness = min(probe_best_fitness, self.global_best_fitness)
                
                probe_pop = self.population.copy()
                probe_vel = self.velocity.copy()
                probe_pbest = self.personal_best.copy()
                probe_pfit = self.personal_best_fitness.copy()
                probe_lbest = self.local_best.copy()
                probe_lfit = self.local_best_fitness.copy()
                probe_gbest = self.global_best.copy() if self.global_best is not None else None
                probe_gfit = self.global_best_fitness
            
            probe_results[strategy_id] = probe_best_fitness
            
            # Restore state
            self.population = saved_pop
            self.velocity = saved_vel
            self.personal_best = saved_personal_best
            self.personal_best_fitness = saved_personal_fitness
            self.local_best = saved_local_best
            self.local_best_fitness = saved_local_fitness
            self.global_best = saved_global_best
            self.global_best_fitness = saved_global_fitness
            self.generation = saved_gen
            self.stagnation_counter = saved_stag
            self.inertia_weight = saved_inertia
            
            if stopping_condition():
                break
        
        return probe_results
    
    def _select_from_probe(self, probe_results):
        """Select best strategy from probe results using rank-based scoring."""
        if not probe_results:
            return 10  # Default to dominant variant
        
        # Rank-based selection: assign ranks (1=best) to each strategy
        sorted_strategies = sorted(probe_results.keys(), key=lambda s: probe_results[s])
        
        # Return the strategy with best (lowest) fitness
        return sorted_strategies[0]
    
    def _probe_and_commit(self, func, stopping_condition):
        """PHASE A: probe all strategies. PHASE B: commit to best."""
        # Probe budget: ~10% of evaluations, min 3 gens, max 8 gens
        probe_gens = min(8, max(3, int(0.05 * self.np)))
        probe_np = min(50, max(10, self.np // 3))
        
        probe_results = self._run_probe_phase(func, stopping_condition, probe_gens, probe_np)
        
        selected = self._select_from_probe(probe_results)
        self.selected_strategy = selected
        self.strategy_commit_gen = self.generation
        
        return selected
    
    # =====================================================================
    # REST OF ORIGINAL ALGORITHM
    # =====================================================================
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            # Reinitialize worst 30% of particles
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
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
        Run the optimizer with probe-and-commit adaptive strategy selection.
        
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
            return self._safe_return()
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # --- PROBE AND COMMIT PHASE ---
        self.selected_strategy = self._probe_and_commit(func, stopping_condition)
        
        # Main optimization loop
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity update (same for all strategies)
            self._velocity_update_batch()
            
            # Execute SELECTED position update strategy
            self._execute_selected_strategy()
            
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
            
            # --- FALLBACK: if selected strategy stagnates, switch to dominant variant_10 ---
            # This handles cases where the probe budget was too small to detect the true best
            if (self.selected_strategy != 10 and 
                self.stagnation_counter > 20 + self.dim // 4):
                self.selected_strategy = 10
        
        return self._safe_return()
    
    def _safe_return(self):
        """Return best found solution with NaN/Infinity safety."""
        if self.global_best is not None and np.isfinite(self.global_best_fitness):
            return self.global_best_fitness, self.global_best.copy()
        
        # Fallback: find best from current population
        valid_mask = np.isfinite(self.current_fitness)
        if np.any(valid_mask):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        # Last resort: return first individual
        return self.current_fitness[0], self.population[0].copy()
```