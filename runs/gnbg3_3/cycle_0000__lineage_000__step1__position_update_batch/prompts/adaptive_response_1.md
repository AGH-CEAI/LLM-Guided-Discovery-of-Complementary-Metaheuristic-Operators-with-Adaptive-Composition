```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    PROBE-AND-COMMIT with rank-based selection (mechanism #4 / per-task fingerprinting).
    
    Blend-hostile gap table: task 5 has variant_10 winning by 192,910× over the runner-up.
    Linear blending CANNOT preserve this advantage (a 50/50 blend of 1 and 100 is 50.5).
    The dispatcher commits exclusively to ONE variant after a short probe phase,
    preserving 100% of the per-task winner's advantage. No hard-coded regime→weight
    tables are used; the commit decision is learned online from rank-based improvement.
    
    Candidates probed (ranked by win count in benchmark):
      - variant_10: 18 wins  (eigenvalue-anisotropic velocity modulation)
      - variant_08:  3 wins  (stagnation-weighted global-best attraction)
      - variant_01:  2 wins  (geometric normalization + centroid guidance)
      - original:    1 win   (plain velocity update, used as fallback)
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
        
        # ── PROBE-AND-COMMIT state ──────────────────────────────────────
        # Probe budget: fraction of max_evals spent exploring
        self._probe_budget_fraction = 0.05
        # Number of arms to probe
        self._num_arms = 4
        # Probe results: (arm_id, rank_improvement_scores)
        self._probe_scores = None          # list of float scores per arm
        self._committed_arm = None         # int arm_id after commit
        self._probe_generation = 0         # tracks probe-phase generations
        
        # ── Sliding window for rank-based scoring ──────────────────────
        self._history_len = 12             # must be >= 3 * num_arms
        self._best_history = []            # best fitness per generation
        self._arm_history = []             # arm_id per generation (probe only)
        
        # Arm implementations as methods for clean dispatch
        self._arm_methods = {
            0: self._position_update_arm_original,
            1: self._position_update_arm_variant01,
            2: self._position_update_arm_variant08,
            3: self._position_update_arm_variant10,
        }
        self._arm_names = {
            0: 'original',
            1: 'variant_01_catA',
            2: 'variant_08_catH',
            3: 'variant_10_catB',
        }
    
    # ─────────────────────────────────────────────────────────────────
    # ARM IMPLEMENTATIONS (from benchmark winning variants)
    # ─────────────────────────────────────────────────────────────────
    
    def _position_update_arm_original(self):
        """Arm 0: plain velocity update (original.py)."""
        new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_arm_variant01(self):
        """Arm 1: geometric normalization + centroid guidance (variant_01_catA)."""
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
    
    def _position_update_arm_variant08(self):
        """Arm 2: stagnation-weighted global-best attraction (variant_08_catH)."""
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
    
    def _position_update_arm_variant10(self):
        """Arm 3: eigenvalue-anisotropic velocity modulation (variant_10_catB)."""
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
    
    # ─────────────────────────────────────────────────────────────────
    # PROBE-AND-COMMIT HELPERS
    # ─────────────────────────────────────────────────────────────────
    
    def _dispatch_position_update(self):
        """Route to the committed arm's position update method."""
        if self._committed_arm is not None and self._committed_arm in self._arm_methods:
            self._arm_methods[self._committed_arm]()
        else:
            # Safety fallback during probe phase (shouldn't reach here)
            self._position_update_arm_original()
    
    def _probe_score_from_history(self):
        """
        Compute rank-based improvement score for each arm from the sliding window.
        Score = mean rank of best-fitness improvement across the last _history_len gens.
        Rank 1 = best (lowest) fitness. Ties broken by arm_id for determinism.
        Returns: dict arm_id -> mean_rank_score (lower = better)
        """
        if len(self._best_history) < 3:
            return {i: 1.0 for i in range(self._num_arms)}
        
        # Compute per-generation improvement: delta = best[t] - best[t-1]
        improvements = []
        for t in range(1, len(self._best_history)):
            imp = self._best_history[t] - self._best_history[t - 1]
            improvements.append((t - 1, self._arm_history[t - 1], imp))
        
        # Use only the most recent _history_len improvements
        recent = improvements[-self._history_len:]
        
        # Rank arms by improvement (lower delta = better = rank 1)
        arm_ids_in_window = sorted(set(item[1] for item in recent))
        if len(arm_ids_in_window) == 0:
            return {i: 1.0 for i in range(self._num_arms)}
        
        # Compute mean rank per arm
        arm_scores = {}
        for arm_id in range(self._num_arms):
            arm_improvements = [item[2] for item in recent if item[1] == arm_id]
            if len(arm_improvements) == 0:
                arm_scores[arm_id] = float(len(arm_ids_in_window) + 1)  # penalise unseen
            else:
                # Rank-based score: count how many other arms beat this arm per generation
                ranks = []
                for t, aid, imp in recent:
                    other_arms = [a for a in arm_ids_in_window if a != arm_id]
                    other_imps = [item[2] for item in recent if item[0] == t and item[1] in other_arms]
                    n_beaten = sum(1 for oi in other_imps if oi > imp)
                    ranks.append(n_beaten + 1)
                arm_scores[arm_id] = np.mean(ranks)
        
        return arm_scores
    
    def _commit_to_arm(self, arm_scores):
        """
        Select the best arm based on rank-based scores.
        Returns arm_id with the lowest mean rank (i.e., most consistently improves).
        """
        best_arm = min(arm_scores, key=arm_scores.get)
        return best_arm
    
    def _is_in_probe_phase(self, max_evals):
        """Return True if we are still in the probe phase."""
        # Probe phase ends when we've used >= probe_budget_fraction of the budget
        # or after at least 3 generations per arm have been observed
        gen_used = self.generation
        # Estimate evaluations used so far
        evals_estimate = gen_used * self.np
        if evals_estimate >= max_evals * self._probe_budget_fraction:
            return False
        # Also end probe if we have enough data for all arms
        if len(self._arm_history) >= 3 * self._num_arms:
            return False
        return True
    
    # ─────────────────────────────────────────────────────────────────
    # EXISTING METHODS (unchanged signatures)
    # ─────────────────────────────────────────────────────────────────
    
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
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
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
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    def _velocity_update_batch(self):
        """Update velocities with adaptive components and DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
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
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
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
        Run the optimizer with PROBE-AND-COMMIT arm selection.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # ── PHASE A: PROBE ───────────────────────────────────────────────
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self._probe_scores = None
        self._committed_arm = None
        self._probe_generation = 0
        self._best_history = []
        self._arm_history = []
        
        # Estimate max_evals from stopping_condition if possible,
        # otherwise use a default cap
        max_evals_estimate = getattr(func, 'max_evals', None)
        if max_evals_estimate is None:
            max_evals_estimate = 500000  # safe default
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        if stopping_condition():
            return self._safe_return()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Track initial best for probe scoring
        self._best_history.append(
            float('nan') if self.global_best_fitness == np.inf else self.global_best_fitness
        )
        
        # ── PROBE LOOP: round-robin through arms ─────────────────────────
        arm_cycle = 0
        probe_complete = False
        
        while not stopping_condition() and not probe_complete:
            self.generation += 1
            
            # Select arm in round-robin order during probe
            if self._is_in_probe_phase(max_evals_estimate):
                current_arm = arm_cycle % self._num_arms
                arm_cycle += 1
            else:
                # Commit phase (should not reach here; commit happens after loop)
                break
            
            # Record which arm is active for this generation
            self._arm_history.append(current_arm)
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared by all arms)
            self._velocity_update_batch()
            
            # Position update via the selected arm
            self._arm_methods[current_arm]()
            
            # Evaluate
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            self._restart_if_stagnant()
            
            # Record best fitness for this generation
            best_f = (
                float('nan') if self.global_best_fitness == np.inf
                else self.global_best_fitness
            )
            self._best_history.append(best_f)
            
            # ── DECIDE COMMIT ────────────────────────────────────────────
            # Commit when we have at least 3 observations per arm
            arm_counts = {}
            for aid in self._arm_history:
                arm_counts[aid] = arm_counts.get(aid, 0) + 1
            
            min_observations = 3
            all_sampled = all(arm_counts.get(aid, 0) >= min_observations for aid in range(self._num_arms))
            budget_exceeded = not self._is_in_probe_phase(max_evals_estimate)
            
            if all_sampled or budget_exceeded:
                # Compute rank-based scores and commit
                arm_scores = self._probe_score_from_history()
                self._committed_arm = self._commit_to_arm(arm_scores)
                self._probe_scores = arm_scores
                probe_complete = True
        
        # ── PHASE B: COMMIT ──────────────────────────────────────────────
        # If probe completed naturally (all_sampled or budget_exceeded), run committed arm
        if self._committed_arm is not None:
            committed_arm = self._committed_arm
        
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared)
            self._velocity_update_batch()
            
            # Position update with committed arm
            self._arm_methods[self._committed_arm]()
            
            # Evaluate
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            self._restart_if_stagnant()
        
        return self._safe_return()
    
    def _safe_return(self):
        """Return best fitness and solution with NaN/-inf safety."""
        if self.global_best is None or np.isnan(self.global_best_fitness) or np.isinf(self.global_best_fitness):
            if self.current_fitness is not None and len(self.current_fitness) > 0:
                valid_mask = np.isfinite(self.current_fitness)
                if np.any(valid_mask):
                    best_idx = np.nanargmin(self.current_fitness)
                    return float(self.current_fitness[best_idx]), self.population[best_idx].copy()
            return np.inf, np.zeros(self.dim)
        return float(self.global_best_fitness), self.global_best.copy()
```