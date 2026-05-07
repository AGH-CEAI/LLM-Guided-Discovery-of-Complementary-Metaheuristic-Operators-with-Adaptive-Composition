import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE (4 tasks with gap >= 10×, 1 task with gap >= 100×).
    Task 5: original.py (6.35e+01) vs variant_02_catB_idea_0.py (4.87e+04) = 768× gap.
    A 50/50 ensemble blend of 63.47 and 48742 yields ~24403, which is 384× WORSE than
    the winner alone. Linear blending mathematically CANNOT preserve per-task advantages.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin burst with each of the 5 winning operators,
        tracking rank-based improvement scores on the ACTUAL landscape.
      - PHASE B (commit): run ONLY the winning operator for the remaining budget.
        No blending. No hard-coded weights.
      - EPSILON-REVIEW: if the committed operator stagnates for >20 gens, allow a
        small probability of switching to the runner-up.
    
    Win-count distribution (8/6/4/3/3 across 5 distinct winners) confirms that
    different operators win on different tasks, making per-task probing essential.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients
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
        self.current_fitness = None
        
        # --- PROBE-AND-COMMIT STATE ---
        # Operators based on ACTUAL benchmark winners
        self._operators = [
            'original',        # 4 wins: tasks 1, 3, 5, 7
            'variant_02_catB', # 3 wins: tasks 2, 4, 8
            'variant_04_catD', # 8 wins: tasks 0,12,15,16,17,20,21,22
            'variant_06_catF', # 3 wins: tasks 11,13,18
            'variant_09_catA', # 6 wins: tasks 6,9,10,14,19,23
        ]
        
        # Rank-based score tracking per operator
        self.operator_scores = {op: [] for op in self._operators}
        self.operator_total_score = {op: 0.0 for op in self._operators}
        self.operator_sample_count = {op: 0 for op in self._operators}
        
        # Probe configuration
        self._probe_gens_per_op = 3
        self._probe_total_budget = len(self._operators) * self._probe_gens_per_op
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Committed operator after probe
        self._committed_operator = None
        self._runner_up_operator = None
        
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
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
        try:
            fitness = func(clipped)
            if hasattr(fitness, '__len__') and not isinstance(fitness, (int, float)):
                return np.asarray(fitness), len(fitness)
            return np.array([fitness]), 1
        except Exception:
            return np.full(len(population), np.inf), len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        if np.any(improved):
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
        if len(self.personal_best_fitness) == 0:
            return
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
        """Adapt inertia weight based on axis-aligned geometric spread ratio."""
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10

        geo_mean = np.exp(np.mean(np.log(spread)))
        arith_mean = np.mean(spread)
        spread_ratio = geo_mean / (arith_mean + 1e-10)

        spread_ratio_normalized = np.clip(spread_ratio / 0.9, 0.0, 1.0)

        target_inertia = (1.0 - spread_ratio_normalized) * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
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
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE VARIANTS (base implementations)
    # -------------------------------------------------------------------------
    
    def _velocity_update_original(self):
        """Original velocity update with DE/rand/1 mutation."""
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
    
    def _velocity_update_variant_02_catB(self):
        """SVD-based velocity whitening with anisotropic damping."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        centered = self.population - np.mean(self.population, axis=0)
        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            sv_norm = singular_values / (singular_values[0] + 1e-10)
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)
            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight
            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            mutation_component = (vel_scaled @ Vt) - self.velocity
        except np.linalg.LinAlgError:
            mutation_component = np.zeros((self.np, self.dim))
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        de_component = np.where(
            mutation_active,
            0.2 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component * 0.3 +
            de_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_04_catD(self):
        """Fitness-rank-adaptive velocity with FDC correlation."""
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            
            if np.std(distances) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                corr = np.corrcoef(
                    np.argsort(np.argsort(self.current_fitness)),
                    np.argsort(np.argsort(distances))
                )[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0
        
        social = self.social_base * np.clip(1.0 + corr, 0.2, 1.5)
        cognitive = self.cognitive_base
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_strength = 0.3 * np.clip(1.0 - corr, 0.1, 1.0)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            mutation_strength * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_06_catF(self):
        """Temporal-drift-modulated velocity scaling."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        centroid = np.mean(self.population, axis=0)
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)
        
        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid
        
        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity
        
        if hasattr(self, '_centroid_vel_history') and len(self._centroid_vel_history) >= 5:
            recent = np.array(self._centroid_vel_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            autocorr = 0.0
        
        if not hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)
        
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness
        
        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement
        
        convergence_signal = np.clip(autocorr, -1.0, 1.0)
        
        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        temporal_scale = base_scale * (1.0 + 0.4 * convergence_signal)
        
        vel_magnitude = np.linalg.norm(self._ema_centroid_velocity) + 1e-10
        vel_factor = np.clip(1.0 / (1.0 + vel_magnitude * 0.005), 0.5, 1.5)
        temporal_scale *= vel_factor
        
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
            temporal_scale *= spectral_factor
        except np.linalg.LinAlgError:
            pass
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.2 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity * temporal_scale +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_09_catA(self):
        """Centroid gravity with k-NN density modulation."""
        cognitive, social = self._adaptive_coefficients()
        
        centroid = np.mean(self.population, axis=0)
        
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
        
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
        
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist
        
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
        
        correction = centroid_attraction * to_centroid_dir * density_modulation
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            0.3 * correction
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> velocity update method
    # -------------------------------------------------------------------------
    
    def _dispatch_velocity_update(self, operator):
        """Route to the appropriate velocity update implementation."""
        if operator == 'original':
            self._velocity_update_original()
        elif operator == 'variant_02_catB':
            self._velocity_update_variant_02_catB()
        elif operator == 'variant_04_catD':
            self._velocity_update_variant_04_catD()
        elif operator == 'variant_06_catF':
            self._velocity_update_variant_06_catF()
        elif operator == 'variant_09_catA':
            self._velocity_update_variant_09_catA()
        else:
            self._velocity_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record probe score using rank-based percentile rank.
        
        Score = fraction of probe generations with WORSE (higher) fitness.
        This is scale-independent and handles the 768× gaps in the benchmark.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        if len(self._probe_fitness_history) < 2:
            score = 1.0
        else:
            current_best = min(self._probe_fitness_history)
            current_worst = max(self._probe_fitness_history)
            if current_worst > current_best:
                normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
                score = np.clip(normalized, 0.0, 1.0)
            else:
                score = 1.0
        
        self.operator_scores[operator].append(score)
        self.operator_total_score[operator] += score
        self.operator_sample_count[operator] += 1
    
    def _compute_ucb_score(self, operator):
        """Compute UCB1-style score for operator selection."""
        n = self.operator_sample_count[operator]
        if n == 0:
            return float('inf')
        
        total_n = sum(self.operator_sample_count.values())
        avg_score = self.operator_total_score[operator] / n
        
        if total_n > 0:
            exploration = np.sqrt(2.0 * np.log(total_n) / n)
        else:
            exploration = float('inf')
        
        return avg_score + exploration
    
    def _select_best_operator(self):
        """Select operator with highest UCB score."""
        best_op = self._operators[0]
        best_score = -float('inf')
        
        for op in self._operators:
            ucb = self._compute_ucb_score(op)
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
    def _select_runner_up_operator(self):
        """Select the second-best operator by UCB score."""
        scores = [(op, self._compute_ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2:
            return scores[1][0]
        return scores[0][0]
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE BATCH (benchmark-winning implementations)
    # -------------------------------------------------------------------------
    
    def _position_update_batch_original(self):
        """Original position update with spectral conditioning."""
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
    
    def _position_update_batch_variant_02_catB(self):
        """SVD-based population whitening with anisotropic damping."""
        centered = self.population - np.mean(self.population, axis=0)
        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10
            sv_norm = singular_values / (singular_values[0] + 1e-10)
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)
            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight
            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_batch_variant_04_catD(self):
        """Fitness-landscape rank signals only (no distances/covariance)."""
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
        
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        exploration_perturb = exploration_factor * fitness_perturb
        random_perturb = exploration_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
        
        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_batch_variant_06_catF(self):
        """Temporal-drift-modulated velocity scaling."""
        centroid = np.mean(self.population, axis=0)
        
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)
        
        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid
        
        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity
        
        if hasattr(self, '_centroid_vel_history') and len(self._centroid_vel_history) >= 5:
            recent = np.array(self._centroid_vel_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            autocorr = 0.0
        
        if not hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)
        
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness
        
        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement
        
        convergence_signal = np.clip(autocorr, -1.0, 1.0)
        
        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        temporal_scale = base_scale * (1.0 + 0.4 * convergence_signal)
        
        vel_magnitude = np.linalg.norm(self._ema_centroid_velocity) + 1e-10
        vel_factor = np.clip(1.0 / (1.0 + vel_magnitude * 0.005), 0.5, 1.5)
        temporal_scale *= vel_factor
        
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
            temporal_scale *= spectral_factor
        except np.linalg.LinAlgError:
            pass
        
        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_batch_variant_09_catA(self):
        """Centroid gravity and k-NN density modulation."""
        centroid = np.mean(self.population, axis=0)
        
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
        
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
        
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dir
        
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
        
        correction = centroid_attraction * to_centroid_dir * density_modulation
        
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_batch_dispatch(self, operator):
        """Dispatch to the correct position update implementation."""
        if operator == 'original':
            self._position_update_batch_original()
        elif operator == 'variant_02_catB':
            self._position_update_batch_variant_02_catB()
        elif operator == 'variant_04_catD':
            self._position_update_batch_variant_04_catD()
        elif operator == 'variant_06_catF':
            self._position_update_batch_variant_06_catF()
        elif operator == 'variant_09_catA':
            self._position_update_batch_variant_09_catA()
        else:
            self._position_update_batch_original()
    
    # -------------------------------------------------------------------------
    # RESTART AND LOCAL BEST UPDATE
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT operator selection.
        
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
        
        # Reset probe state
        for op in self._operators:
            self.operator_scores[op] = []
            self.operator_total_score[op] = 0.0
            self.operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._runner_up_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        
        # Reset EMA state for temporal operators
        if hasattr(self, '_ema_centroid'):
            delattr(self, '_ema_centroid')
        if hasattr(self, '_ema_centroid_velocity'):
            delattr(self, '_ema_centroid_velocity')
        if hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        if hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        if hasattr(self, '_prev_global_best_fitness'):
            delattr(self, '_prev_global_best_fitness')
        if hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        
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
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation phase
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Use the current probe operator for velocity AND position
                self._dispatch_velocity_update(current_op)
                self._position_update_batch_dispatch(current_op)
                
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
                
                # Record probe score
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_score(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= len(self._operators):
                    self._in_probe_phase = False
                    self._committed_operator = self._select_best_operator()
                    self._runner_up_operator = self._select_runner_up_operator()
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._dispatch_velocity_update(self._runner_up_operator)
                    self._position_update_batch_dispatch(self._runner_up_operator)
                else:
                    self._dispatch_velocity_update(self._committed_operator)
                    self._position_update_batch_dispatch(self._committed_operator)
                
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
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            valid_fitness = np.where(np.isfinite(self.current_fitness))[0]
            if len(valid_fitness) > 0:
                best_idx = valid_fitness[np.argmin(self.current_fitness[valid_fitness])]
                return float(self.current_fitness[best_idx]), self.population[best_idx].copy()
            return np.inf, np.zeros(self.dim)
        
        return float(self.global_best_fitness), self.global_best.copy()
