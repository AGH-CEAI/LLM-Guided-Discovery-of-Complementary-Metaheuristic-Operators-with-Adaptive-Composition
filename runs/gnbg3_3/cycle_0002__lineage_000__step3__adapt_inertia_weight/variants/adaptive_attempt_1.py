import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE: Task 5 has variant_08 beating the median
    competitor by 95.6×. A 50/50 ensemble blend of 21.35 and 2039.47 yields
    ~1030, which is 48× WORSE than the winner alone. Linear blending cannot
    preserve per-task advantages when gaps exceed ~10×.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): short round-robin burst with each operator (3 gens
        each), tracking rank-based improvement scores on the ACTUAL landscape.
      - PHASE B (commit): run ONLY the winning operator for the remaining
        budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if the committed operator stagnates, allow a 5%
        probability of switching to the runner-up.
    
    Win-count distribution (7/5/3/3/2/2/2 across 7 winners) confirms that
    different operators win on different tasks, making a single default
    operator suboptimal without probing. The 95.6× gap on Task 5 makes
    ensemble blending mathematically invalid.
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
        self.current_fitness = None
        
        # --- PROBE-AND-COMMIT STATE ---
        self.operator_scores = {}   # {op_id: [score1, score2, ...]}
        self.operator_total_score = {}
        self.operator_sample_count = {}
        self._committed_operator = None
        self._runner_up_operator = None
        
        # Operators = the 7 benchmark winners
        self._operators = [
            'original',
            'variant_01_catA',
            'variant_04_catD',
            'variant_06_catF',
            'variant_07_catG',
            'variant_08_catH',
            'variant_09_catA',
        ]
        for op in self._operators:
            self.operator_scores[op] = []
            self.operator_total_score[op] = 0.0
            self.operator_sample_count[op] = 0
        
        # Probe: 3 gens per operator, minimum 21 total
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
        
        # Temporal state for variant_06_catF inertia (accumulates across gens)
        self._ema_diversity = None
        self._prev_diversity = None
        self._diversity_history = []
        self._fitness_stagnation_window = []
        self._inertia_adjustment_momentum = 0.0
        
        # Bootstrap RNG state for variant_07_catG (shared across probe gens)
        self._rng_state = None
    
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
        
        # Reset temporal state for variant_06
        self._ema_diversity = None
        self._prev_diversity = None
        self._diversity_history = []
        self._fitness_stagnation_window = []
        self._inertia_adjustment_momentum = 0.0
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
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
    
    # -------------------------------------------------------------------------
    # INERTIA WEIGHT ADAPTATION — one method per benchmark winner
    # -------------------------------------------------------------------------
    
    def _adapt_inertia_weight_original(self):
        """Baseline: diversity-based inertia weight adaptation."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_01_catA(self):
        """Adapt inertia weight based on axis-aligned geometric spread ratio.
        
        Low ratio -> anisotropic (elongated) swarm -> increase exploration (higher inertia).
        High ratio -> isotropic (spherical) swarm -> increase exploitation (lower inertia).
        """
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
    
    def _adapt_inertia_weight_variant_04_catD(self):
        """Adapt inertia weight using fitness-rank landscape analysis.
        
        High inertia when ranks are diverse (exploring basin), low inertia when
        ranks compress (stuck in local optimum) or FDC is low (deceptive landscape).
        """
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, len(self.current_fitness) - 1)
        rank_spread = np.std(fitness_ranks) + 1e-10
        q75, q25 = np.percentile(fitness_ranks, [75, 25])
        rank_iqr = q75 - q25 + 1e-10
        
        fdc = 0.0
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            distance_ranks = np.argsort(np.argsort(distances)) / max(1, len(distances) - 1)
            if np.std(distance_ranks) > 1e-10 and np.std(fitness_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, distance_ranks)[0, 1]
        fdc = np.clip(fdc, -0.5, 1.0)
        
        if rank_spread < 0.12 or rank_iqr < 0.15:
            self.inertia_weight = max(0.3, self.inertia_weight * 0.75)
        elif fdc < 0.2:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.85)
        elif rank_spread > 0.30 and rank_iqr > 0.35 and fdc > 0.5:
            self.inertia_weight = min(0.90, self.inertia_weight * 1.08)
        else:
            self.inertia_weight = 0.729 - 0.08 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
        fdc_modulation = 0.6 + 0.4 * np.clip(fdc + 0.5, 0.0, 1.0)
        self.inertia_weight *= fdc_modulation
        self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
    
    def _adapt_inertia_weight_variant_06_catF(self):
        """Adapt inertia weight using temporal dynamics: diversity momentum, rate-of-change, and stagnation.
        
        Uses EMA-smoothed diversity, trend slope, and momentum to modulate inertia.
        Temporal state is accumulated across generations within a probe/commit phase.
        """
        current_diversity = self._compute_diversity()
        
        if self._ema_diversity is None:
            self._ema_diversity = current_diversity
            self._prev_diversity = current_diversity
        
        alpha = 0.2
        self._ema_diversity = alpha * current_diversity + (1 - alpha) * self._ema_diversity
        
        diversity_delta = current_diversity - self._prev_diversity
        self._prev_diversity = current_diversity
        
        self._diversity_history.append(current_diversity)
        if len(self._diversity_history) > 50:
            self._diversity_history.pop(0)
        
        if len(self._fitness_stagnation_window) < 20:
            self._fitness_stagnation_window.append(self.global_best_fitness)
        else:
            self._fitness_stagnation_window.pop(0)
            self._fitness_stagnation_window.append(self.global_best_fitness)
        
        fitness_improving = False
        if len(self._fitness_stagnation_window) >= 5:
            recent_improvement = self._fitness_stagnation_window[0] - self._fitness_stagnation_window[-1]
            fitness_improving = recent_improvement > 1e-6
        
        trend_slope = 0.0
        if len(self._diversity_history) >= 5:
            n = len(self._diversity_history)
            x = np.arange(n)
            x_mean = np.mean(x)
            y = np.array(self._diversity_history)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2) + 1e-10
            trend_slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        
        momentum_decay = 0.8
        self._inertia_adjustment_momentum = (
            momentum_decay * self._inertia_adjustment_momentum +
            (1 - momentum_decay) * np.sign(diversity_delta)
        )
        
        if self._ema_diversity < self.diversity_threshold_low:
            if trend_slope < -1e-6 and not fitness_improving:
                adjustment = 1.15
            elif abs(self._inertia_adjustment_momentum) > 0.5:
                adjustment = 1.1
            else:
                adjustment = 1.05
            self.inertia_weight = min(0.95, self.inertia_weight * adjustment)
        
        elif self._ema_diversity > self.diversity_threshold_high:
            if trend_slope > 1e-6 and fitness_improving:
                adjustment = 0.9
            else:
                adjustment = 0.95
            self.inertia_weight = max(0.4, self.inertia_weight * adjustment)
        
        else:
            base_inertia = 0.729 - 0.1 * (self.generation / 1000)
            if trend_slope < -0.5:
                base_inertia += 0.1
            elif trend_slope > 0.5:
                base_inertia -= 0.05
            self.inertia_weight = np.clip(base_inertia, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_07_catG(self):
        """Adapt inertia weight using bootstrap confidence intervals on diversity.
        
        Uses bootstrap resampling to estimate diversity uncertainty, then makes
        probabilistic decisions about exploration vs exploitation.
        """
        pop = self.population
        np_pop = self.np
        
        n_bootstrap = 50
        bootstrap_diversities = np.zeros(n_bootstrap)
        
        for b in range(n_bootstrap):
            indices = np.random.randint(0, np_pop, size=np_pop)
            sample = pop[indices]
            centroid = np.mean(sample, axis=0)
            distances = np.linalg.norm(sample - centroid, axis=1)
            bootstrap_diversities[b] = np.mean(distances)
        
        ci_low = np.percentile(bootstrap_diversities, 10)
        ci_high = np.percentile(bootstrap_diversities, 90)
        ci_width = ci_high - ci_low
        median_diversity = np.median(bootstrap_diversities)
        
        n_probes = 20
        probe_inertias = np.linspace(0.4, 0.95, n_probes)
        np.random.shuffle(probe_inertias)
        
        expected_diversity = np.zeros(n_probes)
        for i, w in enumerate(probe_inertias):
            vel_mags = np.linalg.norm(self.velocity, axis=1)
            vel_effect = np.mean(vel_mags) * w
            noise_std = ci_width / 2
            expected_diversity[i] = median_diversity + vel_effect + np.random.normal(0, noise_std)
        
        best_idx = np.argmax(expected_diversity)
        candidate_inertia = probe_inertias[best_idx]
        
        rng = np.random.random()
        
        if ci_high < self.diversity_threshold_low:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.1)
        elif ci_low > self.diversity_threshold_high:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.9)
        elif ci_width > 5.0:
            if rng < 0.7:
                self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
            else:
                self.inertia_weight = candidate_inertia
        else:
            decay = 0.1 * (self.generation / 1000)
            self.inertia_weight = candidate_inertia - decay
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_08_catH(self):
        """Hybrid inertia adaptation: combine stagnation signal with spectral condition.
        
        Stagnation dominates when stuck (counter > 15), spectral guides otherwise.
        """
        diversity = self._compute_diversity()
        
        stagnation_penalty = np.exp(-self.stagnation_counter / 30.0)
        
        centered = self.population - np.mean(self.population, axis=0)
        try:
            _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            spectral_stress = np.clip(cond / 50.0, 0.0, 1.0)
        except np.linalg.LinAlgError:
            spectral_stress = 0.5
        
        if self.stagnation_counter > 15:
            if diversity < self.diversity_threshold_low:
                self.inertia_weight = max(0.3, self.inertia_weight * 0.85)
            else:
                self.inertia_weight = min(0.9, self.inertia_weight * 1.02)
        else:
            combined_exploration_pressure = 0.6 * (1.0 - stagnation_penalty) + 0.4 * spectral_stress
            
            if combined_exploration_pressure > 0.5:
                self.inertia_weight = max(0.35, 0.729 - 0.15 * combined_exploration_pressure)
            else:
                self.inertia_weight = min(0.95, 0.729 + 0.10 * (0.5 - combined_exploration_pressure))
        
        self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
    
    def _adapt_inertia_weight_variant_09_catA(self):
        """Adapt inertia weight based on axis-aligned spread ratio (per-dimension).
        
        Detects dimensional collapse: when the swarm collapses non-uniformly
        (spread along one axis >> others), the swarm may miss optima in
        under-explored dimensions. Uses log-scaled response to anisotropy ratio.
        """
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        search_width = self.upper_bound - self.lower_bound + 1e-10
        
        spreads = max_pos - min_pos
        normalized_spreads = spreads / search_width
        
        max_spread = np.max(normalized_spreads)
        min_spread = np.min(normalized_spreads)
        
        axis_ratio = max_spread / (min_spread + 1e-10)
        log_ratio = np.log1p(axis_ratio)
        
        if log_ratio > 5.0:
            self.inertia_weight = 0.92
        elif log_ratio > 3.5:
            self.inertia_weight = 0.85
        elif log_ratio > 2.0:
            self.inertia_weight = 0.78
        elif log_ratio > 1.0:
            self.inertia_weight = 0.72
        else:
            self.inertia_weight = 0.729 - 0.08 * (self.generation / 1000)
        
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> inertia weight adaptation method
    # -------------------------------------------------------------------------
    
    def _dispatch_inertia_adaptation(self, operator):
        """Route to the appropriate inertia weight adaptation implementation."""
        if operator == 'original':
            self._adapt_inertia_weight_original()
        elif operator == 'variant_01_catA':
            self._adapt_inertia_weight_variant_01_catA()
        elif operator == 'variant_04_catD':
            self._adapt_inertia_weight_variant_04_catD()
        elif operator == 'variant_06_catF':
            self._adapt_inertia_weight_variant_06_catF()
        elif operator == 'variant_07_catG':
            self._adapt_inertia_weight_variant_07_catG()
        elif operator == 'variant_08_catH':
            self._adapt_inertia_weight_variant_08_catH()
        elif operator == 'variant_09_catA':
            self._adapt_inertia_weight_variant_09_catA()
        else:
            self._adapt_inertia_weight_original()
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (original PSO + DE/rand/1 mutation — shared across all operators)
    # -------------------------------------------------------------------------
    
    def _velocity_update(self):
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
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
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
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score using rank-based evaluation.
        
        Score = percentile rank of this generation's best fitness among all
        probe generations seen so far. Lower fitness = higher score = better.
        This is scale-independent (no magic constants).
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
        """Select operator with highest UCB score from probe data."""
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
            return self.global_best_fitness, self.global_best
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Use the current probe operator's inertia adaptation
                self._dispatch_inertia_adaptation(current_op)
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                self._velocity_update()
                self._position_update_batch()
                
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
                    # Reset temporal state for variant_06 when entering commit phase
                    self._ema_diversity = None
                    self._prev_diversity = None
                    self._diversity_history = []
                    self._fitness_stagnation_window = []
                    self._inertia_adjustment_momentum = 0.0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._dispatch_inertia_adaptation(self._runner_up_operator)
                else:
                    self._dispatch_inertia_adaptation(self._committed_operator)
                
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                self._velocity_update()
                self._position_update_batch()
                
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
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
