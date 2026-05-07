```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE: Task 5 has the winner (variant_08)
    beating the median competitor by 95.6×. A 50/50 ensemble blend of
    21.35 and the median (~2000) yields ~1000, which is ~47× WORSE than
    the winner alone. Linear blending mathematically cannot preserve
    per-task advantages.
    
    The key insight from the winning variants is that they ALL use
    SIMILAR velocity updates but DIFFERENT inertia weight adaptation
    strategies. The real differentiation is in HOW inertia is computed,
    not in the velocity formula itself.
    
    The bandit selects among 7 distinct inertia adaptation strategies
    each generation using Thompson Sampling with Beta posteriors and
    rank-based rewards. This preserves each strategy's per-task advantage
    because:
      - The arm that wins on THIS landscape gets selected with high
        probability over time.
      - Rank-based rewards are scale-independent (no magic constants).
      - Beta posteriors handle uncertainty naturally.
      - Sliding window tracks non-stationarity without hard-coded weights.
    
    Win-count distribution (7/5/3/3/2/2/2 across 7 winners) confirms
    that different inertia strategies win on different tasks, making
    a single default strategy suboptimal without bandit selection.
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
        
        # --- THOMPSON SAMPLING BANDIT STATE ---
        # One arm per inertia adaptation strategy
        self._arm_names = [
            'original',        # 0: diversity-based (baseline)
            'catA_geo',        # 1: axis-aligned geometric spread ratio
            'catD_fitness',    # 2: fitness-rank landscape analysis
            'catF_temporal',   # 3: temporal dynamics with momentum
            'catG_bootstrap',  # 4: bootstrap confidence intervals
            'catH_hybrid',     # 5: stagnation + spectral condition
            'catA_spread',     # 6: axis-aligned spread ratio (variant_09)
        ]
        n_arms = len(self._arm_names)
        
        # Beta posterior parameters: arm_alpha[i], arm_beta[i]
        # Initialize with uniform prior (alpha=1, beta=1)
        # Use a slightly informative prior favoring original (arm 0)
        # because the gap table shows original wins 7/24 tasks
        self._arm_alpha = np.ones(n_arms)
        self._arm_beta = np.ones(n_arms)
        # Soft prior: original gets a slight boost
        self._arm_alpha[0] = 1.5
        self._arm_beta[0] = 1.0
        
        # Sliding window for rank-based rewards
        # Window size >= 3 * n_arms so each arm is sampled enough
        self._reward_window_size = max(60, 3 * n_arms)
        self._reward_history = []  # list of (arm_idx, rank_score)
        self._gen_counter = 0
        
        # Track per-arm statistics for UCB reporting
        self._arm_sample_count = np.zeros(n_arms)
        self._arm_total_rank_score = np.zeros(n_arms)
        
        # Current selected arm
        self._current_arm = 0
        
        # Warmup phase: force round-robin for first n_arms generations
        self._warmup_gens = n_arms
        self._warmup_arm_idx = 0
        
        # Diversity EMA state (needed by some inertia strategies)
        self._ema_diversity = None
        self._prev_diversity = None
        self._diversity_history = []
        self._fitness_stagnation_window = []
        self._inertia_adjustment_momentum = 0.0
    
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
        
        # Initialize diversity EMA
        current_diversity = self._compute_diversity()
        self._ema_diversity = current_diversity
        self._prev_diversity = current_diversity
        self._diversity_history = [current_diversity]
        self._fitness_stagnation_window = []
        self._inertia_adjustment_momentum = 0.0
    
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
    
    # -------------------------------------------------------------------------
    # INERTIA WEIGHT ADAPTATION STRATEGIES (from benchmark winners)
    # -------------------------------------------------------------------------
    
    def _inertia_original(self):
        """Original diversity-based inertia adaptation."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_catA_geo(self):
        """Axis-aligned geometric spread ratio (variant_01)."""
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
    
    def _inertia_catD_fitness(self):
        """Fitness-rank landscape analysis (variant_04)."""
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, len(self.current_fitness) - 1)
        rank_spread = np.std(fitness_ranks) + 1e-10
        q75, q25 = np.percentile(fitness_ranks, [75, 25])
        rank_iqr = q75 - q25 + 1e-10
        
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            distance_ranks = np.argsort(np.argsort(distances)) / max(1, len(distances) - 1)
            if np.std(distance_ranks) > 1e-10 and np.std(fitness_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, distance_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0
        
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
    
    def _inertia_catF_temporal(self):
        """Temporal dynamics with diversity momentum (variant_06)."""
        current_diversity = self._compute_diversity()
        
        alpha = 0.2
        self._ema_diversity = alpha * current_diversity + (1 - alpha) * (self._ema_diversity or current_diversity)
        
        diversity_delta = current_diversity - (self._prev_diversity or current_diversity)
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
    
    def _inertia_catG_bootstrap(self):
        """Bootstrap confidence intervals (variant_07)."""
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
    
    def _inertia_catH_hybrid(self):
        """Hybrid stagnation + spectral condition (variant_08)."""
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
    
    def _inertia_catA_spread(self):
        """Axis-aligned spread ratio (variant_09)."""
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
    # DISPATCHER: maps arm index -> inertia adaptation method
    # -------------------------------------------------------------------------
    
    def _dispatch_inertia_update(self, arm_idx):
        """Route to the appropriate inertia adaptation implementation."""
        if arm_idx == 0:
            self._inertia_original()
        elif arm_idx == 1:
            self._inertia_catA_geo()
        elif arm_idx == 2:
            self._inertia_catD_fitness()
        elif arm_idx == 3:
            self._inertia_catF_temporal()
        elif arm_idx == 4:
            self._inertia_catG_bootstrap()
        elif arm_idx == 5:
            self._inertia_catH_hybrid()
        elif arm_idx == 6:
            self._inertia_catA_spread()
        else:
            self._inertia_original()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING BANDIT
    # -------------------------------------------------------------------------
    
    def _thompson_sample(self):
        """Sample an arm from Beta posterior distributions."""
        samples = np.random.beta(self._arm_alpha, self._arm_beta)
        arm = int(np.argmax(samples))
        return arm
    
    def _record_reward(self, arm_idx, rank_score):
        """Record a rank-based reward for an arm.
        
        Rank score is the percentile rank of the best fitness in this
        generation relative to all generations in the sliding window.
        This is scale-independent.
        """
        self._reward_history.append((arm_idx, rank_score))
        if len(self._reward_history) > self._reward_window_size:
            self._reward_history.pop(0)
        
        # Update Beta posterior: reward = rank_score (in [0,1])
        # Higher rank_score = better performance = more successes
        # We model rank_score as a Bernoulli with success prob = rank_score
        # Beta-Bernoulli conjugate update:
        #   alpha_new = alpha_old + rank_score
        #   beta_new = beta_old + (1 - rank_score)
        self._arm_alpha[arm_idx] += rank_score
        self._arm_beta[arm_idx] += (1.0 - rank_score)
        
        # Clamp to prevent numerical issues
        self._arm_alpha[arm_idx] = np.clip(self._arm_alpha[arm_idx], 0.1, 1e6)
        self._arm_beta[arm_idx] = np.clip(self._arm_beta[arm_idx], 0.1, 1e6)
        
        self._arm_sample_count[arm_idx] += 1
        self._arm_total_rank_score[arm_idx] += rank_score
    
    def _compute_rank_score(self, best_fitness):
        """Compute rank-based score for the current generation.
        
        Score = fraction of generations in the window with WORSE fitness.
        Higher score = better performance.
        """
        if len(self._reward_history) < 2:
            return 1.0  # First gen gets best score
        
        # Get all best fitness values in the window
        all_best_fitness = [bf for _, (_, bf) in zip(range(len(self._reward_history)), self._reward_history)]
        all_best_fitness.append(best_fitness)
        
        current_best = min(all_best_fitness)
        current_worst = max(all_best_fitness)
        
        if current_worst > current_best:
            normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
            score = np.clip(normalized, 0.0, 1.0)
        else:
            score = 1.0
        
        return score
    
    def _select_arm(self):
        """Select an arm using Thompson Sampling (with warmup)."""
        self._gen_counter += 1
        
        # Warmup phase: force round-robin
        if self._gen_counter <= self._warmup_gens:
            return self._warmup_arm_idx % len(self._arm_names)
        
        # Regular phase: Thompson Sampling
        return self._thompson_sample()
    
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
    # VELOCITY UPDATE (consistent across all arms)
    # -------------------------------------------------------------------------
    
    def _velocity_update_base(self):
        """Base velocity update with DE/rand/1 mutation.
        
        All winning variants use similar velocity updates; the differentiation
        is in the inertia weight adaptation, which is handled by the bandit.
        """
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
        Run the optimizer with Thompson Sampling Bandit over inertia strategies.
        
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
        
        # Reset bandit state
        n_arms = len(self._arm_names)
        self._arm_alpha = np.ones(n_arms)
        self._arm_beta = np.ones(n_arms)
        self._arm_alpha[0] = 1.5
        self._arm_beta[0] = 1.0
        self._reward_history = []
        self._gen_counter = 0
        self._warmup_arm_idx = 0
        self._arm_sample_count = np.zeros(n_arms)
        self._arm_total_rank_score = np.zeros(n_arms)
        
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
            
            # Select arm via Thompson Sampling
            self._current_arm = self._select_arm()
            
            # Adapt neighborhood
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Adapt inertia weight using the selected strategy
            self._dispatch_inertia_update(self._current_arm)
            
            # Update velocity and position
            self._velocity_update_base()
            self._position_update_batch()
            
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
            
            # Record reward for bandit update
            gen_best = float(np.min(self.current_fitness))
            rank_score = self._compute_rank_score(gen_best)
            self._record_reward(self._current_arm, rank_score)
            
            # Advance warmup counter
            if self._gen_counter <= self._warmup_gens:
                self._warmup_arm_idx += 1
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```