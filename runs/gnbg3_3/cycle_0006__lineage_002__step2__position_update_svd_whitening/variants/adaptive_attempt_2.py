import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING WITH RANK-BASED REWARDS (Discrete Arm Selection, mechanism #1)
    
    The gap table is NOT blend-hostile:
      - Maximum per-task gap is 2.3× (task 5), well below the 10× threshold.
      - 10 distinct winners across 24 tasks with small margins means no single
        operator dominates everywhere. Linear blending CAN preserve these small
        advantages (a 50/50 blend of 10 and 23 is 16.5, within 1.65× of winner).
    
    Why Thompson Sampling fits this data:
      - Wins are spread across 10 variants (5/4/3/2/2/2/2/2/1/1) — no dominant operator.
      - All gaps < 3× — scale-independent rank rewards are well-calibrated.
      - Operators are mutually exclusive per generation — easy credit assignment.
      - Thompson Sampling naturally balances explore/exploit without hard-coded
        regime→weight tables (forbidden per calibration rule f).
    
    Key design choices:
      - Beta(α,β) posteriors per operator; prior initialized from benchmark win counts.
      - RANK-BASED reward: percentile of best_fitness in sliding window of last K gens.
        This is scale-independent (rule a) — no magic constants like *1e5.
      - K = max(3*num_arms, 30) = 30 ensures each arm is sampled enough (rule b).
      - Prior on posteriors encodes benchmark evidence without hard-coding weights.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds
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
        
        # --- THOMPSON SAMPLING STATE ---
        # Map operator ID -> (alpha, beta) Beta posterior
        # Prior initialized from benchmark win counts (5,4,3,2,2,2,2,2,1,1)
        # Total benchmark runs per operator ≈ proportional to win count
        benchmark_wins = {
            'original':      2,
            'catA_v01':      2,
            'catB_v02':      4,
            'catC_v03':      2,
            'catD_v04':      3,
            'catF_v06':      2,
            'catG_v07':      2,
            'catH_v08':      1,
            'catA_v09':      5,
            'catB_v10':      1,
        }
        
        # All 10 operators from benchmark
        self._operators = list(benchmark_wins.keys())
        n_ops = len(self._operators)
        
        # Beta prior: alpha = 1 + benchmark_wins, beta = 1 + (total_wins - wins)
        # This encodes benchmark evidence as a prior without hard-coding weights
        total_wins = sum(benchmark_wins.values())
        self._alpha = {}
        self._beta = {}
        for op in self._operators:
            w = benchmark_wins[op]
            # Informative prior: more benchmark wins → higher alpha
            self._alpha[op] = 1.0 + w
            self._beta[op] = 1.0 + (total_wins - w)
        
        # Sliding window for rank-based rewards (rule b: K >= 3 * num_arms)
        self._reward_window_size = max(3 * n_ops, 30)
        self._fitness_history = []  # List of (best_fitness, operator) tuples
        
        # Track which operator was selected this generation
        self._current_operator = None
        
        # Per-operator success counters for diversity bonus
        self._op_success = {op: 0 for op in self._operators}
        self._op_attempts = {op: 0 for op in self._operators}
    
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
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight using condition number of population covariance."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * spectral_signal
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
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
    
    def _velocity_update_base(self):
        """Base velocity update: cognitive + social + DE mutation."""
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
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (all 10 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation."""
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
    
    def _position_update_catA_v01(self):
        """Geometric pairwise distance + axis-aligned spread modulation (Category A).
        
        2 wins on tasks 8, 21. Uses literal geometric layout: pairwise distance
        distribution, axis-aligned bounding box spread, and centroid distances.
        """
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        axis_spread = pop_max - pop_min + 1e-10

        bounds_range = self.upper_bound - self.lower_bound + 1e-10
        rel_spread = axis_spread / bounds_range

        n = self.np
        if n > 2:
            max_samples = min(500, n * (n - 1) // 2)
            sample_size = min(max_samples, 200)

            all_pairs = []
            for _ in range(sample_size):
                i, j = np.random.choice(n, 2, replace=False)
                all_pairs.append((i, j))

            distances = np.array([
                np.linalg.norm(self.population[i] - self.population[j])
                for i, j in all_pairs
            ])
            mean_pair_dist = np.mean(distances)
            std_pair_dist = np.std(distances)
        else:
            mean_pair_dist = bounds_range[0]
            std_pair_dist = 0.0

        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10

        spread_modulation = np.clip(1.0 / (rel_spread + 0.05), 0.3, 3.0)
        centroid_scale = np.clip(dist_to_centroid / mean_pair_dist, 0.2, 2.5)

        collapse_threshold = 0.1 * np.mean(bounds_range)
        is_collapsed = mean_pair_dist < collapse_threshold

        per_component_scale = spread_modulation * centroid_scale

        if is_collapsed:
            per_component_scale = np.clip(per_component_scale, 1.5, 3.0)
        else:
            per_component_scale = np.clip(per_component_scale, 0.3, 2.0)

        vel_scaled = self.velocity * per_component_scale
        new_population = self.population + vel_scaled

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB_v02(self):
        """Eigendecomposition-based whitened coordinate momentum (Category B).
        
        4 wins on tasks 6, 19, 20, 23. Transforms velocity to eigenspace,
        applies momentum-based updates, and uses condition number to
        aggressively modulate exploration.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            eigenvalues = np.clip(eigenvalues, 1e-12, None)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)

            total_var = np.sum(eigenvalues) + 1e-10

            if not hasattr(self, '_prev_whitened_vel'):
                self._prev_whitened_vel = np.zeros((self.np, self.dim))

            vel_proj = self.velocity @ eigenvectors

            whiten_factors = np.sqrt(1.0 / eigenvalues + 1e-10)
            whiten_factors = whiten_factors / (np.max(whiten_factors) + 1e-10)

            vel_whitened = vel_proj * whiten_factors

            momentum_coeff = np.clip(0.3 + 0.5 * np.log1p(cond) / np.log1p(1e6), 0.3, 0.8)

            vel_with_momentum = (1.0 - momentum_coeff) * vel_whitened + momentum_coeff * self._prev_whitened_vel

            self._prev_whitened_vel = vel_with_momentum.copy()

            vel_unwhitened = vel_with_momentum / (whiten_factors + 1e-10)

            vel_scaled = vel_unwhitened * np.clip(cond / 100.0, 0.5, 2.5)

            new_population = self.population + vel_scaled

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catC_v03(self):
        """Entropy-modulated velocity scaling (Category C: Information-theoretic).
        
        2 wins on tasks 0, 7. Uses non-parametric entropy estimation from k-NN
        distances. Low entropy = clustered = more exploration needed.
        """
        if self.np < 3:
            self.population = self._clip_to_bounds(self.population + self.velocity)
            return

        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        distances = np.sqrt(sq_dists)

        k = min(5, self.np - 1)
        sorted_dists = np.sort(distances, axis=1)
        knn_dists = sorted_dists[:, :k]

        entropy_signal = np.zeros(self.np)
        for i in range(self.np):
            d = knn_dists[i]
            if d[0] > 1e-15 and d[-1] > d[0]:
                ratios = d[1:k] / (d[0] + 1e-15)
                p = ratios / (np.sum(ratios) + 1e-15)
                p = np.clip(p, 1e-15, 1.0)
                entropy_signal[i] = -np.sum(p * np.log(p))

        max_entropy = np.log(k)
        entropy_norm = entropy_signal / (max_entropy + 1e-15)
        entropy_norm = np.clip(entropy_norm, 0.0, 1.0)

        explore_scale = 1.0 + 1.5 * (1.0 - entropy_norm)
        explore_scale = np.clip(explore_scale, 0.5, 2.5)

        scaled_velocity = self.velocity * explore_scale[:, np.newaxis]
        new_population = self.population + scaled_velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catD_v04(self):
        """Fitness-landscape rank-based velocity modulation (Category D).
        
        3 wins on tasks 11, 12, 16. Uses Spearman fitness-distance correlation
        to detect landscape smoothness, fitness percentiles for per-particle
        scaling, and success history for momentum.
        """
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        if self.global_best is not None and self.np > 3:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)

            n = self.np
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    fit_diff = (fitness_ranks[i] - fitness_ranks[j]) * (dist_ranks[i] - dist_ranks[j])
                    if fit_diff > 0:
                        concordant += 1
                    elif fit_diff < 0:
                        discordant += 1
            total_pairs = n * (n - 1) / 2
            kendall_tau = (concordant - discordant) / (total_pairs + 1e-10)
            fdc_signal = np.clip(kendall_tau, -1.0, 1.0)
        else:
            fdc_signal = 0.0

        exploit_weight = 0.5 + 0.5 * fdc_signal

        if not hasattr(self, '_fitness_success_count'):
            self._fitness_success_count = np.zeros(self.np)

        improved = self.personal_best_fitness > self.current_fitness
        self._fitness_success_count[improved] += 1
        self._fitness_success_count[~improved] *= 0.9
        success_norm = self._fitness_success_count / (np.max(self._fitness_success_count) + 1.0)

        vel_scale = np.ones(self.np)

        top_mask = fitness_ranks < 0.25
        mid_mask = (fitness_ranks >= 0.25) & (fitness_ranks < 0.75)
        bot_mask = fitness_ranks >= 0.75

        vel_scale[top_mask] = 0.6 + 0.4 * exploit_weight
        vel_scale[mid_mask] = 1.0 + 0.3 * (1.0 - exploit_weight)
        vel_scale[bot_mask] = 1.5 + 0.5 * (1.0 - exploit_weight)

        vel_scale = vel_scale * (1.0 + 0.4 * success_norm)
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        pull_strength = 0.3 * (fitness_ranks ** 1.5)
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = pull_strength[:, np.newaxis] * to_best_dir
        else:
            directional = pull_strength[:, np.newaxis] * np.random.uniform(-1, 1, (self.np, self.dim))

        random_perturb = (0.2 + 0.4 * fitness_ranks[:, np.newaxis]) * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catF_v06(self):
        """SVD-whitening with temporal collapse detection via EMA of condition number.
        
        2 wins on tasks 2, 22. Tracks condition number and singular value ratios
        ACROSS GENERATIONS using exponential moving averages. Detects when
        population is collapsing vs converging well.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            singular_values = np.clip(singular_values, 1e-10, None)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10

            if not hasattr(self, '_ema_cond'):
                self._ema_cond = cond
            alpha_cond = 0.1
            self._ema_cond = alpha_cond * cond + (1 - alpha_cond) * self._ema_cond

            cond_ema_diff = cond - self._ema_cond
            is_collapsing = cond_ema_diff > 0.0

            if len(singular_values) >= 2:
                sv_ratio = singular_values[-1] / (singular_values[0] + 1e-10)
                if not hasattr(self, '_ema_sv_ratio'):
                    self._ema_sv_ratio = sv_ratio
                alpha_sv = 0.05
                self._ema_sv_ratio = alpha_sv * sv_ratio + (1 - alpha_sv) * self._ema_sv_ratio
                is_degenerate = self._ema_sv_ratio > 0.1
            else:
                is_degenerate = False

            if not hasattr(self, '_prev_ema_cond'):
                self._prev_ema_cond = self._ema_cond
            cond_velocity = self._ema_cond - self._prev_ema_cond
            self._prev_ema_cond = self._ema_cond

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            if is_collapsing:
                collapse_boost = 1.0 + 0.5 * np.clip(cond_ema_diff / (self._ema_cond + 1e-10), 0.0, 2.0)
                per_component_scale *= collapse_boost
            elif is_degenerate:
                per_component_scale += 0.3

            momentum_factor = 1.0 - 0.2 * np.sign(cond_velocity) * np.clip(abs(cond_velocity) / (self._ema_cond + 1e-10), 0.0, 1.0)
            per_component_scale *= np.clip(momentum_factor, 0.5, 1.5)

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catG_v07(self):
        """Monte Carlo random subspace SVD for robust population whitening.
        
        2 wins on tasks 10, 14. Uses multiple random orthogonal projections to
        compute SVD transformations in different subspaces, then averages the
        resulting velocity modulations.
        """
        centered = self.population - np.mean(self.population, axis=0)
        n_mc_samples = 15

        try:
            accumulated_velocity = np.zeros_like(self.velocity)

            for _ in range(n_mc_samples):
                R = np.random.randn(self.dim, self.dim)
                Q, _ = np.linalg.qr(R)

                proj_pop = centered @ Q
                proj_vel = self.velocity @ Q

                U, singular_values, Vt = np.linalg.svd(proj_pop, full_matrices=False)
                singular_values = np.clip(singular_values, 1e-10, None)

                if singular_values[0] < 1e-10:
                    continue

                sv_norm = singular_values / (singular_values[0] + 1e-10)

                spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
                spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

                cond = singular_values[0] / (singular_values[-1] + 1e-10)
                blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
                uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
                per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

                vel_scaled = proj_vel * per_component_scale

                accumulated_velocity += vel_scaled @ Q.T

            avg_velocity = accumulated_velocity / n_mc_samples
            new_population = self.population + avg_velocity

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catH_v08(self):
        """Hybrid: SVD spectral conditioning + graph-based clustering detection.
        
        1 win on task 9. Combines SVD whitening with k-NN connected components
        to detect fragmentation/stagnation. Weighting is stagnation-driven.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            sv_norm = singular_values / (singular_values[0] + 1e-10)

            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

            visited = np.zeros(self.np, dtype=bool)
            num_components = 0
            for start in range(self.np):
                if visited[start]:
                    continue
                stack = [start]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    for neighbor in knn_indices[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
                num_components += 1

            fragmentation = num_components / max(1, self.np)
            fragmentation = np.clip(fragmentation, 0.0, 1.0)

            stagnation_signal = np.clip(np.log1p(self.stagnation_counter) / np.log1p(100), 0.0, 1.0)

            exploration_weight = stagnation_signal * 0.5 + 0.25

            cluster_scale = 1.0 + fragmentation * stagnation_signal

            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = (
                uniform_scale * (1.0 - exploration_weight) + 
                spectral_modulation * exploration_weight * cluster_scale
            )

            vel_magnitude_scale = 1.0 + stagnation_signal * 0.5
            vel_magnitude_scale = np.clip(vel_magnitude_scale, 1.0, 1.5)

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale * vel_magnitude_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catA_v09(self):
        """Axis-aligned bounding box normalization + centroid-distance-weighted velocity.
        
        5 wins on tasks 3, 13, 15, 17, 18. Uses literal spatial layout — per-dimension
        bounding box to detect anisotropy, then centroid distance to weight exploration.
        """
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        bb_range = pop_max - pop_min
        bb_range = np.clip(bb_range, 1e-10, None)

        bb_cond = np.max(bb_range) / (np.min(bb_range) + 1e-10)
        bb_cond = np.clip(bb_cond, 1.0, 1000.0)

        centered = self.population - np.mean(self.population, axis=0)
        bb_normalized = centered / bb_range

        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        max_dist = np.max(dist_to_centroid) + 1e-10
        norm_dist = dist_to_centroid / max_dist

        centroid_weights = 1.0 / (norm_dist + 0.1)
        centroid_weights = centroid_weights / (np.max(centroid_weights) + 1e-10)
        centroid_weights = np.clip(centroid_weights, 0.3, 2.5)

        per_dim_spread = bb_normalized / (np.std(bb_normalized, axis=0) + 1e-10)
        per_dim_spread = np.clip(np.abs(per_dim_spread), 0.0, 3.0)

        blend_weight = np.clip((bb_cond - 10.0) / 100.0, 0.0, 0.5)

        vel_scale = centroid_weights[:, np.newaxis] * (1.0 - blend_weight * per_dim_spread)
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        new_population = self.population + vel_scale * self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB_v10(self):
        """Condition-adaptive rank-truncated whitening (variant_10, Category B).
        
        1 win on task 1. Key differences: Tikhonov regularization on
        eigendecomposition, logarithmic spectral modulation, rank-adaptive damping.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-14, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            if eigenvalues[-1] <= 0:
                eigenvalues[-1] = 1e-14

            cond = eigenvalues[0] / eigenvalues[-1]
            total_var = np.sum(eigenvalues) + 1e-10

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim = np.clip(eff_dim, 1, len(eigenvalues))

            rank_ratio = eff_dim / len(eigenvalues)
            reg_strength = 0.01 * (1.0 - rank_ratio) + 0.001
            reg_strength = np.clip(reg_strength, 0.001, 0.1)

            log_eig_ratio = np.log1p(eigenvalues / (eigenvalues[-1] + 1e-14))
            log_cond = np.log1p(cond)

            base_modulation = 1.0 / (log_eig_ratio + 1.0)
            base_modulation = base_modulation / (np.max(base_modulation) + 1e-10)

            blend_weight = np.clip(log_cond / 10.0, 0.0, 0.8)

            uniform_scale = np.clip(1.0 / np.sqrt(log_cond + 1.0), 0.3, 1.5)

            per_component_scale = uniform_scale * (1.0 - blend_weight) + base_modulation * blend_weight

            high_spread_mask = eigenvalues / eigenvalues[0] > 0.5
            per_component_scale[high_spread_mask] *= 0.7

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ eigenvectors.T)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        dispatch_map = {
            'original':      self._position_update_original,
            'catA_v01':      self._position_update_catA_v01,
            'catB_v02':      self._position_update_catB_v02,
            'catC_v03':      self._position_update_catC_v03,
            'catD_v04':      self._position_update_catD_v04,
            'catF_v06':      self._position_update_catF_v06,
            'catG_v07':      self._position_update_catG_v07,
            'catH_v08':      self._position_update_catH_v08,
            'catA_v09':      self._position_update_catA_v09,
            'catB_v10':      self._position_update_catB_v10,
        }
        
        method = dispatch_map.get(operator, self._position_update_original)
        method()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING: rank-based reward update
    # -------------------------------------------------------------------------
    
    def _compute_rank_reward(self, best_fitness):
        """Compute rank-based reward: percentile of best_fitness in sliding window.
        
        This is SCALE-INDEPENDENT (rule a): no magic constants like *1e5.
        Returns value in [0, 1] where 1.0 = best in window, 0.0 = worst.
        """
        if len(self._fitness_history) == 0:
            return 1.0
        
        # Get all fitness values in window
        window_fitness = [f for f, _ in self._fitness_history[-self._reward_window_size:]]
        window_fitness.append(best_fitness)
        
        # Rank: count how many are worse (higher) than current
        n_worse = sum(1 for f in window_fitness if f > best_fitness)
        n_total = len(window_fitness)
        
        # Rank percentile: higher is better
        rank_percentile = n_worse / max(1, n_total - 1)
        return np.clip(rank_percentile, 0.0, 1.0)
    
    def _update_posterior(self, operator, reward):
        """Update Beta posterior for operator based on rank-based reward.
        
        reward in [0, 1]: 1.0 = best in window, 0.0 = worst.
        Success (reward > 0.5) → increase alpha; failure → increase beta.
        """
        # Transform reward to success/failure with noise for exploration
        success_prob = reward + 0.1 * np.random.randn()
        success_prob = np.clip(success_prob, 0.01, 0.99)
        
        if np.random.random() < success_prob:
            # Success: increase alpha (more wins)
            self._alpha[operator] += 0.5
        else:
            # Failure: increase beta (more losses)
            self._beta[operator] += 0.5
        
        # Soft cap to prevent numerical issues
        self._alpha[operator] = np.clip(self._alpha[operator], 0.1, 1000.0)
        self._beta[operator] = np.clip(self._beta[operator], 0.1, 1000.0)
    
    def _thompson_sample(self):
        """Sample from Beta posteriors and return the operator with highest sample."""
        samples = {}
        for op in self._operators:
            alpha = max(self._alpha[op], 0.1)
            beta = max(self._beta[op], 0.1)
            samples[op] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
    # -------------------------------------------------------------------------
    # RESTART & LOCAL BEST
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
        Run the optimizer with Thompson Sampling position-update selection.
        
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
        
        # Reset Thompson Sampling state
        self._fitness_history = []
        self._current_operator = None
        
        # Reset per-operator success counters
        self._op_success = {op: 0 for op in self._operators}
        self._op_attempts = {op: 0 for op in self._operators}
        
        # Reset stateful operators (catB_v02, catD_v04, catF_v06)
        if hasattr(self, '_prev_whitened_vel'):
            del self._prev_whitened_vel
        if hasattr(self, '_fitness_success_count'):
            self._fitness_success_count = np.zeros(self.np)
        if hasattr(self, '_ema_cond'):
            del self._ema_cond
        if hasattr(self, '_ema_sv_ratio'):
            del self._ema_sv_ratio
        if hasattr(self, '_prev_ema_cond'):
            del self._prev_ema_cond
        
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
            if hasattr(self, '_fitness_success_count'):
                self._fitness_success_count = np.zeros(self.np)
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Record initial fitness
        self._fitness_history.append((float(self.global_best_fitness), 'initial'))
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Thompson Sampling: select operator
            self._current_operator = self._thompson_sample()
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
            self._velocity_update_base()
            self._dispatch_position_update(self._current_operator)
            
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
            
            # Record fitness and operator
            gen_best = float(self.global_best_fitness)
            self._fitness_history.append((gen_best, self._current_operator))
            
            # Keep window bounded
            if len(self._fitness_history) > self._reward_window_size + 10:
                self._fitness_history = self._fitness_history[-self._reward_window_size:]
            
            # Compute rank-based reward and update posterior
            reward = self._compute_rank_reward(gen_best)
            self._update_posterior(self._current_operator, reward)
            
            # Track per-operator statistics
            self._op_attempts[self._current_operator] += 1
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
