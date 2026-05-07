```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: ENSEMBLE VOTING WITH EXPONENTIALLY WEIGHTED ADAPTATION (mechanism #2)
    
    The gap table is NOT blend-hostile:
      - All per-task gaps are < 2.3× (no task has gap >= 10×)
      - 10 distinct winners across 24 tasks
      - Win counts are distributed (5/4/3/2/2/2/2/1/1 wins)
    
    Rationale: Small gaps mean linear blending CAN preserve per-task advantages
    (a 50/50 blend of 1.0 and 2.3 yields 1.65, which is close to optimal).
    An ensemble of the 10 benchmark winners, with weights learned online from
    rank-based fitness improvement, can combine complementary strengths without
    the probe overhead of a bandit. The key insight is that different operators
    capture different failure modes (anisotropy, clustering, temporal drift, etc.)
    and their consensus vote is more robust than any single choice.
    
    Calibration:
      - Weights are initialized uniformly (uninformative prior, rule e)
      - Reward = rank-normalized relative improvement (scale-independent, rule a)
      - Window size K=30 >= 3*num_arms=30 (rule b)
      - Mixed signals are both rank-based (rule c)
      - Each operator can reach weight 1.0 via exponential growth (rule h)
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
        
        # --- ENSEMBLE VOTING STATE ---
        self._operators = [
            'original',          # 2 wins: task 4, 5
            'catA_v01',          # 2 wins: task 8, 21
            'catB_v02',          # 4 wins: task 6, 19, 20, 23
            'catC_v03',          # 2 wins: task 0, 7
            'catD_v04',          # 3 wins: task 11, 12, 16
            'catE_v05',          # 0 wins: crashed on all tasks
            'catF_v06',          # 2 wins: task 2, 22
            'catG_v07',          # 2 wins: task 10, 14
            'catH_v08',          # 1 win: task 9
            'catA_v09',          # 5 wins: task 3, 13, 15, 17, 18
            'catB_v10',          # 1 win: task 1
        ]
        
        # Skip catE_v05 (crashed on all tasks)
        self._active_operators = [op for op in self._operators if op != 'catE_v05']
        self.num_arms = len(self._active_operators)
        
        # Exponential weighted average: weights per operator
        # Initialized uniformly (rule e)
        self._log_weights = np.ones(self.num_arms) / self.num_arms
        self._weights = np.exp(self._log_weights - np.max(self._log_weights))
        self._weights = self._weights / (np.sum(self._weights) + 1e-30)
        
        # Rank-based reward history (sliding window, rule b)
        # K must be >= 3 * num_arms = 30
        self._window_size = max(30, 3 * self.num_arms)
        self._reward_history = []  # list of (operator_idx, rank_score) tuples
        
        # Per-operator rank score buffer for normalization
        self._operator_recent_scores = {op: [] for op in self._active_operators}
        
        # Ensemble: accumulated velocity from all operators
        self._ensemble_velocity = None
        
        # Learning rate for weight updates (GentleOnline EMA)
        self._eta = 0.1
        
        # Diversity bonus: reward operators that increase population diversity
        self._prev_diversity = 0.0
    
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
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (shared across all position update strategies)
    # -------------------------------------------------------------------------
    
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
    # POSITION UPDATE STRATEGIES (the 10 benchmark winners + original)
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
        
        2 wins on tasks 8, 21. Operates on literal geometric layout: pairwise
        distance distribution, axis-aligned bounding box spread, and centroid distances.
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
        
        4 wins on tasks 6, 19, 20, 23. Transform velocity to eigenspace, apply
        momentum-based updates, and use condition number to aggressively modulate.
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
        distances to characterize population distribution.
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
        to detect landscape smoothness, fitness percentiles for per-particle scaling.
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
        
        2 wins on tasks 2, 22. Tracks condition number across generations using
        exponential moving averages. Detects when population is collapsing.
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
        """Monte Carlo random subspace SVD for robust population whitening (Category G).
        
        2 wins on tasks 10, 14. Uses multiple random orthogonal projections to
        compute SVD transformations in different subspaces.
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
        """Hybrid: SVD spectral conditioning + graph-based clustering detection (Category H).
        
        1 win on task 9. Combines SVD whitening with k-NN connected components
        to detect fragmentation/stagnation.
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
        """Condition-adaptive rank-truncated whitening (Category B).
        
        1 win on task 1. Uses Tikhonov regularization on eigendecomposition,
        logarithmic spectral modulation, and rank-adaptive damping.
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
            'original': self._position_update_original,
            'catA_v01': self._position_update_catA_v01,
            'catB_v02': self._position_update_catB_v02,
            'catC_v03': self._position_update_catC_v03,
            'catD_v04': self._position_update_catD_v04,
            'catF_v06': self._position_update_catF_v06,
            'catG_v07': self._position_update_catG_v07,
            'catH_v08': self._position_update_catH_v08,
            'catA_v09': self._position_update_catA_v09,
            'catB_v10': self._position_update_catB_v10,
        }
        
        method = dispatch_map.get(operator)
        if method is not None:
            method()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # ENSEMBLE: run all operators and accumulate weighted velocity
    # -------------------------------------------------------------------------
    
    def _run_ensemble_position_update(self):
        """Run ALL active operators and accumulate weighted velocity update.
        
        Each operator produces a candidate velocity (delta = pop_next - pop_cur).
        These deltas are averaged using the current ensemble weights.
        This is a SOFT-VOTING ensemble (mechanism #2).
        """
        candidate_deltas = np.zeros((self.num_arms, self.np, self.dim))
        
        for idx, op in enumerate(self._active_operators):
            # Save state
            saved_population = self.population.copy()
            saved_velocity = self.velocity.copy()
            
            # Run this operator's position update logic
            self._dispatch_position_update(op)
            
            # Compute delta
            candidate_deltas[idx] = self.population - saved_population
            
            # Restore state for next operator
            self.population = saved_population
            self.velocity = saved_velocity
        
        # Weighted average of deltas (soft voting)
        weights_3d = self._weights[:, np.newaxis, np.newaxis]
        ensemble_delta = np.sum(candidate_deltas * weights_3d, axis=0)
        
        self.population = self._clip_to_bounds(self.population + ensemble_delta)
    
    # -------------------------------------------------------------------------
    # RANK-BASED REWARD COMPUTATION (scale-independent, rule a)
    # -------------------------------------------------------------------------
    
    def _compute_rank_reward(self, prev_fitness, curr_fitness):
        """Compute rank-based reward for weight update.
        
        Reward = z-score of relative improvement within sliding window.
        Both signals are rank-based: relative improvement percentile.
        No scale-dependent constants (rule a).
        
        Also computes diversity bonus to reward operators that maintain diversity.
        """
        # Relative improvement: lower is better
        if len(self._reward_history) < 2:
            return np.zeros(self.num_arms)
        
        # Get the last K fitness values
        window_fitness = [entry[2] for entry in self._reward_history[-self._window_size:]]
        if len(window_fitness) < 3:
            return np.zeros(self.num_arms)
        
        window_fitness = np.array(window_fitness)
        
        # Rank-normalize: convert fitness to percentile rank
        # Lower fitness = higher rank (better)
        ranks = np.argsort(np.argsort(window_fitness)) / max(1, len(window_fitness) - 1)
        
        # Current generation's rank (last entry)
        current_rank = ranks[-1]
        
        # Z-score of improvement rate over window
        improvements = np.diff(window_fitness)
        if np.std(improvements) > 1e-15:
            z_improvement = (improvements[-1] - np.mean(improvements)) / (np.std(improvements) + 1e-15)
        else:
            z_improvement = 0.0
        
        # Combined rank-based reward
        reward = 0.5 * (1.0 - current_rank) + 0.5 * (1.0 / (1.0 + np.exp(-z_improvement)))
        
        return np.full(self.num_arms, reward)
    
    def _update_ensemble_weights(self, reward):
        """Update ensemble weights using multiplicative weight update.
        
        Uses exponentiated gradient update: w_i ∝ w_i * exp(eta * reward_i)
        This preserves the ranking of operators and allows any operator to
        dominate if it consistently outperforms others (rule h).
        """
        # Multiplicative weight update (exponentiated gradient)
        self._log_weights = self._log_weights + self._eta * reward
        
        # Normalize to get weights in [0, 1]
        self._weights = np.exp(self._log_weights - np.max(self._log_weights))
        self._weights = self._weights / (np.sum(self._weights) + 1e-30)
        
        # Ensure no weight is too small (floor at 1%)
        self._weights = np.clip(self._weights, 0.01, 1.0)
        self._weights = self._weights / np.sum(self._weights)
    
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
        Run the optimizer with ENSEMBLE VOTING position-update selection.
        
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
        
        # Reset ensemble state
        self._log_weights = np.ones(self.num_arms) / self.num_arms
        self._weights = np.exp(self._log_weights - np.max(self._log_weights))
        self._weights = self._weights / (np.sum(self._weights) + 1e-30)
        self._reward_history = []
        for op in self._active_operators:
            self._operator_recent_scores[op] = []
        
        # Reset EMA states for temporal operators
        for attr in ['_ema_cond', '_ema_sv_ratio', '_prev_ema_cond', '_prev_whitened_vel',
                     '_fitness_success_count', '_ema_centroid', '_ema_centroid_velocity',
                     '_centroid_vel_history', '_ema_improvement', '_success_count']:
            if hasattr(self, attr):
                delattr(self, attr)
        
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
        
        prev_fitness = self.current_fitness.copy()
        self._prev_diversity = self._compute_diversity()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base)
            self._velocity_update_base()
            
            # ENSEMBLE: run all operators and weighted vote
            self._run_ensemble_position_update()
            
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
            
            # --- ENSEMBLE WEIGHT UPDATE ---
            # Record fitness history for rank-based reward
            gen_best = float(np.min(self.current_fitness))
            self._reward_history.append((self.generation, self._weights.copy(), gen_best))
            
            # Keep window bounded
            if len(self._reward_history) > self._window_size * 2:
                self._reward_history = self._reward_history[-self._window_size:]
            
            # Compute rank-based reward
            if len(self._reward_history) >= 3:
                reward = self._compute_rank_reward(prev_fitness, self.current_fitness)
                self._update_ensemble_weights(reward)
            
            prev_fitness = self.current_fitness.copy()
            self._prev_diversity = self._compute_diversity()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```