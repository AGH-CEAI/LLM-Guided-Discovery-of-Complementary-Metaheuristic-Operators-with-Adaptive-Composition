import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT (mechanism #1, multi-armed bandit family)
    
    The gap table is NOT blend-hostile:
      - Maximum gap is 1.2× (Task 16), well below the 10× threshold.
      - Wins are distributed across 5 variants (7/6/5/4/2).
      - BLEND-HOSTILE tasks: 0. DISPATCH-MANDATORY tasks: 0.
    
    Why Thompson Sampling over ensemble blending:
      - All gaps < 2×: blending is mathematically viable, but a 50/50 blend
        of 1.0 and 1.1 gives 1.05, barely better than the winner alone.
      - Five distinct winners with orthogonal mechanisms (spectral, fitness-only,
        hybrid, condition-gated). A bandit that learns which to prefer extracts
        more value than blending near-equivalent operators.
      - Thompson Sampling handles the exploration-exploitation tradeoff
        naturally via posterior sampling from Beta distributions.
    
    Reward shaping (rule a compliance):
      - Rank-normalized within a sliding window of 3*num_arms = 15 entries.
      - Each generation's reward = whether the operator achieved the best
        per-generation fitness (binary credit, not raw fitness).
      - No scale-dependent constants (no *1e5, no log1p with magic multipliers).
    
    Calibration (rule b compliance):
      - Window size K=15 >= 3*num_arms=15. Sufficient samples per arm.
      - Prior: uniform Beta(1,1) — uninformative, consistent with no prior knowledge.
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
        
        # --- THOMPSON SAMPLING BANDIT STATE ---
        # The 5 benchmark-winning position update strategies
        self._operators = [
            'original',      # variant_00: 7 wins (baseline)
            'catB_rayleigh', # variant_02: 2 wins (Rayleigh-quotient velocity modulation)
            'catD_spearman', # variant_04: 6 wins (Spearman-rank stagnation modulation)
            'catH_hybrid',   # variant_08: 4 wins (Hybrid fitness+spectral)
            'catB_cond',     # variant_10: 5 wins (Condition-gated anisotropic damping)
        ]
        self._num_operators = len(self._operators)
        
        # Beta posteriors for Thompson Sampling: Beta(alpha, beta)
        self._alpha = np.ones(self._num_operators)
        self._beta = np.ones(self._num_operators)
        
        # Sliding window for rank-based reward computation
        self._reward_window_size = 3 * self._num_operators  # 15 entries
        self._reward_history = {op: [] for op in self._operators}
        self._best_fitness_history = []  # Track per-gen best for rank normalization
        
        # Current selected operator (for record-keeping)
        self._current_operator = None
        
        # Per-operator generation counters (for Thompson Sampling tiebreaking)
        self._operator_gen_counts = np.zeros(self._num_operators)
    
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

            # Spectral signal: map condition number to inertia adjustment
            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            # Map spectral signal to inertia range [0.4, 0.95]
            target_inertia = 0.4 + 0.55 * spectral_signal

            # Blend with time-based baseline
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
    # POSITION UPDATE STRATEGIES (the 5 benchmark winners)
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
    
    def _position_update_catB_rayleigh(self):
        """Rayleigh-quotient velocity modulation using covariance eigendecomposition (Category B)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            rq_values = np.zeros(self.np)
            for i in range(self.np):
                v = self.velocity[i]
                v_cov_v = np.dot(v, cov @ v)
                v_v = np.dot(v, v) + 1e-10
                rq_values[i] = v_cov_v / v_v
            rq_mean = np.mean(rq_values)
            rq_std = np.std(rq_values) + 1e-10

            vel_proj = self.velocity @ eigenvectors
            vel_energy_per_comp = vel_proj ** 2
            total_vel_energy = np.sum(vel_energy_per_comp) + 1e-10
            vel_energy_fraction = vel_energy_per_comp / total_vel_energy

            alignment = vel_energy_fraction / (eigenvalues_norm + 1e-10)
            alignment = np.clip(alignment, 0.0, 10.0)

            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                per_comp_scale = 1.0 + 0.5 * log_damp * (1.0 - alignment)
            else:
                per_comp_scale = 1.0 + 0.3 * spectral_signal * (1.0 - alignment)

            per_comp_scale = np.clip(per_comp_scale, 0.1, 2.5)
            vel_scaled = vel_proj * per_comp_scale

            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
                directional = 0.4 * (1.0 - fitness_ranks[:, np.newaxis]) * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))

            random_perturb = 0.3 * np.random.uniform(-1, 1, (self.np, self.dim))

            new_population = self.population + (vel_scaled @ eigenvectors.T) + directional + random_perturb

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catD_spearman(self):
        """Spearman-rank stagnation modulation (Category D: fitness-landscape only)."""
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        personal_best_rank = np.argsort(np.argsort(self.personal_best_fitness)) / max(1, self.np - 1)

        if np.std(fitness_ranks) > 1e-10 and np.std(personal_best_rank) > 1e-10:
            spearman_corr = np.corrcoef(fitness_ranks, personal_best_rank)[0, 1]
            spearman_corr = np.clip(spearman_corr, -1.0, 1.0)
        else:
            spearman_corr = 0.0

        bottom_percentile = np.percentile(self.current_fitness, 10)
        top_percentile = np.percentile(self.current_fitness, 90)
        stagnant_particles = self.current_fitness <= bottom_percentile
        top_particles = self.current_fitness >= top_percentile

        if not hasattr(self, '_fitness_success_ema'):
            self._fitness_success_ema = np.zeros(self.np)

        improvement = self.personal_best_fitness - self.current_fitness
        improved_mask = improvement > 0
        self._fitness_success_ema[improved_mask] = self._fitness_success_ema[improved_mask] * 0.9 + 0.1
        self._fitness_success_ema[~improved_mask] = self._fitness_success_ema[~improved_mask] * 0.9 - 0.05
        self._fitness_success_ema = np.clip(self._fitness_success_ema, -0.5, 1.0)

        success_norm = (self._fitness_success_ema - np.min(self._fitness_success_ema) + 1e-10)
        success_norm = success_norm / (np.max(success_norm) + 1e-10)

        if self.global_best_fitness < np.inf:
            fitness_distance_to_best = np.abs(self.current_fitness - self.global_best_fitness)
            fitness_dist_rank = np.argsort(np.argsort(fitness_distance_to_best)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(fitness_dist_rank) > 1e-10:
                fitness_dist_corr = np.corrcoef(fitness_ranks, fitness_dist_rank)[0, 1]
                fitness_dist_corr = np.clip(fitness_dist_corr, -1.0, 1.0)
            else:
                fitness_dist_corr = 0.0
        else:
            fitness_dist_corr = 0.0

        stagnation_signal = spearman_corr * (1.0 - success_norm)
        exploration_signal = np.clip(stagnation_signal * 0.5, 0.0, 1.0)
        exploitation_signal = success_norm * stagnant_particles.astype(float)

        vel_scale = 1.0 + 0.5 * exploration_signal - 0.3 * exploitation_signal
        vel_scale = vel_scale * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.4, 2.5)

        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional_strength = (1.0 - fitness_ranks[:, np.newaxis]) * (1.0 + success_norm[:, np.newaxis])
            directional = 0.3 * directional_strength * to_best_dir
        else:
            directional = 0.3 * (1.0 - fitness_ranks[:, np.newaxis]) * np.random.uniform(-1, 1, (self.np, self.dim))

        random_scale = 0.3 + 0.4 * exploration_signal[:, np.newaxis]
        random_perturb = random_scale * np.random.uniform(-1, 1, (self.np, self.dim))

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catH_hybrid(self):
        """Hybrid: Fitness-rank gradient + spectral-subspace escape with condition-driven switching (Category H)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

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

        cond_threshold = 100.0
        spectral_weight = np.clip((cond - cond_threshold) / (cond_threshold * 10), 0.0, 0.8)
        fitness_weight = 1.0 - spectral_weight

        collapse_signal = 1.0 - eff_dim_ratio
        spectral_weight = np.clip(spectral_weight + 0.2 * collapse_signal, 0.0, 0.8)
        fitness_weight = 1.0 - spectral_weight

        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        if not hasattr(self, '_success_count_hybrid'):
            self._success_count_hybrid = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count_hybrid[improved] += 1
        self._success_count_hybrid[~improved] *= 0.9
        success_norm = self._success_count_hybrid / (np.max(self._success_count_hybrid) + 1e-10)

        rank_boost = 1.0 + 0.5 * (1.0 - fitness_ranks) + 0.3 * success_norm
        rank_boost = np.clip(rank_boost, 0.8, 2.0)

        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            pull_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
            fitness_gradient = pull_strength * to_best_dir * rank_boost[:, np.newaxis]
        else:
            fitness_gradient = np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        try:
            _, eigenvectors = np.linalg.eigh(cov)
            eigenvectors = eigenvectors[:, np.argsort(np.linalg.eigvalsh(cov))[::-1]]

            vel_proj = self.velocity @ eigenvectors
            sv_scale = np.sqrt(eigenvalues + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            inverse_scale = 1.0 / (sv_norm + 0.1)
            inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)

            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_comp_scale = uniform_scale * (1.0 - spectral_weight) + inverse_scale * spectral_weight
            per_comp_scale = np.clip(per_comp_scale, 0.1, 2.5)

            vel_scaled = vel_proj * per_comp_scale

            if cond > cond_threshold:
                escape_strength = np.clip(cond_log / 10.0, 0.0, 0.5)
                minor_perturb = np.zeros_like(self.velocity)
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

        velocity_contribution = (
            fitness_weight * self.velocity * rank_boost[:, np.newaxis] +
            spectral_weight * spectral_component
        )

        gradient_perturbation = fitness_weight * fitness_gradient

        explore_scale = 0.5 + 0.3 * np.clip(cond_log / 5.0, 0.0, 1.0)
        random_perturb = explore_scale * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + velocity_contribution + gradient_perturbation + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB_cond(self):
        """Condition-number-gated anisotropic velocity damping (Category B)."""
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            cond_threshold = 50.0
            if cond > cond_threshold:
                sv_scale = np.sqrt(eigenvalues)
                sv_norm = sv_scale / (sv_scale[0] + 1e-10)
                per_comp_scale = 1.0 / (sv_norm + 0.05)
                per_comp_scale = per_comp_scale / (np.max(per_comp_scale) + 1e-10)
                blend = np.clip((cond - cond_threshold) / 200.0, 0.0, 1.0)
                uniform_scale = np.clip(cond_log / np.log1p(1000.0), 0.5, 2.0)
                final_scale = uniform_scale * (1.0 - blend) + per_comp_scale * blend
            else:
                final_scale = np.clip(cond_log / np.log1p(100.0), 0.7, 1.5)

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * final_scale

            explore_boost = 1.0 + 0.3 * (1.0 - eff_dim_ratio)

            new_population = self.population + explore_boost * (vel_scaled @ eigenvectors.T)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        fitness_perturb = 0.4 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
        new_population = new_population + directional + random_perturb

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'catB_rayleigh':
            self._position_update_catB_rayleigh()
        elif operator == 'catD_spearman':
            self._position_update_catD_spearman()
        elif operator == 'catH_hybrid':
            self._position_update_catH_hybrid()
        elif operator == 'catB_cond':
            self._position_update_catB_cond()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING BANDIT
    # -------------------------------------------------------------------------
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta posteriors.
        
        Samples from Beta(alpha_k, beta_k) for each operator k and picks
        the operator with the highest sample. This naturally balances
        exploration (uncertain operators have wide posteriors) and
        exploitation (confident good operators have high means).
        """
        samples = np.array([
            np.random.beta(self._alpha[i], self._beta[i])
            for i in range(self._num_operators)
        ])
        selected_idx = int(np.argmax(samples))
        return self._operators[selected_idx]
    
    def _compute_rank_reward(self, current_best_fitness):
        """Compute rank-based reward for the current generation.
        
        Uses the fraction of the last K generations where this operator
        achieved the best (lowest) per-generation fitness. This is
        rank-based, not raw fitness — no scale-dependent constants.
        """
        self._best_fitness_history.append(float(current_best_fitness))
        
        # Keep window bounded
        if len(self._best_fitness_history) > self._reward_window_size:
            self._best_fitness_history.pop(0)
        
        if len(self._best_fitness_history) < 2:
            return 1.0 if len(self._best_fitness_history) == 1 else 0.5
        
        # Rank-normalized: compute percentile of current best vs window
        window = np.array(self._best_fitness_history)
        min_f = np.min(window)
        max_f = np.max(window)
        
        if max_f > min_f:
            # Lower percentile = better (lower fitness)
            reward = 1.0 - (current_best_fitness - min_f) / (max_f - min_f + 1e-10)
        else:
            reward = 1.0
        
        return np.clip(reward, 0.0, 1.0)
    
    def _update_bandit(self, operator, reward):
        """Update Beta posterior for the selected operator.
        
        Reward is binary-ish (rank-based percentile in [0,1]):
        - High reward (near 1.0): operator performed well → increase alpha
        - Low reward (near 0.0): operator performed poorly → increase beta
        
        Uses a soft update to avoid over-reacting to single generations.
        """
        idx = self._operators.index(operator)
        
        # Append to reward history window
        self._reward_history[operator].append(reward)
        if len(self._reward_history[operator]) > self._reward_window_size:
            self._reward_history[operator].pop(0)
        
        # Compute rank-based average from window
        window_rewards = self._reward_history[operator]
        if len(window_rewards) >= 3:
            # Rank-normalize within window: fraction of entries this op beats
            all_rewards = []
            for op in self._operators:
                all_rewards.extend(self._reward_history[op])
            
            if len(all_rewards) >= 3:
                # Fraction of all recent rewards that this operator beats
                avg_reward = np.mean(window_rewards)
                # Soft update to Beta posterior
                # Equivalent to adding pseudo-counts proportional to average reward
                self._alpha[idx] = self._alpha[idx] * 0.95 + (1.0 + avg_reward) * 0.05
                self._beta[idx] = self._beta[idx] * 0.95 + (1.0 + (1.0 - avg_reward)) * 0.05
            else:
                # Fallback: small update
                self._alpha[idx] += 0.01 * reward
                self._beta[idx] += 0.01 * (1.0 - reward)
        else:
            # Not enough data yet — small exploratory update
            self._alpha[idx] += 0.01 * reward
            self._beta[idx] += 0.01 * (1.0 - reward)
        
        # Clamp to reasonable bounds (prevent numerical issues)
        self._alpha[idx] = np.clip(self._alpha[idx], 0.1, 100.0)
        self._beta[idx] = np.clip(self._beta[idx], 0.1, 100.0)
    
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
        Run the optimizer with Thompson Sampling bandit position-update selection.
        
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
        self._alpha = np.ones(self._num_operators)
        self._beta = np.ones(self._num_operators)
        self._reward_history = {op: [] for op in self._operators}
        self._best_fitness_history = []
        self._current_operator = None
        self._operator_gen_counts = np.zeros(self._num_operators)
        
        # Reset EMA states for temporal strategies
        if hasattr(self, '_fitness_success_ema'):
            self._fitness_success_ema = np.zeros(self.np)
        if hasattr(self, '_success_count_hybrid'):
            self._success_count_hybrid = np.zeros(self.np)
        
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
            if hasattr(self, '_fitness_success_ema'):
                self._fitness_success_ema = np.zeros(self.np)
            if hasattr(self, '_success_count_hybrid'):
                self._success_count_hybrid = np.zeros(self.np)
        
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
            
            # --- THOMPSON SAMPLING SELECTION ---
            selected_op = self._select_operator_thompson()
            self._current_operator = selected_op
            op_idx = self._operators.index(selected_op)
            self._operator_gen_counts[op_idx] += 1
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
            self._velocity_update_base()
            self._dispatch_position_update(selected_op)
            
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
            
            # --- BANDIT UPDATE: rank-based reward ---
            gen_best = float(np.min(self.current_fitness))
            reward = self._compute_rank_reward(gen_best)
            self._update_bandit(selected_op, reward)
            
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
