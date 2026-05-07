import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT (mechanism #1) with rank-normalized rewards.
    
    Why this fits the gap table:
      - ALL per-task gaps are < 5× (max is 4.8× on Task 5). No blend-hostile tasks.
      - Win count is distributed: 5/4/4/3/2/2/2/1/1 across 9 winners.
      - Each winning variant uses a FUNDAMENTALLY DIFFERENT mechanism
        (SVD, information theory, Spearman-Kendall, MST, temporal autocorrelation,
        bootstrap, hybrid gating). A bandit can exploit these distinct biases.
      - Rank-based scoring within a 30-gen sliding window ensures rewards are
        on the same scale — no scale-dependent magic constants.
    
    Calibration notes:
      - Window size 30 >= 3 * 9 arms (calibration rule b).
      - Rewards are rank-normalized: score = (worst - current) / (worst - best + eps),
        clipped to [0, 1]. No hard-coded multipliers.
      - Gaussian posteriors (mu, precision) updated each generation.
      - Thompson Sampling commits to ONE operator per generation (no blending),
        preserving per-task advantages when they exist.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology
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
        
        # =================================================================
        # THOMPSON SAMPLING BANDIT — rank-normalized, sliding window
        # =================================================================
        # The 9 benchmark-winning position update strategies
        self._operators = [
            'original',        # baseline (0 wins in gap table but included for completeness)
            'variant_02_svd',  # SVD subspace collapse recovery (4 wins)
            'variant_03_info', # information-theoretic entropy + GMM (2 wins)
            'variant_04_rank', # Spearman-Kendall fitness-rank only (4 wins)
            'variant_05_mst',  # MST bridge-breaking topology (3 wins)
            'variant_06_temp', # temporal autocorrelation stagnation (2 wins)
            'variant_07_boot', # bootstrap Monte Carlo uncertainty (1 win)
            'variant_08_hybrid', # improvement-gated hybrid (5 wins)
            'variant_10_power', # subspace power-law SVD (2 wins)
        ]
        
        # Gaussian posterior per arm: (mu, precision)
        self._arm_mu = {op: 0.5 for op in self._operators}
        self._arm_precision = {op: 1.0 for op in self._operators}
        
        # Sliding window of rank-based scores per arm
        self._window_size = 30
        self._arm_scores = {op: [] for op in self._operators}
        
        # Track per-arm sample counts for UCB tiebreaker on initialization
        self._arm_count = {op: 0 for op in self._operators}
        
        # Global rank buffer for score normalization
        self._fitness_buffer = []  # list of best-fitness values per gen
        self._buffer_max = self._window_size + 5
    
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
        """Adapt neighborhood size using information-theoretic signals."""
        fitness = self.current_fitness
        if len(fitness) > 1:
            n_bins = min(20, max(3, len(fitness) // 2))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0]
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_fitness_entropy = np.log(n_bins)
            norm_fitness_ent = fitness_entropy / (max_fitness_entropy + 1e-10)
        else:
            norm_fitness_ent = 0.5

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
            uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
            uniform = uniform + 1e-10
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
            max_kl = np.log(len(eigenvalues_norm))
            norm_kl = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
        except:
            norm_kl = 0.5

        exploration_signal = 0.4 * (1.0 - norm_fitness_ent) + 0.6 * norm_kl
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

        min_ns = 1
        max_ns = max(3, self.np // 4)
        target_ns = int(np.round(min_ns + exploration_signal * (max_ns - min_ns)))
        target_ns = np.clip(target_ns, min_ns, max_ns)

        if target_ns > self.neighborhood_size:
            self.neighborhood_size = min(target_ns, self.neighborhood_size + 1)
        elif target_ns < self.neighborhood_size:
            self.neighborhood_size = max(target_ns, self.neighborhood_size - 1)
    
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
    
    # =========================================================================
    # POSITION UPDATE STRATEGIES — direct implementations from benchmark winners
    # =========================================================================
    
    def _position_update_original(self):
        """Baseline: eigenvalue-anisotropic velocity modulation (original.py)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
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
    
    def _position_update_variant_02_svd(self):
        """SVD Subspace Collapse Recovery (variant_02_catB_idea_0.py) — 4 wins."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            sv_norm = singular_values / (singular_values[0] + 1e-10)
            entropy_sv = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
            max_entropy = np.log(len(singular_values))
            effective_rank = np.exp(entropy_sv)
            eff_rank_ratio = effective_rank / len(singular_values)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)

            collapse_mask = sv_norm < 0.05
            inverse_boost = np.zeros_like(sv_norm)
            inverse_boost[~collapse_mask] = 1.0 / (sv_norm[~collapse_mask] + 0.01)
            inverse_boost[collapse_mask] = 5.0 / (sv_norm[collapse_mask] + 0.01)
            inverse_boost = np.clip(inverse_boost, 0.5, 10.0)
            inverse_boost = inverse_boost / (np.max(inverse_boost) + 1e-10)

            cond_log = np.log1p(cond) / np.log1p(1e6)
            cond_log = np.clip(cond_log, 0.0, 1.0)
            exploration_scale = 0.8 + 0.7 * cond_log

            if eff_rank_ratio < 0.3:
                exploration_scale *= 1.5
            elif eff_rank_ratio < 0.5:
                exploration_scale *= 1.2

            exploration_scale = np.clip(exploration_scale, 0.3, 2.5)

            vel_proj = self.velocity @ Vt.T
            vel_boosted = vel_proj * inverse_boost * exploration_scale
            new_population = self.population + (vel_boosted @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_03_info(self):
        """Information-theoretic: Entropy + KL divergence + GMM (variant_03_catC_idea_0.py) — 2 wins."""
        try:
            fitness = self.current_fitness
            n_bins = min(15, max(3, self.np // 3))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0] + 1e-10
            fitness_entropy = -np.sum(hist * np.log(hist))

            if not hasattr(self, '_info_fitness_entropy_history'):
                self._info_fitness_entropy_history = []
            self._info_fitness_entropy_history.append(fitness_entropy)
            if len(self._info_fitness_entropy_history) > 10:
                self._info_fitness_entropy_history.pop(0)

            if len(self._info_fitness_entropy_history) >= 3:
                recent = np.mean(self._info_fitness_entropy_history[-3:])
                older = np.mean(self._info_fitness_entropy_history[:-3]) if len(self._info_fitness_entropy_history) > 3 else self._info_fitness_entropy_history[0]
                entropy_trend = (recent - older) / (older + 1e-10)
            else:
                entropy_trend = 0.0

            try:
                centered = self.population - np.mean(self.population, axis=0)
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                eigenvalues = np.sort(eigenvalues)[::-1]
                eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
                dim_eff = len(eigenvalues_norm)
                uniform = np.ones(dim_eff) / dim_eff
                kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
                max_kl = np.log(dim_eff)
                kl_normalized = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
                exploration_from_kl = 0.5 + 0.5 * kl_normalized

                U, V = np.linalg.eigh(cov)
                idx = np.argsort(U)[::-1]
                U = U[idx]
                V = V[:, idx]

                vel_proj = self.velocity @ V
                sv_scale = np.sqrt(U + 1e-10)
                sv_norm = sv_scale / (sv_scale[0] + 1e-10)
                inverse_scale = 1.0 / (sv_norm + 0.1)
                inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)
                per_comp_scale = np.clip(inverse_scale, 0.2, 3.0)
                vel_scaled = vel_proj * per_comp_scale

                max_entropy_bins = np.log(n_bins)
                norm_entropy = fitness_entropy / (max_entropy_bins + 1e-10)
                entropy_collapse = 1.0 - np.clip(norm_entropy, 0.0, 1.0)

                trapped_signal = (entropy_collapse * 0.4 +
                                 max(0, -entropy_trend) * 0.3 +
                                 0.3 * kl_normalized)
                trapped_signal = np.clip(trapped_signal, 0.0, 1.0)

                exploration_factor = (1.0 + trapped_signal * 1.5) * exploration_from_kl
                exploration_factor = np.clip(exploration_factor, 0.5, 2.5)

                if self.global_best is not None:
                    to_best = self.global_best - self.population
                    to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                    to_best_dir = to_best / to_best_dist
                else:
                    to_best_dir = np.random.randn(self.np, self.dim)

                if trapped_signal > 0.4:
                    random_perturb = np.random.randn(self.np, self.dim) * trapped_signal * 0.5
                else:
                    random_perturb = np.zeros((self.np, self.dim))

                new_population = self.population + exploration_factor * (vel_scaled @ V.T) + random_perturb

                if entropy_collapse > 0.7 and entropy_trend < -0.1:
                    if self.global_best is not None:
                        n_replace = max(1, int(0.2 * self.np))
                        worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
                        self.population[worst_indices] = np.random.uniform(
                            self.lower_bound, self.upper_bound, (n_replace, self.dim)
                        )
                        self.velocity[worst_indices] = np.random.uniform(
                            self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
                        )
                        self.personal_best_fitness[worst_indices] = np.inf
                        self.personal_best[worst_indices] = self.population[worst_indices]

            except np.linalg.LinAlgError:
                new_population = self.population + self.velocity
        except Exception:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_04_rank(self):
        """Spearman-Kendall fitness-rank modulation (variant_04_catD_idea_0.py) — 4 wins."""
        try:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            if not hasattr(self, '_rank_success_count'):
                self._rank_success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._rank_success_count[improved] += 1
            self._rank_success_count[~improved] *= 0.9
            success_norm = self._rank_success_count / (np.max(self._rank_success_count) + 1.0)

            if self.global_best is not None and self.np > 2:
                dists = np.linalg.norm(self.population - self.global_best, axis=1)
                dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
                if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                    fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
                else:
                    fdc = 0.0
            else:
                fdc = 0.0

            if not hasattr(self, '_rank_fitness_history'):
                self._rank_fitness_history = []
            self._rank_fitness_history.append(np.min(self.current_fitness))
            if len(self._rank_fitness_history) > 30:
                self._rank_fitness_history.pop(0)

            if len(self._rank_fitness_history) >= 2:
                prev_fitness = self._rank_fitness_history[-2]
                curr_fitness = self._rank_fitness_history[-1]
                n = len(self.current_fitness)
                concordant = np.sum((self.current_fitness < prev_fitness) & (self.current_fitness < curr_fitness))
                discordant = np.sum((self.current_fitness > prev_fitness) | (self.current_fitness > curr_fitness))
                kendall_tau = (concordant - discordant) / (n * (n - 1) / 2 + 1e-10)
            else:
                kendall_tau = 0.0

            p10 = np.percentile(self.current_fitness, 10)
            p50 = np.percentile(self.current_fitness, 50)
            p90 = np.percentile(self.current_fitness, 90)
            fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)

            if not hasattr(self, '_rank_prev_spread'):
                self._rank_prev_spread = fitness_spread
            spread_change = abs(fitness_spread - self._rank_prev_spread)
            self._rank_prev_spread = fitness_spread

            is_converging = fitness_spread < 0.1 and spread_change < 0.01
            is_exploring = fitness_spread > 0.5 or spread_change > 0.05

            if self.global_best is not None and hasattr(self, '_rank_prev_global_best'):
                improvement = max(0.0, self._rank_prev_global_best - self.global_best_fitness)
            else:
                improvement = 0.0
            self._rank_prev_global_best = self.global_best_fitness

            if not hasattr(self, '_rank_ema_improvement'):
                self._rank_ema_improvement = 0.0
            self._rank_ema_improvement = 0.3 * improvement + 0.7 * self._rank_ema_improvement

            fdc_signal = np.clip(fdc, -1.0, 1.0)
            exploit_weight = 0.5 * (fdc_signal + 1.0)
            explore_weight = 1.0 - exploit_weight

            if kendall_tau < 0.2:
                global_scale = 1.4
            elif kendall_tau < 0.5:
                global_scale = 1.1
            else:
                global_scale = 0.85

            if is_converging:
                global_scale *= 0.75
            elif is_exploring:
                global_scale *= 1.25

            if self._rank_ema_improvement > 1e-6:
                global_scale *= 0.9
            elif self._rank_ema_improvement < 1e-10:
                global_scale *= 1.2

            rank_modulation = 1.0 + 0.5 * fitness_ranks
            success_modulation = 1.0 + 0.3 * success_norm
            vel_scale = rank_modulation[:, np.newaxis] * global_scale * success_modulation[:, np.newaxis]
            vel_scale = np.clip(vel_scale, 0.3, 2.5)

            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional_strength = 0.4 * (1.0 - fitness_ranks[:, np.newaxis])
                directional = directional_strength * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))

            random_perturb = np.random.uniform(-0.25, 0.25, (self.np, self.dim))
            random_strength = (1.0 - success_norm[:, np.newaxis] * 0.5)
            random_perturb *= random_strength * explore_weight[:, np.newaxis]

            new_population = self.population + vel_scale * self.velocity + directional + random_perturb

        except Exception:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_05_mst(self):
        """MST bridge-breaking for topology-aware exploration (variant_05_catE_idea_0.py) — 3 wins."""
        try:
            diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2) + 1e-10

            n = self.np
            in_mst = np.zeros(n, dtype=bool)
            parent = np.full(n, -1)
            mst_edges = []

            in_mst[0] = True
            for _ in range(n - 1):
                min_edge = (np.inf, -1, -1)
                for i in range(n):
                    if not in_mst[i]:
                        continue
                    for j in range(n):
                        if in_mst[j]:
                            continue
                        if dists[i, j] < min_edge[0]:
                            min_edge = (dists[i, j], i, j)
                if min_edge[1] != -1:
                    in_mst[min_edge[2]] = True
                    parent[min_edge[2]] = min_edge[1]
                    mst_edges.append((min_edge[1], min_edge[2]))

            if len(mst_edges) == 0:
                new_population = self.population + self.velocity
                self.population = self._clip_to_bounds(new_population)
                return

            edge_lengths = np.array([dists[u, v] for u, v in mst_edges])
            mean_len = np.mean(edge_lengths)
            std_len = np.std(edge_lengths) + 1e-10

            bridge_mask = edge_lengths > mean_len + 0.5 * std_len
            bridge_particles = set()
            for idx, (u, v) in enumerate(mst_edges):
                if bridge_mask[idx]:
                    bridge_particles.add(u)
                    bridge_particles.add(v)

            mst_degree = np.zeros(n)
            for u, v in mst_edges:
                mst_degree[u] += 1
                mst_degree[v] += 1

            perturbation = np.zeros((n, self.dim))
            for i in range(n):
                if i not in bridge_particles:
                    continue
                strength = 0.5 / (mst_degree[i] + 1.0)
                direction = np.random.randn(self.dim)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                perturbation[i] = strength * direction

            vel_scale = 1.0 + 0.5 * (1.0 - np.clip(mst_degree / (np.mean(mst_degree) + 1e-10), 0.0, 1.0))
            vel_scale = np.clip(vel_scale, 0.5, 2.0)

            new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + perturbation
            self.population = self._clip_to_bounds(new_population)

        except Exception:
            new_population = self.population + self.velocity
            self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_06_temp(self):
        """Pure temporal/dynamical: Autocorrelation-stagnation momentum (variant_06_catF_idea_0.py) — 2 wins."""
        try:
            if not hasattr(self, '_temp_fitness_history'):
                self._temp_fitness_history = []
            self._temp_fitness_history.append(float(np.min(self.current_fitness)))
            if len(self._temp_fitness_history) > 30:
                self._temp_fitness_history.pop(0)

            if len(self._temp_fitness_history) >= 10:
                improvements = np.diff(self._temp_fitness_history)
                mean_imp = np.mean(improvements)
                var_imp = np.var(improvements) + 1e-10
                autocorr_lag1 = np.sum(improvements[:-1] * improvements[1:]) / ((len(improvements) - 1) * var_imp)
                autocorr_lag1 = np.clip(autocorr_lag1, -1.0, 1.0)
            else:
                autocorr_lag1 = 0.0

            current_centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_temp_ema_centroid'):
                self._temp_ema_centroid = current_centroid.copy()
            self._temp_ema_centroid = 0.7 * self._temp_ema_centroid + 0.3 * current_centroid
            centroid_vel = current_centroid - self._temp_ema_centroid
            centroid_vel_mag = np.linalg.norm(centroid_vel)

            if not hasattr(self, '_temp_centroid_vel_ema_mag'):
                self._temp_centroid_vel_ema_mag = 0.0
            self._temp_centroid_vel_ema_mag = 0.7 * self._temp_centroid_vel_ema_mag + 0.3 * centroid_vel_mag

            if not hasattr(self, '_temp_stagnation_counter'):
                self._temp_stagnation_counter = 0
            if self.global_best is not None and hasattr(self, '_temp_prev_global_best'):
                if abs(self._temp_prev_global_best - self.global_best_fitness) < 1e-10:
                    self._temp_stagnation_counter += 1
                else:
                    self._temp_stagnation_counter = 0
            self._temp_prev_global_best = self.global_best_fitness if self.global_best is not None else 0.0

            stagnation_depth = min(self._temp_stagnation_counter / 50.0, 1.0)

            if not hasattr(self, '_temp_accel_history'):
                self._temp_accel_history = []
            self._temp_accel_history.append(centroid_vel_mag)
            if len(self._temp_accel_history) > 30:
                self._temp_accel_history.pop(0)

            if len(self._temp_accel_history) >= 5:
                accel_array = np.array(self._temp_accel_history)
                accel_trend = np.mean(np.diff(accel_array))
            else:
                accel_trend = 0.0
            accel_signal = -np.sign(accel_trend) * min(abs(accel_trend), 1.0)

            current_diversity = self._compute_diversity()
            if not hasattr(self, '_temp_ema_diversity'):
                self._temp_ema_diversity = current_diversity
            self._temp_ema_diversity = 0.95 * self._temp_ema_diversity + 0.05 * current_diversity
            diversity_ratio = current_diversity / (self._temp_ema_diversity + 1e-10)
            diversity_signal = 1.0 - np.clip(diversity_ratio, 0.0, 2.0)

            stuckness = (
                0.35 * max(0.0, autocorr_lag1) +
                0.25 * stagnation_depth +
                0.20 * max(0.0, accel_signal) +
                0.20 * max(0.0, diversity_signal)
            )
            stuckness = np.clip(stuckness, 0.0, 1.0)

            if stuckness > 0.7:
                base_scale = 1.6
            elif stuckness > 0.4:
                base_scale = 1.2
            else:
                base_scale = 1.0

            if diversity_ratio < 0.8:
                diversity_boost = 1.4
            elif diversity_ratio < 1.0:
                diversity_boost = 1.2
            else:
                diversity_boost = 1.0

            if stuckness > 0.5:
                if centroid_vel_mag > 1e-6:
                    vel_dir = centroid_vel / centroid_vel_mag
                else:
                    vel_dir = np.zeros(self.dim)
                kick_dir = np.random.uniform(-1, 1, self.dim)
                if centroid_vel_mag > 1e-6:
                    kick_dir -= vel_dir * np.dot(kick_dir, vel_dir)
                kick_mag = stuckness * 2.0
                directional_kick = kick_dir * kick_mag
            else:
                directional_kick = np.zeros(self.dim)

            new_population = self.population + base_scale * self.velocity * diversity_boost + directional_kick

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_07_boot(self):
        """Monte Carlo Bootstrap Uncertainty-Driven Exploration (variant_07_catG_idea_0.py) — 1 win."""
        try:
            n_particles = self.np
            dim = self.dim

            n_bootstrap = min(50, n_particles * 2)
            bootstrap_centroids = np.zeros((n_bootstrap, dim))
            for b in range(n_bootstrap):
                indices = np.random.randint(0, n_particles, size=n_particles)
                bootstrap_centroids[b] = np.mean(self.population[indices], axis=0)

            centroid_mean = np.mean(bootstrap_centroids, axis=0)
            centroid_std = np.std(bootstrap_centroids, axis=0) + 1e-10

            uncertainty_radius = np.linalg.norm(centroid_std)
            total_spread = np.std(self.population) + 1e-10
            normalized_uncertainty = uncertainty_radius / (total_spread + 1e-10)

            n_projections = min(20, dim)
            projection_scores = np.zeros(n_projections)
            for p in range(n_projections):
                proj_dir = np.random.randn(dim)
                proj_dir = proj_dir / (np.linalg.norm(proj_dir) + 1e-10)
                proj_pos = self.population @ proj_dir
                proj_vel = self.velocity @ proj_dir
                projection_scores[p] = np.std(proj_pos) / (np.std(proj_vel) + 1e-10)

            best_proj_idx = np.argmax(projection_scores)
            best_proj_score = projection_scores[best_proj_idx]

            random_proj_dir = np.random.randn(dim)
            random_proj_dir = random_proj_dir / (np.linalg.norm(random_proj_dir) + 1e-10)

            if not hasattr(self, '_boot_lhc_grid') or self._boot_lhc_generation + 5 < self.generation:
                grid_size = min(10, n_particles // 3)
                self._boot_lhc_samples = np.random.uniform(
                    self.lower_bound, self.upper_bound, (grid_size, dim)
                )
                for d in range(dim):
                    bins = np.linspace(self.lower_bound, self.upper_bound, grid_size + 1)
                    for i in range(grid_size):
                        self._boot_lhc_samples[i, d] = np.random.uniform(bins[i], bins[i + 1])
                for d in range(dim):
                    np.random.shuffle(self._boot_lhc_samples[:, d])
                self._boot_lhc_generation = self.generation
                self._boot_lhc_fitness = np.full(grid_size, np.inf)

            lhc_dists = np.sum((self.population[:, np.newaxis, :] - self._boot_lhc_samples[np.newaxis, :, :]) ** 2, axis=2)
            closest_lhc = np.argmin(lhc_dists, axis=1)
            lhc_guidance = np.zeros((n_particles, dim))

            for i in range(n_particles):
                lhc_idx = closest_lhc[i]
                direction = self._boot_lhc_samples[lhc_idx] - self.population[i]
                dist = np.linalg.norm(direction) + 1e-10
                lhc_guidance[i] = direction / dist

            n_vel_bootstrap = 30
            vel_std_estimate = np.zeros((n_vel_bootstrap, dim))
            for b in range(n_vel_bootstrap):
                indices = np.random.randint(0, n_particles, size=n_particles)
                vel_std_estimate[b] = np.std(self.velocity[indices], axis=0)

            velocity_confidence = 1.0 / (np.mean(vel_std_estimate, axis=0) + 1e-10)
            velocity_confidence = velocity_confidence / (np.max(velocity_confidence) + 1e-10)
            velocity_confidence = np.clip(velocity_confidence, 0.3, 2.0)

            exploration_factor = 1.0 + 0.5 * normalized_uncertainty
            proj_direction = random_proj_dir * best_proj_score
            vel_scale = velocity_confidence
            lhc_strength = 0.3 * exploration_factor
            lhc_boost = lhc_strength * lhc_guidance

            to_centroid = centroid_mean - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            centroid_pull = 0.2 * exploration_factor * (to_centroid / to_centroid_dist)

            scaled_velocity = self.velocity * vel_scale
            proj_correction = exploration_factor * proj_direction
            random_perturbation = np.random.randn(n_particles, dim) * 0.1 * exploration_factor

            new_population = (
                self.population +
                scaled_velocity +
                lhc_boost +
                centroid_pull +
                random_perturbation
            )
            new_population = self._clip_to_bounds(new_population)

            self._bootstrap_centroid_mean = centroid_mean
            self._bootstrap_uncertainty = normalized_uncertainty
            self._last_random_proj_dir = random_proj_dir

        except Exception:
            noise = np.random.randn(n_particles, self.dim) * 0.5
            new_population = self.population + self.velocity + noise
            new_population = self._clip_to_bounds(new_population)

        self.population = new_population
    
    def _position_update_variant_08_hybrid(self):
        """Hybrid: Improvement-gated fitness-gradient + geometric-spread (variant_08_catH_idea_0.py) — 5 wins."""
        try:
            if self.global_best is not None and hasattr(self, '_hybrid_prev_best_fitness'):
                raw_improvement = self._hybrid_prev_best_fitness - self.global_best_fitness
            else:
                raw_improvement = 0.0
            self._hybrid_prev_best_fitness = self.global_best_fitness if self.global_best is not None else np.inf

            if not hasattr(self, '_hybrid_ema_improvement'):
                self._hybrid_ema_improvement = 0.0
            self._hybrid_ema_improvement = 0.15 * max(0.0, raw_improvement) + 0.85 * self._hybrid_ema_improvement

            if not hasattr(self, '_hybrid_stagnation_ema'):
                self._hybrid_stagnation_ema = 0.0
            self._hybrid_stagnation_ema = 0.1 * self.stagnation_counter + 0.9 * self._hybrid_stagnation_ema

            if self.global_best_fitness > 0:
                normalized_improvement = self._hybrid_ema_improvement / (abs(self.global_best_fitness) + 1e-10)
            else:
                normalized_improvement = self._hybrid_ema_improvement

            if normalized_improvement > 1e-6:
                temporal_scale = 0.7
            elif normalized_improvement > 1e-10:
                temporal_scale = 1.0
            else:
                stagnation_boost = min(2.0, 1.0 + 0.1 * self._hybrid_stagnation_ema)
                temporal_scale = 1.3 * stagnation_boost

            centered = self.population - np.mean(self.population, axis=0)
            sq_dists = np.sum(centered[:, np.newaxis, :] ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            avg_pairwise_dist = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

            search_space_diag = (self.upper_bound - self.lower_bound) * np.sqrt(self.dim)
            normalized_spread = avg_pairwise_dist / search_space_diag

            if not hasattr(self, '_hybrid_spread_history'):
                self._hybrid_spread_history = []
            self._hybrid_spread_history.append(avg_pairwise_dist)
            if len(self._hybrid_spread_history) > 15:
                self._hybrid_spread_history.pop(0)

            if len(self._hybrid_spread_history) >= 5:
                recent_avg = np.mean(self._hybrid_spread_history[-5:])
                older_avg = np.mean(self._hybrid_spread_history[:-5]) if len(self._hybrid_spread_history) > 5 else recent_avg
                spread_trend = (recent_avg - older_avg) / (older_avg + 1e-10)
            else:
                spread_trend = 0.0

            centroid = np.mean(self.population, axis=0)
            to_centroid = self.population - centroid
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist

            repulsion_base = np.clip(1.0 - normalized_spread * 10.0, 0.0, 1.5)
            if spread_trend < -0.1:
                repulsion_base *= (1.0 - spread_trend * 2.0)

            geometric_correction = repulsion_base * to_centroid_dir * to_centroid_dist

            gate_input = np.log10(normalized_improvement + 1e-20)
            gate_sigmoid = 1.0 / (1.0 + np.exp(-gate_input * 2.0))

            temporal_weight = 0.2 + 0.6 * gate_sigmoid
            geometric_weight = 1.0 - temporal_weight

            vel_magnitude = np.linalg.norm(self.velocity, axis=1, keepdims=True) + 1e-10
            vel_direction = self.velocity / vel_magnitude
            temporal_contribution = temporal_scale * vel_magnitude * vel_direction

            geometric_contribution = geometric_correction

            combined_update = temporal_weight * temporal_contribution + geometric_weight * geometric_contribution

            new_population = self.population + combined_update

        except Exception:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_10_power(self):
        """Subspace power-law velocity modulation via SVD (variant_10_catB_idea_0.py) — 2 wins."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            total_var = np.sum(singular_values ** 2) + 1e-10
            sv_norm = singular_values / (singular_values[0] + 1e-10)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)

            cumvar = np.cumsum(singular_values ** 2) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / len(singular_values)

            base_power = 1.5
            power_boost = np.log1p(cond) / np.log1p(1000.0) if cond > 10.0 else 0.0
            power = base_power + 0.8 * power_boost
            power = np.clip(power, 1.0, 3.0)

            power_scale = sv_norm ** power
            power_scale = power_scale / (np.max(power_scale) + 1e-10)

            blend = np.clip((cond - 5.0) / 50.0, 0.0, 1.0)
            per_dir_scale = (1.0 - blend) * sv_norm + blend * power_scale
            per_dir_scale = np.clip(per_dir_scale, 0.1, 2.5)

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_dir_scale

            collapse_threshold = 0.3
            rehabilitation = np.zeros((self.np, self.dim))
            if eff_dim_ratio < collapse_threshold:
                n_weak = max(1, int(len(singular_values) * (1.0 - eff_dim_ratio)))
                weak_directions = Vt[:n_weak]
                weak_sv_norm = sv_norm[:n_weak]
                perturbation_mag = 0.3 * (1.0 - eff_dim_ratio / collapse_threshold)
                perturbation_mag = np.clip(perturbation_mag, 0.05, 0.5)

                for i in range(self.np):
                    for j, (sv_val, wdir) in enumerate(zip(weak_sv_norm, weak_directions)):
                        inv_sv = 1.0 / (sv_val + 0.01)
                        scale = perturbation_mag * inv_sv / (np.sum(1.0 / (weak_sv_norm + 0.01)) + 1e-10)
                        rehabilitation[i] += scale * wdir * np.random.randn()

            new_population = self.population + vel_scaled + rehabilitation

            if cond > 100.0:
                exploration_scale = 1.0 + 0.3 * np.log1p(cond) / np.log1p(1000.0)
                exploration_scale = np.clip(exploration_scale, 1.0, 1.5)
                noise = np.random.randn(*new_population.shape) * 0.1 * exploration_scale
                new_population = new_population + noise

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    # =========================================================================
    # DISPATCHER
    # =========================================================================
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        dispatch_map = {
            'original':         self._position_update_original,
            'variant_02_svd':   self._position_update_variant_02_svd,
            'variant_03_info':  self._position_update_variant_03_info,
            'variant_04_rank':  self._position_update_variant_04_rank,
            'variant_05_mst':   self._position_update_variant_05_mst,
            'variant_06_temp':  self._position_update_variant_06_temp,
            'variant_07_boot':  self._position_update_variant_07_boot,
            'variant_08_hybrid': self._position_update_variant_08_hybrid,
            'variant_10_power': self._position_update_variant_10_power,
        }
        method = dispatch_map.get(operator, self._position_update_original)
        method()
    
    # =========================================================================
    # THOMPSON SAMPLING BANDIT
    # =========================================================================
    
    def _update_bandit_reward(self, operator, rank_score):
        """Update Gaussian posterior with rank-normalized score.
        
        Rank score is in [0, 1] — already scale-independent.
        Gaussian posterior: (mu, precision) updated via Bayesian rules.
        """
        self._arm_scores[operator].append(rank_score)
        if len(self._arm_scores[operator]) > self._window_size:
            self._arm_scores[operator].pop(0)
        
        scores = self._arm_scores[operator]
        if len(scores) >= 1:
            # Bayesian update: conjugate normal with known precision
            observed_mean = np.mean(scores)
            observed_var = np.var(scores) if len(scores) > 1 else 0.01
            observed_precision = 1.0 / max(observed_var, 0.01)
            
            # Posterior precision = prior_precision + n * observed_precision
            new_precision = self._arm_precision[operator] + observed_precision
            # Posterior mean = weighted average
            self._arm_mu[operator] = (
                self._arm_mu[operator] * self._arm_precision[operator] +
                observed_mean * observed_precision
            ) / new_precision
            self._arm_precision[operator] = new_precision
        
        self._arm_count[operator] += 1
    
    def _rank_score_from_fitness(self, current_best_fitness):
        """Compute rank-based score for the current generation.
        
        Score = (worst_in_buffer - current) / (worst_in_buffer - best_in_buffer + eps)
        This is rank-normalized within the sliding window — no scale-dependent constants.
        """
        self._fitness_buffer.append(current_best_fitness)
        if len(self._fitness_buffer) > self._buffer_max:
            self._fitness_buffer.pop(0)
        
        if len(self._fitness_buffer) < 2:
            return 1.0  # First observation gets best score
        
        buf = np.array(self._fitness_buffer)
        worst = np.max(buf)
        best = np.min(buf)
        if worst > best:
            score = (worst - current_best_fitness) / (worst - best + 1e-10)
            return np.clip(score, 0.0, 1.0)
        return 1.0
    
    def _select_operator_thompson(self):
        """Thompson Sampling: sample from Gaussian posterior per arm, pick max."""
        samples = {}
        for op in self._operators:
            # Sample from N(mu, 1/precision)
            std = 1.0 / np.sqrt(max(self._arm_precision[op], 0.01))
            sample = np.random.normal(self._arm_mu[op], max(std, 0.05))
            samples[op] = sample
        
        return max(samples, key=samples.get)
    
    # =========================================================================
    # RESTART & LOCAL BEST
    # =========================================================================
    
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
    
    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    
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
        self._arm_mu = {op: 0.5 for op in self._operators}
        self._arm_precision = {op: 1.0 for op in self._operators}
        self._arm_scores = {op: [] for op in self._operators}
        self._arm_count = {op: 0 for op in self._operators}
        self._fitness_buffer = []
        
        # Reset all operator-specific state
        for attr in ['_info_fitness_entropy_history', '_rank_success_count', '_rank_fitness_history',
                     '_rank_prev_spread', '_rank_ema_improvement', '_temp_fitness_history',
                     '_temp_ema_centroid', '_temp_accel_history', '_temp_ema_diversity',
                     '_boot_lhc_grid', '_hybrid_spread_history', '_hybrid_ema_improvement']:
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
        
        # Track the operator used in the previous generation for reward assignment
        prev_operator = None
        prev_best_fitness = float(np.min(self.current_fitness))
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- SELECT OPERATOR VIA THOMPSON SAMPLING ---
            # Thompson Sampling selects based on posterior samples
            current_operator = self._select_operator_thompson()
            
            # --- ADAPTATION ---
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # --- VELOCITY UPDATE (shared base) ---
            self._velocity_update_base()
            
            # --- POSITION UPDATE (SELECTED BY BANDIT) ---
            self._dispatch_position_update(current_operator)
            
            # --- EVALUATE ---
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
            
            # --- CREDIT ASSIGNMENT: update bandit with rank-based score ---
            gen_best = float(np.min(self.current_fitness))
            rank_score = self._rank_score_from_fitness(gen_best)
            
            # Assign reward to the operator used in the PREVIOUS generation
            # (the one that produced the current best fitness)
            if prev_operator is not None:
                self._update_bandit_reward(prev_operator, rank_score)
            
            # Update tracking for next iteration
            prev_operator = current_operator
            prev_best_fitness = gen_best
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
