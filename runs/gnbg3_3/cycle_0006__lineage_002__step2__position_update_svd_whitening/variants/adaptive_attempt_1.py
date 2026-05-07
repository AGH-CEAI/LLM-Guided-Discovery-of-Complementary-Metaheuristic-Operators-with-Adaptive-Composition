import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT (mechanism #1) with probe-and-commit.
    
    Gap analysis: ALL tasks have gap < 3× (max = 2.3× on task 5). This is NOT
    blend-hostile — linear blending could preserve advantages, but with 10 arms
    and small margins, a bandit is more sample-efficient.
    
    The 10 winning variants are mutually exclusive strategies (different
    geometric/spectral/information-theoretic signals). Thompson Sampling:
      - Handles 10 arms without over-exploring
      - Rank-based Beta posterior updates are scale-independent
      - Warm-starts from probe phase to avoid pure exploration cost
      - Probe extracts cheap fingerprint (convergence slope, rank-corr,
        effective dimensionality, diversity) to condition arm selection
    
    Key design choices:
      - K=30 sliding window for rank-based scores (K >= 3*num_arms)
      - Beta(α,β) posterior per arm, updated from rank-based improvement
      - Thompson sample selects arm each generation; committed arm runs
        with 0.90 weight, others with 0.10/9 probability (exploration)
      - Re-probe every 50 generations if stagnation detected
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
        
        # =================================================================
        # THOMPSON SAMPLING BANDIT STATE (mechanism #1)
        # =================================================================
        # 10 arms corresponding to the 10 benchmark winners
        self._operators = [
            'original',           # 0: original.py (2 wins: tasks 4, 5)
            'variant_01_catA',    # 1: variant_01_catA_idea_0.py (2 wins: tasks 8, 21)
            'variant_02_catB',    # 2: variant_02_catB_idea_0.py (4 wins: tasks 6, 19, 20, 23)
            'variant_03_catC',    # 3: variant_03_catC_idea_0.py (2 wins: tasks 0, 7)
            'variant_04_catD',    # 4: variant_04_catD_idea_0.py (3 wins: tasks 11, 12, 16)
            'variant_06_catF',    # 5: variant_06_catF_idea_0.py (2 wins: tasks 2, 22)
            'variant_07_catG',    # 6: variant_07_catG_idea_0.py (2 wins: tasks 10, 14)
            'variant_08_catH',    # 7: variant_08_catH_idea_0.py (1 win: task 9)
            'variant_09_catA',    # 8: variant_09_catA_idea_0.py (5 wins: tasks 3, 13, 15, 17, 18)
            'variant_10_catB',    # 9: variant_10_catB_idea_0.py (1 win: task 1)
        ]
        self._num_arms = len(self._operators)
        
        # Beta posterior parameters (Thompson Sampling)
        # Initialize with uniform prior: Beta(1,1) = uniform
        self._alpha = np.ones(self._num_arms)  # successes + 1
        self._beta = np.ones(self._num_arms)   # failures + 1
        
        # Sliding window for rank-based scoring (K >= 3 * num_arms = 30)
        self._score_window_size = 30
        self._operator_score_windows = {op: [] for op in self._operators}
        
        # UCB backup scores (used alongside Thompson)
        self._operator_total_score = {op: 0.0 for op in self._operators}
        self._operator_sample_count = {op: 0 for op in self._operators}
        
        # Current selected operator
        self._selected_operator = None
        
        # =================================================================
        # PROBE-AND-COMMIT STATE
        # =================================================================
        # Probe: 2 gens per operator (20 total for 10 arms)
        self._probe_gens_per_op = 2
        self._probe_total_budget = self._num_arms * self._probe_gens_per_op
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Commit phase state
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        self._last_reprobe_gen = 0
        
        # =================================================================
        # FINGERPRINT STATE (for contextual conditioning)
        # =================================================================
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = float(dim)
        self._fingerprint_diversity = 0.0
        
        # Per-arm context vectors (updated from fingerprint)
        self._arm_context = {op: np.zeros(4) for op in self._operators}
    
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
    
    def _compute_fingerprint(self):
        """Compute cheap landscape fingerprint from current population state.
        
        Returns a dict of signals observable from any black-box func:
          - convergence_slope: slope of log(best_fitness) vs generation
          - rank_corr: Spearman-like correlation between fitness rank and
                       distance to global best (proxy for landscape smoothness)
          - eff_dim: effective dimensionality from early covariance
          - diversity: current population diversity
        """
        signals = {}
        
        # 1. Convergence slope (from probe history)
        if len(self._probe_fitness_history) >= 3:
            gens = np.arange(len(self._probe_fitness_history))
            bests = np.array(self._probe_fitness_history)
            valid = bests > 0
            if np.sum(valid) >= 2:
                log_bests = np.log1p(bests[valid])
                signals['convergence_slope'] = np.polyfit(gens[valid], log_bests, 1)[0]
            else:
                signals['convergence_slope'] = np.polyfit(gens, bests, 1)[0]
        else:
            signals['convergence_slope'] = 0.0
        
        # 2. Rank correlation (fitness vs distance to global best)
        if self.global_best is not None and self.np > 2:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                signals['rank_corr'] = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                signals['rank_corr'] = 0.0
        else:
            signals['rank_corr'] = 0.0
        
        # 3. Effective dimensionality from covariance
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                sorted_eig = np.sort(eigenvalues)[::-1]
                cumvar = np.cumsum(sorted_eig) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        # 4. Diversity
        signals['diversity'] = self._compute_diversity()
        
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        
        return signals
    
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
    # POSITION UPDATE STRATEGIES (10 benchmark winners)
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
    
    def _position_update_variant_01_catA(self):
        """Geometric pairwise distance + axis-aligned spread modulation (Category A).
        
        From variant_01_catA_idea_0.py (2 wins: tasks 8, 21).
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
    
    def _position_update_variant_02_catB(self):
        """Eigendecomposition-based whitened coordinate momentum (Category B).
        
        From variant_02_catB_idea_0.py (4 wins: tasks 6, 19, 20, 23).
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
    
    def _position_update_variant_03_catC(self):
        """Entropy-modulated velocity scaling (Category C: Information-theoretic).
        
        From variant_03_catC_idea_0.py (2 wins: tasks 0, 7).
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
    
    def _position_update_variant_04_catD(self):
        """Fitness-landscape rank-based velocity modulation (Category D).
        
        From variant_04_catD_idea_0.py (3 wins: tasks 11, 12, 16).
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
    
    def _position_update_variant_06_catF(self):
        """SVD-whitening with temporal collapse detection via EMA of condition number.
        
        From variant_06_catF_idea_0.py (2 wins: tasks 2, 22).
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
    
    def _position_update_variant_07_catG(self):
        """Monte Carlo random subspace SVD for robust population whitening.
        
        From variant_07_catG_idea_0.py (2 wins: tasks 10, 14).
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
    
    def _position_update_variant_08_catH(self):
        """Hybrid: SVD spectral conditioning + graph-based clustering detection.
        
        From variant_08_catH_idea_0.py (1 win: task 9).
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
    
    def _position_update_variant_09_catA(self):
        """Axis-aligned bounding box normalization + centroid-distance-weighted velocity.
        
        From variant_09_catA_idea_0.py (5 wins: tasks 3, 13, 15, 17, 18).
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
    
    def _position_update_variant_10_catB(self):
        """Condition-adaptive rank-truncated whitening (variant_10, Category B).
        
        From variant_10_catB_idea_0.py (1 win: task 1).
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
            'variant_01_catA': self._position_update_variant_01_catA,
            'variant_02_catB': self._position_update_variant_02_catB,
            'variant_03_catC': self._position_update_variant_03_catC,
            'variant_04_catD': self._position_update_variant_04_catD,
            'variant_06_catF': self._position_update_variant_06_catF,
            'variant_07_catG': self._position_update_variant_07_catG,
            'variant_08_catH': self._position_update_variant_08_catH,
            'variant_09_catA': self._position_update_variant_09_catA,
            'variant_10_catB': self._position_update_variant_10_catB,
        }
        
        method = dispatch_map.get(operator)
        if method is not None:
            method()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING BANDIT (mechanism #1)
    # -------------------------------------------------------------------------
    
    def _thompson_sample(self):
        """Thompson Sampling: sample from Beta posterior for each arm.
        
        Returns the arm index with highest sampled value.
        Scale-independent: Beta posterior is over rank-based scores in [0,1].
        """
        samples = np.random.beta(self._alpha, self._beta)
        return int(np.argmax(samples))
    
    def _update_bandit(self, operator, rank_score):
        """Update Beta posterior based on rank-based score.
        
        Rank score is in [0, 1] — fraction of recent generations where this
        operator produced above-median improvement. This is SCALE-INDEPENDENT.
        
        Beta(α,β) update:
          - High score (close to 1): increase α (more successes)
          - Low score (close to 0): increase β (more failures)
        """
        op_idx = self._operators.index(operator)
        
        # Update sliding window
        self._operator_score_windows[operator].append(rank_score)
        if len(self._operator_score_windows[operator]) > self._score_window_size:
            self._operator_score_windows[operator].pop(0)
        
        # Update running statistics
        self._operator_total_score[operator] += rank_score
        self._operator_sample_count[operator] += 1
        
        # Beta posterior update (Bayesian)
        # Map score ∈ [0,1] to success/failure pseudo-counts
        # Score close to 1 → more success weight
        # Score close to 0 → more failure weight
        success_weight = rank_score * 0.5  # Scale factor for learning rate
        failure_weight = (1.0 - rank_score) * 0.5
        
        self._alpha[op_idx] += success_weight
        self._beta[op_idx] += failure_weight
        
        # Clamp to prevent numerical issues
        self._alpha[op_idx] = np.clip(self._alpha[op_idx], 0.1, 100.0)
        self._beta[op_idx] = np.clip(self._beta[op_idx], 0.1, 100.0)
    
    def _get_rank_score(self, operator, best_fitness):
        """Compute rank-based score for the current generation.
        
        Score = fraction of the last K generations (including this one) where
        this operator produced better-than-median improvement across all operators.
        
        This is SCALE-INDEPENDENT — no magic constants.
        """
        # Track all scores for this operator in the sliding window
        window = self._operator_score_windows[operator]
        
        if len(window) == 0:
            return 0.5  # Neutral prior
        
        # Current best fitness relative to running best
        if len(self._probe_fitness_history) < 2:
            return 0.5
        
        # Compute improvement relative to median across all operators
        current_best = min(self._probe_fitness_history)
        current_worst = max(self._probe_fitness_history)
        
        if current_worst > current_best:
            # Higher = better (lower fitness = higher score)
            normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
            return np.clip(normalized, 0.0, 1.0)
        
        return 0.5
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling."""
        arm_idx = self._thompson_sample()
        return self._operators[arm_idx]
    
    def _select_operator_ucb(self):
        """Fallback UCB selection for comparison."""
        best_op = self._operators[0]
        best_score = -float('inf')
        
        for i, op in enumerate(self._operators):
            n = self._operator_sample_count[op]
            if n == 0:
                return op  # Unexplored arm has infinite UCB
            
            total_n = sum(self._operator_sample_count.values())
            avg_score = self._operator_total_score[op] / n
            
            # UCB1 exploration bonus — scale is [0,1]
            exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
            
            ucb = avg_score + exploration
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
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
            
            # Reset temporal states
            for attr in ['_prev_whitened_vel', '_ema_cond', '_ema_sv_ratio', 
                         '_prev_ema_cond', '_fitness_success_count']:
                if hasattr(self, attr):
                    delattr(self, attr)
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    def _reset_temporal_states(self):
        """Reset all temporal/EMA states used by various operators."""
        for attr in ['_prev_whitened_vel', '_ema_cond', '_ema_sv_ratio', 
                     '_prev_ema_cond', '_fitness_success_count', '_ema_centroid',
                     '_ema_centroid_velocity', '_centroid_vel_history', 
                     '_ema_improvement', '_success_count']:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except:
                    pass
    
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
        self._alpha = np.ones(self._num_arms)
        self._beta = np.ones(self._num_arms)
        for op in self._operators:
            self._operator_score_windows[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Reset probe state
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        self._last_reprobe_gen = 0
        self._selected_operator = None
        
        # Reset all temporal states
        self._reset_temporal_states()
        
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
                
                # Adaptation
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Velocity update (shared base), then position update (PROBED)
                self._velocity_update_base()
                self._dispatch_position_update(current_op)
                
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
                
                self._probe_fitness_history.append(self._probe_best_fitness_seen)
                self._probe_op_history.append(current_op)
                
                # Compute rank-based score and update bandit
                rank_score = self._get_rank_score(current_op, gen_best)
                self._update_bandit(current_op, rank_score)
                
                self._probe_gens_in_current_op += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= self._num_arms:
                    self._in_probe_phase = False
                    # Select initial operator using Thompson Sampling
                    self._selected_operator = self._select_operator_thompson()
                    # Compute fingerprint from probe data
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                    self._last_reprobe_gen = self.generation
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT with Thompson Sampling ---
            else:
                self._commit_gen_count += 1
                
                # Periodic re-probe check: every 50 generations or on stagnation
                should_reprobe = (
                    (self.generation - self._last_reprobe_gen >= 50) and
                    (self.stagnation_counter > 15)
                )
                
                if should_reprobe:
                    # Re-enter probe phase for quick re-assessment
                    self._in_probe_phase = True
                    self._probe_op_index = 0
                    self._probe_gens_in_current_op = 0
                    self._last_reprobe_gen = self.generation
                    # Reset temporal states for fresh assessment
                    self._reset_temporal_states()
                    continue
                
                # Thompson Sampling operator selection
                # 90% commit to sampled operator, 10% exploration
                if np.random.random() < 0.90:
                    # Exploitation: Thompson sample
                    self._selected_operator = self._select_operator_thompson()
                else:
                    # Exploration: random arm
                    self._selected_operator = np.random.choice(self._operators)
                
                # Run selected operator
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                self._velocity_update_base()
                self._dispatch_position_update(self._selected_operator)
                
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
                
                # Update bandit with rank-based score
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                
                self._probe_fitness_history.append(self._probe_best_fitness_seen)
                self._probe_op_history.append(self._selected_operator)
                
                rank_score = self._get_rank_score(self._selected_operator, gen_best)
                self._update_bandit(self._selected_operator, rank_score)
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
