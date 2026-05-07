```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table shows ALL per-task gaps < 5× (max 1.3×), so linear
    blending is mathematically viable but unnecessary. The probe phase
    extracts a cheap LANDSCAPE FINGERPRINT to route to the right
    operator without blending, preserving per-task advantages.
    
    Win distribution: variant_02 (10 wins), variant_03 (6), variant_01 (4),
    variant_04 (2), variant_06 (1), variant_09 (1). Six distinct winners
    across 24 tasks confirm that different operators win on different
    landscapes. The probe (2 gens × 6 arms = 12 total, < 1% of typical
    budget) identifies landscape characteristics that predict success.
    
    Rank-based UCB1 ensures scale-independent reward aggregation.
    Epsilon-review (p=0.05) allows escape from a wrongly-committed operator.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.np = min(max(5 * dim, 20), 300)
        self.neighborhood_size = max(3, dim // 5)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
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
        
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        
        # THE 6 BENCHMARK-WINNING POSITION UPDATE STRATEGIES
        self._operators = [
            'variant_01_catA',  # 4 wins: geometry-modulated velocity scaling
            'variant_02_catB',  # 10 wins: spectral entropy-based velocity modulation
            'variant_03_catC',  # 6 wins: Gaussian entropy / KL-divergence velocity
            'variant_04_catD',  # 2 wins: fitness-rank-modulated velocity scaling
            'variant_06_catF',  # 1 win: fitness-variance temporal tracking
            'variant_09_catA',  # 1 win: convex-hull + principal-axis expansion
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe: 2 gens per operator (12 total for 6 arms)
        self._probe_gens_per_op = 2
        self._epsilon_review = 0.05
        
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
        
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        self._runner_up_operator = None
    
    def _initialize_population(self):
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
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        if len(fitness) < len(population):
            return fitness[:len(pitness)], len(fitness)
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
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
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _compute_fingerprint(self):
        signals = {}
        
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
        
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cumvar = np.cumsum(sorted(eigenvalues, reverse=True)) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        signals['diversity'] = self._compute_diversity()
        
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        
        return signals
    
    def _adapt_inertia_weight(self):
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
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    def _velocity_update_base(self):
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
    # THE 6 BENCHMARK-WINNING POSITION UPDATE STRATEGIES
    # -------------------------------------------------------------------------
    
    def _position_update_variant_01_catA(self):
        """variant_01_catA_idea_0.py (4 wins): Geometry-modulated velocity scaling."""
        centroid = np.mean(self.population, axis=0)

        try:
            from scipy.spatial import ConvexHull
            if self.dim <= 3 and self.np >= self.dim + 2:
                hull = ConvexHull(self.population)
                hull_volume = hull.volume
            else:
                dists = np.linalg.norm(self.population - centroid, axis=1)
                max_dist = np.max(dists) + 1e-10
                hull_volume = max_dist ** self.dim
        except:
            dists = np.linalg.norm(self.population - centroid, axis=1)
            max_dist = np.max(dists) + 1e-10
            hull_volume = max_dist ** self.dim

        if not hasattr(self, '_hull_volume_history'):
            self._hull_volume_history = []
        self._hull_volume_history.append(hull_volume)
        if len(self._hull_volume_history) > 20:
            self._hull_volume_history.pop(0)

        if len(self._hull_volume_history) >= 3:
            recent_vol = np.array(self._hull_volume_history[-3:])
            hull_shrink_rate = (recent_vol[-1] - recent_vol[0]) / (recent_vol[0] + 1e-10)
        else:
            hull_shrink_rate = 0.0

        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        q25, q50, q75 = np.percentile(dists_to_centroid, [25, 50, 75])
        iqr = q75 - q25 + 1e-10
        median_dist = q50 + 1e-10
        spread_normalized = iqr / median_dist

        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_sorted = np.sort(sq_dists, axis=1)
        knn_dists = np.sqrt(knn_sorted[:, :k])
        knn_mean_dist = np.mean(knn_dists, axis=1) + 1e-10
        global_knn_mean = np.mean(knn_mean_dist) + 1e-10
        density_ratio = knn_mean_dist / global_knn_mean

        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        spans = pop_max - pop_min + 1e-10
        max_span = np.max(spans)
        min_span = np.min(spans)
        elongation = max_span / min_span
        elongation = np.clip(elongation / 10.0, 0.5, 2.0)

        if median_dist < 5.0:
            base_scale = 0.7
        elif median_dist < 20.0:
            base_scale = 1.0
        else:
            base_scale = 1.3

        hull_factor = 1.0 - np.clip(hull_shrink_rate * 2.0, -0.5, 0.5)
        elongation_factor = np.clip(2.0 - elongation, 0.5, 1.5)
        geometric_scale = base_scale * hull_factor * elongation_factor
        density_modulation = np.clip(density_ratio, 0.5, 2.0)

        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)
            self._centroid_vel_history = []

        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid
        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

        if not hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)

        if len(self._centroid_vel_history) >= 5:
            recent = np.array(self._centroid_vel_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            autocorr = 0.0

        oscillation_factor = 1.0 + 0.4 * np.clip(autocorr, -1.0, 1.0)

        final_scale = np.clip(geometric_scale * oscillation_factor, 0.5, 2.5)
        vel_adjustment = density_modulation[:, np.newaxis]

        new_population = self.population + final_scale * self.velocity * vel_adjustment
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_02_catB(self):
        """variant_02_catB_idea_0.py (10 wins): Spectral entropy-based velocity modulation."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy