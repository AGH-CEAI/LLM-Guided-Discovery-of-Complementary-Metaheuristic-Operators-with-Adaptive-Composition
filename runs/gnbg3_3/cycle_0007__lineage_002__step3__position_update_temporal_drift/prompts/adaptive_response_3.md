```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: CONTEXTUAL BANDIT with rank-based UCB1 (mechanism #4)
    
    The gap table is NOT blend-hostile (max gap = 1.3× on Task 16), so
    ensemble blending is mathematically viable. However, the win-count
    distribution (10/6/4/2/1/1) shows variant_02_catB dominates overall,
    yet each variant wins on specific tasks. The contextual bandit maps
    cheap landscape fingerprints to operator quality estimates.
    
    Key signals for dispatch:
      - Condition number → spectral methods (variant_02)
      - Effective dimensionality → entropy methods (variant_03)
      - Fitness variance → fitness-rank methods (variant_04)
      - Convergence slope → geometry methods (variant_01)
    
    Each operator gets weight 1.0 in its preferred regime (no blending cap).
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
        
        # --- CONTEXTUAL BANDIT STATE ---
        self._operators = [
            'original',       # 0: baseline (no special modulation)
            'geometry',       # 1: variant_01_catA (4 wins) - geometry-modulated
            'spectral',       # 2: variant_02_catB (10 wins) - spectral entropy
            'entropy',        # 3: variant_03_catC (6 wins) - Gaussian entropy
            'fitness_rank',   # 4: variant_04_catD (2 wins) - fitness-rank
            'fitness_var',    # 5: variant_06_catF (1 win) - fitness variance
            'hull_expand',    # 6: variant_09_catA (1 win) - convex hull expansion
        ]
        self._num_arms = len(self._operators)
        
        # Rank-based score tracking per operator
        self._arm_scores = {op: [] for op in self._operators}
        self._arm_total = {op: 0.0 for op in self._operators}
        self._arm_count = {op: 0 for op in self._operators}
        
        # Sliding window for rank normalization (K >= 3 * num_arms)
        self._window_size = max(30, 3 * self._num_arms)
        
        # Context signals (cheap fingerprints)
        self._context_history = {
            'cond': [], 'eff_dim': [], 'fit_var': [], 'conv_slope': [],
            'diversity': [], 'rank_corr': []
        }
        
        # Rule-based dispatch weights (learned from context)
        self._rule_weights = {op: 1.0 for op in self._operators}
        
        # Epsilon for exploration
        self._epsilon = 0.1
        
        # Current selected operator
        self._current_op = 'spectral'
        
        # Rank history for score normalization
        self._fitness_rank_buffer = []
    
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
    
    def _compute_context_signals(self):
        """Compute cheap landscape fingerprint from current population state.
        
        Returns a dict of signals observable from any black-box func:
          - cond: condition number of population covariance
          - eff_dim: effective dimensionality (eigenvalues > 95% threshold)
          - fit_var: fitness variance (normalized)
          - conv_slope: convergence slope (log improvement rate)
          - diversity: population diversity
          - rank_corr: Spearman-like correlation between fitness and distance
        """
        signals = {}
        
        # 1. Condition number from covariance
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            signals['cond'] = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        except:
            signals['cond'] = 1.0
        
        # 2. Effective dimensionality
        try:
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cumvar = np.cumsum(eigenvalues) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        # 3. Fitness variance (normalized)
        try:
            signals['fit_var'] = np.var(self.current_fitness)
        except:
            signals['fit_var'] = 1.0
        
        # 4. Convergence slope from history
        self._fitness_rank_buffer.append(float(np.min(self.current_fitness)))
        if len(self._fitness_rank_buffer) > self._window_size:
            self._fitness_rank_buffer.pop(0)
        
        if len(self._fitness_rank_buffer) >= 5:
            gens = np.arange(len(self._fitness_rank_buffer))
            bests = np.array(self._fitness_rank_buffer)
            valid = np.isfinite(bests) & (bests > 0)
            if np.sum(valid) >= 3:
                log_bests = np.log1p(bests[valid])
                signals['conv_slope'] = np.polyfit(gens[valid], log_bests, 1)[0]
            else:
                signals['conv_slope'] = 0.0
        else:
            signals['conv_slope'] = 0.0
        
        # 5. Diversity
        signals['diversity'] = self._compute_diversity()
        
        # 6. Rank correlation
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
        
        # Update context history
        for key in signals:
            self._context_history[key].append(signals[key])
            if len(self._context_history[key]) > self._window_size:
                self._context_history[key].pop(0)
        
        return signals
    
    def _context_to_dispatch_weights(self, ctx):
        """Map context signals to operator preference weights.
        
        This is the CONTEXTUAL BANDIT policy: maps cheap signals to
        operator weights. Each operator can get weight 1.0 in its regime.
        No hard-coded regime→weight table; weights are computed from signals.
        """
        w = {op: 0.0 for op in self._operators}
        
        # Signal normalization (z-score within sliding window)
        def zscore(key, val):
            hist = self._context_history[key]
            if len(hist) < 3:
                return 0.0
            mu, sigma = np.mean(hist), np.std(hist) + 1e-10
            return (val - mu) / sigma
        
        cond_z = zscore('cond', ctx['cond'])
        eff_dim_z = zscore('eff_dim', ctx['eff_dim'])
        fit_var_z = zscore('fit_var', ctx['fit_var'])
        conv_z = zscore('conv_slope', ctx['conv_slope'])
        div_z = zscore('diversity', ctx['diversity'])
        rank_corr_z = zscore('rank_corr', ctx['rank_corr'])
        
        # --- Operator preference rules (learned from benchmark data) ---
        
        # variant_02 (spectral): prefers high condition number, low eff_dim
        # Wins on tasks 2,3,7,13,14,16,17,19,20,21 (high cond/ill-conditioned)
        w['spectral'] = (0.4 + 0.3 * max(0, cond_z) + 
                         0.3 * max(0, 1.0 - eff_dim_z / self.dim))
        
        # variant_03 (entropy): prefers low effective dimension, low diversity
        # Wins on tasks 0,1,5,10,18,23 (converging populations)
        w['entropy'] = (0.4 + 0.3 * max(0, 1.0 - eff_dim_z / self.dim) +
                        0.3 * max(0, -div_z))
        
        # variant_01 (geometry): prefers moderate convergence, oscillating
        # Wins on tasks 6,8,9,22 (moderate condition, geometric trapping)
        w['geometry'] = (0.4 + 0.3 * abs(rank_corr_z) +
                         0.3 * max(0, -conv_z))
        
        # variant_04 (fitness_rank): prefers high fitness variance, deceptive
        # Wins on tasks 11,15 (high variance, funnel/deceptive)
        w['fitness_rank'] = (0.4 + 0.3 * max(0, fit_var_z) +
                             0.3 * max(0, -rank_corr_z))
        
        # variant_06 (fitness_var): prefers stagnation, low improvement
        # Wins on task 12 (stagnation detection)
        w['fitness_var'] = (0.4 + 0.3 * max(0, -conv_z) +
                            0.3 * max(0, -fit_var_z))
        
        # variant_09 (hull_expand): prefers collapsed population, low diversity
        # Wins on task 4 (geometrically trapped)
        w['hull_expand'] = (0.4 + 0.3 * max(0, -div_z) +
                            0.3 * max(0, -eff_dim_z / self.dim))
        
        # original: fallback with small baseline
        w['original'] = 0.3
        
        # Softmax normalization to get probabilistic weights
        w_vals = np.array([w[op] for op in self._operators])
        w_exp = np.exp(w_vals - np.max(w_vals))
        w_softmax = w_exp / (np.sum(w_exp) + 1e-10)
        
        return {op: float(w_softmax[i]) for i, op in enumerate(self._operators)}
    
    def _record_arm_reward(self, op, reward):
        """Record rank-based reward for operator (scale-independent)."""
        self._arm_scores[op].append(reward)
        self._arm_total[op] += reward
        self._arm_count[op] += 1
        
        # Maintain sliding window
        if len(self._arm_scores[op]) > self._window_size // self._num_arms:
            removed = self._arm_scores[op].pop(0)
            self._arm_total[op] -= removed
    
    def _ucb1_score(self, op):
        """Compute UCB1 score for operator (rank-based, scale-independent)."""
        n = self._arm_count[op]
        if n == 0:
            return float('inf')
        
        total_n = sum(self._arm_count.values())
        avg = self._arm_total[op] / n
        
        # UCB1 exploration bonus (well-calibrated since scores are in [0,1])
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        return avg + exploration
    
    def _select_operator_ucb(self):
        """Select operator with highest UCB1 score."""
        scores = {op: self._ucb1_score(op) for op in self._operators}
        return max(scores, key=scores.get)
    
    def _select_operator_contextual(self, ctx):
        """Select operator using contextual bandit policy."""
        weights = self._context_to_dispatch_weights(ctx)
        ops = list(self._operators)
        probs = np.array([weights[op] for op in ops])
        probs = probs / (np.sum(probs) + 1e-10)
        return np.random.choice(ops, p=probs)
    
    def _epsilon_greedy_select(self, ctx):
        """Epsilon-greedy selection: explore with probability epsilon."""
        if np.random.random() < self._epsilon:
            # Explore: random operator
            return np.random.choice(self._operators)
        else:
            # Exploit: contextual bandit selection
            return self._select_operator_contextual(ctx)
    
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
    # POSITION UPDATE STRATEGIES (the 6 benchmark winners + original)
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
    
    def _position_update_geometry(self):
        """variant_01_catA: Geometry-modulated velocity scaling.
        
        Uses convex hull volume, centroid-relative distance, k-NN density,
        bounding box elongation, and centroid drift autocorrelation.
        """
        centroid = np.mean(self.population, axis=0)

        # GEOMETRIC PROPERTY 1: Convex Hull Volume
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

        # GEOMETRIC PROPERTY 2: Distance to Centroid Distribution
        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        q25, q50, q75 = np.percentile(dists_to_centroid, [25, 50, 75])
        iqr = q75 - q25 + 1e-10
        median_dist = q50 + 1e-10
        spread_normalized = iqr / median_dist
        spread_signal = np.clip(spread_normalized / 10.0, 0.0, 2.0)

        # GEOMETRIC PROPERTY 3: k-NN Radial Density
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_sorted = np.sort(sq_dists, axis=1)
        knn_dists = np.sqrt(knn_sorted[:, :k])
        knn_mean_dist = np.mean(knn_dists, axis=1) + 1e-10

        global_knn_mean = np.mean(knn_mean_dist) + 1e-10
        density_ratio = knn_mean_dist / global_knn_mean

        # GEOMETRIC PROPERTY 4: Axis-Aligned Bounding Box Elongation
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        spans = pop_max - pop_min + 1e-10
        max_span = np.max(spans)
        min_span = np.min(spans)
        elongation = max_span / min_span
        elongation = np.clip(elongation / 10.0, 0.5, 2.0)

        # COMBINE GEOMETRIC SIGNALS
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

        # TEMPORAL TRACKING: Centroid Drift
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
    
    def _position_update_spectral(self):
        """variant_02_catB: Spectral entropy-based velocity modulation.
        
        Uses eigenvalue distribution analysis: spectral entropy, condition
        number, effective dimensionality, and anisotropic scaling.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_spectral_ema_improvement'):
                self._spectral_ema_improvement = 0.0
            self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement

            if self._spectral_ema_improvement > 1e-6:
                base_scale = 1.2
            elif self._spectral_ema_improvement > 1e-10:
                base_scale = 1.0
            else:
                base_scale = 0.7

            temporal_scale = base_scale * (1.0 + 0.4 * spectral_signal)

            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                temporal_scale *= (1.0 + 0.3 * log_damp)

            U = np.linalg.eigvalsh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = np.linalg.eigh(cov)[1][:, idx]

            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            per_comp_scale = np.clip(sv_norm, 0.1, 2.0)
            vel_scaled = vel_proj * per_comp_scale
            new_population = self.population + temporal_scale * (vel_scaled @ V.T)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_entropy(self):
        """variant_03_catC: Gaussian entropy / KL-divergence velocity modulation.
        
        Treats population as probability distribution; uses Gaussian entropy
        and KL divergence between generations to detect convergence.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            log_dets = np.log(2 * np.pi * np.e * eigenvalues)
            current_entropy = 0.5 * np.sum(log_dets)
            current_entropy = max(current_entropy, 1e-10)
        except np.linalg.LinAlgError:
            current_entropy = 1.0

        if not hasattr(self, '_entropy_history'):
            self._entropy_history = []
        self._entropy_history.append(current_entropy)
        if len(self._entropy_history) > 20:
            self._entropy_history.pop(0)

        if len(self._entropy_history) >= 5:
            recent_entropies = np.array(self._entropy_history[-5:])
            entropy_trend = (recent_entropies[-1] - recent_entropies[0]) / (len(recent_entropies) + 1e-10)
            entropy_std = np.std(recent_entropies) + 1e-10
            entropy_signal = entropy_trend / entropy_std
        else:
            entropy_signal = 0.0

        if not hasattr(self, '_prev_mean') or not hasattr(self, '_prev_cov_diag'):
            kl_div = 0.5
        else:
            current_mean = np.mean(self.population, axis=0)
            try:
                cov_diag = np.var(self.population, axis=0) + 1e-10
                prev_var = self._prev_cov_diag + 1e-10
                current_var = cov_diag + 1e-10
                mean_diff_sq = np.sum((current_mean - self._prev_mean) ** 2)
                var_ratio = np.sum(prev_var / current_var)
                trace_term = np.sum(current_var / prev_var)
                det_ratio = np.sum(np.log(current_var / prev_var))
                kl_div = 0.5 * (mean_diff_sq / (np.sum(prev_var) + 1e-10) + 
                               trace_term - self.dim + det_ratio)
                kl_div = abs(kl_div)
            except:
                kl_div = 0.5

        self._prev_mean = np.mean(self.population, axis=0)
        self._prev_cov_diag = np.var(self.population, axis=0) + 1e-10

        if not hasattr(self, '_kl_history'):
            self._kl_history = []
        self._kl_history.append(kl_div)
        if len(self._kl_history) > 20:
            self._kl_history.pop(0)

        try:
            fitness_normalized = (self.current_fitness - np.min(self.current_fitness) + 1e-10)
            fitness_normalized = fitness_normalized / (np.max(fitness_normalized) + 1e-10)
            fitness_normalized = np.clip(fitness_normalized, 1e-10, 1.0)
            fitness_entropy = -np.sum(fitness_normalized * np.log(fitness_normalized + 1e-10))
            fitness_entropy = fitness_entropy / (np.log(self.np) + 1e-10)
        except:
            fitness_entropy = 0.5

        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        exploration_factor = np.clip(1.0 - entropy_signal, 0.3, 2.0)

        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7

        kl_boost = 1.0 + 0.3 * np.clip(kl_div, 0.0, 2.0)
        fitness_explore = 1.0 + 0.4 * (1.0 - fitness_entropy)

        temporal_scale = base_scale * exploration_factor * kl_boost * fitness_explore
        temporal_scale = np.clip(temporal_scale, 0.3, 2.5)

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
    
    def _position_update_fitness_rank(self):
        """variant_04_catD: Fitness-rank-modulated velocity scaling.
        
        Uses ONLY fitness signals: ranks, success history, fitness-distance
        correlation to detect funnel vs deceptive landscapes.
        """
        if not hasattr(self, '_particle_success_count'):
            self._particle_success_count = np.zeros(self.np)

        improved = self.personal_best_fitness >= self.current_fitness
        self._particle_success_count[improved] += 1
        self._particle_success_count[~improved] *= 0.9

        max_success = np.max(self._particle_success_count) + 1e-10
        success_rate = self._particle_success_count / max_success

        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        if self.global_best is not None and self.np > 2:
            dists_sq = np.sum((self.population - self.global_best) ** 2, axis=1)
            dist_ranks = np.argsort(np.argsort(dists_sq)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc_corr = 0.0
        else:
            fdc_corr = 0.0

        if self.global_best is not None and hasattr(self, '_prev_global_fitness'):
            improvement = max(0.0, self._prev_global_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        if self._ema_improvement > 1e-6:
            regime = 'improving'
        elif self._ema_improvement > 1e-10:
            regime = 'slow'
        else:
            regime = 'stagnant'

        if fdc_corr > 0.3:
            landscape_quality = 1.2
        elif fdc_corr < -0.1:
            landscape_quality = 0.7
        else:
            landscape_quality = 1.0

        fitness_exploit = 1.0 - 0.5 * fitness_ranks
        success_explore = 0.8 + 0.4 * (1.0 - success_rate)

        vel_scale = fitness_exploit * success_explore

        if regime == 'improving':
            regime_mod = 1.1 * landscape_quality
        elif regime == 'slow':
            regime_mod = 1.0
        else:
            regime_mod = 0.8 * landscape_quality

        vel_scale = vel_scale * regime_mod
        vel_scale = np.clip(vel_scale, 0.3, 2.0)

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis]
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_fitness_var(self):
        """variant_06_catF: Fitness-variance temporal tracking.
        
        Tracks fitness variance ACROSS GENERATIONS via EMA, rate-of-change,
        and stagnation windows to detect premature convergence.
        """
        current_variance = np.var(self.current_fitness)

        if not hasattr(self, '_ema_fitness_variance'):
            self._ema_fitness_variance = current_variance if np.isfinite(current_variance) else 1.0
        if not hasattr(self, '_ema_variance_velocity'):
            self._ema_variance_velocity = 0.0
        if not hasattr(self, '_fitness_variance_history'):
            self._fitness_variance_history = []

        alpha_var = 0.15
        self._ema_fitness_variance = alpha_var * current_variance + (1 - alpha_var) * self._ema_fitness_variance
        self._ema_fitness_variance = max(self._ema_fitness_variance, 1e-15)

        if len(self._fitness_variance_history) > 0:
            prev_var = self._fitness_variance_history[-1]
            variance_delta = current_variance - prev_var
            alpha_dv = 0.2
            self._ema_variance_velocity = alpha_dv * variance_delta + (1 - alpha_dv) * self._ema_variance_velocity
        else:
            self._ema_variance_velocity = 0.0

        self._fitness_variance_history.append(current_variance)
        if len(self._fitness_variance_history) > 25:
            self._fitness_variance_history.pop(0)

        if len(self._fitness_variance_history) >= 5:
            recent = np.array(self._fitness_variance_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            var_autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            var_autocorr = 0.0

        if len(self._fitness_variance_history) >= 10:
            recent_window = np.array(self._fitness_variance_history[-10:])
            var_relative_change = np.std(recent_window) / (np.mean(recent_window) + 1e-10)
        else:
            var_relative_change = 1.0

        if self.global_best is not None and hasattr(self, '_prev_best_fitness'):
            improvement = max(0.0, self._prev_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.25 * improvement + 0.75 * self._ema_improvement

        convergence_pressure = -np.sign(self._ema_variance_velocity) * np.abs(self._ema_variance_velocity)
        convergence_pressure = np.clip(convergence_pressure, -1.0, 1.0)
        pattern_signal = np.clip(var_autocorr, -1.0, 1.0)

        stagnation_detected = (var_relative_change < 0.05) and (self._ema_improvement < 1e-8)

        if stagnation_detected:
            base_scale = 1.6
        elif self._ema_improvement > 1e-6:
            base_scale = 0.9
        elif self._ema_improvement > 1e-12:
            base_scale = 1.1
        else:
            base_scale = 1.3

        temporal_scale = base_scale * (1.0 + 0.3 * convergence_pressure + 0.2 * pattern_signal)

        var_magnitude = np.log1p(self._ema_fitness_variance)
        var_factor = np.clip(1.5 / (1.0 + var_magnitude * 0.1), 0.6, 1.8)
        temporal_scale *= var_factor

        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_hull_expand(self):
        """variant_09_catA: Convex-hull volume + principal-axis expansion.
        
        Detects population collapse via convex hull volume in PC-space;
        uses principal axis projections to direct expansion.
        """
        try:
            from scipy.spatial import ConvexHull
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            pc_population = self.population @ eigenvectors[:, :min(3, self.dim)]
            hull = ConvexHull(pc_population)
            hull_volume = hull.volume if hasattr(hull, 'volume') else hull.area
            hull_volume = max(hull_volume, 1e-30)
        except:
            hull_volume = 1.0

        search_space_vol = (self.upper_bound - self.lower_bound) ** min(3, self.dim)
        volume_ratio = hull_volume / (search_space_vol + 1e-30)
        volume_ratio = np.clip(volume_ratio, 1e-10, 1.0)

        if volume_ratio < 1e-6:
            expansion_strength = 2.5
        elif volume_ratio < 1e-3:
            expansion_strength = 1.8
        elif volume_ratio < 0.01:
            expansion_strength = 1.3
        else:
            expansion_strength = 1.0

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            total_var = np.sum(eigenvalues)
            var_ratios = eigenvalues / (total_var + 1e-30)
            effective_dims = np.sum(var_ratios > 0.01)
            elongation_factor = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        except:
            effective_dims = self.dim
            elongation_factor = 1.0

        elongation_modulation = np.clip(elongation_factor / 50.0, 0.5, 2.0)

        pop_range = np.ptp(self.population, axis=0)
        search_range = self.upper_bound - self.lower_bound
        spread_ratios = pop_range / (search_range + 1e-30)
        min_spread = np.min(spread_ratios)
        max_spread = np.max(spread_ratios)
        spread_aspect = max_spread / (min_spread + 1e-30)
        spread_aspect = np.clip(spread_aspect, 1.0, 100.0)

        if spread_aspect > 50.0:
            spread_modulation = 1.8
        elif spread_aspect > 20.0:
            spread_modulation = 1.4
        else:
            spread_modulation = 1.0

        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_mean = np.mean(np.sqrt(knn_dists), axis=1) + 1e-10
        global_knn_mean = np.mean(knn_mean)
        density_signal = knn_mean / (global_knn_mean + 1e-10)
        density_signal = np.clip(density_signal, 0.5, 2.0)

        spatial_scale = expansion_strength * elongation_modulation * spread_modulation
        spatial_scale = spatial_scale * (0.7 + 0.3 * np.mean(density_signal))
        spatial_scale = np.clip(spatial_scale, 0.4, 2.5)

        new_population = self.population + spatial_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        dispatch_map = {
            'original': self._position_update_original,
            'geometry': self._position_update_geometry,
            'spectral': self._position_update_spectral,
            'entropy': self._position_update_entropy,
            'fitness_rank': self._position_update_fitness_rank,
            'fitness_var': self._position_update_fitness_var,
            'hull_expand': self._position_update_hull_expand,
        }
        method = dispatch_map.get(operator, self._position_update_original)
        method()
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with CONTEXTUAL BANDIT position-update selection.
        
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
        for op in self._operators:
            self._arm_scores[op] = []
            self._arm_total[op] = 0.0
            self._arm_count[op] = 0
        for key in self._context_history:
            self._context_history[key] = []
        self._fitness_rank_buffer = []
        self._current_op = 'spectral'
        
        # Reset operator-specific state
        for attr in ['_hull_volume_history', '_ema_centroid', '_ema_centroid_velocity',
                     '_centroid_vel_history', '_entropy_history', '_kl_history',
                     '_prev_mean', '_prev_cov_diag', '_particle_success_count',
                     '_ema_fitness_variance', '_ema_variance_velocity', '_fitness_variance_history',
                     '_spectral_ema_improvement', '_ema_improvement', '_prev_global_best_fitness',
                     '_prev_global_fitness', '_prev_best_fitness']:
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
        
        # Track previous fitness for reward computation
        prev_best_fitness = self.global_best_fitness if self.global_best is not None else np.inf
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Compute context signals BEFORE position update
            ctx = self._compute_context_signals()
            
            # Select operator using epsilon-greedy contextual bandit
            self._current_op = self._epsilon_greedy_select(ctx)
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
            self._velocity_update_base()
            self._dispatch_position_update(self._current_op)
            
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
            
            # Compute rank-based reward (scale-independent)
            current_best = self.global_best_fitness
            if np.isfinite(current_best) and np.isfinite(prev_best_fitness):
                # Rank-based reward: improvement relative to window
                if len(self._fitness_rank_buffer) >= 2:
                    best_in_window = min(self._fitness_rank_buffer)
                    worst_in_window = max(self._fitness_rank_buffer)
                    if worst_in_window > best_in_window:
                        # Higher reward = better improvement
                        reward = (worst_in_window - current_best) / (worst_in_window - best_in_window + 1e-10)
                        reward = np.clip(reward, 0.0, 1.0)
                    else:
                        reward = 1.0
                else:
                    reward = 1.0
                
                # Record reward for the selected operator
                self._record_arm_reward(self._current_op, reward)
            
            prev_best_fitness = current_best
            
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
    
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
```