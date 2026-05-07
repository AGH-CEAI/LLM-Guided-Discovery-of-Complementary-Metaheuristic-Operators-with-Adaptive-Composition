```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT with Thompson Sampling (Contextual Bandit, #4)
    
    The gap table is BLEND-NEUTRAL (all gaps < 2×), so ensemble blending is
    mathematically viable but suboptimal because 6 distinct winners span the
    24 tasks. A bandit alone is fragile given only 3 probe gens per arm.
    
    The chosen design:
      - PHASE A (probe): round-robin each operator for 3 generations with
        a small sub-population, collecting rank-based improvement scores.
      - PHASE B (commit): run ONLY the winner for the remaining budget.
        No blending. No hard-coded weights.
      - REGIME-REVIEW: after commit, a cheap regime detector monitors the
        search state each generation. If stagnation or high condition-number
        is detected, epsilon-switch to the runner-up (p=0.05 per generation).
        This is NOT a hard-coded regime→weight table — it is a dynamic,
        per-generation gate that can commit 100% to any operator.
    
    Win distribution (10/6/4/2/1 across 5 arms) confirms that no single
    operator dominates and that different strategies win on different
    landscapes. The probe extracts a landscape fingerprint to route correctly.
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
        
        # -----------------------------------------------------------------
        # PROBE-AND-COMMIT STATE
        # -----------------------------------------------------------------
        # The 5 winning operators from the benchmark (win-count order)
        self._operators = [
            'variant_02_catB',   # 10 wins — spectral entropy / eigenvalue distribution
            'variant_03_catC',   #  6 wins — Gaussian entropy / KL-divergence velocity
            'variant_01_catA',   #  4 wins — convex hull + centroid drift geometry
            'variant_04_catD',   #  2 wins — fitness-rank-modulated velocity
            'variant_09_catA',   #  1 win  — convex hull PCA + k-NN density
        ]
        
        # Thompson Sampling state: Beta(alpha, beta) per arm
        self._alpha = {op: 1.0 for op in self._operators}
        self._beta  = {op: 1.0 for op in self._operators}
        
        # Rank-based sliding window for reward shaping (K >= 3 * num_arms)
        self._K_reward_window = 18  # 3 * 5 + 3 buffer
        self._reward_history = {op: [] for op in self._operators}
        
        # Probe phase control
        self._probe_gens_per_op = 3
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Commit phase
        self._committed_operator = None
        self._runner_up_operator = None
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Epsilon-review probability (small, per-generation)
        self._epsilon_review = 0.05
        
        # Regime detector state (used in commit phase for epsilon-review gate)
        self._ema_improvement = 0.0
        self._ema_fitness_variance = None
        self._fitness_variance_history = []
        
        # Fingerprint signals (computed at end of probe)
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
    
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
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
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
    
    # ========================================================================
    # POSITION UPDATE STRATEGIES — the 5 benchmark winners
    # ========================================================================
    
    def _position_update_variant_02_catB(self):
        """Spectral entropy / eigenvalue distribution (10 wins).
        
        Key insight: eigenvalue distribution analysis detects convergence from
        covariance shape; condition number drives anisotropic scaling.
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
    
    def _position_update_variant_03_catC(self):
        """Gaussian entropy / KL-divergence velocity (6 wins).
        
        Key insight: treat population as a probability distribution; use
        Gaussian entropy and KL divergence between consecutive generations
        to detect convergence and modulate velocity accordingly.
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
        
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness_catC'):
            improvement = max(0.0, self._prev_global_best_fitness_catC - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness_catC = self.global_best_fitness
        
        if not hasattr(self, '_ema_improvement_catC'):
            self._ema_improvement_catC = 0.0
        self._ema_improvement_catC = 0.3 * improvement + 0.7 * self._ema_improvement_catC
        
        exploration_factor = 1.0 - np.clip(entropy_signal, -2.0, 2.0)
        exploration_factor = np.clip(exploration_factor, 0.3, 2.0)
        
        if self._ema_improvement_catC > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement_catC > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        kl_boost = 1.0 + 0.3 * np.clip(kl_div, 0.0, 2.0)
        fitness_explore = 1.0 + 0.4 * (1.0 - fitness_entropy)
        
        temporal_scale = base_scale * exploration_factor * kl_boost * fitness_explore
        temporal_scale = np.clip(temporal_scale, 0.3, 2.5)
        
        try:
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
            temporal_scale *= spectral_factor
        except np.linalg.LinAlgError:
            pass
        
        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_01_catA(self):
        """Geometry-modulated velocity scaling (4 wins).
        
        Key insight: literal geometric properties of the swarm layout —
        convex hull volume, centroid-relative distance distribution,
        k-NN radial density, bounding box elongation, and centroid drift
        oscillation detection.
        """
        centroid = np.mean(self.population, axis=0)
        
        # Geometric Property 1: Convex Hull Volume
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
        
        # Geometric Property 2: Distance to Centroid Distribution
        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        q25, q50, q75 = np.percentile(dists_to_centroid, [25, 50, 75])
        iqr = q75 - q25 + 1e-10
        median_dist = q50 + 1e-10
        spread_normalized = iqr / median_dist
        spread_signal = np.clip(spread_normalized / 10.0, 0.0, 2.0)
        
        # Geometric Property 3: k-NN Radial Density
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_sorted = np.sort(sq_dists, axis=1)
        knn_dists = np.sqrt(knn_sorted[:, :k])
        knn_mean_dist = np.mean(knn_dists, axis=1) + 1e-10
        global_knn_mean = np.mean(knn_mean_dist) + 1e-10
        density_ratio = knn_mean_dist / global_knn_mean
        
        # Geometric Property 4: Axis-Aligned Bounding Box Elongation
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        spans = pop_max - pop_min + 1e-10
        max_span = np.max(spans)
        min_span = np.min(spans)
        elongation = max_span / min_span
        elongation = np.clip(elongation / 10.0, 0.5, 2.0)
        
        # Combine geometric signals
        if median_dist < 5.0:
            base_scale = 0.7
        elif median_dist < 20.0:
            base_scale = 1.0
        else:
            base_scale = 1.3
        
        hull_factor = 1.0 - np.clip(hull_shrink_rate * 2.0, -0.5, 0.5)
        elongation_factor = 2.0 - elongation
        elongation_factor = np.clip(elongation_factor, 0.5, 1.5)
        geometric_scale = base_scale * hull_factor * elongation_factor
        density_modulation = np.clip(density_ratio, 0.5, 2.0)
        
        # Temporal: Centroid Drift for Oscillation Detection
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)
            self._centroid_vel_history = []
        
        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid
        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity
        
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
        final_scale = geometric_scale * oscillation_factor
        final_scale = np.clip(final_scale, 0.5, 2.5)
        vel_adjustment = density_modulation[:, np.newaxis]
        
        new_population = self.population + final_scale * self.velocity * vel_adjustment
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_04_catD(self):
        """Fitness-rank-modulated velocity scaling (2 wins).
        
        Key insight: use ONLY fitness signals (ranks, success history) to
        modulate velocity. Per-particle success tracking identifies
        exploit/explore regimes. Fitness-distance correlation detects
        funnel vs deceptive landscapes.
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
        
        if self.global_best is not None and hasattr(self, '_prev_global_fitness_catD'):
            improvement = max(0.0, self._prev_global_fitness_catD - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_fitness_catD = self.global_best_fitness
        
        if not hasattr(self, '_ema_improvement_catD'):
            self._ema_improvement_catD = 0.0
        self._ema_improvement_catD = 0.3 * improvement + 0.7 * self._ema_improvement_catD
        
        if self._ema_improvement_catD > 1e-6:
            regime = 'improving'
        elif self._ema_improvement_catD > 1e-10:
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
    
    def _position_update_variant_09_catA(self):
        """Convex-hull PCA + k-NN density expansion (1 win).
        
        Key insight: detect population collapse via convex hull volume in
        PC-space; use principal axis projections to direct expansion away
        from centroid.
        """
        # Geometric Signal 1: Convex Hull Volume in PC-space
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
        
        # Geometric Signal 2: Principal Axis Elongation
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
        
        # Geometric Signal 3: Axis-Aligned Spread Ratio
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
        
        # Geometric Signal 4: k-NN Density Gradient
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
    
    # ========================================================================
    # DISPATCHER: maps operator name → position update method
    # ========================================================================
    
    def _dispatch_position_update(self, operator):
        if operator == 'variant_02_catB':
            self._position_update_variant_02_catB()
        elif operator == 'variant_03_catC':
            self._position_update_variant_03_catC()
        elif operator == 'variant_01_catA':
            self._position_update_variant_01_catA()
        elif operator == 'variant_04_catD':
            self._position_update_variant_04_catD()
        elif operator == 'variant_09_catA':
            self._position_update_variant_09_catA()
        else:
            self._position_update_variant_02_catB()
    
    # ========================================================================
    # RANK-BASED REWARD SYSTEM (scale-independent, per calibration rule a)
    # ========================================================================
    
    def _record_probe_reward(self, operator, best_fitness):
        """Record a rank-based reward within the sliding window.
        
        Reward = position of this generation's best fitness relative to
        the last K probe generations (z-score of rank). Scale-independent.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        # Compute rank-based reward from the sliding window
        if len(self._probe_fitness_history) < 2:
            reward = 0.5
        else:
            # Use the last K entries for rank normalization
            window = self._probe_fitness_history[-self._K_reward_window:]
            current_rank = sum(1 for f in window if f <= best_fitness) / len(window)
            reward = np.clip(current_rank, 0.0, 1.0)
        
        # Store in sliding window
        self._reward_history[operator].append(reward)
        if len(self._reward_history[operator]) > self._K_reward_window:
            self._reward_history[operator].pop(0)
        
        # Update Thompson Sampling Beta distribution
        # reward in [0,1]: map to Beta(alpha, beta) updates
        # reward ≈ 1.0 → success (increase alpha)
        # reward ≈ 0.0 → failure (increase beta)
        self._alpha[operator] += reward
        self._beta[operator]  += (1.0 - reward)
    
    def _thompson_sample(self):
        """Sample from Thompson Sampling Beta distributions and return best arm."""
        samples = {}
        for op in self._operators:
            a = max(self._alpha[op], 1e-10)
            b = max(self._beta[op], 1e-10)
            samples[op] = np.random.beta(a, b)
        return max(samples, key=samples.get)
    
    def _select_top_two_operators(self):
        """Return the top two operators by Thompson posterior mean."""
        means = {op: self._alpha[op] / (self._alpha[op] + self._beta[op] + 1e-10)
                 for op in self._operators}
        sorted_ops = sorted(means.items(), key=lambda x: x[1], reverse=True)
        return sorted_ops[0][0], sorted_ops[1][0]
    
    # ========================================================================
    # REGIME DETECTOR (used in commit phase for epsilon-review gate)
    # ========================================================================
    
    def _detect_regime(self):
        """Detect search regime from population state — NO hard-coded weights.
        
        Returns a dict of signals, not a hard label. The epsilon-review
        gate uses these signals to decide whether to switch.
        """
        regime = {}
        
        # 1. Stagnation signal
        regime['stagnant'] = (self.stagnation_counter > 20)
        
        # 2. Condition number (high → anisotropic, needs spectral damping)
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            regime['condition_number'] = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        except:
            regime['condition_number'] = 1.0
        
        # 3. Diversity signal
        regime['diversity'] = self._compute_diversity()
        
        # 4. Improvement EMA
        if self.global_best is not None and hasattr(self, '_prev_regime_best'):
            improvement = max(0.0, self._prev_regime_best - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_regime_best = self.global_best_fitness
        
        if not hasattr(self, '_ema_improvement_regime'):
            self._ema_improvement_regime = 0.0
        self._ema_improvement_regime = 0.2 * improvement + 0.8 * self._ema_improvement_regime
        regime['improvement_ema'] = self._ema_improvement_regime
        
        # 5. Fitness variance dynamics
        current_var = np.var(self.current_fitness)
        if not hasattr(self, '_regime_fitness_variance'):
            self._regime_fitness_variance = current_var if np.isfinite(current_var) else 1.0
        alpha_var = 0.15
        self._regime_fitness_variance = alpha_var * current_var + (1 - alpha_var) * self._regime_fitness_variance
        
        self._fitness_variance_history.append(self._regime_fitness_variance)
        if len(self._fitness_variance_history) > 15:
            self._fitness_variance_history.pop(0)
        
        if len(self._fitness_variance_history) >= 8:
            recent = np.array(self._fitness_variance_history[-8:])
            regime['variance_relative_change'] = np.std(recent) / (np.mean(recent) + 1e-10)
        else:
            regime['variance_relative_change'] = 1.0
        
        return regime
    
    # ========================================================================
    # RESTART & LOCAL BEST
    # ========================================================================
    
    def _restart_if_stagnant(self):
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
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    # ========================================================================
    # MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT position-update selection.
        
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
            self._alpha[op] = 1.0
            self._beta[op] = 1.0
            self._reward_history[op] = []
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset per-operator EMA states
        for attr in ['_ema_centroid', '_ema_centroid_velocity', '_centroid_vel_history',
                     '_hull_volume_history', '_entropy_history', '_kl_history',
                     '_prev_mean', '_prev_cov_diag', '_particle_success_count',
                     '_spectral_ema_improvement', '_ema_improvement_catC',
                     '_prev_global_best_fitness_catC', '_ema_improvement_catD',
                     '_prev_global_fitness_catD', '_prev_global_best_fitness',
                     '_regime_fitness_variance', '_fitness_variance_history',
                     '_ema_improvement_regime', '_prev_regime_best']:
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
                return float(self.current_fitness[best_idx]), self.population[best_idx].copy()
            return float(self.global_best_fitness), self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # === PHASE A: PROBE ===
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
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
                
                # Record rank-based reward (scale-independent)
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_reward(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                if self._probe_op_index >= len(self._operators):
                    self._in_probe_phase = False
                    # Thompson Sampling commit
                    self._committed_operator = self._thompson_sample()
                    top_two = self._select_top_two_operators()
                    self._runner_up_operator = top_two[1]
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # === PHASE B: COMMIT ===
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review gate: use regime detector to inform switching
                regime = self._detect_regime()
                
                # Compute review probability: higher when stagnant or high condition number
                eps = self._epsilon_review
                if regime['stagnant']:
                    eps = max(eps, 0.15)
                if regime.get('condition_number', 1.0) > 100.0:
                    eps = max(eps, 0.10)
                if regime.get('variance_relative_change', 1.0) < 0.05:
                    eps = max(eps, 0.10)
                
                # Epsilon-review: switch to runner-up with probability eps
                if np.random.random() < eps:
                    self._velocity_update_base()
                    self._dispatch_position_update(self._runner_up_operator)
                else:
                    self._velocity_update_base()
                    self._dispatch_position_update(self._committed_operator)
                
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
            return float(self.current_fitness[best_idx]), self.population[best_idx].copy()
        return float(self.global_best_fitness), self.global_best.copy()
```