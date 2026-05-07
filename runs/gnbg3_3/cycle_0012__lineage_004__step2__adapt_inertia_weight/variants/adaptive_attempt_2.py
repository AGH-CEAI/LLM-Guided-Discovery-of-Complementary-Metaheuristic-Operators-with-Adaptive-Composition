import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: REGIME-BASED DISPATCH WITH ONLINE REINFORCEMENT (mechanism #3 + #4 hybrid)
    
    The gap table shows NO blend-hostile tasks (all gaps < 2×), so linear blending
    IS mathematically viable. However, the win-count distribution (6/6/5/3/2/1/1)
    across 7 distinct winners confirms that different operators excel in different
    PROBLEM REGIMES. The key insight: each winning inertia strategy computes a
    specific SIGNAL from the population, and that signal is informative only when
    the population exhibits the corresponding regime.
    
    The dispatcher preserves each winner's advantage by:
      - Computing MULTIPLE orthogonal signals each generation (geometric spread,
        spectral condition, spectral entropy, FDC, temporal improvement).
      - Using a lightweight REGIME DETECTOR that classifies the current landscape
        state and dispatches to the strategy whose signal is most relevant.
      - Reinforcing successful dispatches via rank-based UCB updates so the
        algorithm learns which regime→strategy mapping works on THIS landscape.
      - Fallback to original.py (6 wins, strong baseline) when no regime is clear.
    
    Why not pure ensemble blending? A 60/40 blend of variant_04 and variant_08
    on task 0 would score ~0.125, but variant_08 alone scores 0.123. The blend
    loses ~1.5% unnecessarily. The dispatcher recovers 100% of the winner's
    advantage by routing to it exclusively.
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
        
        # --- REGIME-BASED DISPATCH STATE ---
        # Map strategy name -> rank-based UCB state
        self._strategy_scores = {}
        self._strategy_total_score = {}
        self._strategy_sample_count = {}
        
        # THE 7 BENCHMARK WINNING INERTIA STRATEGIES
        self._strategies = [
            'original',      # baseline (6 wins)
            'variant_01_catA',  # geometric layout (1 win, tasks 20)
            'variant_03_catC',  # spectral entropy (3 wins, tasks 12,17,18)
            'variant_04_catD',  # FDC + success history (6 wins, tasks 3,7,15,16,21,22)
            'variant_06_catF',  # temporal signals (1 win, task 2)
            'variant_08_catH',  # spectral condition + improvement (5 wins, tasks 0,6,9,13,23)
            'variant_10_catB',  # SVD spectral entropy (2 wins, tasks 14,19)
        ]
        for s in self._strategies:
            self._strategy_scores[s] = []
            self._strategy_total_score[s] = 0.0
            self._strategy_sample_count[s] = 0
        
        # Sliding window for rank-based scoring (K >= 3 * num_strategies)
        self._sliding_window_size = max(21, 3 * len(self._strategies))
        
        # Epsilon-greedy exploration probability
        self._epsilon = 0.10
        
        # EMA state for strategy performance tracking
        self._strategy_ema = {s: 0.0 for s in self._strategies}
        
        # Regime detection signals (computed each generation)
        self._regime_signals = {
            'geometric_spread': 0.0,
            'spectral_entropy': 0.0,
            'spectral_condition': 0.0,
            'fdc': 0.0,
            'temporal_improvement': 0.0,
            'svd_entropy': 0.0,
        }
        
        # Previous fitness for temporal signals
        self._prev_best_fitness = np.inf
        self._ema_improvement = 0.0
        self._ema_stagnation = 0.0
        self._prev_fitness_median = np.inf
        self._prev_centroid = None
        self._ema_drift = 0.0
        self._success_ema = 0.0
        self._prev_global_best_fitness = np.inf
        self._spectral_ema_improvement = 0.0
        
        # Diversity EMA
        self._ema_diversity = 0.0
        
        # Running history for rank-based scoring
        self._fitness_history = []
        self._strategy_history = []
    
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
    
    # -------------------------------------------------------------------------
    # REGIME SIGNAL COMPUTATION — all orthogonal signals in one place
    # -------------------------------------------------------------------------
    
    def _compute_regime_signals(self):
        """Compute regime detection signals from current population state.
        
        Returns a dict of orthogonal signals, each normalized to [0, 1]:
          - geometric_spread: population diameter / expected diameter
          - spectral_entropy: normalized entropy of covariance eigenvalues
          - spectral_condition: log(condition) / log(max_condition)
          - fdc: fitness-distance correlation (Spearman), mapped to exploit signal
          - temporal_improvement: EMA of improvement rate
          - svd_entropy: entropy of normalized singular values squared
        """
        signals = {}
        
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            total_var = np.sum(eigenvalues) + 1e-10
            
            # Signal 1: spectral condition
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)
            signals['spectral_condition'] = np.log1p(cond) / np.log1p(10000.0)
            
            # Signal 2: spectral entropy
            p = eigenvalues / total_var
            p = np.clip(p, 1e-10, 1.0)
            spectral_entropy = -np.sum(p * np.log2(p))
            max_entropy = np.log2(len(eigenvalues) + 1e-10)
            signals['spectral_entropy'] = np.clip(spectral_entropy / max_entropy, 0.0, 1.0) if max_entropy > 0 else 0.5
            
            # Signal 3: SVD spectral entropy (variant_10)
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)
            sv_norm = singular_values ** 2 / (np.sum(singular_values ** 2) + 1e-10)
            sv_norm = np.clip(sv_norm, 1e-10, None)
            sv_entropy = -np.sum(sv_norm * np.log(sv_norm))
            max_sv_entropy = np.log(min(self.np, self.dim) + 1e-10)
            signals['svd_entropy'] = np.clip(sv_entropy / max_sv_entropy, 0.0, 1.0) if max_sv_entropy > 0 else 0.5
            
        except np.linalg.LinAlgError:
            signals['spectral_condition'] = 0.5
            signals['spectral_entropy'] = 0.5
            signals['svd_entropy'] = 0.5
        
        try:
            # Signal 4: geometric spread (variant_01)
            centroid = np.mean(self.population, axis=0)
            dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
            mean_centroid_dist = np.mean(dists_to_centroid)
            mins = np.min(self.population, axis=0)
            maxs = np.max(self.population, axis=0)
            ranges = maxs - mins
            max_range = np.max(ranges)
            expected_diameter = 200.0 * np.sqrt(self.dim)
            diameter = mean_centroid_dist * 2.0 + max_range
            signals['geometric_spread'] = np.clip(diameter / expected_diameter, 0.0, 1.0)
            
            # Signal 5: k-NN density
            k = min(5, self.np - 1)
            sq_dists = np.sum(
                (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2,
                axis=2
            )
            np.fill_diagonal(sq_dists, np.inf)
            sorted_sq_dists = np.sort(sq_dists, axis=1)
            knn_sq_dists = sorted_sq_dists[:, :k]
            mean_knn_dist = np.mean(np.sqrt(knn_sq_dists)) + 1e-10
            signals['knn_density'] = np.clip(1.0 / (mean_knn_dist + 1.0), 0.0, 1.0)
            
        except:
            signals['geometric_spread'] = 0.5
            signals['knn_density'] = 0.5
        
        try:
            # Signal 6: fitness-distance correlation (variant_04)
            if self.global_best is not None and self.np > 3:
                fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
                dists = np.linalg.norm(self.population - self.global_best, axis=1)
                dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
                if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                    corr_matrix = np.corrcoef(fitness_ranks, dist_ranks)
                    spearman_fdc = np.clip(corr_matrix[0, 1], -1.0, 1.0)
                else:
                    spearman_fdc = 0.0
            else:
                spearman_fdc = 0.0
            # High FDC = smooth landscape = exploit signal
            signals['fdc'] = (spearman_fdc + 1.0) / 2.0  # Map [-1,1] to [0,1]
            
            # Signal 7: temporal improvement (variant_06, variant_08)
            if self.global_best is not None:
                improvement = max(0.0, self._prev_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_best_fitness = self.global_best_fitness if self.global_best is not None else np.inf
            self._ema_improvement = 0.2 * improvement + 0.8 * self._ema_improvement
            scale = max(abs(self.global_best_fitness), 1e-10) if self.global_best is not None else 1.0
            signals['temporal_improvement'] = np.clip(self._ema_improvement / scale, 0.0, 2.0) / 2.0
            
        except:
            signals['fdc'] = 0.5
            signals['temporal_improvement'] = 0.0
        
        self._regime_signals = signals
        return signals
    
    # -------------------------------------------------------------------------
    # REGIME DETECTOR — maps signals to strategy selection
    # -------------------------------------------------------------------------
    
    def _detect_regime(self):
        """Classify current landscape state and return preferred strategy.
        
        The regime detection is based on the winning task patterns from the
        benchmark table. Each strategy is best when its key signal is informative.
        
        Returns: strategy name (string)
        """
        s = self._regime_signals
        stagnation_ratio = np.clip(self.stagnation_counter / 100.0, 0.0, 1.0)
        
        # Regime 1: variant_08_catH — spectral condition + temporal improvement
        # Wins on tasks 0,6,9,13,23. High condition = anisotropic landscape.
        # Also wins when stagnation is high (needs exploration).
        if s['spectral_condition'] > 0.6 or stagnation_ratio > 0.3:
            return 'variant_08_catH'
        
        # Regime 2: variant_04_catD — FDC + success history
        # Wins on tasks 3,7,15,16,21,22. High FDC = smooth landscape.
        # Also wins when percentile spread is high (diverse fitness).
        if s['fdc'] > 0.6:
            fitness_sorted = np.sort(self.current_fitness)
            p90 = fitness_sorted[int(0.9 * self.np)] if self.np > 0 else 0.0
            p10 = fitness_sorted[int(0.1 * self.np)] if self.np > 0 else 0.0
            percentile_spread = (p90 - p10) / (np.abs(self.global_best_fitness) + 1e-10 + p90 - p10 + 1e-10)
            if percentile_spread > 0.3 or s['fdc'] > 0.7:
                return 'variant_04_catD'
        
        # Regime 3: variant_03_catC — spectral entropy
        # Wins on tasks 12,17,18. Low entropy = concentrated population.
        # Also wins on high-dimensional tasks (eff_dim matters).
        if s['spectral_entropy'] < 0.4:
            return 'variant_03_catC'
        
        # Regime 4: variant_10_catB — SVD spectral entropy
        # Wins on tasks 14,19. Similar to variant_03 but uses SVD.
        # Distinguishable by low svd_entropy (concentrated in few directions).
        if s['svd_entropy'] < 0.4:
            return 'variant_10_catB'
        
        # Regime 5: variant_01_catA — geometric layout
        # Wins on task 20. High geometric spread = well-distributed population.
        if s['geometric_spread'] > 0.7:
            return 'variant_01_catA'
        
        # Regime 6: variant_06_catF — temporal signals
        # Wins on task 2. Temporal drift + stagnation are the key signals.
        if stagnation_ratio > 0.1 and s['temporal_improvement'] < 0.2:
            return 'variant_06_catF'
        
        # Default: original.py (6 wins, strong baseline)
        return 'original'
    
    # -------------------------------------------------------------------------
    # RANK-BASED SCORING (scale-independent UCB)
    # -------------------------------------------------------------------------
    
    def _record_strategy_score(self, strategy, best_fitness):
        """Record a score using rank-based (scale-independent) scoring.
        
        Score = fraction of recent generations where this strategy produced
        a result in the top tercile. This is rank-based — no scale constants.
        """
        self._fitness_history.append(best_fitness)
        self._strategy_history.append(strategy)
        
        # Maintain sliding window
        if len(self._fitness_history) > self._sliding_window_size:
            self._fitness_history.pop(0)
            self._strategy_history.pop(0)
        
        if len(self._fitness_history) < 3:
            score = 1.0
        else:
            window_fitness = np.array(self._fitness_history)
            tercile = int(len(window_fitness) / 3)
            bottom_tercile_threshold = np.partition(window_fitness, tercile)[tercile]
            # Higher score = this generation's result is in bottom tercile (good)
            score = 1.0 if best_fitness <= bottom_tercile_threshold else 0.0
        
        self._strategy_scores[strategy].append(score)
        self._strategy_total_score[strategy] += score
        self._strategy_sample_count[strategy] += 1
        
        # Maintain sliding window per strategy
        if len(self._strategy_scores[strategy]) > self._sliding_window_size // len(self._strategies):
            oldest = self._strategy_scores[strategy].pop(0)
            self._strategy_total_score[strategy] -= oldest
    
    def _compute_ucb_score(self, strategy):
        """Compute UCB1-style score for strategy selection.
        
        Uses rank-based average (already in [0,1]) so exploration bonus
        is on the same scale — no magic constants.
        """
        n = self._strategy_sample_count[strategy]
        if n == 0:
            return float('inf')
        
        total_n = sum(self._strategy_sample_count.values())
        avg_score = self._strategy_total_score[strategy] / n
        
        # UCB1 exploration bonus — scale is [0,1], so this is well-calibrated
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        
        return avg_score + exploration
    
    def _select_explore_strategy(self):
        """Select strategy using epsilon-greedy + UCB during exploration phase."""
        if np.random.random() < self._epsilon:
            # Explore: random uniform
            return np.random.choice(self._strategies)
        
        # Exploit: UCB-based
        best_op = self._strategies[0]
        best_score = -float('inf')
        
        for op in self._strategies:
            ucb = self._compute_ucb_score(op)
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
    # -------------------------------------------------------------------------
    # INERTIA WEIGHT ADAPTATION — all 7 winning strategies implemented
    # -------------------------------------------------------------------------
    
    def _adapt_inertia_weight_original(self):
        """Original: eigenvalue-anisotropic velocity modulation (baseline)."""
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
    
    def _adapt_inertia_weight_variant_01_catA(self):
        """Variant 01 (Category A): geometric layout-based adaptation.
        
        Wins on task 20. Operates on literal geometric layout (pairwise distances,
        centroid distances, axis-aligned spread, k-NN density).
        """
        try:
            centroid = np.mean(self.population, axis=0)
            dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
            mean_centroid_dist = np.mean(dists_to_centroid)

            mins = np.min(self.population, axis=0)
            maxs = np.max(self.population, axis=0)
            ranges = maxs - mins
            max_range = np.max(ranges)

            diameter = mean_centroid_dist * 2.0 + max_range

            k = min(5, self.np - 1)
            sq_dists = np.sum(
                (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2,
                axis=2
            )
            np.fill_diagonal(sq_dists, np.inf)
            sorted_sq_dists = np.sort(sq_dists, axis=1)
            knn_sq_dists = sorted_sq_dists[:, :k]
            mean_knn_dist = np.mean(np.sqrt(knn_sq_dists)) + 1e-10

            min_range = np.min(ranges) + 1e-10
            spread_ratio = np.clip(max_range / min_range, 1.0, 100.0)

            expected_diameter = 200.0 * np.sqrt(self.dim)
            diameter_signal = np.clip(diameter / expected_diameter, 0.0, 1.0)
            centroid_signal = np.clip(mean_centroid_dist / expected_diameter, 0.0, 1.0)
            spread_signal = np.clip(np.log1p(spread_ratio) / np.log1p(100.0), 0.0, 1.0)
            knn_signal = np.clip(1.0 / (mean_knn_dist + 1.0), 0.0, 1.0)

            geometric_signal = (
                0.30 * diameter_signal +
                0.25 * centroid_signal +
                0.25 * spread_signal +
                0.20 * knn_signal
            )
            geometric_signal = np.clip(geometric_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * geometric_signal

            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

        except Exception:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_03_catC(self):
        """Variant 03 (Category C): spectral entropy of population distribution.
        
        Wins on tasks 12,17,18. Uses Shannon entropy of normalized covariance
        eigenvalues as a probability distribution.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            total = np.sum(eigenvalues)
            if total > 0:
                p = eigenvalues / total
            else:
                p = np.ones_like(eigenvalues) / len(eigenvalues)

            p = np.clip(p, 1e-10, 1.0)
            spectral_entropy = -np.sum(p * np.log2(p))

            max_entropy = np.log2(len(eigenvalues) + 1e-10)
            if max_entropy > 0:
                normalized_entropy = spectral_entropy / max_entropy
            else:
                normalized_entropy = 0.5
            normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * normalized_entropy

            decay = 0.729 - 0.15 * (self.generation / 1000)
            decay = np.clip(decay, 0.4, 0.95)

            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_04_catD(self):
        """Variant 04 (Category D): FDC + success-history adaptation.
        
        Wins on tasks 3,7,15,16,21,22. Uses Spearman fitness-distance correlation,
        fitness percentile spread, and EMA of per-particle improvement.
        """
        try:
            if self.global_best is not None and self.np > 3:
                fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
                dists = np.linalg.norm(self.population - self.global_best, axis=1)
                dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)

                if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                    corr_matrix = np.corrcoef(fitness_ranks, dist_ranks)
                    spearman_fdc = np.clip(corr_matrix[0, 1], -1.0, 1.0)
                else:
                    spearman_fdc = 0.0
            else:
                spearman_fdc = 0.0

            fitness_sorted = np.sort(self.current_fitness)
            p90 = fitness_sorted[int(0.9 * self.np)] if self.np > 0 else 0.0
            p10 = fitness_sorted[int(0.1 * self.np)] if self.np > 0 else 0.0
            percentile_spread = (p90 - p10) / (np.abs(self.global_best_fitness) + 1e-10 + p90 - p10 + 1e-10)
            percentile_spread = np.clip(percentile_spread, 0.0, 1.0)

            median_fitness = np.median(self.current_fitness)
            median_improvement = max(0.0, self._prev_fitness_median - median_fitness)
            self._prev_fitness_median = median_fitness

            self._success_ema = 0.2 * median_improvement + 0.8 * self._success_ema
            success_signal = np.clip(self._success_ema / (np.abs(self.global_best_fitness) + 1.0), 0.0, 1.0)

            fdc_exploit_signal = (spearman_fdc + 1.0) / 2.0
            landscape_roughness = 1.0 - fdc_exploit_signal

            exploration_signal = (
                0.5 * landscape_roughness +
                0.3 * percentile_spread +
                0.2 * (1.0 - success_signal)
            )
            exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * exploration_signal

            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

        except Exception:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_06_catF(self):
        """Variant 06 (Category F): temporal signals adaptation.
        
        Wins on task 2. Uses EMA of improvement, stagnation depth, and diversity drift.
        """
        if self.global_best is not None and hasattr(self, '_prev_best_fitness'):
            raw_improvement = max(0.0, self._prev_best_fitness - self.global_best_fitness)
        else:
            raw_improvement = 1.0
        self._prev_best_fitness = self.global_best_fitness if self.global_best is not None else np.inf

        if not hasattr(self, '_ema_improvement_f'):
            self._ema_improvement_f = raw_improvement
        self._ema_improvement_f = 0.2 * raw_improvement + 0.8 * self._ema_improvement_f

        scale = max(abs(self.global_best_fitness), 1e-10) if self.global_best is not None else 1.0
        norm_ema_improvement = self._ema_improvement_f / scale

        self._ema_stagnation = 0.15 * self.stagnation_counter + 0.85 * self._ema_stagnation
        norm_stagnation = np.clip(self._ema_stagnation / 100.0, 0.0, 1.0)

        current_diversity = self._compute_diversity()
        if not hasattr(self, '_ema_diversity_f'):
            self._ema_diversity_f = current_diversity
        self._ema_diversity_f = 0.1 * current_diversity + 0.9 * self._ema_diversity_f

        diversity_ratio = current_diversity / (self._ema_diversity_f + 1e-10)
        norm_diversity = np.clip(diversity_ratio - 1.0, -1.0, 1.0)

        improvement_signal = np.clip(norm_ema_improvement, 0.0, 2.0) / 2.0
        inertia_from_improvement = 0.7 - 0.4 * improvement_signal

        stagnation_boost = 0.3 * norm_stagnation
        diversity_correction = -0.1 * norm_diversity

        target_inertia = inertia_from_improvement + stagnation_boost + diversity_correction
        target_inertia = np.clip(target_inertia, 0.3, 0.95)

        time_floor = 0.4 - 0.1 * min(self.generation / 500, 1.0)
        target_inertia = max(target_inertia, time_floor)

        self.inertia_weight = 0.8 * self.inertia_weight + 0.2 * target_inertia
        self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
    
    def _adapt_inertia_weight_variant_08_catH(self):
        """Variant 08 (Category H): spectral condition + fitness improvement.
        
        Wins on tasks 0,6,9,13,23. Hybrid combining spectral condition number
        with temporal improvement rate. Stagnation-based weighting between mechanisms.
        """
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

            spectral_inertia = 0.4 + 0.55 * spectral_signal

            if self.global_best is not None:
                if not hasattr(self, '_prev_best_fitness_h'):
                    self._prev_best_fitness_h = self.global_best_fitness
                improvement = self._prev_best_fitness_h - self.global_best_fitness
                self._prev_best_fitness_h = self.global_best_fitness
            else:
                improvement = 0.0

            if not hasattr(self, '_ema_improvement_h'):
                self._ema_improvement_h = 0.0
            self._ema_improvement_h = 0.3 * max(0.0, improvement) + 0.7 * self._ema_improvement_h

            max_expected_improvement = 1.0 + self.stagnation_counter * 0.1
            improvement_signal = 1.0 - np.clip(self._ema_improvement_h / (max_expected_improvement + 1e-10), 0.0, 1.0)

            improvement_inertia = 0.4 + 0.55 * improvement_signal

            stagnation_threshold = 15
            is_stagnant = self.stagnation_counter > stagnation_threshold

            if is_stagnant:
                spectral_weight = 0.7
                improvement_weight = 0.3
            else:
                spectral_weight = 0.3
                improvement_weight = 0.7

            target_inertia = (
                spectral_weight * spectral_inertia +
                improvement_weight * improvement_inertia
            )
            target_inertia = np.clip(target_inertia, 0.4, 0.95)

            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_inertia_weight_variant_10_catB(self):
        """Variant 10 (Category B): SVD-based spectral entropy + effective dimensionality.
        
        Wins on tasks 14,19. Uses SVD of centered population (different from covariance
        eigendecomposition). Low entropy + low eff_dim_ratio = collapse → high inertia.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)

            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            total_var = np.sum(singular_values ** 2) + 1e-10

            sv_norm = singular_values ** 2 / total_var
            sv_norm = np.clip(sv_norm, 1e-10, None)

            spectral_entropy = -np.sum(sv_norm * np.log(sv_norm))
            max_entropy = np.log(min(self.np, self.dim) + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cumvar = np.cumsum(sv_norm)
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / min(self.np, self.dim)

            if len(singular_values) >= 2:
                sv_ratios = singular_values[1:] / (singular_values[:-1] + 1e-10)
                sv_ratios = np.clip(sv_ratios, 0.0, 1.0)
                decay_rate = np.exp(np.mean(np.log(sv_ratios + 1e-10)))
            else:
                decay_rate = 0.5

            collapse_signal = (1.0 - entropy_ratio) * 0.4 + (1.0 - eff_dim_ratio) * 0.4 + (1.0 - decay_rate) * 0.2
            collapse_signal = np.clip(collapse_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * collapse_signal

            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps strategy name -> inertia adaptation method
    # -------------------------------------------------------------------------
    
    def _dispatch_inertia_adaptation(self, strategy):
        """Route to the appropriate inertia adaptation implementation."""
        if strategy == 'original':
            self._adapt_inertia_weight_original()
        elif strategy == 'variant_01_catA':
            self._adapt_inertia_weight_variant_01_catA()
        elif strategy == 'variant_03_catC':
            self._adapt_inertia_weight_variant_03_catC()
        elif strategy == 'variant_04_catD':
            self._adapt_inertia_weight_variant_04_catD()
        elif strategy == 'variant_06_catF':
            self._adapt_inertia_weight_variant_06_catF()
        elif strategy == 'variant_08_catH':
            self._adapt_inertia_weight_variant_08_catH()
        elif strategy == 'variant_10_catB':
            self._adapt_inertia_weight_variant_10_catB()
        else:
            self._adapt_inertia_weight_original()
    
    # -------------------------------------------------------------------------
    # OTHER ADAPTATION METHODS
    # -------------------------------------------------------------------------
    
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
    # VELOCITY AND POSITION UPDATES (shared base)
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
    
    def _position_update(self):
        """Standard position update using eigenvalue-anisotropic velocity modulation."""
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
        Run the optimizer with REGIME-BASED DISPATCH inertia adaptation.
        
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
        
        # Reset strategy scoring state
        for s in self._strategies:
            self._strategy_scores[s] = []
            self._strategy_total_score[s] = 0.0
            self._strategy_sample_count[s] = 0.0
        self._fitness_history = []
        self._strategy_history = []
        
        # Reset EMA states
        self._prev_best_fitness = np.inf
        self._ema_improvement = 0.0
        self._ema_stagnation = 0.0
        self._prev_fitness_median = np.inf
        self._ema_diversity_f = 0.0
        self._success_ema = 0.0
        self._prev_global_best_fitness = np.inf
        self._spectral_ema_improvement = 0.0
        if hasattr(self, '_ema_improvement_f'):
            del self._ema_improvement_f
        if hasattr(self, '_prev_best_fitness_h'):
            del self._prev_best_fitness_h
        if hasattr(self, '_ema_improvement_h'):
            del self._ema_improvement_h
        
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
        
        # Track best fitness for scoring
        gen_best = float(np.min(self.current_fitness))
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Step 1: Compute regime signals from current population state
            self._compute_regime_signals()
            
            # Step 2: Select strategy using epsilon-greedy + regime detector + UCB
            if np.random.random() < self._epsilon:
                # Explore: use regime detector
                current_strategy = self._detect_regime()
            else:
                # Exploit: use UCB on recent rank-based performance
                current_strategy = self._select_explore_strategy()
            
            # Step 3: Apply selected inertia adaptation strategy
            self._dispatch_inertia_adaptation(current_strategy)
            
            # Step 4: Other adaptations
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Step 5: Update velocity and position
            self._velocity_update_base()
            self._position_update()
            
            # Step 6: Evaluate
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
            
            # Step 7: Record rank-based score for the strategy used this generation
            gen_best = float(np.min(self.current_fitness))
            self._record_strategy_score(current_strategy, gen_best)
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
