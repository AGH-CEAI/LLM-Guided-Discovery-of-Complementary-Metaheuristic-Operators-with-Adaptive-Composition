import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE) with Ensemble Adaptation
    
    Adaptation mechanism: #2 ENSEMBLE / VOTING / BLENDING
    
    Rationale: Benchmark shows 4 _adapt_parameters variants each dominate on
    different problem types (geometric/spatial, rank-correlation, autocorrelation,
    geometric-layout). These compute correlated-but-distinct signals. A bandit
    cannot reliably credit-assign between correlated arms. An ensemble that runs
    all strategies and combines their F/Cr adjustments via exponentially-weighted
    soft voting provides robust coverage without regime detection cost. The
    original algorithm (13/24 wins) serves as anchor; ensemble blends in
    specialist signals when they agree, reverting to original when they conflict.
    
    Reward: rank-normalized within sliding window (top-tercile binary).
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)
        self.lower = -100.0
        self.upper = 100.0
        
        # Control parameters
        self.F_base = 0.5
        self.Cr_base = 0.3
        self.F = self.F_base
        self.Cr = self.Cr_base
        
        # Directional memory
        self.direction_memory_size = 5
        self.successful_directions = []
        self.direction_decay = 0.95
        
        # Adaptation tracking
        self.adaptation_window = 10
        self.F_success_history = []
        self.Cr_success_history = []
        
        # --- ENSEMBLE MEMBERS ---
        self.strategy_names = ['original', 'geom_pop', 'rank_fitness', 'autocorr', 'geom_layout']
        # Initialize reward history for rank-based credit
        self.reward_window = 15  # K >= 3 * num_arms = 15
        self.reward_history = {s: [] for s in self.strategy_names}
        # Exponentially-weighted moving average of rewards
        self.ewma_reward = {s: 0.0 for s in self.strategy_names}
        self.ewma_alpha = 0.3  # For EWMA reward tracking
        # Initial weights: original anchored high (it wins 13/24)
        self.strategy_weights = {s: 1.0 for s in self.strategy_names}
        self.strategy_weights['original'] = 3.0  # Prior: original is strong
        
        # --- ORIGINAL EMA STATE ---
        self.success_ewma_short = None
        self.success_ewma_long = None
        self.F_history = []
        self.Cr_history = []
        
        # --- VARIANT STATE: geom_pop ---
        self.geo_centroid_history = []
        self.geo_spread_history = []
        
        # --- VARIANT STATE: rank_fitness ---
        self.fitness_rank_history = []
        self._cached_trial_fitness = None
        
        # --- VARIANT STATE: autocorr ---
        self.success_history = []
        self.F_momentum = 0.0
        self.Cr_momentum = 0.0
        self.prev_best_fitness = np.inf
        self.best_fitness_streak = 0
        
        # --- VARIANT STATE: geom_layout ---
        self.stored_population = None
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
        # Stored references for ensemble strategies
        self.population = None
        self.fitness = None
    
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling for better spread."""
        sampler = np.linspace(self.lower, self.upper, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            indices = np.random.permutation(self.np)
            offsets = np.random.uniform(0, 1, self.np)
            population[:, d] = sampler[indices] + offsets * (sampler[1] - sampler[0])
        
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[np.isnan(fitness)] = np.inf
        
        best_idx = np.argmin(fitness)
        return population, fitness, fitness[best_idx], population[best_idx].copy()
    
    def _clip_to_bounds(self, population):
        """Hard clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        """Create ring neighbor indices for each population member."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid for directional bias."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions for adaptive biasing."""
        if not np.any(successful_mask):
            self.successful_directions = [
                d * self.direction_decay for d in self.successful_directions
            ]
            return
        
        for mutation_vec in successful_mutations[successful_mask]:
            self.successful_directions.append(mutation_vec.copy())
        
        if len(self.successful_directions) > self.direction_memory_size:
            self.successful_directions = self.successful_directions[-self.direction_memory_size:]
    
    def _compute_directional_bias(self, target_indices):
        """Compute directional bias from historical successful mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """Generate mutation vectors using composite strategy."""
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        r1 = ring_prev
        r2 = ring_next
        
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        directional_component = directional_bias
        
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        Cr_individual = np.clip(self.Cr + 0.1 * np.random.randn(np_pop), 0.1, 0.9)
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        """Evaluate batch, handle budget exhaustion gracefully."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with diversity maintenance."""
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    # =====================================================================
    # ENSEMBLE ADAPTATION METHODS
    # =====================================================================
    
    def _update_ensemble_rewards(self, success_rate):
        """
        Update rank-based reward history for each strategy.
        Reward = 1 if this generation's success_rate is in top tercile of
        the last K generations, else 0. Ranks are computed per-strategy
        across their own sliding windows.
        """
        for s in self.strategy_names:
            history = self.reward_history[s]
            history.append(success_rate)
            if len(history) > self.reward_window:
                history.pop(0)
            
            # Rank-normalized reward: top tercile = 1, else 0
            if len(history) >= 3:
                sorted_hist = sorted(history)
                tercile_threshold = sorted_hist[int(2 * len(sorted_hist) // 3)]
                reward = 1.0 if success_rate >= tercile_threshold else 0.0
            else:
                reward = 0.5  # Uninformative prior for first few generations
            
            # Update EWMA reward with smoothing
            if self.ewma_reward[s] == 0.0:
                self.ewma_reward[s] = reward
            else:
                self.ewma_reward[s] = (self.ewma_alpha * reward +
                                       (1 - self.ewma_alpha) * self.ewma_reward[s])
    
    def _compute_ensemble_adjustments(self, population, fitness, improved_mask):
        """
        Compute F and Cr adjustments from all strategies, combine via
        exponentially-weighted soft voting. Returns (F_adj, Cr_adj).
        """
        # Get per-strategy adjustments
        adj = {}
        adj['original'] = self._adapt_original_ema(improved_mask)
        adj['geom_pop'] = self._adapt_geom_pop(population)
        adj['rank_fitness'] = self._adapt_rank_fitness(fitness, improved_mask)
        adj['autocorr'] = self._adapt_autocorr(improved_mask, fitness)
        adj['geom_layout'] = self._adapt_geom_layout(population)
        
        # Update weights based on EWMA reward (reward proportional weighting)
        total_reward = sum(self.ewma_reward[s] for s in self.strategy_names) + 1e-10
        for s in self.strategy_names:
            self.strategy_weights[s] = max(0.1, self.ewma_reward[s] / total_reward)
        
        # Soft-vote: weighted geometric mean of adjustment factors
        # Geometric mean is appropriate for multiplicative factors
        log_F_sum = 0.0
        log_Cr_sum = 0.0
        weight_total = 0.0
        
        for s in self.strategy_names:
            w = self.strategy_weights[s]
            f_adj, c_adj = adj[s]
            # Guard against invalid adjustments
            f_adj = np.clip(f_adj, 0.5, 2.0)
            c_adj = np.clip(c_adj, 0.7, 1.4)
            log_F_sum += w * np.log(f_adj)
            log_Cr_sum += w * np.log(c_adj)
            weight_total += w
        
        if weight_total > 1e-10:
            F_ensemble = np.exp(log_F_sum / weight_total)
            Cr_ensemble = np.exp(log_Cr_sum / weight_total)
        else:
            F_ensemble, Cr_ensemble = 1.0, 1.0
        
        # Sanity bounds
        F_ensemble = np.clip(F_ensemble, 0.5, 1.8)
        Cr_ensemble = np.clip(Cr_ensemble, 0.7, 1.35)
        
        return F_ensemble, Cr_ensemble
    
    # =====================================================================
    # STRATEGY IMPLEMENTATIONS (extracted from benchmark variants)
    # =====================================================================
    
    def _adapt_original_ema(self, improved_mask):
        """Original EMA-based adaptation (anchor strategy)."""
        success_rate = np.mean(improved_mask)
        
        if self.success_ewma_short is None:
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]
            return 1.0, 1.0
        
        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = (alpha_short * success_rate +
                                   (1 - alpha_short) * self.success_ewma_short)
        self.success_ewma_long = (alpha_long * success_rate +
                                  (1 - alpha_long) * self.success_ewma_long)
        
        self.F_history.append(self.F)
        self.Cr_history.append(self.Cr)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)
        
        F_mean = np.mean(self.F_history) + 1e-10
        Cr_mean = np.mean(self.Cr_history) + 1e-10
        F_drift = self.F / F_mean
        Cr_drift = self.Cr / Cr_mean
        
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = (self.success_ewma_short < 0.1 and
                         self.success_ewma_long < 0.15)
        
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        
        if short_trending_up and self.success_ewma_short > 0.25:
            F_adj = 1.05 * momentum_factor
            Cr_adj = 1.08
        elif short_trending_down and self.success_ewma_short < 0.2:
            F_adj = 0.88 / momentum_factor
            Cr_adj = 0.92
        elif both_stagnant:
            F_adj = 0.75
            Cr_adj = 1.15
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adj = 1.0 + 0.08 * delta
            Cr_adj = 1.0 - 0.05 * delta
        
        return F_adj, Cr_adj
    
    def _adapt_geom_pop(self, population):
        """
        Variant 01: geometric/spatial population characteristics.
        Wins on tasks: 19, 21 (combined with variant 09).
        """
        if population is None or len(population) < 2:
            return 1.0, 1.0
        
        # Compute centroid and drift
        centroid = np.mean(population, axis=0)
        self.geo_centroid_history.append(centroid.copy())
        if len(self.geo_centroid_history) > 10:
            self.geo_centroid_history.pop(0)
        
        centroid_drift = 0.0
        if len(self.geo_centroid_history) >= 2:
            centroid_drift = np.linalg.norm(
                self.geo_centroid_history[-1] - self.geo_centroid_history[0]
            )
        
        # Geometric spread
        dist_to_centroid = np.linalg.norm(population - centroid, axis=1)
        mean_dist = np.mean(dist_to_centroid)
        search_range = self.upper - self.lower
        normalized_spread = mean_dist / (search_range + 1e-10)
        
        # k-NN connectivity
        k = min(5, len(population) - 1)
        knn_distances = []
        for i in range(len(population)):
            distances = np.linalg.norm(population - population[i], axis=1)
            distances[i] = np.inf
            knn_distances.append(np.mean(np.partition(distances, k)[:k]))
        mean_knn_dist = np.mean(knn_distances)
        normalized_knn = np.clip(mean_knn_dist / (search_range * 0.5 + 1e-10), 0.0, 1.0)
        
        # Spread trend
        self.geo_spread_history.append(normalized_spread)
        if len(self.geo_spread_history) > 15:
            self.geo_spread_history.pop(0)
        
        spread_trend = 0.0
        if len(self.geo_spread_history) >= 3:
            recent = np.mean(self.geo_spread_history[-2:])
            older = np.mean(self.geo_spread_history[:2])
            spread_trend = (recent - older) / (older + 1e-10)
        
        # Exploration pressure
        exploration_pressure = (1.0 - normalized_spread +
                                (1.0 - normalized_knn) * 0.5)
        exploration_pressure = np.clip(exploration_pressure, 0.0, 2.0)
        
        if exploration_pressure > 1.2:
            F_adj = 1.0 + 0.15 * exploration_pressure
            Cr_adj = 0.90
        elif exploration_pressure < 0.5:
            F_adj = 0.85
            Cr_adj = 1.0 + 0.1 * (0.5 - exploration_pressure)
        else:
            F_adj = 1.0 + 0.05 * spread_trend
            Cr_adj = 1.0 - 0.03 * spread_trend
        
        if centroid_drift > 0.1 * search_range:
            F_adj *= 1.1
            Cr_adj *= 0.95
        
        return F_adj, Cr_adj
    
    def _adapt_rank_fitness(self, fitness, improved_mask):
        """
        Variant 04: Spearman rank correlation and fitness percentile analysis.
        Wins on tasks: 11, 18, 22.
        """
        if fitness is None or len(fitness) < 2:
            return 1.0, 1.0
        
        pop_size = len(fitness)
        current_ranks = np.argsort(np.argsort(fitness)) / max(pop_size - 1, 1)
        
        # Spearman rank correlation
        rank_corr = 0.5
        if len(self.fitness_rank_history) >= 1:
            prev_ranks = self.fitness_rank_history[-1]
            try:
                from scipy.stats import spearmanr
                rc, _ = spearmanr(prev_ranks, current_ranks)
                if not np.isnan(rc):
                    rank_corr = rc
            except Exception:
                pass
        
        # Fitness spread
        fit_25 = np.percentile(fitness, 25)
        fit_75 = np.percentile(fitness, 75)
        fit_mean = np.mean(fitness)
        fit_spread = (fit_75 - fit_25) / (np.abs(fit_mean) + 1e-10)
        
        # Improvement percentile
        improv_percentile = 0.0
        if (np.sum(improved_mask) > 0 and
                self._cached_trial_fitness is not None and
                len(self._cached_trial_fitness) == len(fitness)):
            fitness_diff = fitness - self._cached_trial_fitness
            improvements = fitness_diff[improved_mask]
            if len(improvements) > 0:
                improv_percentile = np.percentile(improvements, 75)
        
        # Regime detection
        high_rank_stability = rank_corr > 0.75
        low_rank_stability = rank_corr < 0.4
        tight_clustering = fit_spread < 0.15
        small_improvements = improv_percentile < 0.25 * (fit_75 - fit_25 + 1e-10)
        
        if high_rank_stability and tight_clustering:
            F_adj = 1.35
            Cr_adj = 0.75
        elif low_rank_stability and improv_percentile > 0:
            F_adj = 0.85
            Cr_adj = 1.1
        elif small_improvements and tight_clustering:
            F_adj = 1.4
            Cr_adj = 0.7
        elif high_rank_stability and not tight_clustering:
            F_adj = 0.9
            Cr_adj = 1.05
        else:
            F_adj = 0.95 + 0.1 * (1.0 - rank_corr)
            Cr_adj = 0.95 + 0.15 * rank_corr
        
        # Update rank history
        self.fitness_rank_history.append(current_ranks.copy())
        if len(self.fitness_rank_history) > 8:
            self.fitness_rank_history.pop(0)
        
        return F_adj, Cr_adj
    
    def _adapt_autocorr(self, improved_mask, fitness):
        """
        Variant 06: autocorrelation-based regime detection and parameter momentum.
        Wins on tasks: 10, 12, 16.
        """
        success_rate = np.mean(improved_mask)
        
        if not hasattr(self, 'success_history'):
            self.success_history = []
            self.success_ewma_short_a = success_rate
            self.success_ewma_long_a = success_rate
            self.F_momentum = 0.0
            self.Cr_momentum = 0.0
            self.prev_best_fitness = np.inf
            self.best_fitness_streak = 0
            self.F_history_a = [self.F]
            self.Cr_history_a = [self.Cr]
        
        alpha_short = 0.4
        alpha_long = 0.15
        self.success_ewma_short_a = (alpha_short * success_rate +
                                     (1 - alpha_short) * self.success_ewma_short_a)
        self.success_ewma_long_a = (alpha_long * success_rate +
                                    (1 - alpha_long) * self.success_ewma_long_a)
        
        self.success_history.append(success_rate)
        if len(self.success_history) > 12:
            self.success_history.pop(0)
        
        # Lag-1 autocorrelation
        autocorr_lag1 = 0.5
        if len(self.success_history) >= 4:
            sh = np.array(self.success_history)
            mean_s = np.mean(sh)
            var_s = np.var(sh) + 1e-10
            autocorr_lag1 = np.mean((sh[:-1] - mean_s) * (sh[1:] - mean_s)) / var_s
            if np.isnan(autocorr_lag1):
                autocorr_lag1 = 0.5
        
        success_derivative = success_rate - self.success_ewma_long_a
        
        # Parameter momentum
        self.F_history_a.append(self.F)
        self.Cr_history_a.append(self.Cr)
        if len(self.F_history_a) > 15:
            self.F_history_a.pop(0)
            self.Cr_history_a.pop(0)
        
        if len(self.F_history_a) >= 3:
            F_velocity = (self.F_history_a[-1] - self.F_history_a[-3]) / 2.0
            Cr_velocity = (self.Cr_history_a[-1] - self.Cr_history_a[-3]) / 2.0
            self.F_momentum = 0.7 * self.F_momentum + 0.3 * F_velocity
            self.Cr_momentum = 0.7 * self.Cr_momentum + 0.3 * Cr_velocity
        
        # Best fitness streak
        if fitness is not None and len(fitness) > 0:
            current_best = np.min(fitness)
            if current_best < self.prev_best_fitness - 1e-10:
                self.best_fitness_streak += 1
            else:
                self.best_fitness_streak = 0
            self.prev_best_fitness = current_best
        
        # Regime detection
        oscillating = autocorr_lag1 < -0.15
        steady_converging = autocorr_lag1 > 0.5 and success_rate > 0.2
        accelerating = success_derivative > 0.05
        decelerating = success_derivative < -0.05
        stagnant = (self.success_ewma_short_a < 0.1 and
                    self.success_ewma_long_a < 0.15)
        improving_streak = self.best_fitness_streak > 10
        
        F_hist_std = np.std(self.F_history_a) + 1e-10
        momentum_damp = np.clip(1.0 - 0.3 * abs(self.F_momentum) / F_hist_std, 0.6, 1.0)
        
        if stagnant or (improving_streak and success_rate < 0.15):
            F_adj = 0.7 * momentum_damp
            Cr_adj = 1.2
            self.F_momentum *= 0.5
            self.Cr_momentum *= 0.5
        elif oscillating:
            F_adj = 0.85 * momentum_damp
            Cr_adj = np.clip(0.9 - 0.2 * autocorr_lag1, 0.7, 1.3)
        elif accelerating and steady_converging:
            F_adj = 1.08 * momentum_damp
            Cr_adj = 1.05
        elif decelerating and not oscillating:
            F_adj = 0.92
            Cr_adj = 0.95
        else:
            delta = self.success_ewma_short_a - self.success_ewma_long_a
            F_adj = 1.0 + 0.1 * delta
            Cr_adj = 1.0 - 0.05 * delta
        
        return F_adj, Cr_adj
    
    def _adapt_geom_layout(self, population):
        """
        Variant 09: geometric properties of population layout.
        Wins on tasks: 14, 15, 23.
        """
        if population is None or len(population) < 2:
            return 1.0, 1.0
        
        self.stored_population = population
        
        dim_min = np.min(population, axis=0)
        dim_max = np.max(population, axis=0)
        dim_spread = dim_max - dim_min
        total_spread = np.sum(dim_spread)
        range_normalized_spread = total_spread / (self.dim * (self.upper - self.lower) + 1e-10)
        
        centroid = np.mean(population, axis=0)
        centroid_normalized = (centroid - self.lower) / (self.upper - self.lower + 1e-10)
        corner_proximity = np.min(centroid_normalized)
        edge_bias = np.max(np.abs(centroid_normalized - 0.5))
        
        distances_to_centroid = np.linalg.norm(population - centroid, axis=1)
        mean_dist = np.mean(distances_to_centroid)
        max_possible_dist = np.sqrt(self.dim) * (self.upper - self.lower) / 2
        compactness = mean_dist / (max_possible_dist + 1e-10)
        
        pairwise_dist = np.linalg.norm(
            population[:, np.newaxis, :] - population[np.newaxis, :, :], axis=2
        )
        np.fill_diagonal(pairwise_dist, np.inf)
        mean_pairwise = np.mean(np.min(pairwise_dist, axis=1))
        max_pairwise = np.max(np.min(pairwise_dist, axis=1))
        clustering_ratio = mean_pairwise / (max_pairwise + 1e-10)
        
        is_tight = compactness < 0.15
        is_dispersed = compactness > 0.4
        is_corner_trapped = corner_proximity < 0.1 or edge_bias > 0.4
        is_clustered = clustering_ratio < 0.3
        
        if is_tight and is_corner_trapped:
            F_adj = 1.30
            Cr_adj = 0.80
        elif is_tight and not is_corner_trapped:
            F_adj = 0.85
            Cr_adj = 1.10
        elif is_dispersed:
            F_adj = 0.80
            Cr_adj = 1.05
        elif is_clustered and not is_tight:
            F_adj = 1.10
            Cr_adj = 0.90
        else:
            F_adj = 1.0 + 0.20 * (range_normalized_spread - 0.3)
            Cr_adj = 1.0 - 0.15 * (range_normalized_spread - 0.3)
        
        return F_adj, Cr_adj
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Ensemble adaptation combining multiple parameter strategies."""
        success_rate = np.mean(improved_mask)
        
        # Update ensemble reward history (rank-based)
        self._update_ensemble_rewards(success_rate)
        
        # Compute ensemble adjustments
        F_adj_ensemble, Cr_adj_ensemble = self._compute_ensemble_adjustments(
            self.population, self.fitness, improved_mask
        )
        
        # Apply ensemble adjustments
        self.F = np.clip(self.F * F_adj_ensemble, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adj_ensemble, 0.1, 0.9)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter > self.stagnation_threshold or
                diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
            # Reset ensemble history on restart (fresh start)
            for s in self.strategy_names:
                self.reward_history[s] = []
                self.ewma_reward[s] = 0.0
                self.strategy_weights[s] = 1.0
            self.strategy_weights['original'] = 3.0
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        # Store references for ensemble strategies
        self.population = population
        self.fitness = fitness
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Cache trial fitness for rank-based strategy
            self._cached_trial_fitness = trial_fitness.copy()
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Update stored references
            self.population = population
            self.fitness = fitness
            
            # Adapt control parameters (ensemble)
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Update stored references after potential restart
            self.population = population
            self.fitness = fitness
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
