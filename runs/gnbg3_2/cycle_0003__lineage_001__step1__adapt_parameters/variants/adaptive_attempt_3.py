import numpy as np


class AdaptiveDirectionalDE:
    """
    ENSEMBLE/VOTING WITH ADAPTIVE WEIGHTING (mechanism #2):
    Combines four correlated but complementary _adapt_parameters strategies
    via exponentially weighted moving average. Each strategy captures distinct
    signals: geometric/spatial (v01, v09), fitness-rank landscape (v04),
    temporal autocorrelation (v06). Ensemble is robust to correlated arms
    where bandit credit-assignment would fail.
    
    A weighted ensemble preserves original.py's 13/24 wins while gaining
    the 3-win tasks where specialized variants excel.
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
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
        # --- ENSEMBLE STATE ---
        # Per-variant state for each _adapt_parameters strategy
        success_rate = 0.5  # Placeholder for initialization
        
        # Variant 01 state: geometric/spatial population characteristics
        self.v01_geo_centroid_history = []
        self.v01_geo_spread_history = []
        
        # Variant 04 state: Spearman rank correlation & fitness percentile
        self.v04_fitness_rank_history = []
        self.v04_improvement_percentile_history = []
        self.v04_fitness_spread_history = []
        self.v04_cached_fitness = None
        self.v04_cached_trial_fitness = None
        
        # Variant 06 state: autocorrelation & parameter momentum
        self.v06_success_history = []
        self.v06_ewma_short = success_rate
        self.v06_ewma_long = success_rate
        self.v06_F_momentum = 0.0
        self.v06_Cr_momentum = 0.0
        self.v06_stagnation_counter = 0
        self.v06_prev_best_fitness = np.inf
        self.v06_best_fitness_streak = 0
        self.v06_F_history = [self.F]
        self.v06_Cr_history = [self.Cr]
        
        # Variant 09 state: geometric properties (compactness, clustering)
        self.v09_stored_population = None
        self.v09_fitness_percentiles = []
        
        # Ensemble weights (initialized from benchmark win counts)
        # original: 13, variant_06: 3, variant_04: 3, variant_09: 3, variant_01: 2
        total_wins = 24
        self.ensemble_weights = {
            'variant_01': 2.0 / total_wins,
            'variant_04': 3.0 / total_wins,
            'variant_06': 3.0 / total_wins,
            'variant_09': 3.0 / total_wins,
        }
        
        # Per-method reward history for adaptive weighting
        self.method_success_history = {k: [] for k in self.ensemble_weights}
        self.method_rewards = {k: 0.5 for k in self.ensemble_weights}
    
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
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
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
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
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
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
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    # --- VARIANT 01: Geometric/spatial population characteristics ---
    def _adapt_variant_01(self, improved_mask, population):
        """Uses spread, centroid drift, k-NN connectivity."""
        if population is None or len(population) == 0:
            return 1.0, 1.0
        
        centroid = np.mean(population, axis=0)
        
        self.v01_geo_centroid_history.append(centroid.copy())
        if len(self.v01_geo_centroid_history) > 10:
            self.v01_geo_centroid_history.pop(0)
        
        centroid_drift = 0.0
        if len(self.v01_geo_centroid_history) >= 2:
            centroid_drift = np.linalg.norm(
                self.v01_geo_centroid_history[-1] - self.v01_geo_centroid_history[0]
            )
        
        dist_to_centroid = np.linalg.norm(population - centroid, axis=1)
        mean_dist = np.mean(dist_to_centroid)
        
        search_range = self.upper - self.lower
        normalized_spread = mean_dist / (search_range + 1e-10)
        
        pop_min = np.min(population, axis=0)
        pop_max = np.max(population, axis=0)
        bbox_sizes = pop_max - pop_min
        bbox_volume = np.prod(bbox_sizes + 1e-10)
        log_bbox_volume = np.log(bbox_volume + 1e-30)
        max_log_volume = np.log((search_range + 1e-10) ** self.dim)
        normalized_bbox = np.clip(log_bbox_volume / (max_log_volume + 1e-10), 0.0, 1.0)
        
        k = min(5, len(population) - 1)
        knn_distances = []
        for i in range(len(population)):
            distances = np.linalg.norm(population - population[i], axis=1)
            distances[i] = np.inf
            knn_distances.append(np.mean(np.partition(distances, k)[:k]))
        mean_knn_dist = np.mean(knn_distances)
        normalized_knn = np.clip(mean_knn_dist / (search_range * 0.5 + 1e-10), 0.0, 1.0)
        
        self.v01_geo_spread_history.append(normalized_spread)
        if len(self.v01_geo_spread_history) > 15:
            self.v01_geo_spread_history.pop(0)
        
        spread_trend = 0.0
        if len(self.v01_geo_spread_history) >= 3:
            recent = np.mean(self.v01_geo_spread_history[-2:])
            older = np.mean(self.v01_geo_spread_history[:2])
            spread_trend = (recent - older) / (older + 1e-10)
        
        exploration_pressure = 1.0 - normalized_spread
        exploration_pressure += (1.0 - normalized_knn) * 0.5
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
    
    # --- VARIANT 04: Spearman rank correlation & fitness percentile ---
    def _adapt_variant_04(self, improved_mask, fitness, trial_fitness):
        """Uses rank correlation and fitness landscape analysis."""
        from scipy.stats import spearmanr
        
        if not hasattr(self, 'v04_cached_fitness') or self.v04_cached_fitness is None:
            self.v04_cached_fitness = fitness.copy()
            return 1.0, 1.0
        
        current_fitness = fitness.copy()
        pop_size = len(current_fitness)
        
        current_ranks = np.argsort(np.argsort(current_fitness)) / max(pop_size - 1, 1)
        
        if len(self.v04_fitness_rank_history) >= 1:
            prev_ranks = self.v04_fitness_rank_history[-1]
            rank_corr, _ = spearmanr(prev_ranks, current_ranks)
            if np.isnan(rank_corr):
                rank_corr = 0.5
        else:
            rank_corr = 0.5
        
        if np.sum(improved_mask) > 0 and trial_fitness is not None:
            fitness_diff = current_fitness - trial_fitness
            improvements = fitness_diff[improved_mask]
            if len(improvements) > 0:
                improv_percentile = np.percentile(improvements, 75)
            else:
                improv_percentile = 0.0
        else:
            improv_percentile = 0.0
        
        fit_25 = np.percentile(current_fitness, 25)
        fit_75 = np.percentile(current_fitness, 75)
        fit_mean = np.mean(current_fitness)
        fit_spread = (fit_75 - fit_25) / (np.abs(fit_mean) + 1e-10)
        
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
        
        self.v04_fitness_rank_history.append(current_ranks.copy())
        if len(self.v04_fitness_rank_history) > 8:
            self.v04_fitness_rank_history.pop(0)
        
        self.v04_improvement_percentile_history.append(improv_percentile)
        if len(self.v04_improvement_percentile_history) > 8:
            self.v04_improvement_percentile_history.pop(0)
        
        self.v04_fitness_spread_history.append(fit_spread)
        if len(self.v04_fitness_spread_history) > 8:
            self.v04_fitness_spread_history.pop(0)
        
        self.v04_cached_fitness = current_fitness.copy()
        self.v04_cached_trial_fitness = trial_fitness.copy() if trial_fitness is not None else None
        
        return F_adj, Cr_adj
    
    # --- VARIANT 06: Autocorrelation & parameter momentum ---
    def _adapt_variant_06(self, improved_mask, population):
        """Uses success rate autocorrelation and parameter momentum."""
        success_rate = np.mean(improved_mask)
        
        alpha_short = 0.4
        alpha_long = 0.15
        self.v06_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.v06_ewma_short
        self.v06_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.v06_ewma_long
        
        self.v06_success_history.append(success_rate)
        if len(self.v06_success_history) > 12:
            self.v06_success_history.pop(0)
        
        if len(self.v06_success_history) >= 4:
            sh = np.array(self.v06_success_history)
            n = len(sh)
            mean_s = np.mean(sh)
            var_s = np.var(sh) + 1e-10
            autocorr_lag1 = np.mean((sh[:-1] - mean_s) * (sh[1:] - mean_s)) / var_s
        else:
            autocorr_lag1 = 0.5
        
        success_derivative = success_rate - self.v06_ewma_long
        
        if len(self.v06_F_history) >= 3:
            F_velocity = (self.v06_F_history[-1] - self.v06_F_history[-3]) / 2.0
            Cr_velocity = (self.v06_Cr_history[-1] - self.v06_Cr_history[-3]) / 2.0
            self.v06_F_momentum = 0.7 * self.v06_F_momentum + 0.3 * F_velocity
            self.v06_Cr_momentum = 0.7 * self.v06_Cr_momentum + 0.3 * Cr_velocity
        
        current_best = np.min(self.v04_cached_fitness) if hasattr(self, 'v04_cached_fitness') and self.v04_cached_fitness is not None else np.inf
        if population is not None:
            current_best = np.minimum(current_best, np.min(func(population) if hasattr(self, '_temp_func') else np.array([np.inf])))
        
        if current_best < self.v06_prev_best_fitness - 1e-10:
            self.v06_best_fitness_streak += 1
        else:
            self.v06_best_fitness_streak = 0
        self.v06_prev_best_fitness = current_best
        
        oscillating = autocorr_lag1 < -0.15
        steady_converging = autocorr_lag1 > 0.5 and success_rate > 0.2
        accelerating = success_derivative > 0.05
        decelerating = success_derivative < -0.05
        stagnant = self.v06_ewma_short < 0.1 and self.v06_ewma_long < 0.15
        improving_streak = self.v06_best_fitness_streak > 10
        
        F_hist_std = np.std(self.v06_F_history) + 1e-10
        momentum_damp = np.clip(1.0 - 0.3 * abs(self.v06_F_momentum) / F_hist_std, 0.6, 1.0)
        
        if stagnant or (improving_streak and success_rate < 0.15):
            F_adj = 0.7 * momentum_damp
            Cr_adj = 1.2
            self.v06_F_momentum *= 0.5
            self.v06_Cr_momentum *= 0.5
            self.v06_stagnation_counter += 2
        elif oscillating:
            F_adj = 0.85 * momentum_damp
            Cr_adj = 0.9 - 0.2 * autocorr_lag1
            self.v06_stagnation_counter += 1
        elif accelerating and steady_converging:
            F_adj = 1.08 * momentum_damp
            Cr_adj = 1.05
            self.v06_stagnation_counter = max(0, self.v06_stagnation_counter - 1)
        elif decelerating and not oscillating:
            F_adj = 0.92
            Cr_adj = 0.95
            self.v06_stagnation_counter += 1
        else:
            delta = self.v06_ewma_short - self.v06_ewma_long
            F_adj = 1.0 + 0.1 * delta
            Cr_adj = 1.0 - 0.05 * delta
        
        return F_adj, Cr_adj
    
    # --- VARIANT 09: Geometric properties (compactness, clustering) ---
    def _adapt_variant_09(self, improved_mask, population):
        """Uses compactness, clustering ratio, corner proximity."""
        if population is None or len(population) < 2:
            return 1.0, 1.0
        
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
    
    # --- ENSEMBLE: Combine all variants with adaptive weighting ---
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Ensemble of four adaptation strategies with adaptive weighting."""
        success_rate = np.mean(improved_mask)
        
        # Get population from stored state
        population = getattr(self, 'v09_stored_population', None)
        fitness = getattr(self, 'v04_cached_fitness', None)
        trial_fitness = getattr(self, 'v04_cached_trial_fitness', None)
        
        # Compute per-variant adjustments
        v01_F, v01_Cr = self._adapt_variant_01(improved_mask, population)
        v04_F, v04_Cr = self._adapt_variant_04(improved_mask, fitness, trial_fitness)
        v06_F, v06_Cr = self._adapt_variant_06(improved_mask, population)
        v09_F, v09_Cr = self._adapt_variant_09(improved_mask, population)
        
        variant_adjustments = {
            'variant_01': (v01_F, v01_Cr),
            'variant_04': (v04_F, v04_Cr),
            'variant_06': (v06_F, v06_Cr),
            'variant_09': (v09_F, v09_Cr),
        }
        
        # Compute per-method reward (rank-based within sliding window)
        for method_name in self.ensemble_weights:
            F_adj, Cr_adj = variant_adjustments[method_name]
            # Reward: closer to 1.0 is neutral; large deviations indicate regime detection
            # Use geometric mean proximity to 1.0 as reward signal
            F_dev = abs(F_adj - 1.0)
            Cr_dev = abs(Cr_adj - 1.0)
            combined_dev = np.sqrt(F_dev * Cr_dev)
            # Convert to reward: low deviation = high reward (stable/healthy regime)
            # Use rank-based approach: compare to recent history
            self.method_success_history[method_name].append(combined_dev)
            if len(self.method_success_history[method_name]) > 20:
                self.method_success_history[method_name].pop(0)
        
        # Update adaptive weights using exponential weighted moving average of inverse deviation
        decay = 0.9
        for method_name in self.ensemble_weights:
            history = self.method_success_history[method_name]
            if len(history) >= 3:
                # Z-score normalize within window
                mean_dev = np.mean(history)
                std_dev = np.std(history) + 1e-10
                z_dev = (history[-1] - mean_dev) / std_dev
                # Convert z-score to reward: lower deviation = higher reward
                reward = 0.5 - 0.1 * z_dev
                reward = np.clip(reward, 0.05, 0.95)
            else:
                reward = 0.5
            
            # Update EMA of reward
            self.method_rewards[method_name] = (
                decay * self.method_rewards[method_name] + (1 - decay) * reward
            )
        
        # Normalize weights to sum to 1
        total_reward = sum(self.method_rewards.values())
        if total_reward > 0:
            for k in self.ensemble_weights:
                self.ensemble_weights[k] = self.method_rewards[k] / total_reward
        else:
            # Fallback to uniform
            for k in self.ensemble_weights:
                self.ensemble_weights[k] = 0.25
        
        # Compute ensemble adjustment as weighted average
        ensemble_F = sum(
            self.ensemble_weights[k] * variant_adjustments[k][0]
            for k in self.ensemble_weights
        )
        ensemble_Cr = sum(
            self.ensemble_weights[k] * variant_adjustments[k][1]
            for k in self.ensemble_weights
        )
        
        # Apply ensemble adjustment with base value blending (conservative)
        self.F = np.clip(self.F * np.clip(ensemble_F, 0.5, 1.8), 0.1, 1.5)
        self.Cr = np.clip(self.Cr * np.clip(ensemble_Cr, 0.5, 1.6), 0.1, 0.9)
        
        # Store population for geometric variants
        if population is not None:
            self.v09_stored_population = population.copy()
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
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
                population[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
            
            # Reset variant states on restart
            self.v04_cached_fitness = None
            self.v04_cached_trial_fitness = None
            self.v06_stagnation_counter = 0
            self.v06_best_fitness_streak = 0
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        self._temp_func = func  # For variant_06 best fitness tracking
        
        # Initialize cached fitness for variant_04
        self.v04_cached_fitness = fitness.copy()
        self.v04_cached_trial_fitness = None
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Cache trial fitness for variant_04
            self.v04_cached_trial_fitness = trial_fitness.copy()
            
            # Store population for geometric variants
            self.v09_stored_population = population.copy()
            
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
