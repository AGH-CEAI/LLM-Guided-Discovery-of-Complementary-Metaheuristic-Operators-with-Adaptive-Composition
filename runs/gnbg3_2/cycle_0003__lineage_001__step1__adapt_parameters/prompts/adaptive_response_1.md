```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    ENSEMBLE ADAPTATION (Mechanism #2) — Blends five correlated _adapt_parameters
    strategies via confidence-weighted voting. Confidence updates use rank-normalized
    success rates within a sliding window, avoiding scale-dependent magic constants.
    
    Rationale: Benchmark shows no single strategy dominates (original wins 54%,
    others win 2-3 tasks each). Variants compute correlated but distinct signals
    (geometric, rank-based, autocorrelation). A bandit cannot distinguish near-
    identical arms; an ensemble gives smooth, robust blending without credit assignment.
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
        
        # Ensemble adaptation state
        self.reward_window = 12  # k >= 3 * num_arms (4 arms here)
        
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
    
    def _adapt_parameters_variant_01(self, improved_mask):
        """Variant 01: Geometric/spatial population characteristics."""
        population = self.population if hasattr(self, 'population') else None

        if not hasattr(self, 'geo_centroid_history'):
            self.geo_centroid_history = []
            self.geo_spread_history = []

        if population is not None and len(population) > 0:
            centroid = np.mean(population, axis=0)

            self.geo_centroid_history.append(centroid.copy())
            if len(self.geo_centroid_history) > 10:
                self.geo_centroid_history.pop(0)

            centroid_drift = 0.0
            if len(self.geo_centroid_history) >= 2:
                centroid_drift = np.linalg.norm(self.geo_centroid_history[-1] - self.geo_centroid_history[0])

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

            self.geo_spread_history.append(normalized_spread)
            if len(self.geo_spread_history) > 15:
                self.geo_spread_history.pop(0)

            spread_trend = 0.0
            if len(self.geo_spread_history) >= 3:
                recent = np.mean(self.geo_spread_history[-2:])
                older = np.mean(self.geo_spread_history[:2])
                spread_trend = (recent - older) / (older + 1e-10)

            exploration_pressure = 1.0 - normalized_spread
            exploration_pressure += (1.0 - normalized_knn) * 0.5
            exploration_pressure = np.clip(exploration_pressure, 0.0, 2.0)

            if exploration_pressure > 1.2:
                F_adjustment = 1.0 + 0.15 * exploration_pressure
                Cr_adjustment = 0.90
            elif exploration_pressure < 0.5:
                F_adjustment = 0.85
                Cr_adjustment = 1.0 + 0.1 * (0.5 - exploration_pressure)
            else:
                F_adjustment = 1.0 + 0.05 * spread_trend
                Cr_adjustment = 1.0 - 0.03 * spread_trend

            if centroid_drift > 0.1 * search_range:
                F_adjustment *= 1.1
                Cr_adjustment *= 0.95

            self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
        else:
            self.F = np.clip(self.F * 0.98 + self.F_base * 0.02, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * 0.98 + self.Cr_base * 0.02, 0.1, 0.9)
    
    def _adapt_parameters_variant_04(self, improved_mask):
        """Variant 04: Spearman rank correlation and fitness percentile landscape analysis."""
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return
        
        if not hasattr(self, '_cached_fitness') or self._cached_fitness is None:
            return
        current_fitness = self._cached_fitness
        pop_size = len(current_fitness)

        if not hasattr(self, 'fitness_rank_history'):
            self.fitness_rank_history = []
            self.improvement_percentile_history = []
            self.fitness_spread_history = []

        current_ranks = np.argsort(np.argsort(current_fitness)) / max(pop_size - 1, 1)

        if len(self.fitness_rank_history) >= 1:
            prev_ranks = self.fitness_rank_history[-1]
            rank_corr, _ = spearmanr(prev_ranks, current_ranks)
            if np.isnan(rank_corr):
                rank_corr = 0.5
        else:
            rank_corr = 0.5

        if np.sum(improved_mask) > 0 and hasattr(self, '_cached_trial_fitness'):
            trial_fitness = self._cached_trial_fitness
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
            F_adjustment = 1.35
            Cr_adjustment = 0.75
        elif low_rank_stability and improv_percentile > 0:
            F_adjustment = 0.85
            Cr_adjustment = 1.1
        elif small_improvements and tight_clustering:
            F_adjustment = 1.4
            Cr_adjustment = 0.7
        elif high_rank_stability and not tight_clustering:
            F_adjustment = 0.9
            Cr_adjustment = 1.05
        else:
            F_adjustment = 0.95 + 0.1 * (1.0 - rank_corr)
            Cr_adjustment = 0.95 + 0.15 * rank_corr

        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)

        self.fitness_rank_history.append(current_ranks.copy())
        if len(self.fitness_rank_history) > 8:
            self.fitness_rank_history.pop(0)

        self.improvement_percentile_history.append(improv_percentile)
        if len(self.improvement_percentile_history) > 8:
            self.improvement_percentile_history.pop(0)

        self.fitness_spread_history.append(fit_spread)
        if len(self.fitness_spread_history) > 8:
            self.fitness_spread_history.pop(0)
    
    def _adapt_parameters_variant_06(self, improved_mask, F_used, Cr_used):
        """Variant 06: Autocorrelation-based regime detection and parameter momentum."""
        success_rate = np.mean(improved_mask)

        if not hasattr(self, 'success_history'):
            self.success_history = []
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.F_momentum = 0.0
            self.Cr_momentum = 0.0
            self.v06_stagnation_counter = 0
            self.prev_best_fitness = np.inf
            self.best_fitness_streak = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        alpha_short = 0.4
        alpha_long = 0.15
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        self.success_history.append(success_rate)
        if len(self.success_history) > 12:
            self.success_history.pop(0)

        if len(self.success_history) >= 4:
            sh = np.array(self.success_history)
            mean_s = np.mean(sh)
            var_s = np.var(sh) + 1e-10
            autocorr_lag1 = np.mean((sh[:-1] - mean_s) * (sh[1:] - mean_s)) / var_s
        else:
            autocorr_lag1 = 0.5

        success_derivative = success_rate - self.success_ewma_long

        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)

        if len(self.F_history) >= 3:
            F_velocity = (self.F_history[-1] - self.F_history[-3]) / 2.0
            Cr_velocity = (self.Cr_history[-1] - self.Cr_history[-3]) / 2.0
            self.F_momentum = 0.7 * self.F_momentum + 0.3 * F_velocity
            self.Cr_momentum = 0.7 * self.Cr_momentum + 0.3 * Cr_velocity

        current_best = np.min(self.population.fitness) if hasattr(self, 'population') else np.inf
        if current_best < self.prev_best_fitness - 1e-10:
            self.best_fitness_streak += 1
        else:
            self.best_fitness_streak = 0
        self.prev_best_fitness = current_best

        oscillating = autocorr_lag1 < -0.15
        steady_converging = autocorr_lag1 > 0.5 and success_rate > 0.2
        accelerating = success_derivative > 0.05
        decelerating = success_derivative < -0.05
        stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
        improving_streak = self.best_fitness_streak > 10

        F_hist_std = np.std(self.F_history) + 1e-10
        momentum_damp = np.clip(1.0 - 0.3 * abs(self.F_momentum) / F_hist_std, 0.6, 1.0)

        if stagnant or (improving_streak and success_rate < 0.15):
            F_adjustment = 0.7 * momentum_damp
            Cr_adjustment = 1.2
            self.F_momentum *= 0.5
            self.Cr_momentum *= 0.5
            self.v06_stagnation_counter += 2
        elif oscillating:
            F_adjustment = 0.85 * momentum_damp
            Cr_adjustment = 0.9 - 0.2 * autocorr_lag1
            self.v06_stagnation_counter += 1
        elif accelerating and steady_converging:
            F_adjustment = 1.08 * momentum_damp
            Cr_adjustment = 1.05
            self.v06_stagnation_counter = max(0, self.v06_stagnation_counter - 1)
        elif decelerating and not oscillating:
            F_adjustment = 0.92
            Cr_adjustment = 0.95
            self.v06_stagnation_counter += 1
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.1 * delta
            Cr_adjustment = 1.0 - 0.05 * delta

        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    def _adapt_parameters_variant_09(self, improved_mask):
        """Variant 09: Geometric properties of population layout."""
        if not hasattr(self, 'stored_population') or self.stored_population is None:
            self.F = np.clip(self.F * 1.0, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * 1.0, 0.1, 0.9)
            return

        pop = self.stored_population
        if len(pop) < 2:
            return

        dim_min = np.min(pop, axis=0)
        dim