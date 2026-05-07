import numpy as np


class AdaptiveDirectionalDE:
    """
    ADAPTIVE ENSEMBLE WITH REGIME-CONDITIONAL WEIGHTING (Mechanism #2 + #3 hybrid)
    
    Rationale: The benchmark shows variant_06 (temporal signals) winning 11 tasks,
    variant_07 (bootstrap confidence) winning 6, original (EMA) winning 6, and
    variant_01 (geometry) winning 1. These variants capture fundamentally different
    aspects: temporal dynamics, statistical uncertainty, and spatial structure.
    A plain bandit cannot credit-assign reliably across such different signals.
    Instead, we run all strategies every generation and combine them via regime-
    conditional weighted ensemble. The regime detector (rule-based) chooses which
    strategy to favor based on measurable signals (success rate, diversity, etc.).
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
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
        # === ENSEMBLE STATE ===
        # Initialize state for all four adaptation strategies
        # variant_06 state (temporal autocorrelation, rate-of-change, drift)
        self.success_history_06 = []
        self.F_ewma_06 = self.F
        self.Cr_ewma_06 = self.Cr
        self.stagnation_window = []
        self.last_best = np.inf
        self.ewma_fast = None
        self.ewma_slow = None
        self.stagnation_counter_06 = 0
        
        # variant_07 state (bootstrap confidence estimation)
        self.bootstrap_means = []
        self.bootstrap_ci_width = []
        self.sample_buffer = None
        
        # variant_01 state (population geometry)
        self.pop_geometry_history = {
            'pairwise_means': [], 'pairwise_stds': [],
            'axis_spreads': [], 'aspect_ratios': [], 'centroid_dists': []
        }
        
        # original state (EMA dynamics)
        self.success_ewma_short = None
        self.success_ewma_long = None
        self.F_history = [self.F]
        self.Cr_history = [self.Cr]
        self.stagnation_counter_orig = 0
        
        # === REGIME DETECTION STATE ===
        self.regime_history = []
        self.variant_performance = np.ones(4)  # Per-variant performance tracking
        self.variant_weights_history = []
        self.ensemble_rewards = []  # Rank-based rewards for each variant
        
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
        """Greedy selection with spectral diversity modulation."""
        improved_mask = trial_fitness < fitness
        improvement = fitness - trial_fitness

        centered = population - np.mean(population, axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]

        if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
            condition = eigvals[0] / eigvals[-1]
        else:
            condition = 1.0

        diversity_bonus = np.log1p(condition)
        fit_range = np.ptp(fitness)
        fit_range = max(fit_range, 1e-10)
        scaled_improvement = improvement / fit_range

        accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
        accept_mask = improved_mask | (scaled_improvement > accept_threshold)

        new_population = population.copy()
        new_fitness = fitness.copy()

        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]

        return new_population, new_fitness, accept_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _detect_regime(self, success_rate, diversity, population):
        """Detect current search regime using measurable signals."""
        # Initialize EMA state if needed
        if self.success_ewma_short is None:
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
        
        # Update EMAs
        self.success_ewma_short = 0.3 * success_rate + 0.7 * self.success_ewma_short
        self.success_ewma_long = 0.1 * success_rate + 0.9 * self.success_ewma_long
        
        # Compute autocorrelation if we have enough history
        autocorr = 0.0
        if len(self.success_history_06) >= 4:
            series = np.array(self.success_history_06)
            var_series = np.var(series)
            if var_series > 1e-12:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr = np.clip(autocorr, -1.0, 1.0)
        
        # Stagnation detection
        current_best = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.inf
        improved_this_gen = current_best < self.last_best
        self.last_best = current_best
        
        self.stagnation_window.append(1 if improved_this_gen else 0)
        if len(self.stagnation_window) > 12:
            self.stagnation_window.pop(0)
        
        stagnation_ratio = np.mean(self.stagnation_window)
        
        # Compute rate of change
        rate_of_change = self.success_ewma_short - self.success_ewma_long
        
        # Regime classification
        is_oscillating = autocorr > 0.4 and abs(rate_of_change) < 0.05
        is_declining = rate_of_change < -0.02
        is_stagnant = stagnation_ratio < 0.2
        is_converging = rate_of_change > 0.02 and stagnation_ratio > 0.5
        
        # Compute diversity metrics for geometry detection
        pop_min = np.min(population, axis=0)
        pop_max = np.max(population, axis=0)
        axis_spreads = pop_max - pop_min
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads) + 1e-10
        aspect_ratio = max_spread / min_spread
        is_elongated = aspect_ratio > 5.0
        
        # Bootstrap uncertainty (variant_07 signal)
        if self.sample_buffer is not None:
            n = len(self.sample_buffer)
            if n >= 5:
                boot_rates = []
                for _ in range(30):
                    indices = np.random.randint(0, n, size=n)
                    boot_rates.append(np.mean(self.sample_buffer[indices]))
                ci_width = np.percentile(boot_rates, 90) - np.percentile(boot_rates, 10)
            else:
                ci_width = 0.2
        else:
            ci_width = 0.2
        
        high_uncertainty = ci_width > 0.25
        
        # Determine regime
        if is_oscillating or (is_stagnant and stagnation_ratio < 0.1):
            current_regime = 'stagnant'
        elif is_converging and not is_elongated and not high_uncertainty:
            current_regime = 'converging'
        elif high_uncertainty or (is_declining and is_stagnant):
            current_regime = 'high_uncertainty'
        elif is_elongated:
            current_regime = 'elongated'
        else:
            current_regime = 'neutral'
        
        self.regime_history.append(current_regime)
        if len(self.regime_history) > 30:
            self.regime_history.pop(0)
        
        return current_regime, stagnation_ratio
    
    def _compute_variant_06_adjustment(self, success_rate, improved_mask):
        """variant_06: temporal autocorrelation, rate-of-change, and drift."""
        self.success_history_06.append(success_rate)
        if len(self.success_history_06) > 20:
            self.success_history_06.pop(0)
        
        # EMA for rate-of-change
        if self.ewma_fast is None:
            self.ewma_fast = success_rate
            self.ewma_slow = success_rate
        self.ewma_fast = 0.4 * success_rate + 0.6 * self.ewma_fast
        self.ewma_slow = 0.15 * success_rate + 0.85 * self.ewma_slow
        rate_of_change = self.ewma_fast - self.ewma_slow
        
        # Autocorrelation
        n_hist = len(self.success_history_06)
        autocorr = 0.0
        if n_hist >= 4:
            series = np.array(self.success_history_06)
            var_series = np.var(series)
            if var_series > 1e-12:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr = np.clip(autocorr, -1.0, 1.0)
        
        # Parameter drift
        self.F_ewma_06 = 0.7 * self.F_ewma_06 + 0.3 * self.F
        self.Cr_ewma_06 = 0.7 * self.Cr_ewma_06 + 0.3 * self.Cr
        F_drift_ratio = self.F / (self.F_ewma_06 + 1e-12)
        Cr_drift_ratio = self.Cr / (self.Cr_ewma_06 + 1e-12)
        
        stagnation_ratio = np.mean(self.stagnation_window) if len(self.stagnation_window) > 0 else 0.5
        
        # Regime detection
        is_oscillating = autocorr > 0.4 and abs(rate_of_change) < 0.05
        is_declining = rate_of_change < -0.02
        is_stagnant = stagnation_ratio < 0.2
        is_converging = rate_of_change > 0.02 and stagnation_ratio > 0.5
        
        if is_oscillating:
            F_adj = 1.25 * np.clip(F_drift_ratio, 0.8, 1.3)
            Cr_adj = 0.85
            self.stagnation_counter_06 += 1
        elif is_declining and is_stagnant:
            F_adj = 1.15 * (1.1 - stagnation_ratio) / np.clip(F_drift_ratio, 0.8, 1.4)
            Cr_adj = 0.88
            self.stagnation_counter_06 += 1
        elif is_stagnant and stagnation_ratio < 0.1:
            F_adj = 0.7
            Cr_adj = 1.2
            self.stagnation_counter_06 += 2
        elif is_converging:
            F_adj = 0.95 * np.clip(F_drift_ratio, 0.9, 1.1)
            Cr_adj = 1.05
            self.stagnation_counter_06 = max(0, self.stagnation_counter_06 - 1)
        else:
            F_adj = 1.0 + 0.1 * rate_of_change
            Cr_adj = 1.0 - 0.05 * rate_of_change
        
        self.stagnation_counter_06 = max(0, self.stagnation_counter_06 - 0.1)
        return F_adj, Cr_adj
    
    def _compute_variant_07_adjustment(self, success_rate, improved_mask):
        """variant_07: bootstrap confidence estimation of success rate."""
        n = len(improved_mask)
        n_success = int(success_rate * n)
        
        if self.sample_buffer is None:
            self.sample_buffer = improved_mask.copy().astype(float)
        
        n_bootstrap = max(50, 3 * n)
        bootstrap_success_rates = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            indices = np.random.randint(0, n, size=n)
            bootstrap_success_rates[b] = np.mean(self.sample_buffer[indices])
        
        bootstrap_mean = np.mean(bootstrap_success_rates)
        bootstrap_std = np.std(bootstrap_success_rates)
        ci_lower = np.percentile(bootstrap_success_rates, 5)
        ci_upper = np.percentile(bootstrap_success_rates, 95)
        ci_width = ci_upper - ci_lower
        
        self.bootstrap_means.append(bootstrap_mean)
        self.bootstrap_ci_width.append(ci_width)
        if len(self.bootstrap_means) > 20:
            self.bootstrap_means.pop(0)
            self.bootstrap_ci_width.pop(0)
        
        if len(self.bootstrap_means) >= 5:
            recent_uncertainty = np.mean(self.bootstrap_ci_width[-5:])
            long_term_uncertainty = np.mean(self.bootstrap_ci_width)
            uncertainty_trend = recent_uncertainty / (long_term_uncertainty + 1e-10)
        else:
            uncertainty_trend = 1.0
        
        robust_success = bootstrap_mean
        exploration_noise_scale = np.clip(ci_width * 2.0, 0.05, 0.4)
        
        high_success = robust_success > 0.3
        low_success = robust_success < 0.15
        high_uncertainty = ci_width > 0.25
        rising_uncertainty = uncertainty_trend > 1.2
        
        if high_success and not high_uncertainty:
            F_adj = 1.08
            Cr_adj = 1.05
            F_noise = np.random.normal(0, 0.02)
            Cr_noise = np.random.normal(0, 0.02)
        elif low_success or high_uncertainty or rising_uncertainty:
            F_adj = 0.82 - 0.1 * bootstrap_std
            Cr_adj = 0.88
            F_noise = np.random.normal(0, exploration_noise_scale)
            Cr_noise = np.random.normal(0, exploration_noise_scale * 0.5)
        else:
            delta = robust_success - 0.2
            F_adj = 1.0 + 0.1 * delta
            Cr_adj = 1.0 - 0.05 * delta
            F_noise = np.random.normal(0, exploration_noise_scale * 0.5)
            Cr_noise = np.random.normal(0, exploration_noise_scale * 0.3)
        
        self.sample_buffer = improved_mask.copy().astype(float)
        return F_adj + F_noise, Cr_adj + Cr_noise
    
    def _compute_variant_01_adjustment(self, success_rate, improved_mask):
        """variant_01: population geometry and spatial structure."""
        pop = getattr(self, '_current_population', None)
        if pop is None:
            return 1.0, 1.0
        
        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        axis_spreads = pop_max - pop_min
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads) + 1e-10
        aspect_ratio = max_spread / min_spread
        
        n_geo = min(20, len(pop))
        indices = np.random.choice(len(pop), n_geo, replace=False)
        subpop = pop[indices]
        diffs = subpop[:, np.newaxis, :] - subpop[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        triu_indices = np.triu_indices(n_geo, k=1)
        pairwise_mean = np.mean(pairwise_dists[triu_indices])
        
        centroid = np.mean(pop, axis=0)
        centroid_dists = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = np.mean(centroid_dists)
        
        history = self.pop_geometry_history
        history['pairwise_means'].append(pairwise_mean)
        history['axis_spreads'].append(max_spread)
        history['aspect_ratios'].append(aspect_ratio)
        history['centroid_dists'].append(mean_centroid_dist)
        
        for key in history:
            if len(history[key]) > 10:
                history[key].pop(0)
        
        hist_pm = np.mean(history['pairwise_means'])
        hist_as = np.mean(history['axis_spreads'])
        
        diversity_ratio = pairwise_mean / (hist_pm + 1e-10)
        spread_ratio = max_spread / (hist_as + 1e-10)
        
        is_expanded = diversity_ratio > 1.1 or spread_ratio > 1.15
        is_contracted = diversity_ratio < 0.9 or spread_ratio < 0.85
        is_highly_elongated = aspect_ratio > 5.0
        is_near_spherical = aspect_ratio < 2.0
        
        if is_expanded:
            F_adj = 0.88
        elif is_contracted:
            F_adj = 1.12
        else:
            delta = diversity_ratio - 1.0
            F_adj = 1.0 - 0.15 * delta
        
        if success_rate > 0.3:
            F_adj = F_adj * 0.95 + 1.0 * 0.05
        elif success_rate < 0.1:
            F_adj *= 1.08
        
        if is_highly_elongated:
            Cr_adj = 1.10
        elif is_near_spherical:
            Cr_adj = 0.92
        else:
            aspect_deviation = (aspect_ratio - 2.0) / 8.0
            aspect_deviation = np.clip(aspect_deviation, -0.2, 0.2)
            Cr_adj = 1.0 + 0.15 * aspect_deviation
        
        if success_rate > 0.35:
            Cr_adj *= 0.97
        elif success_rate < 0.15:
            Cr_adj *= 1.05
        
        return F_adj, Cr_adj
    
    def _compute_original_adjustment(self, success_rate):
        """original: EMA dynamics and parameter drift."""
        if self.success_ewma_short is None:
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
        
        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long
        
        self.F_history.append(self.F)
        self.Cr_history.append(self.Cr)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)
        
        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
        
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        
        if short_trending_up and self.success_ewma_short > 0.25:
            F_adj = 1.05 * momentum_factor
            Cr_adj = 1.08
            self.stagnation_counter_orig = max(0, self.stagnation_counter_orig - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            F_adj = 0.88 / momentum_factor
            Cr_adj = 0.92
            self.stagnation_counter_orig += 1
        elif both_stagnant:
            F_adj = 0.75
            Cr_adj = 1.15
            self.stagnation_counter_orig += 2
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adj = 1.0 + 0.08 * delta
            Cr_adj = 1.0 - 0.05 * delta
        
        return F_adj, Cr_adj
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Ensemble adaptation with regime-conditional weighting."""
        success_rate = np.mean(improved_mask)
        pop = getattr(self, '_current_population', None)
        diversity = self._compute_diversity(pop) if pop is not None else 1.0
        
        # Detect regime
        current_regime, stagnation_ratio = self._detect_regime(success_rate, diversity, pop)
        
        # Compute adjustments from all four strategies
        F06, Cr06 = self._compute_variant_06_adjustment(success_rate, improved_mask)
        F07, Cr07 = self._compute_variant_07_adjustment(success_rate, improved_mask)
        F01, Cr01 = self._compute_variant_01_adjustment(success_rate, improved_mask)
        F_orig, Cr_orig = self._compute_original_adjustment(success_rate)
        
        # Regime-conditional weights (based on benchmark win patterns)
        # variant_06 dominates overall → favor it when converging
        # variant_07 excels in uncertain/stagnant regimes
        # variant_01 handles geometric challenges
        # original is a reliable fallback
        regime_weights = {
            'converging': np.array([0.50, 0.20, 0.15, 0.15]),      # variant_06 dominant
            'high_uncertainty': np.array([0.20, 0.50, 0.15, 0.15]), # variant_07 dominant
            'elongated': np.array([0.20, 0.15, 0.50, 0.15]),       # variant_01 dominant
            'stagnant': np.array([0.30, 0.35, 0.15, 0.20]),        # variant_07 + 06
            'neutral': np.array([0.35, 0.25, 0.20, 0.20]),         # Blend all
        }
        
        regime_based_weights = regime_weights.get(current_regime, regime_weights['neutral'])
        
        # Performance-based adjustment (rank-based, no scale dependency)
        regime_to_variant = {
            'converging': 0, 'high_uncertainty': 1, 'elongated': 2,
            'stagnant': 1, 'neutral': 0, 'exploring': 3
        }
        aligned_variant = regime_to_variant.get(current_regime, 3)
        
        # Rank-based reward: increment aligned variant on success, decrement on failure
        reward_signal = 1.0 if success_rate > 0.2 else -0.5
        self.variant_performance[aligned_variant] += reward_signal
        self.variant_performance = np.clip(self.variant_performance * 0.995, 0.1, None)
        
        # Normalize performance to weights
        perf_weights = self.variant_performance / np.sum(self.variant_performance)
        
        # Final weights: blend regime-based and performance-based
        final_weights = 0.7 * regime_based_weights + 0.3 * perf_weights
        
        # Stability smoothing: prevent oscillation if regime is stable
        if len(self.variant_weights_history) >= 3:
            recent_weights = np.array(self.variant_weights_history[-2:])
            if np.allclose(recent_weights[0], recent_weights[1], atol=0.1):
                final_weights = 0.8 * final_weights + 0.2 * np.mean(recent_weights, axis=0)
        
        self.variant_weights_history.append(final_weights.copy())
        if len(self.variant_weights_history) > 10:
            self.variant_weights_history.pop(0)
        
        # Compute weighted ensemble adjustments
        all_F = np.array([F06, F07, F01, F_orig])
        all_Cr = np.array([Cr06, Cr07, Cr01, Cr_orig])
        
        F_ensemble = np.sum(final_weights * all_F)
        Cr_ensemble = np.sum(final_weights * all_Cr)
        
        # Apply adjustments with clipping
        self.F = np.clip(self.F * F_ensemble, 0.1, 2.0)
        self.Cr = np.clip(self.Cr * Cr_ensemble, 0.05, 0.95)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter_orig += 1
        else:
            self.stagnation_counter_orig = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter_orig > self.stagnation_threshold or 
            diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter_orig = 0
            self.successful_directions = []
            
            # Reset ensemble state on restart
            self.success_history_06 = []
            self.bootstrap_means = []
            self.bootstrap_ci_width = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter_orig = 0
        self.generation = 0
        self._current_population = population
        self.population_fitness = fitness
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            mutant_batch, F_used = self._mutate_batch(population, fitness, ring_prev, ring_next)
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update state for adaptation strategies
            self._current_population = population
            self.population_fitness = fitness
            
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
