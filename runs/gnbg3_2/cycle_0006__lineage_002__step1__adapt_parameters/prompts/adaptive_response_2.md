```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    META-CONTROLLED ENSEMBLE ADAPTATION (mechanism #9 + #2 hybrid):
    Combines three complementary adaptation strategies via a meta-controller
    that learns optimal blending weights based on observed regime and performance.
    
    Why this fits the benchmark:
    - variant_06 (temporal) dominates on hard/late tasks (11 wins)
    - original (EMA-based) excels on easy/early tasks (6 wins)
    - variant_07 (bootstrap) wins on stagnation-prone tasks (6 wins)
    - The meta-controller learns which blend is best for each regime,
      avoiding the need to hard-code regime boundaries.
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
        
        # Meta-controller state for regime-aware ensemble adaptation
        self._init_meta_controller()
    
    def _init_meta_controller(self):
        """Initialize state for the meta-controlled ensemble adaptation."""
        # Regime detection state
        self.regime = "neutral"
        self.stagnation_counter = 0
        self.improvement_history = []
        self.diversity_history = []
        self.initial_fitness_gap = None
        
        # Ensemble weights (temporal, bootstrap, geometry, original)
        # Prior: original and variant_06 are strong, variant_07 moderate
        self.variant_weights = np.array([0.30, 0.20, 0.15, 0.35])
        
        # Per-variant performance tracking for weight adaptation
        self.variant_performance = {
            'temporal': [],
            'bootstrap': [],
            'geometry': [],
            'original': []
        }
        self.variant_rewards_window = 15  # Must be >= 3 * num_variants
        
        # Per-variant adaptation state (initialized lazily)
        self._variant_state = {}
    
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
    
    def _detect_regime(self, population, fitness, best_fitness, success_rate):
        """Detect current optimization regime from multiple signals."""
        # Progress ratio: how much improvement has been made
        if self.initial_fitness_gap is None:
            worst_fitness = np.max(fitness)
            self.initial_fitness_gap = max(worst_fitness - best_fitness, 1e-10)
        
        current_gap = np.max(fitness) - best_fitness
        progress_ratio = 1.0 - np.clip(current_gap / self.initial_fitness_gap, 0.0, 1.0)
        
        # Diversity ratio: current vs historical diversity
        current_diversity = self._compute_diversity(population)
        self.diversity_history.append(current_diversity)
        if len(self.diversity_history) > 20:
            self.diversity_history.pop(0)
        
        avg_diversity = np.mean(self.diversity_history) if self.diversity_history else current_diversity
        diversity_ratio = current_diversity / (avg_diversity + 1e-10)
        
        # Improvement tracking
        self.improvement_history.append(success_rate)
        if len(self.improvement_history) > 20:
            self.improvement_history.pop(0)
        
        recent_improvement = np.mean(self.improvement_history[-5:]) if len(self.improvement_history) >= 5 else success_rate
        longterm_improvement = np.mean(self.improvement_history) if self.improvement_history else success_rate
        
        # Regime classification
        is_stagnant = recent_improvement < 0.05 and self.stagnation_counter > 10
        is_converging = progress_ratio > 0.5 and recent_improvement > 0.15
        is_exploring = diversity_ratio > 1.1 or progress_ratio < 0.2
        
        if is_stagnant:
            regime = "stagnation"
        elif is_converging:
            regime = "converging"
        elif is_exploring:
            regime = "exploring"
        else:
            regime = "neutral"
        
        return regime, progress_ratio, diversity_ratio
    
    def _update_variant_weights(self, regime, F_adjustments, Cr_adjustments):
        """Update ensemble weights based on regime and observed performance."""
        # Compute reward signals (rank-based within sliding window)
        all_adjustments = list(zip(
            F_adjustments['temporal'], F_adjustments['bootstrap'], 
            F_adjustments['geometry'], F_adjustments['original']
        ))
        
        # Rank-based reward: lower |adjustment| indicates stability (good)
        # Convert to reward where moderate adjustments are best
        rewards = {}
        for name, adj in [('temporal', F_adjustments['temporal']),
                          ('bootstrap', F_adjustments['bootstrap']),
                          ('geometry', F_adjustments['geometry']),
                          ('original', F_adjustments['original'])]:
            # Reward: closer to 1.0 is better (stable adaptation)
            stability = 1.0 - min(abs(adj - 1.0), 0.5)
            rewards[name] = stability
        
        # Store rewards in sliding window
        for name, reward in rewards.items():
            self.variant_rewards_window[name].append(reward)
            if len(self.variant_rewards_window[name]) > self.variant_rewards_window:
                self.variant_rewards_window[name].pop(0)
        
        # Compute rank-based scores (z-score normalization within window)
        scores = {}
        for name in rewards:
            window = self.variant_rewards_window[name]
            if len(window) >= 3:
                mean_reward = np.mean(window)
                std_reward = np.std(window) + 1e-10
                scores[name] = (rewards[name] - mean_reward) / std_reward
            else:
                scores[name] = 0.0
        
        # Regime-based prior adjustments
        regime_prior = {'temporal': 0.0, 'bootstrap': 0.0, 'geometry': 0.0, 'original': 0.0}
        if regime == "stagnation":
            regime_prior['bootstrap'] = 0.15  # Bootstrap handles uncertainty well
            regime_prior['geometry'] = 0.10
        elif regime == "converging":
            regime_prior['temporal'] = 0.15   # Temporal excels in late stages
            regime_prior['original'] = 0.10
        elif regime == "exploring":
            regime_prior['original'] = 0.15   # Original is robust for exploration
            regime_prior['geometry'] = 0.10
        
        # Combine scores with regime prior and update weights
        combined_scores = {name: scores.get(name, 0.0) + regime_prior[name] 
                          for name in ['temporal', 'bootstrap', 'geometry', 'original']}
        
        # Softmax to get new weights
        score_array = np.array([combined_scores['temporal'], combined_scores['bootstrap'],
                               combined_scores['geometry'], combined_scores['original']])
        exp_scores = np.exp(score_array - np.max(score_array))  # Numerically stable
        new_weights = exp_scores / np.sum(exp_scores)
        
        # Blend with previous weights (conservative learning rate)
        learning_rate = 0.15
        self.variant_weights = (1 - learning_rate) * self.variant_weights + learning_rate * new_weights
        
        # Ensure valid weights
        self.variant_weights = np.clip(self.variant_weights, 0.05, 0.60)
        self.variant_weights = self.variant_weights / np.sum(self.variant_weights)
        
        return dict(zip(['temporal', 'bootstrap', 'geometry', 'original'], self.variant_weights))
    
    def _adapt_temporal_variant(self, improved_mask, F_used, Cr_used):
        """Variant 06-inspired: temporal autocorrelation and rate-of-change."""
        n = len(improved_mask)
        success_rate = np.mean(improved_mask) if n > 0 else 0.0
        
        state = self._variant_state.setdefault('temporal', {
            'ewma_fast': success_rate, 'ewma_slow': success_rate,
            'F_ewma': self.F, 'Cr_ewma': self.Cr,
            'stagnation_window': [], 'last_best': np.inf,
            'success_history': [], 'stagnation_counter': 0
        })
        
        # Update EMAs
        state['ewma_fast'] = 0.4 * success_rate + 0.6 * state['ewma_fast']
        state['ewma_slow'] = 0.15 * success_rate + 0.85 * state['ewma_slow']
        rate_of_change = state['ewma_fast'] - state['ewma_slow']
        
        # Parameter drift
        state['F_ewma'] = 0.7 * state['F_ewma'] + 0.3 * F_used
        state['Cr_ewma'] = 0.7 * state['Cr_ewma'] + 0.3 * Cr_used
        F_drift_ratio = self.F / (state['F_ewma'] + 1e-12)
        Cr_drift_ratio = self.Cr / (state['Cr_ewma'] + 1e-12)
        
        # Stagnation tracking
        current_best = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.inf
        improved_this_gen = current_best < state['last_best']
        state['last_best'] = current_best
        
        state['stagnation_window'].append(1 if improved_this_gen else 0)
        if len(state['stagnation_window']) > 12:
            state['stagnation_window'].pop(0)
        
        stagnation_ratio = np.mean(state['stagnation_window'])
        is_oscillating = abs(rate_of_change) < 0.05 and stagnation_ratio > 0.3
        is_stagnant = stagnation_ratio < 0.2
        is_converging = rate_of_change > 0.02 and stagnation_ratio > 0.5
        
        # Compute adjustments
        if is_oscillating:
            F_adj = 1.25 * np.clip(F_drift_ratio, 0.8, 1.3)
            Cr_adj = 0.85
        elif is_stagnant:
            F_adj = 0.7 if stagnation_ratio < 0.1 else 1.15 * (1.1 - stagnation_ratio) / np.clip(F_drift_ratio, 0.8, 1.4)
            Cr_adj = 1.2 if stagnation_ratio < 0.1 else 0.88
        elif is_converging:
            F_adj = 0.95 * np.clip(F_drift_ratio, 0.9, 1.1)
            Cr_adj = 1.05
        else:
            F_adj = 1.0 + 0.1 * rate_of_change
            Cr_adj = 1.0 - 0.05 * rate_of_change
        
        return F_adj, Cr_adj
    
    def _adapt_bootstrap_variant(self, improved_mask, F_used, Cr_used):
        """Variant 07-inspired: bootstrap confidence estimation."""
        n = len(improved_mask)
        if n == 0:
            return 1.0, 1.0
        
        success_rate = np.mean(improved_mask)
        
        state = self._variant_state.setdefault('bootstrap', {
            'bootstrap_means': [], 'bootstrap_ci_width': [],
            'sample_buffer': improved_mask.copy().astype(float)
        })
        
        # Bootstrap resampling
        sample_buffer = state['sample_buffer']
        n_bootstrap = max(50, 3 * n)
        bootstrap_success_rates = np.zeros(n_bootstrap)
        
        for b in range(n_bootstrap):
            indices = np.random.randint(0, n, size=n)
            bootstrap_success_rates[b] = np.mean(sample_buffer[indices])
        
        bootstrap_mean = np.mean(bootstrap_success_rates)
        bootstrap_std = np.std(bootstrap_success_rates)
        ci_lower, ci_upper = np.percentile(bootstrap_success_rates, [5, 95])
        ci_width = ci_upper - ci_lower
        
        # Store history
        state['bootstrap_means'].append(bootstrap_mean)
        state['bootstrap_ci_width'].append(ci_width)
        if len(state['bootstrap_means']) > 20:
            state['bootstrap_means'].pop(0)
            state['bootstrap_ci_width'].pop(0)
        
        # Uncertainty trend
        if len(state['bootstrap_means']) >= 5:
            recent_uncertainty = np.mean(state['bootstrap_ci_width'][-5:])
            long_term_uncertainty = np.mean(state['bootstrap_ci_width'])
            uncertainty_trend = recent_uncertainty / (long_term_uncertainty + 1e-10)
        else:
            uncertainty_trend = 1.0
        
        # Regime detection
        high_success = bootstrap_mean > 0.3
        low_success = bootstrap_mean < 0.15
        high_uncertainty = ci_width > 0.25
        rising_uncertainty = uncertainty_trend > 1.2
        
        # Compute adjustments with stochastic perturbation
        exploration_noise_scale = np.clip(ci_width * 2.0, 0.05, 0.4)
        
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
            delta = bootstrap_mean - 0.2
            F_adj = 1.0 + 0.1 * delta
            Cr_adj = 1.0 - 0.05 * delta
            F_noise = np.random.normal(0, exploration_noise_scale * 0.5)
            Cr_noise = np.random.normal(0, exploration_noise_scale * 0.3)
        
        # Update sample buffer
        state['sample_buffer'] = improved_mask.copy().astype(float)
        
        return F_adj + F_noise, Cr_adj + Cr_noise
    
    def _adapt_geometry_variant(self, improved_mask, F_used, Cr_used):
        """Variant 01-inspired: population geometry-based adaptation."""
        n = len(improved_mask)
        if n == 0:
            return 1.0, 1.0
        
        success_rate = np.mean(improved_mask)
        pop = getattr(self, '_current_population', None)
        
        if pop is None or len(pop) < 4:
            return 1.0, 1.0
        
        state = self._variant_state.setdefault('geometry', {'history': None})
        
        if state['history'] is None:
            state['history'] = {k: [] for k in ['pairwise_means', 'pairwise_stds', 
                                                   'axis_spreads', 'aspect_ratios', 
                                                   'centroid_dists']}
        
        # Compute geometric features
        pop_min, pop_max = np.min(pop, axis=0), np.max(pop, axis=0)
        axis_spreads = pop_max - pop_min
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads) + 1e-10
        aspect_ratio = max_spread / min_spread
        
        # Pairwise distances (subsample)
        n_geo = min(20, len(pop))
        indices = np.random.choice(len(pop), n_geo, replace=False)
        subpop = pop[indices]
        diffs = subpop[:, np.newaxis, :] - subpop[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        triu_indices = np.triu_indices(n_geo, k=1)
        pairwise_mean = np.mean(pairwise_dists[triu_indices])
        pairwise_std = np.std(pairwise_dists[triu_indices])
        
        centroid = np.mean(pop, axis=0)
        centroid_dists = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = np.mean(centroid_dists)
        
        # Store history
        history = state['history']
        history['pairwise_means'].append(pairwise_mean)
        history['pairwise_stds'].append(pairwise_std)
        history['axis_spreads'].append(max_spread)
        history['aspect_ratios'].append(aspect_ratio)
        history['centroid_dists'].append(mean_centroid_dist)
        
        for key in history:
            if len(history[key]) > 10:
                history[key].pop(0)
        
        # Compute ratios
        hist_pm = np.mean(history['pairwise_means']) if history['pairwise_means'] else pairwise_mean
        hist_as = np.mean(history['axis_spreads']) if history['axis_spreads'] else max_spread
        
        diversity_ratio = pairwise_mean / (hist_pm + 1e-10)
        spread_ratio = max_spread / (hist_as + 1e-10)
        
        # Regime detection
        is_expanded = diversity_ratio > 1.1 or spread_ratio > 1.15
        is_contracted = diversity_ratio < 0.9 or spread_ratio < 0.85
        is_highly_elongated = aspect_ratio > 5.0
        is_near_spherical = aspect_ratio < 2.0
        
        # F adjustment
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
        
        # Cr adjustment
        if is_highly_elongated:
            Cr_adj = 1.10
        elif is_near_spherical:
            Cr_adj = 0.92
        else:
            aspect_deviation = np.clip((aspect_ratio - 2.0) / 8.0, -0.2, 0.2)
            Cr_adj = 1.0 + 0.15 * aspect_deviation
        
        if success_rate > 0.35:
            Cr_adj *= 0.97
        elif success_rate < 0.15:
            Cr_adj *= 1.05
        
        return F_adj, Cr_adj
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Meta-controlled ensemble adaptation combining three strategies."""
        n = len(improved_mask)
        
        # Safety check for empty batch
        if n == 0:
            self.F = np.clip(self.F * 1.0, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * 1.0, 0.1, 0.9)
            return
        
        success_rate = np.mean(improved_mask)
        pop = getattr(self, '_current_population', None)
        
        # Detect current regime
        if pop is not None and hasattr(self, 'population_fitness'):
            fitness = self.population_fitness
            best_fitness = np.min(fitness)
            regime, progress_ratio, diversity_ratio = self._detect_regime(pop, fitness, best_fitness, success_rate)
        else:
            regime = "neutral"
            progress_ratio = 0.0
            diversity_ratio = 1.0
        
        # Update stagnation counter
        if success_rate < 0.05:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        # Get adjustments from all variants
        F_temporal, Cr_temporal = self._adapt_temporal_variant(improved_mask, F_used, Cr_used)
        F_bootstrap, Cr_bootstrap = self._adapt_bootstrap_variant(improved_mask, F_used, Cr_used)
        F_geometry, Cr_geometry = self._adapt_geometry_variant(improved_mask, F_used, Cr_used)
        
        # Original method (EMA-based)
        state_orig = self._variant_state.setdefault('original', {
            'ewma_short': success_rate, 'ewma_long': success_rate,
            'F_history': [self.F], 'Cr_history': [self.Cr]
        })
        
        alpha_short, alpha_long = 0.3, 0.1
        state_orig['ewma_short'] = alpha_short * success_rate + (1 - alpha_short) * state_orig['ewma_short']
        state_orig['ewma_long'] = alpha_long * success_rate + (1 - alpha_long) * state_orig['ewma_long']
        
        state_orig['F_history'].append(F_used)
        state_orig['Cr_history'].append(Cr_used)
        if len(state_orig['F_history']) > 15:
            state_orig['F_history'].pop(0)
            state_orig['Cr_history'].pop(0)
        
        F_mean = np.mean(state_orig['F_history'])
        Cr_mean = np.mean(state_orig['Cr_history'])
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)
        
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        short_up = state_orig['ewma_short'] > state_orig['ewma_long']
        short_down = state_orig['ewma_short'] < state_orig['ewma_long']
        both_stagnant = state_orig['ewma_short'] < 0.1 and state_orig['ewma_long'] < 0.15
        
        if short_up and state_orig['ewma_short'] > 0.25:
            F_original = 1.05 * momentum_factor
            Cr_original = 1.08
        elif short_down and state_orig['ewma_short'] < 0.2:
            F_original = 0.88 / momentum_factor
            Cr_original = 0.92
        elif both_stagnant:
            F_original = 0.75
            Cr_original = 1.15
        else:
            delta = state_orig['ewma_short'] - state_orig['ewma_long']
            F_original = 1.0 + 0.08 * delta
            Cr_original = 1.0 - 0.05 * delta
        
        # Collect all adjustments
        F_adjustments = {
            'temporal': F_temporal, 'bootstrap': F_bootstrap,
            'geometry': F_geometry, 'original': F_original
        }
        Cr_adjustments = {
            'temporal': Cr_temporal, 'bootstrap': Cr_bootstrap,
            'geometry': Cr_geometry, 'original': Cr_original
        }
        
        # Update variant weights based on regime
        weights = self._update_variant_weights(regime, F_adjustments, Cr_adjustments)
        
        # Ensemble combination
        F_combined = (weights['temporal'] * F_temporal + 
                     weights['bootstrap'] * F_bootstrap +
                     weights['geometry'] * F_geometry + 
                     weights['original'] * F_original)
        
        Cr_combined = (weights['temporal'] * Cr_temporal + 
                      weights['bootstrap'] * Cr_bootstrap +
                      weights['geometry'] * Cr_geometry + 
                      weights['original'] * Cr_original)
        
        # Apply with safety bounds
        self.F = np.clip(self.F * F_combined, 0.05, 2.0)
        self.Cr = np.clip(self.Cr * Cr_combined, 0.05, 0.95)
        
        # Safety: reset if numerical issues
        if np.isnan(self.F) or self.F < 0.01:
            self.F = self.F_base
        if np.isnan(self.Cr) or self.Cr < 0.01:
            self.Cr = self.Cr_base
    
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
                population[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
            
            # Reset variant states on restart
            self._variant_state = {}
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        self.population_fitness = fitness
        
        # Initialize meta-controller with initial fitness gap
        worst = np.max(fitness)
        self.initial_fitness_gap = max(worst - best_fitness, 1e-10)
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Store current population for geometry-based adaptation
            self._current_population = population
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(population, fitness, ring_prev, ring_next)
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            self.population_fitness = fitness
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Meta-controlled ensemble adaptation
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
```