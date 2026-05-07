```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    ADAPTIVE ENSEMBLE OF PARAMETER ADAPTATION STRATEGIES
    
    Mechanism #2 (Ensemble/Voting): Combines 4 correlated but complementary
    adaptation methods via softmax-weighted blending. The benchmark shows
    no single method dominates (variant_06=11 wins, original=6, variant_07=6,
    variant_01=1), and all methods compute similar quantities (F/Cr adjustments)
    from different signals. Ensemble blending is more robust than discrete
    selection when arms are correlated and credit assignment is noisy.
    
    The 4 ensemble members:
    - Strategy 0 (original): EMA dynamics with trend detection
    - Strategy 1 (variant_01): Population geometry features
    - Strategy 2 (variant_06): Temporal autocorrelation + drift
    - Strategy 3 (variant_07): Bootstrap confidence estimation
    
    All methods compute: F_adjustment, Cr_adjustment
    Ensemble output: weighted average of adjustments, weights = softmax(recent_success)
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
        
        # ============================================================
        # ENSEMBLE STATE: per-strategy tracking for blending
        # ============================================================
        self.num_strategies = 4
        self.strategy_successes = [[] for _ in range(self.num_strategies)]
        self.strategy_window = 15  # K >= 3 * num_strategies per calibration rule (b)
        self.strategy_F_adjustments = [1.0] * self.num_strategies
        self.strategy_Cr_adjustments = [1.0] * self.num_strategies
        
        # Per-strategy internal state (initialized lazily)
        self._strategy_states = [{} for _ in range(self.num_strategies)]
        
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
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    # =================================================================
    # ENSEMBLE STRATEGY IMPLEMENTATIONS
    # Each computes its own F_adjustment and Cr_adjustment
    # All inputs are rank-free; outputs are multiplicative factors
    # =================================================================
    
    def _strategy_original(self, improved_mask, F_used, Cr_used, state):
        """Strategy 0: EMA dynamics with trend detection (original adaptation)."""
        success_rate = np.mean(improved_mask)
        
        if 'success_ewma_short' not in state:
            state['success_ewma_short'] = success_rate
            state['success_ewma_long'] = success_rate
            state['F_history'] = [self.F]
            state['Cr_history'] = [self.Cr]
        
        alpha_short, alpha_long = 0.3, 0.1
        state['success_ewma_short'] = alpha_short * success_rate + (1 - alpha_short) * state['success_ewma_short']
        state['success_ewma_long'] = alpha_long * success_rate + (1 - alpha_long) * state['success_ewma_long']
        
        state['F_history'].append(F_used)
        state['Cr_history'].append(Cr_used)
        if len(state['F_history']) > 15:
            state['F_history'].pop(0)
            state['Cr_history'].pop(0)
        
        F_mean = np.mean(state['F_history'])
        Cr_mean = np.mean(state['Cr_history'])
        F_drift = self.F / (F_mean + 1e-10)
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        
        short_trending_up = state['success_ewma_short'] > state['success_ewma_long']
        short_trending_down = state['success_ewma_short'] < state['success_ewma_long']
        both_stagnant = state['success_ewma_short'] < 0.1 and state['success_ewma_long'] < 0.15
        
        if short_trending_up and state['success_ewma_short'] > 0.25:
            F_adj = 1.05 * momentum_factor
            Cr_adj = 1.08
        elif short_trending_down and state['success_ewma_short'] < 0.2:
            F_adj = 0.88 / momentum_factor
            Cr_adj = 0.92
        elif both_stagnant:
            F_adj = 0.75
            Cr_adj = 1.15
        else:
            delta = state['success_ewma_short'] - state['success_ewma_long']
            F_adj = 1.0 + 0.08 * delta
            Cr_adj = 1.0 - 0.05 * delta
        
        return np.clip(F_adj, 0.5, 1.5), np.clip(Cr_adj, 0.8, 1.2), state
    
    def _strategy_geometry(self, improved_mask, F_used, Cr_used, state):
        """Strategy 1: Population geometry features (variant_01)."""
        success_rate = np.mean(improved_mask)
        pop = getattr(self, '_current_population', None)
        
        if pop is None:
            return 1.0, 1.0, state
        
        if 'pairwise_means' not in state:
            state['pairwise_means'] = []
            state['axis_spreads'] = []
            state['aspect_ratios'] = []
        
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
        
        state['pairwise_means'].append(pairwise_mean)
        state['axis_spreads'].append(max_spread)
        state['aspect_ratios'].append(aspect_ratio)
        
        for key in ['pairwise_means', 'axis_spreads', 'aspect_ratios']:
            if len(state[key]) > 10:
                state[key].pop(0)
        
        if len(state['pairwise_means']) >= 2:
            hist_pm = np.mean(state['pairwise_means'][:-1]) if len(state['pairwise_means']) > 1 else pairwise_mean
            hist_as = np.mean(state['axis_spreads'][:-1]) if len(state['axis_spreads']) > 1 else max_spread
            diversity_ratio = pairwise_mean / (hist_pm + 1e-10)
            spread_ratio = max_spread / (hist_as + 1e-10)
        else:
            diversity_ratio = 1.0
            spread_ratio = 1.0
        
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
            aspect_deviation = np.clip((aspect_ratio - 2.0) / 8.0, -0.2, 0.2)
            Cr_adj = 1.0 + 0.15 * aspect_deviation
        
        if success_rate > 0.35:
            Cr_adj *= 0.97
        elif success_rate < 0.15:
            Cr_adj *= 1.05
        
        return np.clip(F_adj, 0.5, 1.5), np.clip(Cr_adj, 0.8, 1.2), state
    
    def _strategy_temporal(self, improved_mask, F_used, Cr_used, state):
        """Strategy 2: Temporal autocorrelation + drift (variant_06)."""
        success_rate = np.mean(improved_mask)
        
        if 'success_history' not in state:
            state['success_history'] = []
            state['F_ewma'] = self.F
            state['Cr_ewma'] = self.Cr
            state['stagnation_window'] = []
            state['last_best'] = np.inf
            state['ewma_fast'] = success_rate
            state['ewma_slow'] = success_rate
        
        state['success_history'].append(success_rate)
        if len(state['success_history']) > 20:
            state['success_history'].pop(0)
        
        current_best = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.inf
        improved_this_gen = current_best < state['last_best']
        state['last_best'] = current_best
        
        state['stagnation_window'].append(1 if improved_this_gen else 0)
        if len(state['stagnation_window']) > 12:
            state['stagnation_window'].pop(0)
        
        alpha_fast, alpha_slow = 0.4, 0.15
        state['ewma_fast'] = alpha_fast * success_rate + (1 - alpha_fast) * state['ewma_fast']
        state['ewma_slow'] = alpha_slow * success_rate + (1 - alpha_slow) * state['ewma_slow']
        rate_of_change = state['ewma_fast'] - state['ewma_slow']
        
        n_hist = len(state['success_history'])
        autocorr = 0.0
        if n_hist >= 4:
            series = np.array(state['success_history'])
            mean_series = np.mean(series)
            var_series = np.var(series)
            if var_series > 1e-12:
                corr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr = np.clip(corr, -1.0, 1.0)
        
        state['F_ewma'] = 0.7 * state['F_ewma'] + 0.3 * F_used
        state['Cr_ewma'] = 0.7 * state['Cr_ewma'] + 0.3 * Cr_used
        F_drift_ratio = self.F / (state['F_ewma'] + 1e-12)
        stagnation_ratio = np.mean(state['stagnation_window'])
        
        is_oscillating = autocorr > 0.4 and abs(rate_of_change) < 0.05
        is_declining = rate_of_change < -0.02
        is_stagnant = stagnation_ratio < 0.2
        
        if is_oscillating:
            F_adj = 1.25 * np.clip(F_drift_ratio, 0.8, 1.3)
            Cr_adj = 0.85
        elif is_declining and is_stagnant:
            F_adj = 1.15 * (1.1 - stagnation_ratio) / np.clip(F_drift_ratio, 0.8, 1.4)
            Cr_adj = 0.88
        elif is_stagnant and stagnation_ratio < 0.1:
            F_adj = 0.7
            Cr_adj = 1.2
        elif rate_of_change > 0.02 and stagnation_ratio > 0.5:
            F_adj = 0.95 * np.clip(F_drift_ratio, 0.9, 1.1)
            Cr_adj = 1.05
        else:
            F_adj = 1.0 + 0.1 * rate_of_change
            Cr_adj = 1.0 - 0.05 * rate_of_change
        
        return np.clip(F_adj, 0.5, 1.5), np.clip(Cr_adj, 0.7, 1.3), state
    
    def _strategy_bootstrap(self, improved_mask, F_used, Cr_used, state):
        """Strategy 3: Bootstrap confidence estimation (variant_07)."""
        n = len(improved_mask)
        success_rate = np.mean(improved_mask)
        
        if 'sample_buffer' not in state:
            state['bootstrap_means'] = []
            state['bootstrap_ci_width'] = []
            state['sample_buffer'] = improved_mask.copy().astype(float)
        
        n_bootstrap = max(50, 3 * n)
        bootstrap_success_rates = np.zeros(n_bootstrap)
        
        for b in range(n_bootstrap):
            indices = np.random.randint(0, n, size=n)
            bootstrap_success_rates[b] = np.mean(state['sample_buffer'][indices])
        
        bootstrap_mean = np.mean(bootstrap_success_rates)
        bootstrap_std = np.std(bootstrap_success_rates)
        ci_lower = np.percentile(bootstrap_success_rates, 5)
        ci_upper = np.percentile(bootstrap_success_rates, 95)
        ci_width = ci_upper - ci_lower
        
        state['bootstrap_means'].append(bootstrap_mean)
        state['bootstrap_ci_width'].append(ci_width)
        if len(state['bootstrap_means']) > 20:
            state['bootstrap_means'].pop(0)
            state['bootstrap_ci_width'].pop(0)
        
        if len(state['bootstrap_means']) >= 5:
            recent_uncertainty = np.mean(state['bootstrap_ci_width'][-5:])
            long_term_uncertainty = np.mean(state['bootstrap_ci_width'])
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
            F_adj = 1.08 + np.random.normal(0, 0.02)
            Cr_adj = 1.05 + np.random.normal(0, 0.02)
        elif low_success or high_uncertainty or rising_uncertainty:
            F_adj = (0.82 - 0.1 * bootstrap_std) + np.random.normal(0, exploration_noise_scale)
            Cr_adj = 0.88 + np.random.normal(0, exploration_noise_scale * 0.5)
        else:
            delta = robust_success - 0.2
            F_adj = (1.0 + 0.1 * delta) + np.random.normal(0, exploration_noise_scale * 0.5)
            Cr_adj = (1.0 - 0.05 * delta) + np.random.normal(0, exploration_noise_scale * 0.3)
        
        state['sample_buffer'] = improved_mask.copy().astype(float)
        
        return np.clip(F_adj, 0.5, 1.5), np.clip(Cr_adj, 0.8, 1.2), state
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """
        ENSEMBLE ADAPTATION: run all 4 strategies, blend their outputs.
        
        Steps:
        1. Each strategy computes its F_adj, Cr_adj and returns updated state
        2. Measure per-strategy success using rank-based reward in sliding window
        3. Compute softmax weights over strategies
        4. Blend adjustments: F *= weighted_avg(F_adjs), Cr *= weighted_avg(Cr_adjs)
        """
        # Store population for geometry-based strategy
        self._current_population = self.population
        
        # Run each strategy and collect adjustments
        F_adjustments = []
        Cr_adjustments = []
        
        # Strategy 0: original EMA
        f_adj, c_adj, self._strategy_states[0] = self._strategy_original(
            improved_mask, F_used, Cr_used, self._strategy_states[0])
        F_adjustments.append(f_adj)
        Cr_adjustments.append(c_adj)
        
        # Strategy 1: geometry
        f_adj, c_adj, self._strategy_states[1] = self._strategy_geometry(
            improved_mask, F_used, Cr_used, self._strategy_states[1])
        F_adjustments.append(f_adj)
        Cr_adjustments.append(c_adj)
        
        # Strategy 2: temporal autocorrelation
        f_adj, c_adj, self._strategy_states[2] = self._strategy_temporal(
            improved_mask, F_used, Cr_used, self._strategy_states[2])
        F_adjustments.append(f_adj)
        Cr_adjustments.append(c_adj)
        
        # Strategy 3: bootstrap confidence
        f_adj, c_adj, self._strategy_states[3] = self._strategy_bootstrap(
            improved_mask, F_used, Cr_used, self._strategy_states[3])
        F_adjustments.append(f_adj)
        Cr_adjustments.append(c_adj)
        
        # Record per-strategy success for adaptive weighting
        success_rate = np.mean(improved_mask)
        for s in range(self.num_strategies):
            self.strategy_successes[s].append(success_rate)
            if len(self.strategy_successes[s]) > self.strategy_window:
                self.strategy_successes[s].pop(0)
        
        # Compute rank-based reward: each strategy gets credit based on its
        # adjustment quality. Use relative deviation from neutral (1.0) as proxy.
        # Strategies that suggest larger deviations when success is low are exploring;
        # strategies that suggest small deviations when success is high are exploiting.
        # We use a simple heuristic: strategy is rewarded when its adjustment direction
        # matches the trend (F should be lower when success is declining, higher when rising).
        strategy_rewards = np.zeros(self.num_strategies)
        
        for s in range(self.num_strategies):
            # Base reward: mean success rate within window
            if len(self.strategy_successes[s]) > 0:
                base_reward = np.mean(self.strategy_successes[s])
            else:
                base_reward = 0.2
            
            # Adjustment magnitude bonus: strategies that make decisive adjustments
            # when the signal is clear should be rewarded more
            adj_magnitude = abs(F_adjustments[s] - 1.0) + abs(Cr_adjustments[s] - 1.0)
            
            # Signal clarity bonus: high uncertainty in success rate → bold adjustments
            if len(self.strategy_successes[s]) >= 3:
                signal_clarity = 1.0 - np.std(self.strategy_successes[s])
            else:
                signal_clarity = 0.5
            
            strategy_rewards[s] = base_reward + 0.1 * adj_magnitude * signal_clarity
        
        # Ensure minimum exploration: add small epsilon to all rewards
        strategy_rewards = strategy_rewards + 0.05
        
        # Softmax weighting: temperature controls how peaked the distribution is
        # Lower temp = more weight on best; higher temp = more uniform
        temperature = 1.5
        rewards_shifted = strategy_rewards - np.max(strategy_rewards)  # numerically stable
        exp_weights = np.exp(rewards_shifted / temperature)
        softmax_weights = exp_weights / np.sum(exp_weights)
        
        # Blend adjustments
        blended_F = np.sum(np.array(F_adjustments) * softmax_weights)
        blended_Cr = np.sum(np.array(Cr_adjustments) * softmax_weights)
        
        # Apply blended adjustments
        self.F = np.clip(self.F * blended_F, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * blended_Cr, 0.1, 0.9)
        
        # Store for diagnostics (optional, doesn't affect behavior)
        self.strategy_F_adjustments = F_adjustments
        self.strategy_Cr_adjustments = Cr_adjustments
        self.strategy_weights = softmax_weights.copy()
    
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
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Store population for ensemble strategies
            self.population = population
            self.population_fitness = fitness
            
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
            
            # Ensemble adaptation
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
```