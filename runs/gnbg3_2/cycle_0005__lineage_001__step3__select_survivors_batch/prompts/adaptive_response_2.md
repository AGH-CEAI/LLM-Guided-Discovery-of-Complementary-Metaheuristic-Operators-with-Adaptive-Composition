```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    ADAPTIVE ENSEMBLE SELECTION (Mechanism #2)
    
    Combines 5 survivor selection strategies via exponentially-weighted voting.
    Each strategy's weight is adapted online based on its recent success rate,
    enabling the algorithm to favor strategies that work in the current regime
    without requiring explicit credit assignment between correlated diversity metrics.
    
    Win distribution from benchmarks: catA(6), catB(6), catF(4), catC(3), catD(3)
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
        
        # ─── ENSEMBLE SELECTION MECHANISM ───
        # Initialize strategy weights based on benchmark win counts
        # catA(6), catB(6), catF(4), catC(3), catD(3) → total 22 wins
        self.strategy_names = ['geometric', 'spectral', 'temporal', 'entropy', 'rank_adaptive']
        self.strategy_weights = np.array([6.0, 6.0, 4.0, 3.0, 3.0])
        self.strategy_weights = self.strategy_weights / self.strategy_weights.sum()
        
        # Per-strategy success tracking (sliding window)
        self.success_window_size = max(15, self.np // 2)
        self.strategy_success_counts = np.zeros(5)
        self.strategy_total_counts = np.zeros(5)
        
        # EMA state for temporal strategy (variant_06)
        self._init_temporal_state()
        
        # EMA state for rank-adaptive strategy (variant_04)
        self._init_rank_adaptive_state()
        
    def _init_temporal_state(self):
        """Initialize temporal regime tracking state."""
        self.gen_improvement_ema = None
        self.pop_improvement_ema_short = 0.0
        self.pop_improvement_long = 0.0
        self.temporal_regime = 'neutral'
        self.regime_patience = 0
        self.stagnation_window = []
        
    def _init_rank_adaptive_state(self):
        """Initialize rank-adaptive regime tracking state."""
        self.sel_ewma_short = None
        self.sel_ewma_long = None
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling."""
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
        """Create ring neighbor indices."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions."""
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
        """Compute directional bias from historical mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """Composite mutation: ring-based + directional."""
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
        """Evaluate batch with graceful budget handling."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    # ─────────────────────────────────────────────────────────────
    # ENSEMBLE SELECTION STRATEGIES
    # Each strategy returns: (selection_mask, strategy_id)
    # ─────────────────────────────────────────────────────────────
    
    def _strategy_geometric(self, population, fitness, trials, trial_fitness):
        """Variant_01: Greedy with centroid-distance geometric diversity bonus."""
        improved_mask = trial_fitness < fitness

        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        current_centroid = np.sum(population * weights[:, np.newaxis], axis=0)

        current_distances = np.linalg.norm(population - current_centroid, axis=1)
        current_avg_spread = np.mean(current_distances)

        min_spread_threshold = 1e-6
        use_geometric_bonus = current_avg_spread > min_spread_threshold

        selection_mask = improved_mask.copy()

        if use_geometric_bonus:
            for i in range(len(population)):
                if trial_fitness[i] < fitness[i]:
                    temp_weights = weights.copy()
                    temp_weights[i] = 1.0 / (trial_fitness[i] - np.min(fitness) + 1e-10)
                    temp_weights = temp_weights / np.sum(temp_weights)
                    temp_centroid = np.sum(
                        np.where(np.arange(len(population))[:, np.newaxis] == i,
                                 trials[i], population) * temp_weights[:, np.newaxis],
                        axis=0
                    )

                    old_dist_i = current_distances[i]
                    new_dist_i = np.linalg.norm(trials[i] - temp_centroid)

                    diversity_gain = new_dist_i - old_dist_i
                    normalized_gain = diversity_gain / (current_avg_spread + 1e-10)
                    diversity_bonus = 0.02 * normalized_gain

                    composite_score = trial_fitness[i] + diversity_bonus
                    selection_mask[i] = composite_score < fitness[i]

        return selection_mask, 0
    
    def _strategy_spectral(self, population, fitness, trials, trial_fitness):
        """Variant_10: Greedy with spectral diversity modulation (eigenvalue spread)."""
        improved_mask = trial_fitness < fitness
        improvement = fitness - trial_fitness

        try:
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0
        except Exception:
            condition = 1.0

        diversity_bonus = np.log1p(condition)

        fit_range = np.ptp(fitness)
        fit_range = max(fit_range, 1e-10)
        scaled_improvement = improvement / fit_range

        accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
        selection_mask = improved_mask | (scaled_improvement > accept_threshold)

        return selection_mask, 1
    
    def _strategy_temporal(self, population, fitness, trials, trial_fitness):
        """Variant_06: Temporal regime-adaptive selection with EMA-based stagnation detection."""
        improved_mask = trial_fitness < fitness

        if self.gen_improvement_ema is None:
            self.gen_improvement_ema = np.zeros(self.np)

        improvement_mag = trial_fitness - fitness
        improvement_mag = np.where(improved_mask, improvement_mag, 0.0)

        alpha_indiv = 0.3
        self.gen_improvement_ema = alpha_indiv * improvement_mag + (1 - alpha_indiv) * self.gen_improvement_ema

        pop_improvement = np.sum(np.where(improved_mask, fitness - trial_fitness, 0.0))

        alpha_short = 0.4
        alpha_long = 0.1
        self.pop_improvement_ema_short = alpha_short * pop_improvement + (1 - alpha_short) * self.pop_improvement_ema_short
        self.pop_improvement_long = alpha_long * pop_improvement + (1 - alpha_long) * self.pop_improvement_long

        self.stagnation_window.append(np.mean(improved_mask))
        if len(self.stagnation_window) > 12:
            self.stagnation_window.pop(0)

        window_mean = np.mean(self.stagnation_window) if self.stagnation_window else 0.25
        ema_trend = self.pop_improvement_ema_short - self.pop_improvement_long

        if ema_trend > 0.01 and window_mean > 0.25:
            self.temporal_regime = 'improving'
            self.regime_patience = 0
        elif ema_trend < -0.005 or window_mean < 0.15:
            self.regime_patience += 1
            if self.regime_patience > 5:
                self.temporal_regime = 'stagnant'
        else:
            self.temporal_regime = 'neutral'
            self.regime_patience = max(0, self.regime_patience - 1)

        temporal_bonus = 0.1 * self.gen_improvement_ema
        effective_fitness = fitness - temporal_bonus
        effective_trial_fitness = trial_fitness.copy()

        if self.temporal_regime == 'improving':
            selection_mask = trial_fitness < fitness
        elif self.temporal_regime == 'stagnant':
            temporal_improvement_mask = effective_trial_fitness < effective_fitness
            poor_temporal_mask = self.gen_improvement_ema < np.percentile(self.gen_improvement_ema, 30)
            selection_mask = temporal_improvement_mask | (improved_mask & poor_temporal_mask)
        else:
            selection_mask = trial_fitness < fitness

        return selection_mask, 2
    
    def _strategy_entropy(self, population, fitness, trials, trial_fitness):
        """Variant_03: Information-theoretic selection using entropy and KL divergence."""
        improved_mask = trial_fitness < fitness

        fitness_flat = fitness.ravel()
        valid_fitness = fitness_flat[~np.isinf(fitness_flat)]

        if len(valid_fitness) >= 10:
            hist, bin_edges = np.histogram(valid_fitness, bins='auto', density=True)
            hist = np.maximum(hist, 1e-10)
            bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
            population_entropy = -np.sum(hist * np.log(hist)) * bin_width
        else:
            population_entropy = 1.0

        selection_mask = improved_mask.copy()

        for i in range(len(population)):
            if improved_mask[i]:
                trial_val = trial_fitness[i]
                parent_val = fitness[i]

                sigma = np.std(valid_fitness) + 1e-10

                p_parent = np.exp(-0.5 * ((parent_val - valid_fitness) / sigma) ** 2)
                p_trial = np.exp(-0.5 * ((trial_val - valid_fitness) / sigma) ** 2)

                p_parent = np.maximum(p_parent.mean(), 1e-10)
                p_trial = np.maximum(p_trial.mean(), 1e-10)

                kl_gain = np.log(p_trial / p_parent + 1e-10)

                fitness_range = np.max(valid_fitness) - np.min(valid_fitness) + 1e-10
                norm_improvement = (parent_val - trial_val) / fitness_range

                alpha = 0.7
                score = alpha * norm_improvement + (1 - alpha) * kl_gain

                selection_mask[i] = score > 0

        return selection_mask, 3
    
    def _strategy_rank_adaptive(self, population, fitness, trials, trial_fitness):
        """Variant_04: Adaptive greedy/rank-based selection with regime detection."""
        improved_mask_greedy = trial_fitness < fitness
        improvement_rate = np.mean(improved_mask_greedy)

        if self.sel_ewma_short is None:
            self.sel_ewma_short = improvement_rate
            self.sel_ewma_long = improvement_rate

        self.sel_ewma_short = 0.3 * improvement_rate + 0.7 * self.sel_ewma_short
        self.sel_ewma_long = 0.1 * improvement_rate + 0.9 * self.sel_ewma_long

        n = len(fitness)
        if n > 3:
            ranks_f = np.argsort(np.argsort(fitness)) + 1
            ranks_tf = np.argsort(np.argsort(trial_fitness)) + 1
            std_f = np.std(fitness)
            std_tf = np.std(trial_fitness)
            if std_f > 1e-12 and std_tf > 1e-12:
                cov = np.mean((fitness - np.mean(fitness)) * (trial_fitness - np.mean(trial_fitness)))
                spearman_proxy = cov / (std_f * std_tf + 1e-12)
            else:
                spearman_proxy = 1.0
        else:
            spearman_proxy = 1.0

        short_trending_up = self.sel_ewma_short > self.sel_ewma_long
        both_stagnant = self.sel_ewma_short < 0.08 and self.sel_ewma_long < 0.12
        low_correlation = spearman_proxy < 0.25

        use_rank_selection = (not short_trending_up) or both_stagnant or low_correlation

        if use_rank_selection:
            combined_pop = np.vstack([population, trials])
            combined_fitness = np.concatenate([fitness, trial_fitness])

            n_pop = len(population)

            rank_order = np.argsort(combined_fitness)
            ranks = np.empty_like(rank_order)
            ranks[rank_order] = np.arange(len(rank_order)) + 1

            selected_indices = np.argsort(ranks)[:n_pop]

            new_population = combined_pop[selected_indices]
            new_fitness = combined_fitness[selected_indices]
            
            # Return mask indicating which trials were selected
            selection_mask = selected_indices >= n_pop
            return selection_mask, 4
        else:
            return improved_mask_greedy, 4
    
    # ─────────────────────────────────────────────────────────────
    # ENSEMBLE AGGREGATION
    # ─────────────────────────────────────────────────────────────
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Ensemble selection: run all strategies, aggregate via weighted voting.
        Strategy weights are adapted based on recent per-strategy success rates.
        """
        pop_size = len(population)
        
        # Run all strategies and collect their selection masks
        strategies = [
            self._strategy_geometric,
            self._strategy_spectral,
            self._strategy_temporal,
            self._strategy_entropy,
            self._strategy_rank_adaptive
        ]
        
        all_masks = []
        for strategy_fn in strategies:
            try:
                mask, sid = strategy_fn(population, fitness, trials, trial_fitness)
                if len(mask) == pop_size:
                    all_masks.append((mask, sid))
            except Exception:
                pass
        
        if not all_masks:
            # Fallback to standard greedy
            improved_mask = trial_fitness < fitness
            new_population = population.copy()
            new_fitness = fitness.copy()
            new_population[improved_mask] = trials[improved_mask]
            new_fitness[improved_mask] = trial_fitness[improved_mask]
            return new_population, new_fitness, improved_mask
        
        # Build ensemble decision matrix: rows = individuals, cols = strategies
        n_strategies = len(all_masks)
        decision_matrix = np.zeros((pop_size, n_strategies), dtype=float)
        
        for col_idx, (mask, sid) in enumerate(all_masks):
            decision_matrix[:, col_idx] = mask.astype(float)
        
        # Compute weighted vote for each individual
        # Weight each strategy by its current weight
        weights_vec = self.strategy_weights[:n_strategies]
        weighted_votes = decision_matrix @ weights_vec
        
        # Ensemble selection: accept if weighted vote exceeds 0.5
        # This means on average more than half the (weighted) strategies vote yes
        ensemble_threshold = 0.5
        final_selection = weighted_votes >= ensemble_threshold
        
        # Count successes per strategy for weight adaptation
        for col_idx, (mask, sid) in enumerate(all_masks):
            # Count how many selections this strategy contributed that were accepted
            strategy_accepted = mask & final_selection
            n_accepted = np.sum(strategy_accepted)
            n_improved = np.sum(mask)
            
            self.strategy_success_counts[sid] += n_accepted
            self.strategy_total_counts[sid] += max(n_improved, 1)
        
        # Adapt strategy weights using exponential weighted average of success rates
        self._adapt_strategy_weights()
        
        # Apply ensemble selection
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[final_selection] = trials[final_selection]
        new_fitness[final_selection] = trial_fitness[final_selection]
        
        # Track which individuals improved (for downstream adaptation)
        improved_mask = final_selection
        
        return new_population, new_fitness, improved_mask
    
    def _adapt_strategy_weights(self):
        """
        Adapt strategy weights using rank-normalized success rates.
        Uses z-score normalization within sliding window to avoid scale dependence.
        """
        # Compute per-strategy success rates
        success_rates = np.zeros(5)
        valid_mask = self.strategy_total_counts > 0
        success_rates[valid_mask] = (
            self.strategy_success_counts[valid_mask] / 
            self.strategy_total_counts[valid_mask]
        )
        
        # Rank-normalize success rates (convert to percentile ranks)
        # This avoids scale-dependent magic constants
        n_strategies = 5
        ranks = np.argsort(np.argsort(-success_rates)) + 1  # Higher success = lower rank (1 = best)
        rank_normalized = (ranks - 1) / (n_strategies - 1)  # Map to [0, 1]
        
        # Compute new weights as softmax of rank-normalized rates
        # Temperature controls how peaked the distribution is
        temperature = 2.0
        raw_weights = np.exp(-temperature * rank_normalized)
        
        # Blend with prior (benchmark-based) weights for stability
        prior_weights = np.array([6.0, 6.0, 4.0, 3.0, 3.0])
        prior_weights = prior_weights / prior_weights.sum()
        
        blend_factor = 0.7  # How much to trust the online adaptation
        new_weights = blend_factor * raw_weights / raw_weights.sum() + \
                      (1 - blend_factor) * prior_weights
        
        # Smooth weight changes to avoid erratic switching
        self.strategy_weights = 0.8 * self.strategy_weights + 0.2 * new_weights
        self.strategy_weights = self.strategy_weights / self.strategy_weights.sum()
        
        # Decay counters periodically to allow adaptation to new regimes
        decay_rate = 0.95
        self.strategy_success_counts *= decay_rate
        self.strategy_total_counts *= decay_rate
        self.strategy_total_counts = np.maximum(self.strategy_total_counts, 0.1)
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics."""
        success_rate = np.mean(improved_mask)

        if not hasattr(self, 'success_ewma_short'):
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)

        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)

        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15

        momentum_factor = np.clip(F_drift, 0.7, 1.4)

        if short_trending_up and self.success_ewma_short > 0.25:
            F_adjustment = 1.05 * momentum_factor
            Cr_adjustment = 1.08
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            F_adjustment = 0.88 / momentum_factor
            Cr_adjustment = 0.92
            self.stagnation_counter += 1
        elif both_stagnant:
            F_adjustment = 0.75
            Cr_adjustment = 1.15
            self.stagnation_counter += 2
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.08 * delta
            Cr_adjustment = 1.0 - 0.05 * delta

        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
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
            
            # Reset ensemble state on restart
            self._init_temporal_state()
            self._init_rank_adaptive_state()
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop with ensemble selection."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
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
            
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
```