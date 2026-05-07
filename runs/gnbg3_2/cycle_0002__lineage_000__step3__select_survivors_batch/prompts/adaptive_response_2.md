```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    // Mechanism #4: Contextual Bandit with Regime Detection
    // 
    // The 5 variants use fundamentally different selection signals (geometric
    // distances, SVD condition number, k-NN connectivity, temporal EMA) and
    // specialize on different problem regimes. A contextual bandit maps
    // population state features to strategy weights, switching emphasis rather
    // than making hard discrete choices. Rank-normalized rewards within a
    // sliding window prevent scale-dependent credit assignment.
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
        
        # --- Contextual Bandit State ---
        # 4 arms: [geometric, SVD, connectivity, temporal]
        self.num_strategies = 4
        self.strategy_names = ['geometric', 'SVD', 'connectivity', 'temporal']
        
        # Prior weights from benchmark win counts (normalized to sum=1)
        # geometric=10, original=5, connectivity=4, temporal=4, SVD=1
        # Approximate: geometric=0.42, SVD=0.04, connectivity=0.17, temporal=0.17
        # plus 0.2 for original (baseline)
        self.strategy_weights = np.array([0.40, 0.04, 0.18, 0.18])
        self.strategy_weights = self.strategy_weights / self.strategy_weights.sum()
        
        # Sliding window for rank-normalized rewards (K >= 3 * num_strategies)
        self.reward_window_size = 15
        self.reward_history = {i: [] for i in range(self.num_strategies)}
        
        # Regime detection features
        self.cond_number_ema = None
        self.diversity_ema = None
        self.improvement_ema = None
        self.ema_alpha = 0.2
        
        # Per-strategy performance tracking
        self.strategy_fitness_sum = np.zeros(self.num_strategies)
        self.strategy_count = np.zeros(self.num_strategies) + 1e-6
        
        # Temporal state for variant_06-style tracking
        self.fitness_ema = None
        self.fitness_ema_alpha = 0.3
        self.prev_fitness = None
        self.improvement_streak = None
        self.generation_age = None
        
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
        
        # Initialize temporal state
        self.fitness_ema = fitness.copy()
        self.prev_fitness = fitness.copy()
        self.improvement_streak = np.zeros(len(fitness))
        self.generation_age = np.zeros(len(fitness))
        
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
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _compute_condition_number(self, population):
        """Compute SVD condition number for regime detection."""
        try:
            centered = population - np.mean(population, axis=0)
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            eps = 1e-12
            return s[0] / max(s[-1], eps)
        except Exception:
            return 1.0
    
    def _compute_regime_features(self, population, fitness, prev_fitness):
        """Extract features for contextual bandit regime detection."""
        features = {}
        
        # Diversity
        div = self._compute_diversity(population)
        features['diversity'] = div
        if self.diversity_ema is None:
            self.diversity_ema = div
        else:
            self.diversity_ema = self.ema_alpha * div + (1 - self.ema_alpha) * self.diversity_ema
        features['diversity_ema'] = self.diversity_ema
        
        # Condition number
        cond = self._compute_condition_number(population)
        if self.cond_number_ema is None:
            self.cond_number_ema = cond
        else:
            self.cond_number_ema = self.ema_alpha * cond + (1 - self.ema_alpha) * self.cond_number_ema
        features['condition_number'] = cond
        features['cond_ema'] = self.cond_number_ema
        
        # Improvement rate
        current_best = np.min(fitness)
        prev_best = np.min(prev_fitness) if prev_fitness is not None else current_best
        improvement = prev_best - current_best
        if self.improvement_ema is None:
            self.improvement_ema = improvement
        else:
            self.improvement_ema = self.ema_alpha * improvement + (1 - self.ema_alpha) * self.improvement_ema
        features['improvement'] = improvement
        features['improvement_ema'] = self.improvement_ema
        
        # Stagnation indicator (is improvement near zero?)
        features['is_stagnant'] = abs(self.improvement_ema) < 1e-8
        
        # High condition number indicator
        features['high_cond'] = self.cond_number_ema > 30.0
        
        # Low diversity indicator
        features['low_diversity'] = self.diversity_ema < 10.0
        
        return features
    
    def _compute_strategy_reward(self, strategy_idx, fitness_improvement, diversity_change):
        """
        Compute rank-normalized reward for a strategy.
        Uses relative improvement (delta / |incumbent|) and z-score for diversity.
        """
        # Clamp to prevent extreme values
        fitness_improvement = np.clip(fitness_improvement, -1e6, 1e6)
        diversity_change = np.clip(diversity_change, -1e6, 1e6)
        
        # Use relative fitness improvement (stable scale)
        incumbent = np.maximum(np.abs(fitness_improvement), 1e-10)
        rel_fitness = fitness_improvement / incumbent if incumbent > 1e-10 else 0.0
        
        # Z-score for diversity change (handles varying scales across problems)
        history = self.reward_history[strategy_idx]
        if len(history) >= 3:
            recent_diversity = [h[1] for h in history[-10:]]
            mean_d = np.mean(recent_diversity)
            std_d = np.std(recent_diversity) + 1e-10
            z_diversity = diversity_change / std_d
        else:
            z_diversity = 0.0
        
        # Combined reward: fitness is primary, diversity is secondary
        reward = 0.7 * rel_fitness + 0.3 * z_diversity
        return np.clip(reward, -5.0, 5.0)
    
    def _update_strategy_weights(self, rewards):
        """
        Update strategy weights using exponential weighted averaging.
        Each strategy gets credit for its contribution to the blended decision.
        """
        for i, reward in enumerate(rewards):
            # Add to sliding window
            self.reward_history[i].append(reward)
            if len(self.reward_history[i]) > self.reward_window_size:
                self.reward_history[i].pop(0)
        
        # Compute rank-based scores within window
        if len(self.reward_history[0]) >= 3:
            window_rewards = np.array([
                np.mean(self.reward_history[i][-5:]) for i in range(self.num_strategies)
            ])
            
            # Rank transformation (higher rank = better)
            ranks = np.argsort(np.argsort(window_rewards))
            rank_scores = ranks.astype(float) / self.num_strategies
            
            # Exponential weighted update
            learning_rate = 0.3
            self.strategy_weights = (
                (1 - learning_rate) * self.strategy_weights +
                learning_rate * rank_scores
            )
            
            # Renormalize
            self.strategy_weights = self.strategy_weights / (self.strategy_weights.sum() + 1e-10)
    
    def _get_contextual_weights(self, features):
        """
        Compute context-adjusted strategy weights based on regime detection.
        These are priors that get multiplied with the learned bandit weights.
        """
        context_multipliers = np.ones(self.num_strategies)
        
        # Strategy indices: 0=geometric, 1=SVD, 2=connectivity, 3=temporal
        
        if features['is_stagnant']:
            # Stagnation: favor geometric diversity (variant_01) and original baseline
            context_multipliers[0] *= 1.5  # geometric
            context_multipliers[3] *= 1.2  # temporal
        elif features['high_cond']:
            # Subspace collapse: favor SVD (variant_02)
            context_multipliers[1] *= 2.0  # SVD
            context_multipliers[0] *= 1.3  # geometric also helps
        elif features['low_diversity']:
            # Low diversity: favor connectivity (variant_05) and geometric
            context_multipliers[2] *= 1.5  # connectivity
            context_multipliers[0] *= 1.3  # geometric
        else:
            # Normal regime: favor temporal (variant_06) for exploitation
            context_multipliers[3] *= 1.2  # temporal
        
        # Combine bandit weights with context priors
        combined = self.strategy_weights * context_multipliers
        combined = combined / (combined.sum() + 1e-10)
        
        return combined
    
    # --- Strategy Implementations ---
    
    def _select_geometric(self, population, fitness, trials, trial_fitness):
        """Variant_01: Greedy + geometric diversity via pairwise distances."""
        improved_mask = trial_fitness < fitness
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        not_improved_mask = ~improved_mask
        if np.any(not_improved_mask):
            all_pairwise = np.linalg.norm(
                population[:, np.newaxis, :] - population[np.newaxis, :, :],
                axis=2
            )
            np.fill_diagonal(all_pairwise, np.inf)
            avg_dist_to_others = np.mean(all_pairwise, axis=1)
            
            trial_to_member_dist = np.linalg.norm(
                trials[not_improved_mask] - population[not_improved_mask],
                axis=1
            )
            
            geometric_acceptance = trial_to_member_dist > avg_dist_to_others[not_improved_mask]
            indices_to_update = np.where(not_improved_mask)[0][geometric_acceptance]
            new_population[indices_to_update] = trials[indices_to_update]
            new_fitness[indices_to_update] = trial_fitness[indices_to_update]
        
        return new_population, new_fitness, improved_mask
    
    def _select_svd(self, population, fitness, trials, trial_fitness):
        """Variant_02: SVD condition number for subspace preservation."""
        improved_mask = trial_fitness < fitness
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        centered = new_population - np.mean(new_population, axis=0)
        
        try:
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            eps = 1e-12
            cond_number = s[0] / max(s[-1], eps)
            cond_threshold = 50.0
            
            if cond_number > cond_threshold and len(trials) > 0:
                minor_threshold_idx = max(0, len(s) // 3)
                minor_directions = Vt[minor_threshold_idx:]
                
                if minor_directions.shape[0] > 0:
                    trial_projections = trials @ minor_directions.T
                    trial_minor_energy = np.sum(trial_projections ** 2, axis=1)
                    current_projections = new_population @ minor_directions.T
                    current_minor_energy = np.sum(current_projections ** 2, axis=1)
                    improvement_in_minor = trial_minor_energy - current_minor_energy
                    
                    not_improved = ~improved_mask
                    if np.any(not_improved):
                        candidates = np.where(not_improved)[0]
                        candidate_improvement = improvement_in_minor[candidates]
                        n_select = min(2, len(candidates))
                        top_indices = candidates[np.argsort(candidate_improvement)[-n_select:]]
                        
                        for idx in top_indices:
                            if improvement_in_minor[idx] > 0.01 * np.mean(s):
                                new_population[idx] = trials[idx]
                                new_fitness[idx] = trial_fitness[idx]
        except Exception:
            pass
        
        return new_population, new_fitness, improved_mask
    
    def _select_connectivity(self, population, fitness, trials, trial_fitness):
        """Variant_05: k-NN graph connectivity preservation."""
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        pop_size = len(population)
        combined = np.vstack([population, trials])
        
        k = min(5, pop_size - 1)
        diffs = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
        
        knn_connectivity = np.zeros(2 * pop_size)
        for i in range(2 * pop_size):
            sorted_idx = np.argpartition(dist_matrix[i], k + 1)[:k + 1]
            neighbor_dists = dist_matrix[i, sorted_idx]
            knn_connectivity[i] = np.mean(neighbor_dists)
        
        pop_conn = knn_connectivity[:pop_size]
        trial_conn = knn_connectivity[pop_size:]
        
        improved_mask = trial_fitness < fitness
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        not_improved = ~improved_mask
        if np.any(not_improved):
            fitness_diffs = fitness - trial_fitness
            max_diff = np.max(fitness_diffs[not_improved]) + 1e-10
            near_tie_threshold = 0.2 * max_diff
            near_tie_mask = not_improved & (fitness_diffs < near_tie_threshold)
            
            if np.any(near_tie_mask):
                idx_nt = np.where(near_tie_mask)[0]
                conn_advantage = pop_conn[idx_nt] - trial_conn[idx_nt]
                connectivity_threshold = 0.05 * pop_conn[idx_nt]
                accept_connectivity = conn_advantage > connectivity_threshold
                accept_idx = idx_nt[accept_connectivity]
                new_population[accept_idx] = trials[accept_idx]
                new_fitness[accept_idx] = trial_fitness[accept_idx]
                improved_mask[accept_idx] = True
        
        return new_population, new_fitness, improved_mask
    
    def _select_temporal(self, population, fitness, trials, trial_fitness):
        """Variant_06: Temporal EMA with velocity and age tracking."""
        if self.fitness_ema is None:
            self.fitness_ema = fitness.copy()
        if self.prev_fitness is None:
            self.prev_fitness = fitness.copy()
        if self.improvement_streak is None:
            self.improvement_streak = np.zeros(len(fitness))
        if self.generation_age is None:
            self.generation_age = np.zeros(len(fitness))
        
        self.generation_age += 1
        
        fitness_velocity = fitness - self.prev_fitness
        
        improvement_boost = np.clip(self.improvement_streak / 10.0, 0, 0.2)
        adaptive_alpha = self.fitness_ema_alpha - improvement_boost
        adaptive_alpha = np.clip(adaptive_alpha, 0.1, 0.5)
        
        new_ema = np.zeros_like(fitness)
        for i in range(len(fitness)):
            new_ema[i] = adaptive_alpha[i] * fitness[i] + (1 - adaptive_alpha[i]) * self.fitness_ema[i]
        
        trial_ema = np.array([adaptive_alpha[i] * trial_fitness[i] + 
                              (1 - adaptive_alpha[i]) * self.fitness_ema[i] 
                              for i in range(len(trial_fitness))])
        
        velocity_bonus = np.zeros(len(fitness))
        improving_mask = fitness_velocity < 0
        velocity_bonus[improving_mask] = 0.1 * np.abs(fitness_velocity[improving_mask])
        velocity_bonus = np.clip(velocity_bonus, 0, 0.5)
        
        age_penalty = np.clip(self.generation_age / 50.0, 0, 0.3)
        
        combined_score_current = self.fitness_ema - velocity_bonus + age_penalty
        combined_score_trial = trial_ema
        
        temporal_select_mask = combined_score_trial < combined_score_current
        raw_improved_mask = trial_fitness < fitness
        
        accept_mask = temporal_select_mask | raw_improved_mask
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]
        
        self.fitness_ema[accept_mask] = trial_ema[accept_mask]
        self.improvement_streak[accept_mask] += 1
        self.improvement_streak[~accept_mask] = np.maximum(0, self.improvement_streak[~accept_mask] - 1)
        self.generation_age[accept_mask] = 0
        
        self.prev_fitness = fitness.copy()
        
        return new_population, new_fitness, accept_mask
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Contextual Bandit Selection: Blend all 4 strategies using context-adjusted
        weights. Each strategy contributes to the final decision proportionally.
        """
        # Compute regime features for context
        features = self._compute_regime_features(population, fitness, self.prev_fitness)
        
        # Get context-adjusted weights
        context_weights = self._get_contextual_weights(features)
        
        # Run all 4 strategies
        strategies = [
            self._select_geometric,
            self._select_svd,
            self._select_connectivity,
            self._select_temporal
        ]
        
        strategy_populations = []
        strategy_fitness = []
        strategy_masks = []
        
        for strat_fn in strategies:
            pop, fit, mask = strat_fn(population.copy(), fitness.copy(), 
                                      trials.copy(), trial_fitness.copy())
            strategy_populations.append(pop)
            strategy_fitness.append(fit)
            strategy_masks.append(mask)
        
        # Compute diversity for each strategy's result
        diversities = [self._compute_diversity(pop) for pop in strategy_populations]
        
        # Compute fitness improvement for each strategy
        base_best = np.min(fitness)
        fitness_improvements = [base_best - np.min(fit) for fit in strategy_fitness]
        
        # Compute rewards (rank-normalized)
        rewards = [
            self._compute_strategy_reward(i, fitness_improvements[i], diversities[i] - self.diversity_ema)
            for i in range(self.num_strategies)
        ]
        
        # Update bandit weights
        self._update_strategy_weights(rewards)
        
        # Final weights: combine bandit + context
        final_weights = 0.6 * self.strategy_weights + 0.4 * context_weights
        final_weights = final_weights / (final_weights.sum() + 1e-10)
        
        # Weighted blend of populations using fitness as the primary signal
        # For each individual, pick the strategy with highest weight among those
        # that accepted this individual, with fitness as tiebreaker
        final_population = population.copy()
        final_fitness = fitness.copy()
        final_improved_mask = np.zeros(len(fitness), dtype=bool)
        
        for i in range(len(population)):
            # Collect which strategies accepted this individual
            accepting_strategies = []
            for s in range(self.num_strategies):
                if strategy_masks[s][i]:
                    accepting_strategies.append((s, strategy_fitness[s][i], final_weights[s]))
            
            if accepting_strategies:
                # Sort by weight (descending), then by fitness (ascending)
                accepting_strategies.sort(key=lambda x: (-x[2], x[1]))
                best_s, best_fit, _ = accepting_strategies[0]
                final_population[i] = strategy_populations[best_s][i]
                final_fitness[i] = strategy_fitness[best_s][i]
                final_improved_mask[i] = True
        
        # Also ensure strict improvements are always accepted
        strict_improved = trial_fitness < fitness
        final_population[strict_improved] = trials[strict_improved]
        final_fitness[strict_improved] = trial_fitness[strict_improved]
        final_improved_mask = final_improved_mask | strict_improved
        
        return final_population, final_fitness, final_improved_mask
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
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
        self.Cr_history.append(F_used)
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
            
            # Reset temporal state on restart
            if self.fitness_ema is not None:
                self.fitness_ema[replace_indices] = np.inf
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
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