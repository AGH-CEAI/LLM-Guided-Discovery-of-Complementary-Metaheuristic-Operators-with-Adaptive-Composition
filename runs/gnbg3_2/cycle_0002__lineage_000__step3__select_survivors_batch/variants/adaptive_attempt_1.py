import numpy as np


class AdaptiveDirectionalDE:
    """
    ADAPTIVE MECHANISM: #2 — Ensemble/Voting with Exponentially Weighted Majority.
    
    Rationale: Benchmark wins are spread across 5 variants (10/5/4/4/1) with no 
    single operator dominating (>50%). The variants compute correlated but distinct 
    diversity signals (geometric, graph-connectivity, temporal, SVD). A bandit 
    cannot reliably credit-assign across correlated operators. Ensemble voting 
    combines their complementary strengths without exploration cost.
    
    Adaptive Directional Differential Evolution (ADDE)
    
    A DE variant with:
    - Ring topology for mutation parent selection
    - Directional memory that biases mutation toward successful directions
    - Adaptive F and Cr based on historical success rates
    - Composite mutation: directional + rand/1 blended adaptively
    - Diversity-triggered restart mechanism
    - ENSEMBLE survivor selection: combines geometric, graph-connectivity,
      temporal, and SVD-based strategies via weighted voting
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
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
        
        # =========================================================
        # ENSEMBLE ADAPTATION STATE
        # =========================================================
        # Four selection strategies: geometric, graph, temporal, SVD
        self._ensemble_strategies = [
            self._select_geometric,
            self._select_graph_connectivity,
            self._select_temporal,
            self._select_svd,
        ]
        self._strategy_names = ['geometric', 'graph', 'temporal', 'svd']
        self._num_strategies = len(self._ensemble_strategies)
        
        # Exponential weighted moving average weights (log-odds)
        # Initialize with slight prior favoring geometric (10 wins vs 5/4/4/1)
        self._log_weights = np.log(np.array([3.0, 2.0, 2.0, 1.0, 1.0])[:self._num_strategies])
        
        # Per-strategy reward history (sliding window for rank-normalization)
        self._reward_history = {i: [] for i in range(self._num_strategies)}
        self._reward_window = 15  # K >= 3 * num_strategies (rule b)
        
        # Track per-generation performance signals for reward computation
        self._prev_best_fitness = None
        
        # Persistent state for temporal strategy
        self._temporal_state = None
        
    def _normalize_rewards(self, raw_rewards):
        """
        Rank-based normalization within sliding window (rule a).
        Returns z-scores relative to window, clipped to [-2, 2].
        """
        if len(raw_rewards) < 2:
            return np.zeros(len(raw_rewards))
        arr = np.array(raw_rewards)
        mean = np.mean(arr)
        std = np.std(arr) + 1e-10
        z_scores = (arr - mean) / std
        return np.clip(z_scores, -2.0, 2.0)
    
    def _update_ensemble_weights(self, generation_improvement, fitness_improvements):
        """
        Update ensemble weights using exponentially weighted majority.
        
        Reward signal: combination of
        - global fitness improvement (relative to incumbent)
        - per-strategy improvement contribution (fraction of improvements 
          attributable to each strategy's acceptances)
        """
        if generation_improvement <= 0:
            # No improvement this generation — penalize all, especially those
            # that accepted non-improvers (contribute to stagnation)
            penalty = -0.5 / self._num_strategies
            for i in range(self._num_strategies):
                self._log_weights[i] += penalty
            return
        
        # Compute per-strategy relative contribution to improvement
        # Strategy i gets reward proportional to its share of accepted improvements
        total_improvement = np.sum(np.abs(fitness_improvements))
        if total_improvement > 0:
            rel_rewards = np.abs(fitness_improvements) / total_improvement
        else:
            rel_rewards = np.ones(self._num_strategies) / self._num_strategies
        
        # Also reward strategies that produced ANY improvement
        any_improvement_bonus = (np.array(fitness_improvements) < 0).astype(float)
        combined_rewards = 0.6 * rel_rewards + 0.4 * any_improvement_bonus
        
        # Add to history and normalize
        for i in range(self._num_strategies):
            self._reward_history[i].append(combined_rewards[i])
            if len(self._reward_history[i]) > self._reward_window:
                self._reward_history[i].pop(0)
        
        # Compute normalized reward for each strategy
        normalized_rewards = np.array([
            np.mean(self._normalize_rewards(self._reward_history[i]))
            for i in range(self._num_strategies)
        ])
        
        # Weighted majority update: reward * learning_rate
        learning_rate = 0.1
        self._log_weights += learning_rate * normalized_rewards
        
        # Soft-max normalization to prevent collapse
        max_weight = 5.0
        self._log_weights = np.clip(self._log_weights, -max_weight, max_weight)
    
    def _get_ensemble_weights(self):
        """Convert log-weights to probability distribution."""
        w = np.exp(self._log_weights - np.max(self._log_weights))
        return w / np.sum(w)
    
    def _select_geometric(self, population, fitness, trials, trial_fitness):
        """Geometric diversity maintenance via pairwise distances."""
        improved_mask = trial_fitness < fitness
        accept_mask = improved_mask.copy()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Accept fitness improvements unconditionally
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        # For non-improving trials, apply geometric diversity selection
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
            accept_mask[indices_to_update] = True
        
        return new_population, new_fitness, accept_mask
    
    def _select_graph_connectivity(self, population, fitness, trials, trial_fitness):
        """Graph-connectivity-aware selection using k-NN topology."""
        new_population = population.copy()
        new_fitness = fitness.copy()
        accept_mask = np.zeros(len(population), dtype=bool)
        
        pop_size = len(population)
        combined = np.vstack([population, trials])
        
        k = min(5, pop_size - 1)
        
        diffs = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2) + 1e-12)
        
        sorted_indices = np.argpartition(dist_matrix, k + 1, axis=1)
        knn_connectivity = np.zeros(2 * pop_size)
        for i in range(2 * pop_size):
            neighbor_dists = dist_matrix[i, sorted_indices[i, :k + 1]]
            knn_connectivity[i] = np.mean(neighbor_dists)
        
        pop_conn = knn_connectivity[:pop_size]
        trial_conn = knn_connectivity[pop_size:]
        
        improved_mask = trial_fitness < fitness
        
        # Phase 1: Clear fitness improvements
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        accept_mask[improved_mask] = True
        
        # Phase 2: Connectivity-preserving replacement for near-ties
        not_improved = ~improved_mask
        if np.any(not_improved):
            fitness_diffs = fitness - trial_fitness
            max_diff = np.max(np.abs(fitness_diffs[not_improved])) + 1e-10
            near_tie_threshold = 0.2 * max_diff
            near_tie_mask = not_improved & (np.abs(fitness_diffs) < near_tie_threshold)
            
            if np.any(near_tie_mask):
                idx_nt = np.where(near_tie_mask)[0]
                conn_advantage = pop_conn[idx_nt] - trial_conn[idx_nt]
                connectivity_threshold = 0.05 * (pop_conn[idx_nt] + 1e-10)
                accept_connectivity = conn_advantage > connectivity_threshold
                
                accept_idx = idx_nt[accept_connectivity]
                new_population[accept_idx] = trials[accept_idx]
                new_fitness[accept_idx] = trial_fitness[accept_idx]
                accept_mask[accept_idx] = True
        
        return new_population, new_fitness, accept_mask
    
    def _select_temporal(self, population, fitness, trials, trial_fitness):
        """Temporal-aware selection using EMA of fitness and rate-of-improvement."""
        if self._temporal_state is None:
            self._temporal_state = {
                'fitness_ema': fitness.copy(),
                'prev_fitness': fitness.copy(),
                'improvement_streak': np.zeros(len(fitness)),
                'generation_age': np.zeros(len(fitness)),
                'alpha': 0.3,
            }
        
        state = self._temporal_state
        state['generation_age'] += 1
        
        fitness_velocity = fitness - state['prev_fitness']
        
        improvement_boost = np.clip(state['improvement_streak'] / 10.0, 0, 0.2)
        adaptive_alpha = np.clip(state['alpha'] - improvement_boost, 0.1, 0.5)
        
        new_ema = adaptive_alpha * fitness + (1 - adaptive_alpha) * state['fitness_ema']
        
        trial_ema = adaptive_alpha * trial_fitness + (1 - adaptive_alpha) * state['fitness_ema']
        
        velocity_bonus = np.zeros(len(fitness))
        improving_mask = fitness_velocity < 0
        velocity_bonus[improving_mask] = 0.1 * np.abs(fitness_velocity[improving_mask])
        velocity_bonus = np.clip(velocity_bonus, 0, 0.5)
        
        age_penalty = np.clip(state['generation_age'] / 50.0, 0, 0.3)
        
        combined_score_current = state['fitness_ema'] - velocity_bonus + age_penalty
        combined_score_trial = trial_ema
        
        temporal_select_mask = combined_score_trial < combined_score_current
        raw_improved_mask = trial_fitness < fitness
        accept_mask = temporal_select_mask | raw_improved_mask
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]
        
        state['fitness_ema'][accept_mask] = trial_ema[accept_mask]
        state['improvement_streak'][accept_mask] += 1
        state['improvement_streak'][~accept_mask] = np.maximum(
            0, state['improvement_streak'][~accept_mask] - 1
        )
        state['generation_age'][accept_mask] = 0
        state['prev_fitness'] = fitness.copy()
        
        return new_population, new_fitness, accept_mask
    
    def _select_svd(self, population, fitness, trials, trial_fitness):
        """SVD condition number-based selection for subspace exploration."""
        improved_mask = trial_fitness < fitness
        accept_mask = improved_mask.copy()
        
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
                                accept_mask[idx] = True
        except Exception:
            pass
        
        return new_population, new_fitness, accept_mask
    
    def _ensemble_select(self, population, fitness, trials, trial_fitness):
        """
        Ensemble survivor selection: run all strategies and vote.
        
        For each candidate (i), each strategy votes accept/reject based on its
        internal logic. Weights are exponentially weighted based on historical
        contribution to fitness improvement. Ties broken by weighted average of
        fitness values.
        """
        np_pop = len(population)
        weights = self._get_ensemble_weights()
        
        # Track per-strategy acceptances and fitness predictions
        strategy_acceptances = np.zeros((self._num_strategies, np_pop), dtype=bool)
        strategy_fitness = np.zeros((self._num_strategies, np_pop))
        
        for s_idx, strategy_fn in enumerate(self._ensemble_strategies):
            _, _, accept_mask = strategy_fn(
                population.copy(), fitness.copy(), 
                trials.copy(), trial_fitness.copy()
            )
            strategy_acceptances[s_idx] = accept_mask
            # Fitness prediction: strategy's accepted fitness or current fitness
            strategy_fitness[s_idx] = np.where(accept_mask, trial_fitness, fitness)
        
        # Weighted vote per candidate
        weighted_votes = np.sum(
            strategy_acceptances * weights[:, np.newaxis], axis=0
        )
        
        # Decision: accept if weighted vote > 0.5 (majority threshold)
        accept_threshold = 0.5
        ensemble_accept = weighted_votes >= accept_threshold
        
        # For candidates where ensemble is undecided (near 0.5), 
        # use weighted median of strategy fitness predictions
        near_tie = np.abs(weighted_votes - accept_threshold) < 0.15
        if np.any(near_tie):
            # Weighted median: sort by fitness, pick weight-median
            for i in np.where(near_tie)[0]:
                preds = strategy_fitness[:, i]
                sorted_idx = np.argsort(preds)
                cumsum = np.cumsum(weights[sorted_idx])
                median_idx = sorted_idx[np.searchsorted(cumsum, 0.5)]
                ensemble_accept[i] = (trial_fitness[i] <= preds[median_idx])
        
        # Build final population
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[ensemble_accept] = trials[ensemble_accept]
        new_fitness[ensemble_accept] = trial_fitness[ensemble_accept]
        
        # Compute per-strategy improvement contributions for weight update
        fitness_improvements = np.zeros(self._num_strategies)
        for s_idx in range(self._num_strategies):
            accepted_by_s = strategy_acceptances[s_idx] & ensemble_accept
            if np.any(accepted_by_s):
                fitness_improvements[s_idx] = np.sum(
                    fitness[accepted_by_s] - trial_fitness[accepted_by_s]
                )
        
        return new_population, new_fitness, ensemble_accept, fitness_improvements
    
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
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        fitness = np.where(np.isinf(fitness), 1e20, fitness)
        
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
        """
        Generate mutation vectors using composite strategy:
        - Ring-based rand/1 component
        - Directional component from historical memory
        """
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
        
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
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
        fitness = np.where(np.isinf(fitness), 1e20, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Wrapper for ensemble selection (backward compatibility)."""
        new_pop, new_fit, accept_mask, _ = self._ensemble_select(
            population, fitness, trials, trial_fitness
        )
        return new_pop, new_fit, accept_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics and parameter drift detection."""
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
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = 1e20
            self.stagnation_counter = 0
            self.successful_directions = []
            # Reset temporal state on restart
            self._temporal_state = None
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        self._prev_best_fitness = best_fitness
        
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
            
            # Ensemble selection with weight update
            population, fitness, improved_mask, fitness_improvements = self._ensemble_select(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update ensemble weights based on this generation's performance
            gen_best = np.min(fitness)
            generation_improvement = self._prev_best_fitness - gen_best
            self._update_ensemble_weights(generation_improvement, fitness_improvements)
            self._prev_best_fitness = gen_best
            
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
