import numpy as np


class AdaptiveDirectionalDE:
    """
    ADAPTIVE MECHANISM: #2 (Ensemble/Voting with Performance-Weighted Aggregation)
    
    The five _select_survivors_batch variants are NOT highly correlated — they
    measure geometric distance, spectral condition number, k-NN graph connectivity,
    temporal EMA trends, and greedy fitness. An ensemble of their decisions is more
    robust than credit-assigning to any single variant, especially since the "best"
    variant changes per-task and per-regime. Weights are updated via exponentially
    weighted moving averages of rank-normalized success rates within a sliding window.
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
        
        # ── ENSEMBLE MECHANISM ────────────────────────────────────────────────
        # Define the five survivor-selection strategies as bound methods
        self._survivor_strategies = [
            self._select_survivors_greedy,
            self._select_survivors_geometric,
            self._select_survivors_spectral,
            self._select_survivors_connectivity,
            self._select_survivors_temporal,
        ]
        self._strategy_names = [
            "greedy", "geometric", "spectral", "connectivity", "temporal"
        ]
        self._n_strategies = len(self._survivor_strategies)
        
        # Per-strategy EWMA weight (initialized to uniform; original has prior weight)
        self._strategy_weights = np.ones(self._n_strategies)
        # Slightly boost the original greedy as prior (it wins on 5 tasks including
        # the hardest ones — tasks 11,16,17 — giving us a reasonable default)
        self._strategy_weights[0] = 1.5
        self._strategy_weights = self._strategy_weights / self._strategy_weights.sum()
        
        # Sliding window for rank-normalized reward
        self._reward_window = max(15, 3 * self._n_strategies)  # Rule (b): K >= 3*arms
        self._per_strategy_rewards = {i: [] for i in range(self._n_strategies)}
        
        # EMA smoothing for weight updates
        self._weight_ema_alpha = 0.2
        
        # Track ensemble "vote margin" history for diagnostics / potential fallback
        self._vote_margin_history = []
        # ── END ENSEMBLE MECHANISM ────────────────────────────────────────────
    
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENSEMBLE MECHANISM — Core method
    # ─────────────────────────────────────────────────────────────────────────
    
    def _select_survivors_ensemble(self, population, fitness, trials, trial_fitness):
        """
        Run all five survivor-selection strategies and aggregate their decisions
        via performance-weighted voting. Weights are updated each generation using
        rank-normalized relative improvement within a sliding window.
        """
        np_pop = len(population)
        
        # Collect per-strategy acceptance masks and per-strategy fitness improvements
        strategy_masks = []
        strategy_improvements = []
        
        for strat_idx, strategy_fn in enumerate(self._survivor_strategies):
            mask, _, _ = strategy_fn(population.copy(), fitness.copy(),
                                     trials.copy(), trial_fitness.copy())
            # Acceptance = where strategy deviated from population
            accept_mask = np.any(mask != population, axis=1)
            strategy_masks.append(accept_mask)
            
            # Per-strategy "reward": mean relative fitness improvement for accepted trials
            accepted_trials = accept_mask & (trial_fitness < fitness)
            if np.any(accepted_trials):
                delta = (fitness[accepted_trials] - trial_fitness[accepted_trials])
                incumbent_mag = np.abs(fitness[accepted_trials]) + 1e-10
                rel_improvement = np.mean(delta / incumbent_mag)
            else:
                rel_improvement = 0.0
            strategy_improvements.append(rel_improvement)
        
        # ── Step 1: Rank-normalize rewards within the sliding window ─────────
        all_rewards = []
        for s_idx in range(self._n_strategies):
            all_rewards.extend(self._per_strategy_rewards[s_idx])
        
        if len(all_rewards) >= self._reward_window:
            # z-score normalization over the full reward pool
            reward_array = np.array(all_rewards)
            mean_r = np.mean(reward_array)
            std_r = np.std(reward_array) + 1e-8
            z_all = (reward_array - mean_r) / std_r
            
            # Split back into per-strategy z-scores
            z_idx = 0
            z_scores = []
            for s_idx in range(self._n_strategies):
                k = len(self._per_strategy_rewards[s_idx])
                z_slice = z_all[z_idx:z_idx + k] if k > 0 else np.array([0.0])
                # Mean z-score for this strategy (proxy for its recent normalized performance)
                z_scores.append(float(np.mean(z_slice)))
                z_idx += k
        else:
            # Not enough data yet — use raw relative improvements, min-max scaled
            raw = np.array(strategy_improvements)
            r_min, r_max = raw.min(), raw.max()
            if r_max > r_min + 1e-12:
                z_scores = (raw - r_min) / (r_max - r_min)
            else:
                z_scores = np.zeros(self._n_strategies)
        
        # ── Step 2: Update sliding reward window ─────────────────────────────
        for s_idx, improvement in enumerate(strategy_improvements):
            self._per_strategy_rewards[s_idx].append(improvement)
            # Keep window bounded
            if len(self._per_strategy_rewards[s_idx]) > self._reward_window:
                self._per_strategy_rewards[s_idx].pop(0)
        
        # ── Step 3: Update strategy weights via EMA ──────────────────────────
        # Blend current z-scores into the EWMA weight proxy
        # We maintain pseudo-counts that represent cumulative normalized performance
        if not hasattr(self, '_strategy_counts'):
            self._strategy_counts = np.ones(self._n_strategies)
        
        # Soft update: move counts toward z-scores
        self._strategy_counts = (1 - self._weight_ema_alpha) * self._strategy_counts + \
                                 self._weight_ema_alpha * (z_scores + 1.0)
        
        # Convert counts to weights (softmax-like)
        raw_weights = self._strategy_counts
        raw_weights = np.clip(raw_weights, 0.01, None)  # Floor to avoid collapse
        self._strategy_weights = raw_weights / raw_weights.sum()
        
        # ── Step 4: Weighted vote across strategies ──────────────────────────
        vote_score = np.zeros(np_pop)
        for s_idx, mask in enumerate(strategy_masks):
            vote_score += self._strategy_weights[s_idx] * mask.astype(float)
        
        # Ensemble decision threshold: accept if weighted vote > 0.5
        # (meaning more than half the weighted vote mass says "accept")
        vote_threshold = 0.5
        ensemble_accept = vote_score > vote_threshold
        
        # Also always accept if ANY strategy accepted AND the trial strictly improves fitness
        strict_improvement = trial_fitness < fitness
        fallback_accept = strict_improvement & np.any(np.vstack(strategy_masks), axis=0)
        
        final_accept = ensemble_accept | fallback_accept
        
        # Record vote margin for diagnostics
        self._vote_margin_history.append(np.mean(vote_score))
        if len(self._vote_margin_history) > 50:
            self._vote_margin_history.pop(0)
        
        # Build new population
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[final_accept] = trials[final_accept]
        new_fitness[final_accept] = trial_fitness[final_accept]
        
        return new_population, new_fitness, final_accept
    
    # ─────────────────────────────────────────────────────────────────────────
    # Individual survivor strategies (from the benchmark)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _select_survivors_greedy(self, population, fitness, trials, trial_fitness):
        """Strategy 0 — Original greedy selection."""
        improved_mask = trial_fitness < fitness
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        return new_population, new_fitness, improved_mask
    
    def _select_survivors_geometric(self, population, fitness, trials, trial_fitness):
        """Strategy 1 — Geometric diversity maintenance via pairwise distances."""
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
    
    def _select_survivors_spectral(self, population, fitness, trials, trial_fitness):
        """Strategy 2 — Spectral selection via SVD condition number."""
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
    
    def _select_survivors_connectivity(self, population, fitness, trials, trial_fitness):
        """Strategy 3 — Graph-connectivity-aware selection via k-NN topology."""
        new_population = population.copy()
        new_fitness = fitness.copy()
        pop_size = len(population)
        combined = np.vstack([population, trials])
        
        k = min(5, pop_size - 1)
        diffs = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
        
        knn_connectivity = np.zeros(2 * pop_size)
        sorted_indices = np.argpartition(dist_matrix, k + 1, axis=1)
        for i in range(2 * pop_size):
            neighbor_dists = dist_matrix[i, sorted_indices[i, :k + 1]]
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
    
    def _select_survivors_temporal(self, population, fitness, trials, trial_fitness):
        """Strategy 4 — Temporal-aware selection using EMA and rate-of-improvement."""
        if not hasattr(self, '_temp_fitness_ema'):
            self._temp_fitness_ema = fitness.copy()
            self._temp_fitness_ema_alpha = 0.3
            self._temp_prev_fitness = fitness.copy()
            self._temp_improvement_streak = np.zeros(len(fitness))
            self._temp_generation_age = np.zeros(len(fitness))
        
        self._temp_generation_age += 1
        fitness_velocity = fitness - self._temp_prev_fitness
        
        improvement_boost = np.clip(self._temp_improvement_streak / 10.0, 0, 0.2)
        adaptive_alpha = np.clip(self._temp_fitness_ema_alpha - improvement_boost, 0.1, 0.5)
        
        new_ema = adaptive_alpha * fitness + (1 - adaptive_alpha) * self._temp_fitness_ema
        
        trial_ema = adaptive_alpha * trial_fitness + (1 - adaptive_alpha) * self._temp_fitness_ema
        
        velocity_bonus = np.zeros(len(fitness))
        improving_mask = fitness_velocity < 0
        velocity_bonus[improving_mask] = 0.1 * np.abs(fitness_velocity[improving_mask])
        velocity_bonus = np.clip(velocity_bonus, 0, 0.5)
        
        age_penalty = np.clip(self._temp_generation_age / 50.0, 0, 0.3)
        
        combined_score_current = new_ema - velocity_bonus + age_penalty
        combined_score_trial = trial_ema
        
        temporal_select_mask = combined_score_trial < combined_score_current
        raw_improved_mask = trial_fitness < fitness
        accept_mask = temporal_select_mask | raw_improved_mask
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]
        
        self._temp_fitness_ema[accept_mask] = trial_ema[accept_mask]
        self._temp_improvement_streak[accept_mask] += 1
        self._temp_improvement_streak[~accept_mask] = np.maximum(
            0, self._temp_improvement_streak[~accept_mask] - 1
        )
        self._temp_generation_age[accept_mask] = 0
        self._temp_prev_fitness = fitness.copy()
        
        return new_population, new_fitness, accept_mask
    
    # ─────────────────────────────────────────────────────────────────────────
    # End individual strategies
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
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
            
            # Reset temporal state on restart (it is generation-dependent)
            if hasattr(self, '_temp_fitness_ema'):
                self._temp_fitness_ema = fitness.copy()
                self._temp_prev_fitness = fitness.copy()
                self._temp_improvement_streak = np.zeros(len(fitness))
                self._temp_generation_age = np.zeros(len(fitness))
        
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
            
            # ── Use ensemble survivor selection ──────────────────────────────
            population, fitness, improved_mask = self._select_survivors_ensemble(
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
