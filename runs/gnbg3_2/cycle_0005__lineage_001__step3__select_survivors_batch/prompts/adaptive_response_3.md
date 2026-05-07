```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE) — ENSEMBLE VOTING (Mechanism #2)
    
    The benchmark shows 7 different survivor selection strategies each winning 1-6 tasks,
    with no clear dominant strategy. The variants are highly correlated (all measuring
    diversity via different metrics: geometric, spectral, information-theoretic, etc.).
    An ensemble that VOTES across all strategies captures complementary signals without
    requiring credit assignment between correlated arms.
    
    A weighted majority vote aggregates decisions, with weights adapted via exponential
    weighted majority to emphasize strategies that perform well on the current problem.
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
        
        # ===== ENSEMBLE VOTING MECHANISM =====
        # Define ensemble members: each is a survivor selection method
        self._ensemble_methods = [
            self._select_survivors_greedy,           # baseline (index 0)
            self._select_survivors_spectral,         # variant_10 (index 1) - 6 wins
            self._select_survivors_geometric,        # variant_01 (index 2) - 6 wins
            self._select_survivors_temporal,         # variant_06 (index 3) - 4 wins
            self._select_survivors_rank_adaptive,    # variant_04 (index 4) - 3 wins
            self._select_survivors_info_theoretic,   # variant_03 (index 5) - 3 wins
        ]
        self._method_names = [
            'greedy', 'spectral', 'geometric', 'temporal', 'rank_adaptive', 'info_theoretic'
        ]
        n_methods = len(self._ensemble_methods)
        
        # Exponential weighted majority: initialize with prior from benchmark wins
        # Win counts: greedy=1, spectral=6, geometric=6, temporal=4, rank_adaptive=3, info_theoretic=3
        # Total = 23, use soft prior (add 1 for Laplace smoothing)
        prior_wins = np.array([2, 7, 7, 5, 4, 4], dtype=float)
        self._ensemble_weights = prior_wins / prior_wins.sum()
        
        # Track per-method performance for weight adaptation
        self._method_rewards = {i: [] for i in range(n_methods)}
        self._reward_window = 8  # Sliding window for rank normalization
        
        # Track ensemble consensus signal
        self._consensus_history = []
        
        # Initialize temporal state for methods that need it
        self._init_ensemble_state()
    
    def _init_ensemble_state(self):
        """Initialize stateful components for each ensemble member."""
        # Temporal method state (variant_06)
        self._temporal_ewma_short = 0.0
        self._temporal_ewma_long = 0.0
        self._temporal_regime = 'neutral'
        self._temporal_patience = 0
        self._stagnation_window = []
        self._gen_improvement_ema = None
        
        # Rank-adaptive method state (variant_04)
        self._sel_ewma_short = 0.0
        self._sel_ewma_long = 0.0
    
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
    
    # =============================================================================
    # ENSEMBLE MEMBER: 0 - Greedy (baseline)
    # =============================================================================
    def _select_survivors_greedy(self, population, fitness, trials, trial_fitness):
        """Greedy selection: baseline method."""
        improved_mask = trial_fitness < fitness
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        return new_population, new_fitness, improved_mask
    
    # =============================================================================
    # ENSEMBLE MEMBER: 1 - Spectral diversity (variant_10, 6 wins)
    # =============================================================================
    def _select_survivors_spectral(self, population, fitness, trials, trial_fitness):
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
    
    # =============================================================================
    # ENSEMBLE MEMBER: 2 - Geometric diversity (variant_01, 6 wins)
    # =============================================================================
    def _select_survivors_geometric(self, population, fitness, trials, trial_fitness):
        """Greedy selection with centroid-distance geometric diversity maintenance."""
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        current_centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        
        current_distances = np.linalg.norm(population - current_centroid, axis=1)
        current_avg_spread = np.mean(current_distances)
        
        min_spread_threshold = 1e-6
        use_geometric_bonus = current_avg_spread > min_spread_threshold
        
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
                    
                    if composite_score < fitness[i]:
                        new_population[i] = trials[i]
                        new_fitness[i] = trial_fitness[i]
                        improved_mask[i] = True
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
    # =============================================================================
    # ENSEMBLE MEMBER: 3 - Temporal regime (variant_06, 4 wins)
    # =============================================================================
    def _select_survivors_temporal(self, population, fitness, trials, trial_fitness):
        """Temporal regime-adaptive selection with EMA-based stagnation detection."""
        improved_mask = trial_fitness < fitness
        
        if self._gen_improvement_ema is None:
            self._gen_improvement_ema = np.zeros(self.np)
        
        improvement_mag = trial_fitness - fitness
        improvement_mag = np.where(improved_mask, improvement_mag, 0.0)
        
        alpha_indiv = 0.3
        self._gen_improvement_ema = alpha_indiv * improvement_mag + (1 - alpha_indiv) * self._gen_improvement_ema
        
        pop_improvement = np.sum(np.where(improved_mask, fitness - trial_fitness, 0.0))
        
        alpha_short = 0.4
        alpha_long = 0.1
        self._temporal_ewma_short = alpha_short * pop_improvement + (1 - alpha_short) * self._temporal_ewma_short
        self._temporal_ewma_long = alpha_long * pop_improvement + (1 - alpha_long) * self._temporal_ewma_long
        
        self._stagnation_window.append(np.mean(improved_mask))
        if len(self._stagnation_window) > 12:
            self._stagnation_window.pop(0)
        
        window_mean = np.mean(self._stagnation_window)
        ema_trend = self._temporal_ewma_short - self._temporal_ewma_long
        
        if ema_trend > 0.01 and window_mean > 0.25:
            self._temporal_regime = 'improving'
            self._temporal_patience = 0
        elif ema_trend < -0.005 or window_mean < 0.15:
            self._temporal_patience += 1
            if self._temporal_patience > 5:
                self._temporal_regime = 'stagnant'
        else:
            self._temporal_regime = 'neutral'
            self._temporal_patience = max(0, self._temporal_patience - 1)
        
        temporal_bonus = 0.1 * self._gen_improvement_ema
        effective_fitness = fitness - temporal_bonus
        effective_trial_fitness = trial_fitness.copy()
        
        if self._temporal_regime == 'improving':
            selection_mask = trial_fitness < fitness
        elif self._temporal_regime == 'stagnant':
            temporal_improvement_mask = effective_trial_fitness < effective_fitness
            poor_temporal_mask = self._gen_improvement_ema < np.percentile(self._gen_improvement_ema, 30)
            selection_mask = temporal_improvement_mask | (improved_mask & poor_temporal_mask)
        else:
            selection_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[selection_mask] = trials[selection_mask]
        new_fitness[selection_mask] = trial_fitness[selection_mask]
        
        return new_population, new_fitness, improved_mask
    
    # =============================================================================
    # ENSEMBLE MEMBER: 4 - Rank-adaptive (variant_04, 3 wins)
    # =============================================================================
    def _select_survivors_rank_adaptive(self, population, fitness, trials, trial_fitness):
        """Adaptive selection: greedy when improving, rank-based when stagnant."""
        improved_mask_greedy = trial_fitness < fitness
        improvement_rate = np.mean(improved_mask_greedy)
        
        self._sel_ewma_short = 0.3 * improvement_rate + 0.7 * self._sel_ewma_short
        self._sel_ewma_long = 0.1 * improvement_rate + 0.9 * self._sel_ewma_long
        
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
        
        short_trending_up = self._sel_ewma_short > self._sel_ewma_long
        both_stagnant = self._sel_ewma_short < 0.08 and self._sel_ewma_long < 0.12
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
            improved_mask = selected_indices >= n_pop
            
            return new_population, new_fitness, improved_mask
        else:
            new_population = population.copy()
            new_fitness = fitness.copy()
            improved_mask = trial_fitness < fitness
            new_population[improved_mask] = trials[improved_mask]
            new_fitness[improved_mask] = trial_fitness[improved_mask]
            
            return new_population, new_fitness, improved_mask
    
    # =============================================================================
    # ENSEMBLE MEMBER: 5 - Information-theoretic (variant_03, 3 wins)
    # =============================================================================
    def _select_survivors_info_theoretic(self, population, fitness, trials, trial_fitness):
        """Information-theoretic survivor selection using entropy and KL divergence."""
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
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
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
                
                if score > 0:
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_fitness[i]
        
        return new_population, new_fitness, improved_mask
    
    # =============================================================================
    # ENSEMBLE VOTING MECHANISM
    # =============================================================================
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Ensemble voting across all survivor selection strategies.
        
        Each method votes on each individual. We use weighted majority vote
        with rank-normalized rewards for weight adaptation.
        """
        n_methods = len(self._ensemble_methods)
        np_pop = len(population)
        
        # Step 1: Collect votes from each method
        method_votes = np.zeros((n_methods, np_pop), dtype=bool)
        method_selections = np.zeros((n_methods, np_pop), dtype=bool)
        
        for m in range(n_methods):
            _, _, method_selections[m] = self._ensemble_methods[m](
                population.copy(), fitness.copy(), trials.copy(), trial_fitness.copy()
            )
            method_votes[m] = method_selections[m]
        
        # Step 2: Compute weighted vote for each individual
        # Weight each method by its current ensemble weight
        weighted_votes = np.zeros(np_pop)
        for m in range(n_methods):
            weighted_votes += self._ensemble_weights[m] * method_votes[m].astype(float)
        
        # Ensemble decision: accept if weighted vote exceeds 0.5 (majority)
        # Alternative: accept if any method with weight > 0.15 votes yes
        min_weight_threshold = 0.15
        ensemble_decision = np.zeros(np_pop, dtype=bool)
        
        # Majority voting
        ensemble_decision = weighted_votes >= 0.5
        
        # Also accept if a high-weight method alone votes yes (safety net)
        for m in range(n_methods):
            if self._ensemble_weights[m] > min_weight_threshold:
                ensemble_decision |= method_votes[m]
        
        # Step 3: Compute reward for each method (rank-normalized)
        # Reward = relative improvement when method selected vs when it didn't
        improvement = fitness - trial_fitness
        improvement = np.where(np.isnan(improvement) | np.isinf(improvement), 0.0, improvement)
        
        for m in range(n_methods):
            # For each method, compute average improvement in selections it made
            if np.any(method_votes[m]):
                method_avg_improvement = np.mean(improvement[method_votes[m]])
            else:
                method_avg_improvement = 0.0
            
            self._method_rewards[m].append(method_avg_improvement)
        
        # Step 4: Update ensemble weights using exponential weighted majority
        # Rank-normalize rewards within sliding window
        reward_window = self._reward_window
        eta = 0.1  # Learning rate for weight update
        
        for m in range(n_methods):
            if len(self._method_rewards[m]) >= 3:
                # Get last K rewards and compute rank (1 = best)
                recent_rewards = np.array(self._method_rewards[m][-reward_window:])
                if len(recent_rewards) >= 3:
                    # Rank normalization: convert to percentile rank [0, 1]
                    ranks = np.argsort(np.argsort(-recent_rewards)) / len(recent_rewards)
                    # Use mean rank as normalized reward (higher = better)
                    normalized_reward = np.mean(ranks)
                else:
                    normalized_reward = 0.5
            else:
                normalized_reward = 0.5  # Uninformative prior for early steps
            
            # Exponential weighted majority update
            # Multiply weight by exp(eta * (2 * reward - 1)) to reward good methods
            weight_update = np.exp(eta * (2 * normalized_reward - 1))
            self._ensemble_weights[m] *= weight_update
        
        # Normalize weights to sum to 1
        weight_sum = np.sum(self._ensemble_weights)
        if weight_sum > 0:
            self._ensemble_weights /= weight_sum
        else:
            # Fallback to uniform if all weights collapse
            self._ensemble_weights = np.ones(n_methods) / n_methods
        
        # Ensure no weight is too small (minimum 0.05)
        self._ensemble_weights = np.maximum(self._ensemble_weights, 0.05)
        self._ensemble_weights /= np.sum(self._ensemble_weights)
        
        # Step 5: Apply ensemble decision
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[ensemble_decision] = trials[ensemble_decision]
        new_fitness[ensemble_decision] = trial_fitness[ensemble_decision]
        
        # Track consensus level for diagnostics
        consensus = np.mean(np.sum(method_votes, axis=0) == n_methods)  # Fraction unanimous
        self._consensus_history.append(consensus)
        if len(self._consensus_history) > 50:
            self._consensus_history.pop(0)
        
        return new_population, new_fitness, ensemble_decision
    
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
            
            # Reset temporal state on restart
            self._init_ensemble_state()
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
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