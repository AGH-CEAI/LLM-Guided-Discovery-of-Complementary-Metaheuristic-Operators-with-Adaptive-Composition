import numpy as np


class AdaptiveDirectionalDE:
    """
    ENSEMBLE VOTING WITH EXPONENTIALLY WEIGHTED SUCCESS (Mechanism #2)
    
    Runs all 6 winning _select_survivors_batch strategies every generation,
    aggregates their acceptance decisions via weighted majority vote,
    and adapts strategy weights using rank-normalized success history.
    
    Rationale: Benchmark shows 6 strategies winning across 24 tasks with
    moderate spread (6,6,4,3,3,1 wins). These compute genuinely different
    signals (geometric/spatial/entropy/rank/spectral/temporal) — NOT
    rescalings — so a bandit cannot reliably credit-assign. Ensemble voting
    is robust on correlated-but-distinct heuristics and has no cold-start.
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
        # ENSEMBLE VOTING MECHANISM (Mechanism #2)
        # ============================================================
        # Strategy names and configs
        self._strategy_names = [
            'geometric',    # variant_01: centroid-distance diversity
            'spectral',     # variant_10: eigenvalue condition modulation
            'temporal',     # variant_06: EMA regime detection
            'rank',         # variant_04: Spearman correlation fallback
            'entropy',      # variant_03: KL divergence information gain
            'centrality',   # variant_05: graph betweenness
        ]
        self._num_strategies = len(self._strategy_names)
        
        # Initialize EWM weights: start with uniform prior biased toward
        # strategies that won more in benchmarks (catA=6, catB=6, catF=4,
        # catC=3, catD=3 wins). Use log-odds to avoid extreme initial bias.
        # log-odds: win_count / total_wins -> prior probability
        win_counts = np.array([6, 6, 4, 3, 3, 1], dtype=float)
        total_wins = win_counts.sum()
        # Soft prior: mix uniform (0.5 weight) with benchmark-based (0.5 weight)
        uniform_prior = 1.0 / self._num_strategies
        benchmark_prior = win_counts / total_wins
        self._strategy_weights = 0.5 * uniform_prior + 0.5 * benchmark_prior
        self._strategy_weights /= self._strategy_weights.sum()
        
        # Per-strategy success history (for rank-normalized EWM)
        self._strategy_success_rates = [0.0] * self._num_strategies
        self._strategy_ewm_short = [0.0] * self._num_strategies
        self._strategy_ewm_long = [0.0] * self._num_strategies
        self._strategy_n_improved = [0] * self._num_strategies
        
        # Sliding window for rank-normalization (Rule b: K >= 3 * num_arms)
        self._rank_window_size = max(18, 3 * self._num_strategies)
        self._strategy_recent_wins = [[] for _ in range(self._num_strategies)]
        
        # Ensemble voting threshold (majority: > 0.5 weighted vote)
        self._vote_threshold = 0.5
        
        # Fallback: if all strategies agree on reject, use greedy
        self._use_greedy_fallback = True
        
        # Track whether centrality strategy is available (needs scipy)
        self._centrality_available = self._check_scipy_availability()
        
    def _check_scipy_availability(self):
        """Check if scipy is available for graph centrality strategy."""
        try:
            from scipy.sparse.csgraph import shortest_path
            from scipy.sparse import csr_matrix
            return True
        except ImportError:
            return False
    
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
    
    # ==================================================================
    # STRATEGY 0: Geometric diversity (variant_01_catA_idea_0)
    # ==================================================================
    def _strategy_geometric(self, population, fitness, trials, trial_fitness):
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

        accept_mask = improved_mask.copy()

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
                        accept_mask[i] = True

        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]

        return new_population, new_fitness, accept_mask
    
    # ==================================================================
    # STRATEGY 1: Spectral diversity (variant_10_catB_idea_0)
    # ==================================================================
    def _strategy_spectral(self, population, fitness, trials, trial_fitness):
        """Greedy selection with spectral diversity modulation via condition number."""
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
    
    # ==================================================================
    # STRATEGY 2: Temporal regime-adaptive (variant_06_catF_idea_0)
    # ==================================================================
    def _strategy_temporal(self, population, fitness, trials, trial_fitness):
        """Temporal regime-adaptive selection with EMA-based stagnation detection."""
        improved_mask = trial_fitness < fitness

        if not hasattr(self, 'gen_improvement_ema'):
            self.gen_improvement_ema = np.zeros(self.np)
            self.pop_improvement_ema_short = 0.0
            self.pop_improvement_long = 0.0
            self.temporal_regime = 'neutral'
            self.regime_patience = 0
            self.stagnation_window = []

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

        window_mean = np.mean(self.stagnation_window)
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

        new_population = population.copy()
        new_fitness = fitness.copy()

        new_population[selection_mask] = trials[selection_mask]
        new_fitness[selection_mask] = trial_fitness[selection_mask]

        return new_population, new_fitness, selection_mask
    
    # ==================================================================
    # STRATEGY 3: Rank-based fallback (variant_04_catD_idea_0)
    # ==================================================================
    def _strategy_rank(self, population, fitness, trials, trial_fitness):
        """Adaptive selection: greedy when improving, rank-based when stagnant."""
        improved_mask_greedy = trial_fitness < fitness
        improvement_rate = np.mean(improved_mask_greedy)

        if not hasattr(self, 'sel_ewma_short'):
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
            improved_mask = selected_indices >= n_pop

            return new_population, new_fitness, improved_mask
        else:
            new_population = population.copy()
            new_fitness = fitness.copy()

            improved_mask = trial_fitness < fitness
            new_population[improved_mask] = trials[improved_mask]
            new_fitness[improved_mask] = trial_fitness[improved_mask]

            return new_population, new_fitness, improved_mask
    
    # ==================================================================
    # STRATEGY 4: Information-theoretic entropy (variant_03_catC_idea_0)
    # ==================================================================
    def _strategy_entropy(self, population, fitness, trials, trial_fitness):
        """Information-theoretic survivor selection using entropy and KL divergence."""
        improved_mask = trial_fitness < fitness

        fitness_flat = fitness.ravel()
        valid_fitness = fitness_flat[~np.isinf(fitness_flat)]
        valid_fitness = valid_fitness[~np.isnan(valid_fitness)]

        if len(valid_fitness) >= 10:
            hist, bin_edges = np.histogram(valid_fitness, bins='auto', density=True)
            hist = np.maximum(hist, 1e-10)
            bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
            population_entropy = -np.sum(hist * np.log(hist)) * bin_width
        else:
            population_entropy = 1.0

        new_population = population.copy()
        new_fitness = fitness.copy()
        accept_mask = np.zeros(len(population), dtype=bool)

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
                    accept_mask[i] = True

        return new_population, new_fitness, accept_mask
    
    # ==================================================================
    # STRATEGY 5: Graph betweenness centrality (variant_05_catE_idea_0)
    # ==================================================================
    def _strategy_centrality(self, population, fitness, trials, trial_fitness):
        """Greedy selection biased by graph betweenness centrality."""
        pop_size, dim = population.shape

        new_population = population.copy()
        new_fitness = fitness.copy()
        improved_mask = np.zeros(pop_size, dtype=bool)

        if not self._centrality_available:
            # Fallback to greedy if scipy unavailable
            improved_mask = trial_fitness < fitness
            new_population[improved_mask] = trials[improved_mask]
            new_fitness[improved_mask] = trial_fitness[improved_mask]
            return new_population, new_fitness, improved_mask

        try:
            from scipy.sparse.csgraph import shortest_path
            from scipy.sparse import csr_matrix

            k = max(2, min(5, pop_size // 4))
            dist_matrix = np.zeros((pop_size, pop_size))
            for i in range(pop_size):
                diffs = population - population[i]
                dist_matrix[i] = np.sqrt(np.sum(diffs * diffs, axis=1))
            np.fill_diagonal(dist_matrix, np.inf)

            knn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
            row_idx = np.repeat(np.arange(pop_size), k)
            col_idx = knn_indices.flatten()
            data = np.ones(len(row_idx))
            adj = csr_matrix((data, (row_idx, col_idx)), shape=(pop_size, pop_size))
            adj = adj + adj.T
            adj = (adj > 0).astype(float)

            dists, preds = shortest_path(adj, directed=False, unweighted=True, return_predecessors=True)
            dists = np.nan_to_num(dists, nan=np.inf)

            n_paths = np.zeros((pop_size, pop_size))
            for start in range(pop_size):
                for end in range(pop_size):
                    if start != end and dists[start, end] < np.inf:
                        current = end
                        while current != start:
                            prev = preds[start, current]
                            if prev < 0:
                                break
                            n_paths[start, prev] += 1
                            n_paths[prev, end] += 1
                            current = prev

            centrality = np.zeros(pop_size)
            for i in range(pop_size):
                for j in range(i + 1, pop_size):
                    if dists[i, j] < np.inf:
                        c_ij = n_paths[i, j] + n_paths[j, i]
                        centrality[i] += c_ij
                        centrality[j] += c_ij
            centrality = centrality / (centrality.max() + 1e-10)

            for i in range(pop_size):
                trial_f = trial_fitness[i]
                parent_f = fitness[i]

                if trial_f < parent_f:
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_f
                    improved_mask[i] = True
                elif np.abs(trial_f - parent_f) < 1e-14:
                    trial_centrality = 0.5
                    if np.random.random() < (centrality[i] - trial_centrality + 0.5):
                        new_population[i] = trials[i]
                        new_fitness[i] = trial_f
                        improved_mask[i] = True

        except Exception:
            # Fallback on any error
            improved_mask = trial_fitness < fitness
            new_population[improved_mask] = trials[improved_mask]
            new_fitness[improved_mask] = trial_fitness[improved_mask]

        return new_population, new_fitness, improved_mask
    
    # ==================================================================
    # ENSEMBLE AGGREGATION
    # ==================================================================
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        ENSEMBLE VOTING: Run all strategies, aggregate decisions via
        weighted majority vote with rank-normalized EWM weights.
        
        This is the adaptive core. Each strategy votes accept/reject for
        each individual. Votes are weighted by exponentially-weighted
        success history, rank-normalized within a sliding window.
        """
        np_pop = len(population)
        
        # Collect acceptance masks from all strategies
        strategy_masks = []
        strategy_n_accepted = []
        
        # Strategy 0: Geometric
        _, _, mask0 = self._strategy_geometric(population, fitness, trials, trial_fitness)
        strategy_masks.append(mask0)
        strategy_n_accepted.append(np.sum(mask0))
        
        # Strategy 1: Spectral
        _, _, mask1 = self._strategy_spectral(population, fitness, trials, trial_fitness)
        strategy_masks.append(mask1)
        strategy_n_accepted.append(np.sum(mask1))
        
        # Strategy 2: Temporal
        _, _, mask2 = self._strategy_temporal(population, fitness, trials, trial_fitness)
        strategy_masks.append(mask2)
        strategy_n_accepted.append(np.sum(mask2))
        
        # Strategy 3: Rank-based
        _, _, mask3 = self._strategy_rank(population, fitness, trials, trial_fitness)
        strategy_masks.append(mask3)
        strategy_n_accepted.append(np.sum(mask3))
        
        # Strategy 4: Entropy
        _, _, mask4 = self._strategy_entropy(population, fitness, trials, trial_fitness)
        strategy_masks.append(mask4)
        strategy_n_accepted.append(np.sum(mask4))
        
        # Strategy 5: Centrality
        _, _, mask5 = self._strategy_centrality(population, fitness, trials, trial_fitness)
        strategy_masks.append(mask5)
        strategy_n_accepted.append(np.sum(mask5))
        
        # ---- Compute weighted vote for each individual ----
        strategy_masks = np.array(strategy_masks, dtype=bool)  # shape: (6, np_pop)
        strategy_n_accepted = np.array(strategy_n_accepted, dtype=float)
        
        # Normalize weights to sum to 1
        weights = self._strategy_weights / (self._strategy_weights.sum() + 1e-10)
        
        # Weighted vote: sum of weights for strategies that accept each individual
        weighted_votes = np.zeros(np_pop)
        for s in range(self._num_strategies):
            weighted_votes += weights[s] * strategy_masks[s].astype(float)
        
        # Ensemble decision: accept if weighted vote >= threshold
        ensemble_accept = weighted_votes >= self._vote_threshold
        
        # ---- Update strategy success tracking (rank-normalized EWM) ----
        # For each strategy, record whether it agreed with the greedy decision
        greedy_mask = trial_fitness < fitness
        
        for s in range(self._num_strategies):
            # Strategy "success" = did it vote the same as greedy?
            # This avoids needing a separate reward signal
            strategy_agree = (strategy_masks[s] == greedy_mask)
            success_rate = np.mean(strategy_agree)
            
            # Update EWM state
            alpha_s = 0.3
            alpha_l = 0.1
            self._strategy_ewm_short[s] = alpha_s * success_rate + (1 - alpha_s) * self._strategy_ewm_short[s]
            self._strategy_ewm_long[s] = alpha_l * success_rate + (1 - alpha_l) * self._strategy_ewm_long[s]
            
            # Record in sliding window for rank-normalization
            self._strategy_recent_wins[s].append(success_rate)
            if len(self._strategy_recent_wins[s]) > self._rank_window_size:
                self._strategy_recent_wins[s].pop(0)
        
        # ---- Rank-normalize within sliding window and update weights ----
        if len(self._strategy_recent_wins[0]) >= 3:
            # Compute mean EWM for each strategy in the window
            window_means = []
            for s in range(self._num_strategies):
                if len(self._strategy_recent_wins[s]) > 0:
                    window_means.append(np.mean(self._strategy_recent_wins[s]))
                else:
                    window_means.append(0.5)
            window_means = np.array(window_means)
            
            # Rank-normalize: convert to ranks, then to normalized scores
            ranks = np.argsort(np.argsort(-window_means)) + 1  # 1 = best
            rank_scores = ranks / (self._num_strategies + 1)  # Normalize to (0,1)
            
            # Blend rank score with current EWM for smooth adaptation
            # EWM_short captures recent dynamics, rank captures relative standing
            blend_alpha = 0.6
            for s in range(self._num_strategies):
                ewm_score = self._strategy_ewm_short[s]
                combined = blend_alpha * ewm_score + (1 - blend_alpha) * rank_scores[s]
                self._strategy_success_rates[s] = combined
            
            # Convert to weights via softmax (temperature controls sharpness)
            temperature = 2.0
            raw_weights = np.array(self._strategy_success_rates)
            exp_weights = np.exp(temperature * (raw_weights - raw_weights.max()))
            new_weights = exp_weights / exp_weights.sum()
            
            # Smooth weight update: don't jump too fast
            adapt_rate = 0.1
            self._strategy_weights = (1 - adapt_rate) * self._strategy_weights + adapt_rate * new_weights
            
            # Ensure positive weights
            self._strategy_weights = np.maximum(self._strategy_weights, 0.01)
            self._strategy_weights /= self._strategy_weights.sum()
        
        # ---- Apply ensemble decision ----
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Use ensemble decision
        new_population[ensemble_accept] = trials[ensemble_accept]
        new_fitness[ensemble_accept] = trial_fitness[ensemble_accept]
        
        # Fallback: if ensemble rejects everything that greedy would accept,
        # use greedy for those individuals (ensures progress)
        greedy_only = greedy_mask & (~ensemble_accept)
        if self._use_greedy_fallback and np.any(greedy_only):
            new_population[greedy_only] = trials[greedy_only]
            new_fitness[greedy_only] = trial_fitness[greedy_only]
            ensemble_accept |= greedy_only
        
        return new_population, new_fitness, ensemble_accept
    
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
            
            # Reset temporal strategy state on restart
            if hasattr(self, 'gen_improvement_ema'):
                self.gen_improvement_ema = np.zeros(self.np)
            if hasattr(self, 'sel_ewma_short'):
                self.sel_ewma_short = 0.1
                self.sel_ewma_long = 0.1
        
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
