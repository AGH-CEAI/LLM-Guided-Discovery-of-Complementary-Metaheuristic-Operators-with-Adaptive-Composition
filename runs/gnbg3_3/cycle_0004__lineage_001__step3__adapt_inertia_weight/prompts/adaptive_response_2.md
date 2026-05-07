```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT over 7 inertia adaptation strategies (mechanism #1)
    
    The gap table is NOT blend-hostile:
      - Maximum Gap(2nd) = 5.6× (Task 5), far below the 100× dispatch-mandatory threshold.
      - All Gap(med) values are <50×, so linear blending is mathematically viable.
    
    Why Thompson Sampling over the alternatives:
      - Win distribution is spread (7/5/3/3/2/2/2 across 7 winners), confirming no single
        operator dominates everywhere. A bandit learns which strategy fits the current landscape.
      - Gaps are small enough that exploration cost is acceptable — we can afford to try
        sub-optimal operators briefly before converging.
      - Thompson Sampling balances exploration/exploitation naturally via posterior sampling,
        without needing an explicit exploration constant (unlike UCB).
      - Rank-based reward shaping ensures scale-independence per calibration rule (a).
    
    The 7 arms correspond to the 7 winning inertia adaptation strategies from the benchmark:
      [spectral_v2, info_theory, fitness_rank, graph_topology, diameter_ratio,
       spectral_v1, original_decay]
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self.current_fitness = None
        
        # --- BANDIT STATE FOR INERTIA ADAPTATION ---
        self._inertia_strategies = [
            'spectral_v2',   # variant_02_catB_idea_0: spectral + effective dim
            'info_theory',   # variant_03_catC_idea_0: KL divergence + entropy
            'fitness_rank',  # variant_04_catD_idea_0: rank IQR + improvement rate
            'graph_topology',# variant_05_catE_idea_0: k-NN connectivity + MST spread
            'diameter_ratio',# variant_09_catA_idea_0: diameter / mean pairwise
            'spectral_v1',   # variant_10_catB_idea_0: condition number only
            'original_decay',# original.py: axis-aligned geometric spread
        ]
        self._n_strategies = len(self._inertia_strategies)
        
        # Beta posterior parameters for Thompson Sampling
        # alpha = wins + 1 (Laplace prior), beta = losses + 1
        self._alpha = {s: 1.0 for s in self._inertia_strategies}
        self._beta = {s: 1.0 for s in self._inertia_strategies}
        
        # Sliding window of rank-based scores per strategy
        self._score_window_size = min(30, max(3 * self._n_strategies, 15))
        self._strategy_scores = {s: [] for s in self._inertia_strategies}
        
        # Current selected strategy
        self._selected_strategy = 'original_decay'  # start with baseline
        
        # Fallback counter per strategy (resets on use)
        self._strategy_fallback = {s: 0 for s in self._inertia_strategies}
        
        # History for rank-based scoring
        self._recent_fitness_history = []
        self._recent_history_size = 10
    
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING BANDIT FOR INERTIA STRATEGY SELECTION
    # -------------------------------------------------------------------------
    
    def _compute_rank_score(self, fitness):
        """Compute rank-based score for the current generation.
        
        Score = where current best falls in the recent fitness distribution.
        Scale-independent: uses relative rank, not absolute fitness.
        """
        if len(self._recent_fitness_history) == 0:
            return 0.5  # neutral
        
        recent_best = min(self._recent_fitness_history)
        recent_worst = max(self._recent_fitness_history)
        
        if recent_worst <= recent_best:
            return 1.0  # perfect
        
        # Normalize: 1.0 = best in window, 0.0 = worst in window
        score = (recent_worst - fitness) / (recent_worst - recent_best + 1e-10)
        return float(np.clip(score, 0.0, 1.0))
    
    def _update_bandit(self, strategy, fitness):
        """Update bandit posterior with rank-based reward.
        
        Uses sliding window + Beta posterior for Thompson Sampling.
        """
        # Add to sliding window
        self._strategy_scores[strategy].append(fitness)
        if len(self._strategy_scores[strategy]) > self._score_window_size:
            self._strategy_scores[strategy].pop(0)
        
        # Compute rank score within strategy's window
        if len(self._strategy_scores[strategy]) >= 3:
            window = self._strategy_scores[strategy]
            w_best = min(window)
            w_worst = max(window)
            if w_worst > w_best:
                rank_in_window = (w_worst - fitness) / (w_worst - w_best + 1e-10)
            else:
                rank_in_window = 1.0
        else:
            rank_in_window = 0.5
        
        # Update Beta posterior: win if rank > 0.5 (top half), else loss
        if rank_in_window > 0.5:
            self._alpha[strategy] += 1.0
        else:
            self._beta[strategy] += 1.0
        
        # Reset fallback counter on use
        self._strategy_fallback[strategy] = 0
    
    def _select_strategy_thompson(self):
        """Select strategy using Thompson Sampling from Beta posteriors."""
        samples = {}
        for s in self._inertia_strategies:
            # Sample from Beta distribution
            a = max(self._alpha[s], 1e-10)
            b = max(self._beta[s], 1e-10)
            samples[s] = np.random.beta(a, b)
        
        # Pick strategy with highest sample
        selected = max(samples, key=samples.get)
        
        # Fallback: if a strategy hasn't been used in a while, boost its sample
        for s in self._inertia_strategies:
            self._strategy_fallback[s] += 1
        
        max_fallback = max(self._strategy_fallback.values())
        if max_fallback > 3 * self._n_strategies:
            # Force selection of least-used strategy
            least_used = min(self._strategy_fallback, key=self._strategy_fallback.get)
            selected = least_used
            self._strategy_fallback = {s: 0 for s in self._inertia_strategies}
        
        return selected
    
    # -------------------------------------------------------------------------
    # INERTIA ADAPTATION STRATEGIES (the 7 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _adapt_inertia_weight(self):
        """Dispatch to the bandit-selected inertia adaptation strategy."""
        strategy = self._selected_strategy
        
        if strategy == 'spectral_v2':
            self._inertia_spectral_v2()
        elif strategy == 'info_theory':
            self._inertia_info_theory()
        elif strategy == 'fitness_rank':
            self._inertia_fitness_rank()
        elif strategy == 'graph_topology':
            self._inertia_graph_topology()
        elif strategy == 'diameter_ratio':
            self._inertia_diameter_ratio()
        elif strategy == 'spectral_v1':
            self._inertia_spectral_v1()
        else:  # original_decay
            self._inertia_original_decay()
    
    def _inertia_spectral_v2(self):
        """Variant 02: spectral + effective dimensionality."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-14, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_normalized = np.clip(cond / 500.0, 0.0, 1.0)

            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = np.clip(eff_dim / self.dim, 0.0, 1.0)

            spectral_explore_signal = cond_normalized * 0.6 + (1.0 - eff_dim_ratio) * 0.4
            base_inertia = 0.4 + 0.35 * eff_dim_ratio
            target_inertia = base_inertia * (1.0 - 0.45 * spectral_explore_signal)

            decay = 0.729 - 0.08 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            fallback = 0.729 - 0.08 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + fallback * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_info_theory(self):
        """Variant 03: KL divergence + entropy (information-theoretic)."""
        f_min = np.min(self.current_fitness)
        p_fit = self.current_fitness - f_min + 1e-10
        p_fit = p_fit / (np.sum(p_fit) + 1e-10)

        entropy_fit = -np.sum(p_fit * np.log(p_fit + 1e-10))
        max_entropy = np.log(self.np) if self.np > 1 else 1.0
        normalized_entropy_fit = np.clip(entropy_fit / (max_entropy + 1e-10), 0.0, 1.0)

        kl_div = 0.0
        n_bins = min(10, max(3, self.np // 5))

        for d in range(self.dim):
            hist, _ = np.histogram(self.population[:, d], bins=n_bins, density=True)
            hist = np.clip(hist, 1e-10, None)
            uniform_prob = 1.0 / n_bins
            kl_div += np.sum(hist * np.log(hist / (uniform_prob + 1e-10)))

        kl_normalized = np.clip(kl_div / 5.0, 0.0, 1.0)

        exploration_signal = normalized_entropy_fit * (1.0 - kl_normalized)
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

        target_inertia = 0.95 - 0.55 * exploration_signal

        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_fitness_rank(self):
        """Variant 04: rank IQR + improvement rate."""
        sorted_indices = np.argsort(self.current_fitness)
        ranks = np.zeros(self.np)
        ranks[sorted_indices] = np.arange(self.np) / max(1, self.np - 1)

        q75_idx = int(0.75 * (self.np - 1))
        q25_idx = int(0.25 * (self.np - 1))
        sorted_ranks = np.sort(ranks)
        rank_iqr = sorted_ranks[q75_idx] - sorted_ranks[q25_idx]
        rank_iqr = np.clip(rank_iqr, 0.01, 1.0)

        rank_spread = rank_iqr / 0.5

        if not hasattr(self, '_prev_fitness_for_inertia'):
            self._prev_fitness_for_inertia = self.current_fitness.copy()

        prev_fitness = self._prev_fitness_for_inertia
        improved_mask = self.current_fitness < prev_fitness
        improvement_rate = np.sum(improved_mask) / max(1, self.np)

        self._prev_fitness_for_inertia = self.current_fitness.copy()

        rank_spread_clamped = np.clip(rank_spread, 0.1, 2.0)
        improvement_clamped = np.clip(improvement_rate, 0.01, 1.0)

        target_inertia = 0.4 + 0.55 * (rank_spread_clamped * improvement_clamped)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_graph_topology(self):
        """Variant 05: k-NN connectivity + MST spread."""
        npops = self.np
        k = min(5, npops - 1)

        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)

        knn_indices = np.argsort(sq_dists, axis=1)[:, 1:k+1]
        knn_dists = np.take_along_axis(sq_dists, knn_indices, axis=1)

        threshold = knn_dists[:, -1:]
        connection_counts = np.sum(sq_dists[:, 1:] <= threshold, axis=1)
        connectivity_ratio = np.mean(connection_counts >= (k // 2 + 1))

        candidate_edges = []
        candidate_weights = []
        for i in range(npops):
            for j in knn_indices[i]:
                if i < j:
                    candidate_edges.append((i, j))
                    candidate_weights.append(np.sqrt(sq_dists[i, j]))

        if len(candidate_weights) >= npops - 1:
            sorted_edge_indices = np.argsort(candidate_weights)
            parent = np.arange(npops)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            mst_total_weight = 0.0
            edges_added = 0
            for idx in sorted_edge_indices:
                if edges_added >= npops - 1:
                    break
                i, j = candidate_edges[idx]
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    mst_total_weight += candidate_weights[idx]
                    edges_added += 1

            mst_avg_edge = mst_total_weight / max(edges_added, 1)
        else:
            mst_avg_edge = np.mean(knn_dists)

        pop_scale = np.mean(np.std(self.population, axis=0)) + 1e-10
        mst_spread_normalized = np.clip(mst_avg_edge / (pop_scale * 3.0), 0.0, 1.0)

        connectivity_component = 0.4 + 0.4 * connectivity_ratio
        spread_component = 0.95 - 0.4 * mst_spread_normalized

        target_inertia = connectivity_component * 0.6 + spread_component * 0.4
        target_inertia = np.clip(target_inertia, 0.4, 0.95)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_diameter_ratio(self):
        """Variant 09: diameter / mean pairwise distance."""
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, -np.inf)
        diameter = np.sqrt(np.max(sq_dists)) + 1e-10

        upper_sq = np.triu(sq_dists, k=1)
        mean_pairwise_sq = np.sum(upper_sq) / max(1, self.np * (self.np - 1) / 2)
        mean_pairwise = np.sqrt(max(mean_pairwise_sq, 0)) + 1e-10

        spread_ratio = diameter / (mean_pairwise + 1e-10)
        spread_ratio_normalized = np.clip(spread_ratio / 5.0, 0.0, 1.0)

        target_inertia = spread_ratio_normalized * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_spectral_v1(self):
        """Variant 10: condition number only."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * spectral_signal

            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _inertia_original_decay(self):
        """Original: axis-aligned geometric spread ratio."""
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10

        geo_mean = np.exp(np.mean(np.log(spread)))
        arith_mean = np.mean(spread)
        spread_ratio = geo_mean / (arith_mean + 1e-10)
        spread_ratio_normalized = np.clip(spread_ratio / 0.9, 0.0, 1.0)

        target_inertia = (1.0 - spread_ratio_normalized) * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    def _velocity_update_base(self):
        """Base velocity update: cognitive + social + DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _position_update(self):
        """Position update: standard PSO with inertia + ring topology."""
        new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            if self.global_best is not None:
                perturbation = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + perturbation)
                self.personal_best[0] = self.population[0]
                self.personal_best_fitness[0] = np.inf
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with Thompson Sampling bandit for inertia adaptation.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Reset bandit state
        self._alpha = {s: 1.0 for s in self._inertia_strategies}
        self._beta = {s: 1.0 for s in self._inertia_strategies}
        self._strategy_scores = {s: [] for s in self._inertia_strategies}
        self._strategy_fallback = {s: 0 for s in self._inertia_strategies}
        self._recent_fitness_history = []
        self._selected_strategy = 'original_decay'
        
        # Reset strategy-specific state
        if hasattr(self, '_prev_fitness_for_inertia'):
            del self._prev_fitness_for_inertia
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Update recent history
        gen_best = float(np.min(self.current_fitness))
        self._recent_fitness_history.append(gen_best)
        if len(self._recent_fitness_history) > self._recent_history_size:
            self._recent_fitness_history.pop(0)
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Select inertia strategy via Thompson Sampling
            self._selected_strategy = self._select_strategy_thompson()
            
            # Inertia adaptation (bandit-selected strategy)
            self._adapt_inertia_weight()
            
            # Topology adaptation
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity and position update
            self._velocity_update_base()
            self._position_update()
            
            # Evaluation
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            self._restart_if_stagnant()
            
            # Update recent history for rank scoring
            gen_best = float(np.min(self.current_fitness))
            self._recent_fitness_history.append(gen_best)
            if len(self._recent_fitness_history) > self._recent_history_size:
                self._recent_fitness_history.pop(0)
            
            # Update bandit with rank-based reward
            rank_score = self._compute_rank_score(gen_best)
            self._update_bandit(self._selected_strategy, gen_best)
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```