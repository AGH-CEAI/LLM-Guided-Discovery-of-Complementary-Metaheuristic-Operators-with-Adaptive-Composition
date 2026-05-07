import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTIVE CROSSOVER SELECTION via PROBE-AND-COMMIT (mechanism #4).
    
    Per-task gap table analysis:
      - 6 tasks with gap >= 100x (DISPATCH-MANDATORY)
      - 10 tasks with gap >= 10x (BLEND-HOSTILE)
      - 5 distinct winners across non-trivial tasks
    
    Mechanism: A short exploratory probe (~5% of budget) tests all 5
    crossover operators on the ACTUAL landscape. The operator that wins
    the probe (rank-based success rate) is COMMITTED for the remaining
    budget with weight 1.0. This preserves per-task winner advantages
    because no blending occurs — a 50/50 blend of values 1 and 100 is
    50.5, not 1. This is a contextual bandit where context = cheap
    landscape fingerprint computed once at the start.
    
    Operators mapped from winning variants:
      0: spectral_crossover  (variant_02, 6 wins)  — PCA eigendecomposition
      1: ema_crossover       (variant_06, 6 wins)  — temporal EMA-driven CR
      2: graph_crossover     (variant_05, 3 wins)  — k-NN graph topology
      3: geometry_crossover  (variant_01, 1 win)   — axis-aligned spread
      4: binomial_crossover  (original, 6 wins)    — classic DE/binomial
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool with softmax-converted success history
        self.strategy_names = ['rand', 'best', 'current_to_rand_best', 'local_ring', 'two_rail']
        self.strategy_scores = np.ones(len(self.strategy_names))
        self._strategy_funcs = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # ── CROSSOVER OPERATOR REGISTRY ──────────────────────────────────────
        # All 5 operators are implemented as methods; the probe selects ONE.
        self._crossover_operators = [
            self._crossover_spectral,
            self._crossover_ema,
            self._crossover_graph,
            self._crossover_geometry,
            self._crossover_binomial,
        ]
        self._crossover_names = [
            'spectral', 'ema', 'graph', 'geometry', 'binomial'
        ]
        self._committed_operator = None   # Set after probe
        self._probe_results = None        # For diagnostics
        
    # ══════════════════════════════════════════════════════════════════════════
    # CROSSOVER OPERATORS (from winning variants)
    # ══════════════════════════════════════════════════════════════════════════
    
    def _crossover_spectral(self, population, mutants):
        """Spectral crossover — variant_02_catB (6 wins). PCA eigendecomposition."""
        NP, dim = population.shape
        centroid = population.mean(axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / NP

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return np.where(
                np.random.rand(NP, dim) < self.CR,
                mutants,
                population
            )

        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = np.clip(eigenvalues, 1e-12, None)

        pop_pc = centered @ eigenvectors
        mut_pc = (mutants - centroid) @ eigenvectors

        total_eig = eigenvalues.sum()
        eig_weights = eigenvalues / total_eig if total_eig > 0 else np.ones(dim) / dim
        base_cr = np.clip(self.CR * np.power(eig_weights, 0.1), 0.01, 0.99)

        crossover_mask = np.random.rand(NP, dim) < base_cr
        if dim > 1:
            force_cross_idx = np.random.randint(0, dim, size=NP)
            crossover_mask[np.arange(NP), force_cross_idx] = True

        trial_pc = np.where(crossover_mask, mut_pc, pop_pc)
        trials = trial_pc @ eigenvectors.T + centroid
        return trials

    def _crossover_ema(self, population, mutants):
        """EMA-driven crossover — variant_06_catF (6 wins). Temporal CR adaptation."""
        NP, dim = population.shape

        if not hasattr(self, '_ema_crossover_success'):
            self._ema_crossover_success = 0.5
            self._ema_fitness_change = 0.0
            self._prev_mean_fitness = None
            self._crossover_success_history = []
            self._CR_ema = self.CR
            self._momentum_buffer = 0.0

        current_mean_fitness = population.mean(axis=0).sum()
        if self._prev_mean_fitness is not None:
            fitness_drift = current_mean_fitness - self._prev_mean_fitness
            alpha_drift = 0.2
            self._ema_fitness_change = (1 - alpha_drift) * self._ema_fitness_change + alpha_drift * fitness_drift
        self._prev_mean_fitness = current_mean_fitness

        target_correction = 0.0
        success_rate = self._ema_crossover_success
        if success_rate > 0.4:
            target_correction -= 0.08 * (success_rate - 0.4)
        else:
            target_correction += 0.12 * (0.4 - success_rate)

        drift_magnitude = abs(self._ema_fitness_change)
        if drift_magnitude > 1e-6:
            target_correction += 0.05 * np.sign(drift_magnitude)

        self._momentum_buffer = 0.7 * self._momentum_buffer + 0.3 * target_correction
        self._CR_ema = np.clip(self._CR_ema + self._momentum_buffer, 0.1, 0.98)

        individual_CR = np.random.uniform(
            max(0.1, self._CR_ema - 0.15),
            min(0.98, self._CR_ema + 0.15),
            size=NP
        )

        r_dim = np.random.randint(0, dim, size=NP)
        mask = np.zeros((NP, dim), dtype=bool)
        for i in range(NP):
            mask[i, r_dim[i]] = True
            n_mutate = int(np.ceil(individual_CR[i] * dim))
            other_dims = np.random.choice(dim, size=n_mutate, replace=False)
            mask[i, other_dims] = True

        trials = np.where(mask, mutants, population)
        self.CR = self._CR_ema
        return trials

    def _crossover_graph(self, population, mutants):
        """Graph-distance-adaptive crossover — variant_05_catE (3 wins). k-NN topology."""
        NP, dim = population.shape
        k = max(2, min(5, NP // 4))

        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        graph_dist = np.zeros(NP)
        for i in range(NP):
            mutant = mutants[i]
            target = population[i]
            direct_dist = np.linalg.norm(mutant - target)
            mutant_neighbors = set(knn_indices[i])

            second_order = set()
            for neighbor in mutant_neighbors:
                second_order.update(knn_indices[neighbor])
            second_order.discard(i)
            second_order = second_order - mutant_neighbors

            if len(second_order) == 0:
                graph_penalty = 1.5
            else:
                connectivity_factor = len(second_order) / float(k * k)
                graph_penalty = 1.0 / (1.0 + connectivity_factor)

            graph_dist[i] = direct_dist * graph_penalty

        max_dist = graph_dist.max()
        norm_dist = graph_dist / max_dist if max_dist > 1e-10 else np.zeros(NP)

        adaptive_CR = self.CR * (1.0 - 0.3 * norm_dist)
        adaptive_CR = np.clip(adaptive_CR, 0.1, 0.95)

        trials = np.empty_like(population)
        jr = np.random.randint(0, dim, size=NP)

        for i in range(NP):
            cr = adaptive_CR[i]
            mask = np.random.rand(dim) < cr
            mask[jr[i]] = True
            trials[i] = np.where(mask, mutants[i], population[i])

        return trials

    def _crossover_geometry(self, population, mutants):
        """Geometry-driven crossover — variant_01_catA (1 win). Axis-aligned spread."""
        NP, dim = population.shape
        dim_min = population.min(axis=0)
        dim_max = population.max(axis=0)
        spread = dim_max - dim_min
        spread = np.where(spread < 1e-10, 1e-10, spread)
        cr_per_dim = spread / spread.sum()
        rand_matrix = np.random.rand(NP, dim)
        mask = rand_matrix < cr_per_dim[np.newaxis, :]
        no_mutant_dims = ~mask.any(axis=1)
        forced_dims = np.random.randint(0, dim, size=no_mutant_dims.sum())
        mask[no_mutant_dims, forced_dims] = True
        trials = np.where(mask, mutants, population)
        return trials

    def _crossover_binomial(self, population, mutants):
        """Classic binomial crossover — original.py (6 wins)."""
        NP, dim = population.shape
        CR = np.clip(self.CR + np.random.randn(NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(NP, dim) < CR[:, np.newaxis]
        mask[np.arange(NP), np.random.randint(0, dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)

    # ══════════════════════════════════════════════════════════════════════════
    # PROBE-AND-COMMIT DISPATCHER
    # ══════════════════════════════════════════════════════════════════════════

    def _run_crossover_probe(self, func, budget):
        """
        Probe phase: test all 5 crossover operators on the actual landscape.
        Uses rank-based success rate to avoid scale-dependent rewards.
        Returns the index of the winning operator.
        """
        num_operators = len(self._crossover_operators)
        # Probe with a smaller sub-population for speed
        probe_NP = max(4 * self.dim, 20)
        probe_NP = min(probe_NP, self.NP)

        # Initialize probe population
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (probe_NP, self.dim))
        pop += np.random.randn(probe_NP, self.dim) * 5.0
        pop = np.clip(pop, low, high)
        fitness = func(pop)

        # Track per-operator statistics using RANK-based improvement
        # (not raw fitness — avoids scale dependence)
        operator_success_counts = np.zeros(num_operators)
        operator_trial_counts = np.zeros(num_operators)
        
        # Sliding window of best-fitness ranks per operator for z-scoring
        operator_rank_history = [[] for _ in range(num_operators)]

        probe_gens = max(3, int(budget))
        
        for gen in range(probe_gens):
            # Round-robin: each operator gets a turn this generation
            op_idx = gen % num_operators
            
            # Generate mutants using compass mutation (consistent across operators)
            fitness_ranks = self._compute_fitness_ranking(fitness)
            mutants = np.empty_like(pop)
            for i in range(probe_NP):
                mutants[i] = self._compass_directions(pop, i, fitness_ranks)

            # Apply the current operator
            crossover_func = self._crossover_operators[op_idx]
            trials = crossover_func(pop, mutants)
            trials = np.clip(trials, low, high)
            trial_fitness = func(trials)

            # Selection
            improved = trial_fitness < fitness
            pop = np.where(improved[:, np.newaxis], trials, pop)
            fitness = np.where(improved, trial_fitness, fitness)

            # Record rank-based improvement signal for this operator
            # Rank improvement: how much did the best rank improve?
            current_best_rank = np.argmin(fitness) / max(probe_NP - 1, 1)
            operator_rank_history[op_idx].append(current_best_rank)
            if len(operator_rank_history[op_idx]) > 5:
                operator_rank_history[op_idx].pop(0)

            # Count successful improvements for this operator
            if improved.any():
                operator_success_counts[op_idx] += improved.sum()
            operator_trial_counts[op_idx] += probe_NP

        # Compute rank-based success rate per operator
        # Use z-score normalization across the window to avoid scale magic
        operator_scores = np.zeros(num_operators)
        for op in range(num_operators):
            if operator_trial_counts[op] > 0:
                raw_rate = operator_success_counts[op] / operator_trial_counts[op]
            else:
                raw_rate = 0.0
            
            # Z-score over recent history for this operator
            hist = operator_rank_history[op]
            if len(hist) >= 3:
                mean_rank = np.mean(hist)
                std_rank = np.std(hist) + 1e-8
                z_score = (hist[-1] - mean_rank) / std_rank
                # Combine raw rate with z-score: higher z = better rank improvement
                operator_scores[op] = raw_rate + 0.2 * z_score
            else:
                operator_scores[op] = raw_rate

        # Commit to the operator with the highest score
        winner = int(np.argmax(operator_scores))
        
        # Store probe results for diagnostics
        self._probe_results = {
            'operator_scores': operator_scores.copy(),
            'operator_names': self._crossover_names,
            'winner': winner,
            'winner_name': self._crossover_names[winner],
        }
        
        return winner

    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL METHODS (unchanged from AdaptiveCompassDE)
    # ══════════════════════════════════════════════════════════════════════════

    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(fitness))
        return ranks / max(len(fitness) - 1, 1)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        i_rank = fitness_ranks[i]
        
        candidates = []
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
        p = 0.1
        top_p = int(np.ceil(p * self.NP))
        pbest_candidates = np.argsort(fitness_ranks)[:top_p]
        pbest = population[np.random.choice(pbest_candidates)]
        dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
        score_C = i_rank - fitness_ranks[pbest_candidates].min()
        
        candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
        best_dir, _ = max(candidates, key=lambda x: x[1])
        return best_dir
    
    def _mutate_compass_batch(self, population, fitness):
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        for i in range(self.NP):
            mutants[i] = self._compass_directions(population, i, fitness_ranks)
        return mutants
    
    def _mutate_rand(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_best(self, population, idx, F):
        best_idx = np.argmin(population.sum(axis=1))
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 2, replace=False)]
        return population[best_idx] + F * (population[r[0]] - population[r[1]])
    
    def _mutate_current_to_rand_best(self, population, idx, F):
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[i] + F * (population[r[0]] - population[i]) + F * (population[r[1]] - population[r[2]])
    
    def _mutate_local_ring(self, population, idx, F):
        left = (idx - 2) % self.NP
        right = (idx + 2) % self.NP
        others = [left, (idx - 1) % self.NP, (idx + 1) % self.NP, right]
        r = np.array(others)[np.random.choice(4, 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_two_rail(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[r[0]] + population[r[1]] + F * (population[r[2]] - population[r[3]]) - population[idx]
    
    def _select_strategy_batch(self):
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        return mutants, strategy_used
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        n_strategies = len(self.strategy_scores)
        n = len(strategy_used)

        if not hasattr(self, '_ema_rank_improvement'):
            self._ema_rank_improvement = np.zeros(n_strategies)
            self._ema_success_rate = np.zeros(n_strategies)
            self._rank_improvement_history = [[] for _ in range(n_strategies)]
            self._success_rate_history = [[] for _ in range(n_strategies)]
            self._momentum = 0.3

        rank_improvement = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()

        success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                success_rate[s] = improved[mask].sum() / mask.sum()

        for s in range(n_strategies):
            self._rank_improvement_history[s].append(rank_improvement[s])
            self._success_rate_history[s].append(success_rate[s])
            if len(self._rank_improvement_history[s]) > 10:
                self._rank_improvement_history[s].pop(0)
            if len(self._success_rate_history[s]) > 10:
                self._success_rate_history[s].pop(0)

        alpha = 0.3
        for s in range(n_strategies):
            self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
            self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]

        rank_var = np.zeros(n_strategies)
        success_var = np.zeros(n_strategies)
        for s in range(n_strategies):
            if len(self._rank_improvement_history[s]) >= 3:
                rank_var[s] = np.var(self._rank_improvement_history[s])
            if len(self._success_rate_history[s]) >= 3:
                success_var[s] = np.var(self._success_rate_history[s])

        eps = 1e-6
        rank_weight = np.where(rank_var > eps, 1.0 / (rank_var + eps), 1.0 / eps)
        success_weight = np.where(success_var > eps, 1.0 / (success_var + eps), 1.0 / eps)

        total_rank_weight = rank_weight.sum()
        total_success_weight = success_weight.sum()
        if total_rank_weight > 0:
            rank_weight /= total_rank_weight
        if total_success_weight > 0:
            success_weight /= total_success_weight

        hybrid_signal = 0.5 * rank_weight * self._ema_rank_improvement + 0.5 * success_weight * self._ema_success_rate

        mean_signal = hybrid_signal.mean()
        if mean_signal > 0:
            comparative_advantage = (hybrid_signal - mean_signal) / (mean_signal + eps)
        else:
            comparative_advantage = np.zeros(n_strategies)

        combined_var = rank_var + success_var
        confidence = np.exp(-combined_var * 5)

        delta = np.zeros(n_strategies)
        for s in range(n_strategies):
            delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])

        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _adapt_parameters(self, improvement_rate):
        if not hasattr(self, '_graph_population'):
            self._graph_population = None
            self._graph_metrics_history = []

        if not hasattr(self, '_current_population'):
            return
        pop = self._current_population
        if pop is None or len(pop) < 5:
            return

        NP, dim = pop.shape
        k = max(2, min(5, NP // 10))
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
        avg_edge_length = np.mean(np.sqrt(nearest_sq_dists))

        clustering_coeffs = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest_indices[i])
            if len(neighbors) < 2:
                clustering_coeffs[i] = 0.0
                continue
            edges_among_neighbors = 0
            possible_edges = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj:
                        possible_edges += 1
                        if nj in set(nearest_indices[ni]):
                            edges_among_neighbors += 1
            clustering_coeffs[i] = edges_among_neighbors / max(possible_edges, 1)

        avg_clustering = np.mean(clustering_coeffs)
        graph_diameter = np.max(np.sqrt(nearest_sq_dists))

        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
            'graph_diameter': graph_diameter
        }
        self._graph_metrics_history.append(current_metrics)

        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)

        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]
            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
            clustering_trend = recent_clustering[-1] - recent_clustering[0]
        else:
            edge_length_trend = 0.0
            clustering_trend = 0.0

        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * (edge_length_trend < -0.01)
        clustering_factor += 0.08 * (clustering_trend > 0.05)
        self.F = np.clip(self.F * clustering_factor, 0.3, 2.0)

        median_edge = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        if avg_edge_length > median_edge * 0.5:
            self.CR = np.clip(self.CR * 1.08, 0.1, 0.95)
        else:
            self.CR = np.clip(self.CR * 0.93, 0.1, 0.95)

        if improvement_rate > 0.15:
            self.F = np.clip(self.F * 1.05, 0.3, 2.0)
        elif improvement_rate < 0.03:
            self.F = np.clip(self.F * 1.12, 0.3, 2.0)

        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    def _compute_diversity(self, population):
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _update_archive(self, population, fitness, improved_mask):
        improved_pop = population[improved_mask]
        if len(improved_pop) > 0:
            self.archive.append(improved_pop)
            total_len = sum(len(a) for a in self.archive)
            while total_len > self.archive_max:
                oldest = self.archive.pop(0)
                total_len -= len(oldest)
    
    def _inject_diversity(self, population, fitness):
        if len(self.archive) < 2:
            return population
        archive_concat = np.vstack(self.archive)
        if len(archive_concat) < self.NP // 4:
            return population
        
        n_replace = max(self.NP // 5, 2)
        worst_indices = np.argsort(fitness)[-n_replace:]
        
        for idx in worst_indices:
            if np.random.rand() < 0.5 and len(archive_concat) > 0:
                donor_idx = np.random.randint(len(archive_concat))
                donor = archive_concat[donor_idx]
                noise = np.random.randn(self.dim) * 10.0
                population[idx] = np.clip(donor + noise, -100.0, 100.0)
            else:
                population[idx] = np.random.uniform(-100.0, 100.0, self.dim)
        
        return population
    
    def _check_stagnation(self, best_fitness):
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
    def _restart_if_needed(self, population, fitness):
        NP, dim = population.shape
        k = max(2, min(5, NP // 8))
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]

        connectivity = np.zeros(NP)
        for i in range(NP):
            neighbors = nearest_indices[i]
            for n in neighbors:
                connectivity[n] += 1

        clustering = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest_indices[i])
            if len(neighbors) < 2:
                clustering[i] = 0.0
                continue
            edges = 0
            possible = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj:
                        possible += 1
                        if nj in set(nearest_indices[ni]):
                            edges += 1
            clustering[i] = edges / max(possible, 1)

        fitness_ranks = self._compute_fitness_ranking(fitness)
        isolation_score = (1.0 - connectivity / (connectivity.max() + 1e-10)) * 0.5 + \
                          (1.0 - clustering) * 0.3 + \
                          fitness_ranks * 0.2

        n_replace = max(NP // 4, 3)
        replace_indices = np.argsort(isolation_score)[-n_replace:]
        n_best = min(5, NP)
        best_indices = np.argsort(fitness)[:n_best]

        new_individuals = []
        for idx in replace_indices:
            best_source = None
            best_graph_dist = -1
            for b_idx in best_indices:
                n1 = set(nearest_indices[idx])
                n2 = set(nearest_indices[b_idx])
                common = len(n1 & n2)
                graph_dist_approx = 1.0 / (common + 0.1)
                if graph_dist_approx > best_graph_dist:
                    best_graph_dist = graph_dist_approx
                    best_source = b_idx

            if best_source is None:
                best_source = best_indices[0]

            source = population[best_source]
            direction = population[best_source] - population[idx]
            mutation_scale = np.random.uniform(0.8, 1.5)
            local_edges = sq_dists[idx, nearest_indices[idx]]
            avg_local_dist = np.mean(np.sqrt(local_edges))

            new_point = source + mutation_scale * self.F * direction + \
                        np.random.randn(dim) * avg_local_dist * 0.5
            new_individuals.append(np.clip(new_point, -100.0, 100.0))

        for i, idx in enumerate(replace_indices):
            population[idx] = new_individuals[i]

        best_idx = np.argmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        self.stagnation_counter = 0

        return population, f_opt, x_opt

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CALL — PROBE-AND-COMMIT ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def __call__(self, func, stopping_condition):
        """
        Run optimization with PROBE-AND-COMMIT crossover selection.
        
        PHASE A (PROBE): ~5% of budget — test all 5 crossover operators.
        PHASE B (COMMIT): remaining ~95% — use the winning operator exclusively.
        
        The committed operator gets weight 1.0 (no blending), preserving
        per-task winner advantages up to 10000x as seen in the gap table.
        """
        # ── PHASE A: PROBE ────────────────────────────────────────────────────
        # Estimate total budget from stopping_condition behavior
        # We use a conservative probe: 5 generations, round-robin over 5 operators
        probe_budget = max(3, int(self.max_iter * 0.05))
        
        self._committed_operator = self._run_crossover_probe(func, probe_budget)
        
        # ── PHASE B: COMMIT ───────────────────────────────────────────────────
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        # Reset EMA state for the committed operator (clean slate)
        if hasattr(self, '_ema_crossover_success'):
            delattr(self, '_ema_crossover_success')
        if hasattr(self, '_ema_fitness_change'):
            delattr(self, '_ema_fitness_change')
        if hasattr(self, '_prev_mean_fitness'):
            delattr(self, '_prev_mean_fitness')
        if hasattr(self, '_CR_ema'):
            delattr(self, '_CR_ema')
        if hasattr(self, '_momentum_buffer'):
            delattr(self, '_momentum_buffer')
        
        iteration = 0
        committed_op_stagnation = 0  # Track stagnation of committed operator
        
        while not stopping_condition():
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            # ── COMMITTED CROSSOVER OPERATOR (weight = 1.0) ──────────────────
            crossover_func = self._crossover_operators[self._committed_operator]
            trials = crossover_func(population, mutants)
            trials = np.clip(trials, -100.0, 100.0)
            # ─────────────────────────────────────────────────────────────────
            
            # Evaluate
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.full(len(trials), np.nan)
            
            # Selection
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
                committed_op_stagnation = 0
            else:
                committed_op_stagnation += 1
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            self._adapt_parameters(improvement_rate)
            
            # Diversity injection
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check — with ε-switch to runner-up operator
            if self._check_stagnation(f_opt):
                # ε-greedy switch: small probability of trying runner-up
                if committed_op_stagnation > 30 and np.random.rand() < 0.05:
                    # Find runner-up (second-best from probe)
                    probe_scores = self._probe_results['operator_scores']
                    sorted_ops = np.argsort(probe_scores)[::-1]
                    runner_up = sorted_ops[1] if len(sorted_ops) > 1 else self._committed_operator
                    
                    # Commit to runner-up temporarily
                    self._committed_operator = runner_up
                    committed_op_stagnation = 0
                    
                    # Reset EMA state for new operator
                    for attr in ['_ema_crossover_success', '_ema_fitness_change', 
                                 '_prev_mean_fitness', '_CR_ema', '_momentum_buffer']:
                        if hasattr(self, attr):
                            delattr(self, attr)
                else:
                    population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                    fitness = func(population)
                    if len(fitness) < self.NP:
                        fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
