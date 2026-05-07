```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: PROBE-AND-COMMIT (mechanism #4 / contextual bandit).
    
    BLEND-HOSTILE TABLE JUSTIFICATION:
    The per-task gap table shows 6 tasks with winner beats runner-up by >=100×
    (tasks 1, 5, 6, 12, 16, 17). Linear blending of two values where one is 100×
    better yields a result ~50× worse than the winner — ensemble blending is
    mathematically excluded. The 5 distinct winners (original, variant_01,
    variant_02, variant_05, variant_06) with large gaps mandate a dispatch-style
    mechanism that commits to ONE crossover operator at a time.
    
    PROBE-AND-COMMIT PATTERN:
    PHASE A (PROBE, ~5-8% of budget): Run each of 5 crossover strategies in
    round-robin fashion for a short burst. Extract a cheap landscape fingerprint
    (convergence slope, effective dimensionality, diversity ratio). Record which
    strategy achieved the best rank-normalized improvement.
    
    PHASE B (COMMIT): Use the probe-winning strategy exclusively for the
    remaining budget. This preserves 100% of the per-task winner's advantage.
    
    Fallback: If probe is inconclusive (all strategies within 5% of each other),
    default to the original binomial crossover (original.py was a strong 6-win
    variant). After commit, use a small stagnation-override: if the committed
    strategy stagnates for >30 generations, allow one re-probe with the same
    short budget.
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
        
        # ============================================================
        # CROSSOVER ADAPTATION STATE (PROBE-AND-COMMIT)
        # ============================================================
        # Probe parameters
        self._probe_budget_pct = 0.06  # 6% of budget for probing
        self._probe_min_gens = 4       # Minimum probe generations
        self._probe_max_gens = 12      # Maximum probe generations
        
        # Probe tracking: per-crossover-strategy performance
        self._crossover_arm_names = [
            'original',      # variant: standard binomial (baseline)
            'catA_spread',   # variant_01: geometry-driven spread
            'catB_spectral', # variant_02: spectral PCA
            'catE_knn',      # variant_05: k-NN graph distance
            'catF_temporal'  # variant_06: EMA temporal adaptation
        ]
        self._n_arms = len(self._crossover_arm_names)
        
        # Probe state
        self._probe_active = True
        self._probe_start_iter = 0
        self._probe_arm_trial_count = np.zeros(self._n_arms)  # How many times each arm was tried
        self._probe_arm_improvement = np.zeros(self._n_arms)  # Sum of rank-normalized improvements
        self._probe_arm_best_fitness = np.full(self._n_arms, np.inf)  # Best fitness per arm
        self._probe_consecutive_no_improve = 0
        
        # Commit state
        self._committed_arm = None  # Index of chosen arm
        self._commit_start_iter = 0
        self._stagnation_override_threshold = 35  # Generations before allowing re-probe
        self._stagnation_since_commit = 0
        self._allow_reprobe = True
        
        # Landscape fingerprint (computed from probe data)
        self._fingerprint = None
        
        # EMA state for temporal crossover arm
        self._ema_crossover_success = 0.5
        self._ema_fitness_change = 0.0
        self._prev_mean_fitness = None
        self._CR_ema = self.CR
        self._momentum_buffer = 0.0
    
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / max(self.NP - 1, 1)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        """Sample 3 candidate mutation directions, score by rank improvement potential."""
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
        """Vectorized compass mutation: compute direction scores, select best per individual."""
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
        """Softmax-weighted strategy selection."""
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        """Use adaptive strategy pool to generate mutants."""
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        
        return mutants, strategy_used
    
    # ==================================================================
    # CROSSOVER VARIANTS (from benchmark winners)
    # ==================================================================
    
    def _crossover_original(self, population, mutants):
        """Baseline binomial crossover (original.py)."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catA_spread(self, population, mutants):
        """Geometry-driven crossover: adapt CR per dimension by axis-aligned spread."""
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
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catB_spectral(self, population, mutants):
        """Spectral crossover using eigenvalue-weighted PCA decomposition."""
        NP, dim = population.shape
        
        centroid = population.mean(axis=0)
        centered = population - centroid
        
        cov = (centered.T @ centered) / NP
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return self._crossover_original(population, mutants)
        
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        
        pop_pc = centered @ eigenvectors
        mut_pc = (mutants - centroid) @ eigenvectors
        
        total_eig = eigenvalues.sum()
        if total_eig > 0:
            eig_weights = eigenvalues / total_eig
        else:
            eig_weights = np.ones(dim) / dim
        
        base_cr = np.clip(self.CR * np.power(eig_weights, 0.1), 0.01, 0.99)
        crossover_mask = np.random.rand(NP, dim) < base_cr
        
        if dim > 1:
            force_cross_idx = np.random.randint(0, dim, size=NP)
            crossover_mask[np.arange(NP), force_cross_idx] = True
        
        trial_pc = np.where(crossover_mask, mut_pc, pop_pc)
        trials = trial_pc @ eigenvectors.T + centroid
        
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catE_knn(self, population, mutants):
        """Graph-distance-adaptive binomial crossover using k-NN topology."""
        NP, dim = population.shape
        
        k = max(2, min(5, NP // 4))
        
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        graph_dist = np.zeros(NP)
        
        for i in range(NP):
            target = population[i]
            mutant = mutants[i]
            
            direct_dist = np.linalg.norm(mutant - target)
            
            mutant_neighbors = set(knn_indices[i])
            
            if len(mutant_neighbors) == 0:
                graph_dist[i] = direct_dist
                continue
            
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
        if max_dist > 1e-10:
            norm_dist = graph_dist / max_dist
        else:
            norm_dist = np.zeros(NP)
        
        adaptive_CR = self.CR * (1.0 - 0.3 * norm_dist)
        adaptive_CR = np.clip(adaptive_CR, 0.1, 0.95)
        
        trials = np.empty_like(population)
        jr = np.random.randint(0, dim, size=NP)
        
        for i in range(NP):
            cr = adaptive_CR[i]
            mask = np.random.rand(dim) < cr
            mask[jr[i]] = True
            trials[i] = np.where(mask, mutants[i], population[i])
        
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catF_temporal(self, population, mutants):
        """Binomial crossover with temporal EMA-driven CR adaptation."""
        NP, dim = population.shape
        
        current_mean_fitness = population.mean(axis=0).sum()
        if self._prev_mean_fitness is not None:
            fitness_drift = current_mean_fitness - self._prev_mean_fitness
            alpha_drift = 0.2
            self._ema_fitness_change = (1 - alpha_drift) * self._ema_fitness_change + alpha_drift * fitness_drift
        
        self._prev_mean_fitness = current_mean_fitness
        
        success_rate = self._ema_crossover_success
        target_correction = 0.0
        
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
        
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_batch(self, population, mutants):
        """Dispatch to the committed crossover strategy (or original as fallback)."""
        if self._committed_arm is None:
            return self._crossover_original(population, mutants)
        
        if self._committed_arm == 0:
            return self._crossover_original(population, mutants)
        elif self._committed_arm == 1:
            return self._crossover_catA_spread(population, mutants)
        elif self._committed_arm == 2:
            return self._crossover_catB_spectral(population, mutants)
        elif self._committed_arm == 3:
            return self._crossover_catE_knn(population, mutants)
        elif self._committed_arm == 4:
            return self._crossover_catF_temporal(population, mutants)
        else:
            return self._crossover_original(population, mutants)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        """Hybrid credit: rank improvement + success rate, weighted by signal confidence."""
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
        """Adapt F and CR based on k-NN graph topology of the population."""
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

        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
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
        """Population spread: average pairwise Euclidean distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _update_archive(self, population, fitness, improved_mask):
        """Maintain archive of improved solutions for diversity injection."""
        improved_pop = population[improved_mask]
        if len(improved_pop) > 0:
            self.archive.append(improved_pop)
            total_len = sum(len(a) for a in self.archive)
            while total_len > self.archive_max:
                oldest = self.archive.pop(0)
                total_len -= len(oldest)
    
    def _inject_diversity(self, population, fitness):
        """Replace worst individuals with archive + random samples."""
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
        """Detect stagnation and low diversity."""
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
    def _restart_if_needed(self, population, fitness):
        """Restart based on k-NN graph connectivity analysis."""
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
        isolation_score = (1.0 - connectivity / max(connectivity.max(), 1e-10)) * 0.5 + \
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
    
    # ==================================================================
    # PROBE-AND-COMMIT CROSSOVER ADAPTATION
    # ==================================================================
    
    def _compute_landscape_fingerprint(self, pop, fitness):
        """Extract cheap landscape features from current population state.
        
        These are observable from any black-box func — no oracle knowledge used.
        Returns a dict of normalized features.
        """
        NP, dim = pop.shape
        
        # Feature 1: Effective dimensionality (from covariance PCA)
        # Number of eigenvalues explaining >95% of variance
        centroid = pop.mean(axis=0)
        centered = pop - centroid
        cov = (centered.T @ centered) / NP
        try:
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            total_var = eigvals.sum()
            cumsum = np.cumsum(eigvals)
            eff_dim = np.searchsorted(cumsum, 0.95 * total_var) + 1
            eff_dim_norm = eff_dim / dim
        except:
            eff_dim_norm = 0.5
        
        # Feature 2: Diversity ratio (avg pairwise dist / search space diagonal)
        centroid_dist = np.linalg.norm(pop - centroid, axis=1).mean()
        space_diag = np.sqrt(dim) * 200.0  # [-100, 100]^dim
        diversity_ratio = centroid_dist / space_diag
        
        # Feature 3: Fitness rank entropy (higher = more uniform = more exploration needed)
        ranks = self._compute_fitness_ranking(fitness)
        # Compute probability mass in bins
        n_bins = min(10, NP)
        bin_counts = np.zeros(n_bins)
        for r in ranks:
            b = min(int(r * n_bins), n_bins - 1)
            bin_counts[b] += 1
        bin_probs = bin_counts / NP
        bin_probs = bin_probs[bin_probs > 0]
        rank_entropy = -np.sum(bin_probs * np.log(bin_probs + 1e-10))
        rank_entropy_norm = rank_entropy / np.log(n_bins + 1e-10)
        
        # Feature 4: Spread ratio (max - min) / space diagonal per dimension
        per_dim_spread = pop.max(axis=0) - pop.min(axis=0)
        avg_spread_ratio = np.mean(per_dim_spread / 200.0)
        
        return {
            'eff_dim': eff_dim_norm,
            'diversity': diversity_ratio,
            'rank_entropy': rank_entropy_norm,
            'spread': avg_spread_ratio
        }
    
    def _eval_probe_arm(self, arm_idx, pop, fitness, prev_best_fitness):
        """Evaluate a single crossover arm and return rank-normalized improvement.
        
        Uses rank-based reward to avoid scale-dependent magic constants (Rule a).
        Returns: (rank_normalized_improvement, current_best_fitness_for_this_arm)
        """
        NP, dim = pop.shape
        
        # Generate mutants using the current mutation strategy
        mutants_compass = self._mutate_compass_batch(pop, fitness)
        mutants_ensemble, strategy_used = self._mutate_ensemble_batch(pop, fitness)
        blend_ratio = np.random.rand(NP, 1)
        mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
        
        # Apply the specific crossover arm under evaluation
        if arm_idx == 0:
            trials = self._crossover_original(pop, mutants)
        elif arm_idx == 1:
            trials = self._crossover_catA_spread(pop, mutants)
        elif arm_idx == 2:
            trials = self._crossover_catB_spectral(pop, mutants)
        elif arm_idx == 3:
            trials = self._crossover_catE_knn(pop, mutants)
        elif arm_idx == 4:
            trials = self._crossover_catF_temporal(pop, mutants)
        else:
            trials = self._crossover_original(pop, mutants)
        
        # Clip to bounds
        trials = np.clip(trials, -100.0, 100.0)
        
        return trials, prev_best_fitness
    
    def _run_probe_round(self, func, pop, fitness, prev_best_fitness):
        """Run one generation with round-robin crossover arm selection.
        
        Each arm gets one trial per round. Returns arm-specific trial fitness.
        """
        NP, dim = pop.shape
        trials_per_arm = np.empty((self._n_arms, NP, dim))
        trial_fitness_per_arm = np.empty((self._n_arms, NP))
        
        for arm_idx in range(self._n_arms):
            trials_per_arm[arm_idx], _ = self._eval_probe_arm(arm_idx, pop, fitness, prev_best_fitness)
            trial_fitness_per_arm[arm_idx] = func(trials_per_arm[arm_idx])
        
        return trials_per_arm, trial_fitness_per_arm
    
    def _compute_rank_reward(self, all_fitness, chosen_arm, improved_mask):
        """Compute rank-based reward for the chosen arm.
        
        Uses relative rank improvement within the batch — scale-independent.
        """
        NP = len(improved_mask)
        
        # Compute rank of each arm's best trial fitness
        arm_best_fitness = np.array([np.min(all_fitness[i]) for i in range(self._n_arms)])
        
        # Rank arms by their best fitness (1 = best)
        arm_order = np.argsort(arm_best_fitness)
        arm_ranks = np.empty(self._n_arms)
        arm_ranks[arm_order] = np.arange(self._n_arms) + 1
        
        # Compute improvement rate for the chosen arm
        chosen_improvement = np.mean(improved_mask[chosen_arm]) if hasattr(improved_mask, 'shape') else improved_mask
        
        # Rank-based reward: higher rank = better
        rank_reward = 1.0 - (arm_ranks[chosen_arm] - 1) / max(self._n_arms - 1, 1)
        
        return rank_reward, arm_best_fitness[chosen_arm]
    
    def _decide_commit(self):
        """After probe phase, decide which arm to commit to.
        
        Uses rank-based selection from probe data. Falls back to original
        if probe is inconclusive.
        """
        # Check if any arm got enough samples
        min_samples = 2
        valid_arms = self._probe_arm_trial_count >= min_samples
        
        if not np.any(valid_arms):
            # Not enough data — commit to original (strong baseline)
            self._committed_arm = 0
            return
        
        # Compute average rank-based improvement per arm
        avg_improvement = np.zeros(self._n_arms)
        for a in range(self._n_arms):
            if self._probe_arm_trial_count[a] > 0:
                avg_improvement[a] = self._probe_arm_improvement[a] / self._probe_arm_trial_count[a]
        
        # Rank arms by average improvement
        arm_order = np.argsort(avg_improvement)[::-1]  # Descending
        best_arm = arm_order[0]
        second_arm = arm_order[1] if len(arm_order) > 1 else best_arm
        
        # Check if best arm is significantly better than second best
        # Use relative margin — scale-independent
        if self._probe_arm_trial_count[best_arm] > 0 and self._probe_arm_trial_count[second_arm] > 0:
            best_avg = avg_improvement[best_arm]
            second_avg = avg_improvement[second_arm]
            
            if second_avg > 1e-10:
                margin = (best_avg - second_avg) / (abs(second_avg) + 1e-10)
            else:
                margin = best_avg - second_avg
            
            # If margin is small (< 5%), probe is inconclusive — default to original
            if abs(margin) < 0.05:
                self._committed_arm = 0  # Default to original (6-win baseline)
            else:
                self._committed_arm = int(best_arm)
        else:
            self._committed_arm = 0
        
        # Compute and store landscape fingerprint at commit time
        self._fingerprint = self._compute_landscape_fingerprint(self._current_population, self._current_fitness)
    
    def _update_crossover_ema_success(self, improvement_rate):
        """Update EMA of crossover success rate (for catF temporal arm)."""
        alpha = 0.3
        self._ema_crossover_success = (1 - alpha) * self._ema_crossover_success + alpha * improvement_rate
    
    # ==================================================================
    # MAIN OPTIMIZATION LOOP
    # ==================================================================
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        prev_best_fitness = f_opt
        
        # Store current state for adaptation methods
        self._current_population = population
        self._current_fitness = fitness
        
        # Estimate total budget from max_iter
        total_budget = getattr(self, 'max_iter', 20000)
        probe_max_gens = max(self._probe_min_gens, int(total_budget * self._probe_budget_pct))
        probe_max_gens = min(probe_max_gens, self._probe_max_gens)
        
        # Track probe progress
        probe_round = 0
        probe_generations_done = 0
        probe_best_per_arm = np.full(self._n_arms, np.inf)
        probe_improvement_per_arm = np.zeros(self._n_arms)
        
        while not stopping_condition():
            self._current_population = population
            self._current_fitness = fitness
            
            # ============================================================
            # PHASE A: PROBE (round-robin crossover arm evaluation)
            # ============================================================
            if self._probe_active and probe_generations_done < probe_max_gens:
                # Run one probe round with all arms
                trials_per_arm, trial_fitness_per_arm = self._run_probe_round(
                    func, population, fitness, prev_best_fitness
                )
                
                # Evaluate each arm
                for arm_idx in range(self._n_arms):
                    arm_trials = trials_per_arm[arm_idx]
                    arm_trial_fitness = trial_fitness_per_arm[arm_idx]
                    
                    # Selection for this arm's trials
                    better = arm_trial_fitness < fitness
                    
                    # Rank-based improvement signal
                    improvement_count = better.sum()
                    improvement_rate = improvement_count / self.NP
                    
                    # Update probe stats
                    self._probe_arm_trial_count[arm_idx] += 1
                    self._probe_arm_improvement[arm_idx] += improvement_rate
                    
                    # Track best fitness seen by this arm
                    arm_best = np.min(arm_trial_fitness)
                    if arm_best < self._probe_arm_best_fitness[arm_idx]:
                        self._probe_arm_best_fitness[arm_idx] = arm_best
                    
                    # Count generations with improvement for this arm
                    if improvement_count > 0:
                        probe_improvement_per_arm[arm_idx] += 1
                
                probe_generations_done += 1
                probe_round += 1
                
                # Update EMA for temporal arm
                avg_improvement = np.mean([np.mean(trial_fitness_per_arm[i] < fitness) for i in range(self._n_arms)])
                self._update_crossover_ema_success(avg_improvement)
                
                # Check if probe should end
                if probe_generations_done >= probe_max_gens:
                    self._probe_active = False
                    self._decide_commit()
                    self._commit_start_iter = iteration
                
                # Advance iteration without full update (probe-only step)
                iteration += 1
                if stopping_condition():
                    break
                continue
            
            # ============================================================
            # PHASE B: COMMIT (use the winning crossover strategy)
            # ============================================================
            
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            # Crossover using COMMITTED strategy
            trials = self._crossover_batch(population, mutants)
            
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
                self._stagnation_since_commit = 0
            else:
                self._stagnation_since_commit += 1
            
            # Update EMA crossover success rate
            improvement_rate = np.mean(improved_mask)
            self._update_crossover_ema_success(improvement_rate)
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation
            self._update_archive(population, fitness, improved_mask)
            self._adapt_parameters(improvement_rate)
            
            # Diversity injection
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check with re-probe option
            if self._check_stagnation(f_opt):
                # Allow re-probe if committed strategy stagnates
                if (self._committed_arm is not None and 
                    self._stagnation_since_commit > self._stagnation_override_threshold and
                    self._allow_reprobe):
                    # Reset probe state
                    self._probe_active = True
                    self._probe_arm_trial_count = np.zeros(self._n_arms)
                    self._probe_arm_improvement = np.zeros(self._n_arms)
                    self._probe_arm_best_fitness = np.full(self._n_arms, np.inf)
                    probe_generations_done = 0
                    probe_improvement_per_arm = np.zeros(self._n_arms)
                    self._stagnation_since_commit = 0
                    self._allow_reprobe = False  # Only one re-probe per run
                else:
                    population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                    fitness = func(population)
                    if len(fitness) < self.NP:
                        fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
```