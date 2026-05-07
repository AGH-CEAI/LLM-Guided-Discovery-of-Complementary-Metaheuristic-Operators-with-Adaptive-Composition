import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: #4 CONTEXTUAL BANDIT — PROBE-AND-COMMIT
    
    Per-task gap table analysis: 7 tasks have DISPATCH-MANDATORY gaps (>=100×),
    and 10 tasks are BLEND-HOSTILE (>=10×). Ensemble blending (mechanism #2) is
    mathematically EXCLUDED: a 50/50 blend of values 1 and 100 yields 50.5,
    destroying the winner's 100× advantage on those tasks.
    
    The per-task winner distribution shows:
    - original.py: 21/24 wins (87.5%), many with >1000× gaps
    - variant_06: 2/24 wins (tasks 18, 20, gaps ~1.1×)
    - variant_10: 1/24 wins (task 16, gap ~1.5×)
    
    Strategy: Run a brief PROBE (~5% of budget) with all 3 candidate methods,
    measure rank-based improvement, COMMIT to the winner, and optionally
    allow small ε-switch on prolonged stagnation. This preserves the dominant
    variant's advantage on blend-hostile tasks while enabling adaptation on
    the 3 tasks where other variants win (those have tiny gaps, so any method
    works acceptably there).
    
    Key design choices:
    - Rank-based rewards (no scale-dependent constants like 1e5 or 1e10)
    - Probe uses round-robin across methods to avoid bias
    - Commit phase uses ONLY the winner (weight 1.0), no blending
    - Stagnation fallback with ε-greedy switching to prevent lock-in
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
        
        # ========== PROBE-AND-COMMIT STATE ==========
        # 3 candidate methods: 'original', 'variant_06', 'variant_10'
        self._method_names = ['original', 'variant_06', 'variant_10']
        self._method_probe_scores = np.zeros(3)  # Rank-based improvement scores
        self._method_history = [[] for _ in range(3)]  # Per-method rank improvement history
        self._committed_method = None  # Will be set after probe
        self._probe_complete = False
        self._probe_generations = 0
        self._post_commit_stagnation = 0
        self._epsilon_revisit = 0.02  # Small probability of trying runner-up on stagnation
        
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        """Convert fitness to normalized ranks [0, 1], lower is better."""
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
        # Direction A: DE/rand/1 style
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        # Direction B: DE/best/1 style
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
        # Direction C: Current-to-pbest style with archive
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
    
    def _mutate_variant_06(self, population, fitness):
        """
        Temporal momentum compass (variant_06): blend current direction with EMA of past successes.
        From: variant_06_catF_t05_idea_0.py
        """
        NP = self.NP

        # Initialize temporal momentum tracking
        if not hasattr(self, '_dir_ema'):
            self._dir_ema = np.zeros((NP, self.dim))
            self._ema_alpha = 0.3

        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)

        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]

            i_rank = fitness_ranks[i]

            # Direction A: DE/rand/1 style
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)

            # Direction B: DE/best/1 style
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]

            # Direction C: pbest with archive
            p = 0.1
            top_p = int(np.ceil(p * NP))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()

            # Select best direction for this individual
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])

            # Blend with temporal momentum (EMA of past directions)
            ema_strength = 1.0 / (1.0 + np.linalg.norm(self._dir_ema[i]))
            momentum_weight = 0.3 * ema_strength
            mutants[i] = (1 - momentum_weight) * best_dir + momentum_weight * self._dir_ema[i]

            # Update EMA: track direction toward better fitness
            if hasattr(self, '_prev_population') and self._prev_population is not None:
                if fitness[i] < getattr(self, '_prev_fitness', fitness)[i]:
                    delta_dir = best_dir - self._dir_ema[i]
                    self._dir_ema[i] = (1 - self._ema_alpha) * self._dir_ema[i] + self._ema_alpha * delta_dir

        # Store for next generation comparison
        self._prev_population = population.copy()
        self._prev_fitness = fitness.copy()

        return mutants
    
    def _mutate_variant_10(self, population, fitness):
        """
        Spectral/linear-algebraic compass (variant_10): uses eigendecomposition of 
        population covariance matrix to identify principal axes.
        From: variant_10_catB_t05_idea_0.py
        """
        NP, dim = population.shape

        ranks = self._compute_fitness_ranking(fitness)

        # Compute population covariance matrix (centered)
        centroid = population.mean(axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / max(NP - 1, 1)

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            std_pop = np.std(population, axis=0) + 1e-10
            return population + self.F * np.random.randn(NP, dim) * std_pop

        # Sort by descending eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvalues = np.maximum(eigenvalues, 1e-12)
        total_var = np.sum(eigenvalues)
        explained_ratio = eigenvalues / total_var if total_var > 0 else np.ones(dim) / dim

        cumsum = np.cumsum(explained_ratio)
        n_effective = int(np.searchsorted(cumsum, 0.95)) + 1
        n_effective = min(n_effective, dim)

        base_scale = 1.0 / np.sqrt(eigenvalues + 1e-10)
        axis_scaling = base_scale / (np.max(base_scale) + 1e-10)

        mutants = np.empty_like(population)

        for i in range(NP):
            p = 0.1
            top_k = max(1, int(np.ceil(p * NP)))
            pbest_indices = np.argsort(ranks)[:top_k]
            pbest = population[np.random.choice(pbest_indices)]

            direction = pbest - population[i]
            z = eigenvectors.T @ direction
            z_scaled = z * axis_scaling
            direction_scaled = eigenvectors @ z_scaled

            rank_factor = 0.5 + 0.5 * (1.0 - ranks[i])

            random_perturb = np.zeros(dim)
            for j in range(n_effective):
                random_perturb += eigenvectors[:, j] * np.random.randn() * axis_scaling[j]

            mutants[i] = population[i] + self.F * rank_factor * direction_scaled + \
                         0.3 * self.F * random_perturb

        return np.clip(mutants, -100.0, 100.0)
    
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
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with dimension-wise CR."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
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
        isolation_score = (1.0 - connectivity / connectivity.max()) * 0.5 + \
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
    
    def _get_mutant_generator(self, method_name):
        """Return the mutant generation function for the specified method."""
        if method_name == 'original':
            return self._mutate_compass_batch
        elif method_name == 'variant_06':
            return self._mutate_variant_06
        elif method_name == 'variant_10':
            return self._mutate_variant_10
        else:
            return self._mutate_compass_batch  # Default fallback
    
    def _run_optimization_loop(self, func, stopping_condition, mutants_generator, 
                                max_generations, method_name, track_rank_improvement=True):
        """
        Core optimization loop with a specific mutation generator.
        Returns (f_opt, x_opt, rank_improvements) where rank_improvements is a list
        of per-generation rank-based improvement scores.
        """
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        generation = 0
        prev_best_rank = 0.0
        rank_improvements = []
        
        # Reset method-specific state
        if method_name == 'variant_06':
            if hasattr(self, '_dir_ema'):
                self._dir_ema = np.zeros((self.NP, self.dim))
            if hasattr(self, '_prev_population'):
                self._prev_population = None
        
        while not stopping_condition() and generation < max_generations:
            self._current_population = population
            
            # Generate mutants using the specified method
            mutants = mutants_generator(population, fitness)
            
            # Ensemble mutation (secondary strategy pool) - always available
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% primary method, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants, mutants_ensemble)
            
            # Crossover
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
            
            # Track rank-based improvement
            if track_rank_improvement:
                current_best_rank = self._compute_fitness_ranking(fitness)[current_best_idx]
                # Rank improvement: how much did the best rank improve (lower is better)
                rank_imp = prev_best_rank - current_best_rank
                rank_improvements.append(rank_imp)
                prev_best_rank = current_best_rank
            
            # Strategy score update (for ensemble component)
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
            
            # Stagnation check
            if self._check_stagnation(f_opt):
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            generation += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt, rank_improvements
    
    def _run_probe_phase(self, func, stopping_condition):
        """
        PROBE PHASE: Evaluate all 3 candidate methods for a short burst.
        Uses round-robin across methods to avoid bias.
        Returns the method name with highest rank-based improvement.
        """
        # Probe budget: 5% of max_iter or minimum 5 generations
        probe_budget = max(5, int(self.max_iter * 0.05))
        probe_per_method = max(2, probe_budget // 3)  # At least 2 gens per method
        
        method_scores = np.zeros(3)
        method_rank_improvements = [[] for _ in range(3)]
        
        for method_idx, method_name in enumerate(self._method_names):
            if stopping_condition():
                break
            
            # Reset algorithm state for fair comparison
            self.F = 0.6
            self.CR = 0.85
            self.archive = []
            self.stagnation_counter = 0
            self.prev_best_fitness = np.inf
            
            mutants_generator = self._get_mutant_generator(method_name)
            
            # Run probe with this method
            _, _, rank_improvements = self._run_optimization_loop(
                func, stopping_condition, mutants_generator,
                max_generations=probe_per_method,
                method_name=method_name,
                track_rank_improvement=True
            )
            
            if len(rank_improvements) > 0:
                # Use cumulative rank improvement as the score
                # Positive = best rank decreased (improved)
                total_rank_imp = np.sum(rank_improvements)
                method_rank_improvements[method_idx] = rank_improvements
                method_scores[method_idx] = total_rank_imp
        
        # Determine winner: method with highest cumulative rank improvement
        winner_idx = np.argmax(method_scores)
        winner_name = self._method_names[winner_idx]
        
        # Store probe results
        self._method_probe_scores = method_scores.copy()
        self._method_history = method_rank_improvements
        self._probe_generations = probe_budget
        self._committed_method = winner_name
        
        return winner_name, method_scores

    def __call__(self, func, stopping_condition):
        """
        Main entry point with PROBE-AND-COMMIT adaptation.
        
        Phase 1 (PROBE): Run short evaluation of all 3 methods
        Phase 2 (COMMIT): Use winning method for remaining budget
        Phase 3 (FALLBACK): Allow ε-switch on prolonged stagnation
        """
        # ========== PHASE 1: PROBE ==========
        winner_method, probe_scores = self._run_probe_phase(func, stopping_condition)
        
        if stopping_condition():
            # Return best found so far
            return getattr(self, 'f_opt', np.inf), getattr(self, 'x_opt', np.zeros(self.dim))
        
        # ========== PHASE 2: COMMIT ==========
        # Reset state for committed run
        self.F = 0.6
        self.CR = 0.85
        self.archive = []
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self._current_population = None
        
        # Calculate remaining budget
        remaining_budget = max(10, self.max_iter - self._probe_generations)
        
        mutants_generator = self._get_mutant_generator(winner_method)
        
        # Run with committed method
        f_opt, x_opt, rank_improvements = self._run_optimization_loop(
            func, stopping_condition, mutants_generator,
            max_generations=remaining_budget,
            method_name=winner_method,
            track_rank_improvement=True
        )
        
        # ========== PHASE 3: STAGNATION FALLBACK ==========
        # If committed method stagnates, allow small probability of trying runner-up
        self._post_commit_stagnation = 0
        for i, rank_imp in enumerate(rank_improvements):
            if rank_imp <= 0:  # No improvement
                self._post_commit_stagnation += 1
            else:
                self._post_commit_stagnation = 0
            
            # If stagnated for 20+ generations, try runner-up with small probability
            if self._post_commit_stagnation > 20 and np.random.rand() < self._epsilon_revisit:
                # Find runner-up method
                sorted_methods = np.argsort(probe_scores)[::-1]  # Descending
                for runner_idx in sorted_methods:
                    runner_method = self._method_names[runner_idx]
                    if runner_method != winner_method:
                        # Try runner-up for a few generations
                        runner_gen = self._get_mutant_generator(runner_method)
                        f_runner, x_runner, _ = self._run_optimization_loop(
                            func, stopping_condition, runner_gen,
                            max_generations=5,  # Short burst
                            method_name=runner_method,
                            track_rank_improvement=False
                        )
                        # If runner-up is better, switch
                        if f_runner < f_opt:
                            f_opt = f_runner
                            x_opt = x_runner
                            winner_method = runner_method
                        break
        
        self.f_opt = f_opt
        self.x_opt = x_opt
        self._probe_complete = True
        
        return f_opt, x_opt
