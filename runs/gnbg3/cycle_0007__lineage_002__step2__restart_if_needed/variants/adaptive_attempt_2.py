import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Probe-and-Commit (mechanism #3/4 hybrid).
    
    Per-task gap analysis: ALL gaps are < 3× (max observed: 2.7× on task 17).
    No task is blend-hostile (no gap >= 10×), so linear blending is mathematically
    viable. However, the per-task winner table shows DISTINCT winners across
    tasks (11 wins for variant_08, 10 wins for original), meaning no single
    strategy is universally optimal. Probe-and-commit is chosen because:
    (a) it learns which restart strategy suits the CURRENT landscape cheaply,
    (b) it commits fully to the winner (no blending dilution), and
    (c) the 2.7× max gap means even a partial blend would lose ~1.7× accuracy.
    
    The probe uses rank-based improvement (scale-independent) over ~8% of budget,
    then commits to the empirically better restart strategy for the remaining run.
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
        
        # --- RESTART STRATEGY SELECTION (Probe-and-Commit) ---
        # Strategy 0: original k-NN graph connectivity restart
        # Strategy 1: variant_08 fitness-variance + spatial-clustering hybrid restart
        self._restart_strategies = [self._restart_original, self._restart_variant08]
        self._selected_restart = 0  # 0=original, 1=variant_08
        self._probe_budget = max(int(0.08 * self.max_iter), 10)
        self._probe_iter = 0
        self._probe_rank_improvement = [0.0, 0.0]  # cumulative rank-based improvement per strategy
        self._probe_counts = [0, 0]
        self._committed = False
        self._restart_stagnation_count = 0
        self._fallback_prob = 0.10  # small prob to try other strategy if stagnating
    
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / (self.NP - 1) if self.NP > 1 else np.zeros_like(fitness)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        """Sample 3 candidate mutation directions, score by rank improvement potential."""
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
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
            'graph_diameter': np.max(np.sqrt(nearest_sq_dists))
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
    
    # -------------------------------------------------------------------------
    # RESTART STRATEGY 0: Original k-NN graph connectivity analysis
    # -------------------------------------------------------------------------
    def _restart_original(self, population, fitness):
        """Restart based on k-NN graph connectivity analysis (original.py)."""
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
        self.stagnation_counter = 0
        return population, fitness[best_idx], population[best_idx].copy()
    
    # -------------------------------------------------------------------------
    # RESTART STRATEGY 1: Fitness-variance + spatial-clustering hybrid (variant_08)
    # -------------------------------------------------------------------------
    def _restart_variant08(self, population, fitness):
        """Hybrid restart: combine fitness-variance and spatial-clustering signals with data-driven weighting."""
        if len(population) < 4:
            best_idx = np.nanargmin(fitness)
            return population, np.nanmin(fitness), population[best_idx].copy()

        if not hasattr(self, '_fitness_cv_history'):
            self._fitness_cv_history = []
            self._spatial_cv_history = []

        valid_fitness = fitness[~np.isnan(fitness)]
        if len(valid_fitness) < 2:
            fitness_diversity = 1.0
        else:
            cv = np.std(valid_fitness) / (np.abs(np.mean(valid_fitness)) + 1e-10)
            fitness_cv = np.clip(cv, 0.0, 10.0)
            self._fitness_cv_history.append(fitness_cv)
            if len(self._fitness_cv_history) > 15:
                self._fitness_cv_history.pop(0)
            if len(self._fitness_cv_history) >= 3:
                baseline_cv = np.mean(self._fitness_cv_history[:3])
                current_cv = np.mean(self._fitness_cv_history[-3:])
                fitness_diversity = current_cv / (baseline_cv + 1e-10)
            else:
                fitness_diversity = 1.0

        centroid = population.mean(axis=0)
        dists = np.linalg.norm(population - centroid, axis=1)
        spatial_spread = np.mean(dists)
        self._spatial_cv_history.append(spatial_spread)
        if len(self._spatial_cv_history) > 15:
            self._spatial_cv_history.pop(0)
        if len(self._spatial_cv_history) >= 3:
            baseline_spread = np.mean(self._spatial_cv_history[:3])
            current_spread = np.mean(self._spatial_cv_history[-3:])
            spatial_diversity = current_spread / (baseline_spread + 1e-10)
        else:
            spatial_diversity = 1.0

        fitness_severity = 1.0 / (fitness_diversity + 1e-10)
        spatial_severity = 1.0 / (spatial_diversity + 1e-10)
        total_severity = fitness_severity + spatial_severity + 1e-10
        w_fitness = fitness_severity / total_severity
        w_spatial = spatial_severity / total_severity

        stagnation_signal = w_fitness * fitness_severity + w_spatial * spatial_severity

        if stagnation_signal < 0.5 or fitness_diversity < 0.3 or spatial_diversity < 0.3:
            intensity = np.clip(1.0 - stagnation_signal, 0.2, 0.8)
            n_replace = max(int(self.NP * intensity), 2)
            worst_idx = np.argsort(fitness)[-n_replace:]
            archive_blend = np.clip(intensity, 0.3, 0.7)

            for idx in worst_idx:
                if np.random.rand() < archive_blend and len(self.archive) > 0:
                    arch = np.vstack(self.archive)
                    donor = arch[np.random.randint(len(arch))]
                    noise_scale = 15.0 * (1.0 - intensity)
                    population[idx] = np.clip(donor + np.random.randn(self.dim) * noise_scale, -100.0, 100.0)
                else:
                    population[idx] = np.random.uniform(-100.0, 100.0, self.dim)

            self._fitness_cv_history = self._fitness_cv_history[-3:]
            self._spatial_cv_history = self._spatial_cv_history[-3:]

        best_idx = np.nanargmin(fitness)
        self.stagnation_counter = 0
        return population, fitness[best_idx], population[best_idx].copy()
    
    # -------------------------------------------------------------------------
    # PROBE PHASE: Evaluate both restart strategies with rank-based credit
    # -------------------------------------------------------------------------
    def _run_probe(self, population, fitness, current_best_fitness):
        """Run a short probe evaluating both restart strategies with rank-based improvement."""
        if self._probe_iter >= self._probe_budget:
            return None, None, None

        # Alternate between strategy 0 and 1
        strat_idx = self._probe_iter % 2

        # Create a copy to test the strategy
        pop_test = population.copy()
        fit_test = fitness.copy()

        # Apply restart strategy
        pop_after, f_after, x_after = self._restart_strategies[strat_idx](pop_test, fit_test)

        # Re-evaluate
        fit_after = np.array([np.nan if np.isnan(f) else f for f in fit_test])

        # Rank-based improvement (scale-independent)
        # Lower rank = better. If trial rank < current rank, that's an improvement.
        current_ranks = self._compute_fitness_ranking(fit_test)
        after_ranks = self._compute_fitness_ranking(fit_after)

        # Average rank improvement: positive means the restart led to better ranks
        rank_improvement = np.mean(current_ranks - after_ranks)

        # Accumulate
        self._probe_rank_improvement[strat_idx] += rank_improvement
        self._probe_counts[strat_idx] += 1

        self._probe_iter += 1

        # Check if probe is done
        if self._probe_iter >= self._probe_budget:
            # Commit to the better strategy
            avg0 = self._probe_rank_improvement[0] / max(self._probe_counts[0], 1)
            avg1 = self._probe_rank_improvement[1] / max(self._probe_counts[1], 1)
            self._selected_restart = 0 if avg0 >= avg1 else 1
            self._committed = True
            self._restart_stagnation_count = 0

        return pop_after, f_after, x_after
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
            # PROBE PHASE: evaluate both restart strategies
            if not self._committed and self._probe_iter < self._probe_budget:
                pop_after, f_after, x_after = self._run_probe(population.copy(), fitness.copy(), f_opt)
                if pop_after is not None:
                    # Update state from probe
                    population = pop_after
                    fitness = func(population)
                    if len(fitness) < self.NP:
                        fitness = np.full(self.NP, np.nan)
                    current_best_idx = np.nanargmin(fitness)
                    if fitness[current_best_idx] < f_opt:
                        f_opt = fitness[current_best_idx]
                        x_opt = population[current_best_idx].copy()
                    iteration += 1
                    if stopping_condition():
                        break
                    continue
            
            # COMPASS + ENSEMBLE mutation (unchanged from original)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            trials = self._crossover_batch(population, mutants)
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.full(len(trials), np.nan)
            
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
            self._update_strategy_scores(strategy_used, improved_mask)
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            self._adapt_parameters(improvement_rate)
            
            # Store current population for parameter adaptation
            self._current_population = population
            
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # STAGNATION → use SELECTED restart strategy
            if self._check_stagnation(f_opt):
                # Determine which strategy to use
                strat_to_apply = self._selected_restart
                
                # Small fallback probability if current strategy is stagnating
                if self._committed and self._restart_stagnation_count > 30:
                    if np.random.rand() < self._fallback_prob:
                        strat_to_apply = 1 - self._selected_restart
                        self._restart_stagnation_count = 0
                
                # Apply the chosen restart strategy
                population, f_opt, x_opt = self._restart_strategies[strat_to_apply](population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
                
                self._restart_stagnation_count += 1
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
