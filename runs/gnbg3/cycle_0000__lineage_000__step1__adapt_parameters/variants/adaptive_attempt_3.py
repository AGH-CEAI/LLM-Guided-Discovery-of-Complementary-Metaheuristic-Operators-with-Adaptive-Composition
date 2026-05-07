import numpy as np


class AdaptiveCompassDE:
    """
    Mechanism #4 (Contextual Bandit) with PROBE-AND-COMMIT.
    
    Per-task gap table is BLEND-HOSTILE: 6 tasks have winner/runner-up gap >= 100×,
    and the worst-case gap is 822,177× (task 4). Linear ensemble blending
    mathematically cannot preserve these per-task advantages — a 50/50 blend of
    1e-8 and 1e-2 yields 5e-3, not 1e-8. The dispatcher MUST commit to ONE
    adaptation strategy (weight 1.0) per run, selected from a short probe burst.
    
    The 4 adaptation strategies are:
      - strategy_0 (original): improvement-rate-based F/CR adaptation
      - strategy_1 (variant_01): geometric spatial spread (axis-aligned range, centroid distance)
      - strategy_2 (variant_05): k-NN graph topology (clustering coefficient, edge length)
      - strategy_3 (variant_07): bootstrap confidence intervals of improvement rate
    
    Probe phase (~5% of budget):
      - Run a small sub-population (NP//2) for probe_iters generations
      - All 4 strategies run in parallel with round-robin per-generation
      - Extract a cheap landscape fingerprint: convergence rate, rank entropy,
        effective dimensionality, stagnation index
      - Track per-strategy empirical win rate (rank-based, not raw fitness)
    
    Commit phase (~95% of budget):
      - Commit exclusively to the probe-winning strategy (weight 1.0, no blending)
      - Run the full algorithm with that strategy for the remaining budget
      - Optionally allow epsilon-switch if stagnation persists for >60 generations
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool
        self.strategy_names = ['original', 'catA_spread', 'catE_graph', 'catG_bootstrap']
        self._strategy_funcs = [
            self._adapt_original,
            self._adapt_catA_spread,
            self._adapt_catE_graph,
            self._adapt_catG_bootstrap,
        ]
        self._strategy_scores = np.ones(len(self.strategy_names))
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Probe-and-commit state
        self._committed_strategy = None  # index of chosen strategy
        self._probe_completed = False
        self._probe_iterations = 0
        self._probe_budget = 0
        self._probe_fingerprints = {}  # per-strategy fingerprint data
        
        # Epsilon-switch for post-commit contingency
        self._epsilon_switch_prob = 0.02  # small probability of trying runner-up
        self._post_stagnation_threshold = 60
        
    # ─────────────────────────────────────────────────────────────────
    # Probe helpers
    # ─────────────────────────────────────────────────────────────────
    
    def _compute_landscape_fingerprint(self, pop, fitness_history, gen):
        """Extract cheap landscape features observable from any black-box func."""
        fingerprint = {}
        
        # 1. Convergence rate: slope of log(best) vs generation (last 3 points)
        if len(fitness_history) >= 2:
            recent = fitness_history[-min(3, len(fitness_history)):]
            bests = [f[0] for f in recent]  # best fitness per gen
            if len(bests) >= 2 and bests[-1] > 0 and bests[0] > 0:
                log_bests = np.log(np.maximum(bests, 1e-12))
                x = np.arange(len(log_bests))
                # Simple linear regression slope
                if np.std(x) > 0:
                    cov = np.cov(x, log_bests)
                    fingerprint['convergence_slope'] = cov[0, 1] / np.var(x)
                else:
                    fingerprint['convergence_slope'] = 0.0
            else:
                fingerprint['convergence_slope'] = 0.0
        else:
            fingerprint['convergence_slope'] = 0.0
        
        # 2. Rank entropy: how uniformly spread are the fitness ranks?
        current_fitness = fitness_history[-1] if fitness_history else pop
        if hasattr(current_fitness, '__len__') and len(current_fitness) > 1:
            order = np.argsort(current_fitness)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(current_fitness))
            ranks = ranks / max(len(current_fitness) - 1, 1)
            # Entropy of rank distribution (discretized into 5 bins)
            bins = np.linspace(0, 1, 6)
            hist, _ = np.histogram(ranks, bins=bins)
            probs = hist / max(hist.sum(), 1)
            probs = probs[probs > 0]
            fingerprint['rank_entropy'] = -np.sum(probs * np.log(probs + 1e-12))
        else:
            fingerprint['rank_entropy'] = 0.0
        
        # 3. Effective dimensionality: fraction of PCA eigenvalues > 95% threshold
        if pop.shape[0] >= 3 and pop.shape[1] >= 2:
            centered = pop - pop.mean(axis=0)
            try:
                cov = np.cov(centered.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.sort(eigvals)[::-1]
                eigvals = eigvals[eigvals > 1e-12]
                if len(eigvals) > 0:
                    total_var = eigvals.sum()
                    cumsum = np.cumsum(eigvals)
                    n_important = np.searchsorted(cumsum, 0.95 * total_var) + 1
                    fingerprint['effective_dim_frac'] = n_important / max(self.dim, 1)
                else:
                    fingerprint['effective_dim_frac'] = 1.0
            except Exception:
                fingerprint['effective_dim_frac'] = 1.0
        else:
            fingerprint['effective_dim_frac'] = 1.0
        
        # 4. Stagnation index: fraction of recent generations with no improvement
        if len(fitness_history) >= 2:
            improvements = []
            for i in range(1, len(fitness_history)):
                prev_best = fitness_history[i-1][0]
                curr_best = fitness_history[i][0]
                improvements.append(1.0 if prev_best - curr_best > 1e-10 else 0.0)
            window = min(5, len(improvements))
            fingerprint['stagnation_index'] = 1.0 - np.mean(improvements[-window:])
        else:
            fingerprint['stagnation_index'] = 0.0
        
        # 5. Population spread: normalized mean distance to centroid
        centroid = pop.mean(axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        expected = 200.0 * np.sqrt(self.dim / max(self.NP, 1))
        fingerprint['normalized_spread'] = distances.mean() / max(expected, 1e-6)
        
        fingerprint['generation'] = gen
        return fingerprint
    
    def _rank_based_score(self, fitness_a, fitness_b):
        """Return 1 if fitness_a is better than fitness_b in rank terms, 0.5 tie, 0 otherwise.
        Uses rank normalization within a sliding window to avoid scale dependence."""
        # Rank each fitness array
        order_a = np.argsort(fitness_a)
        ranks_a = np.empty_like(order_a, dtype=float)
        ranks_a[order_a] = np.arange(len(fitness_a))
        
        order_b = np.argsort(fitness_b)
        ranks_b = np.empty_like(order_b, dtype=float)
        ranks_b[order_b] = np.arange(len(fitness_b))
        
        # Compare best ranks
        best_a = ranks_a[0]
        best_b = ranks_b[0]
        if best_a < best_b:
            return 1.0
        elif best_a > best_b:
            return 0.0
        else:
            return 0.5
    
    def _run_probe_phase(self, func, budget):
        """Run a short probe burst with all 4 strategies in round-robin.
        Returns the index of the winning strategy and per-strategy fingerprints."""
        probe_NP = max(self.NP // 2, 4)
        probe_iters = max(5, int(budget * 0.05))  # 5% of budget or at least 5
        
        # Initialize probe population
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (probe_NP, self.dim))
        pop += np.random.randn(probe_NP, self.dim) * 5.0
        pop = np.clip(pop, low, high)
        fitness = func(pop)
        
        if len(fitness) < probe_NP:
            fitness = np.full(probe_NP, np.nan)
        
        # Per-strategy tracking
        strategy_best_fitness = [np.inf] * len(self.strategy_names)
        strategy_fitness_history = [[] for _ in range(len(self.strategy_names))]
        strategy_win_counts = np.zeros(len(self.strategy_names))
        strategy_probe_counts = np.zeros(len(self.strategy_names))
        
        # Store population per strategy for their respective _adapt_* methods
        strategy_pops = {i: pop.copy() for i in range(len(self.strategy_names))}
        
        best_overall = fitness[np.nanargmin(fitness)]
        fitness_history = [(float(np.nanmin(fitness)),)]  # (best_fitness,)
        
        for gen in range(probe_iters):
            # Round-robin: each generation uses a different strategy
            strat_idx = gen % len(self.strategy_names)
            
            # Run one generation with the current strategy
            # (simplified DE loop: mutate, cross, select)
            mutants = self._probe_mutate(strategy_pops[strat_idx], strat_idx)
            trials = self._probe_crossover(strategy_pops[strat_idx], mutants)
            trials = np.clip(trials, low, high)
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trial_fitness = np.full(len(trials), np.nan)
            
            # Selection
            better = trial_fitness < fitness
            new_pop = np.where(better[:, np.newaxis], trials, pop)
            new_fit = np.where(better, trial_fitness, fitness)
            
            # Update per-strategy tracking
            strategy_pops[strat_idx] = new_pop
            
            current_best = float(np.nanmin(new_fit))
            fitness_history.append((current_best,))
            
            if current_best < best_overall:
                best_overall = current_best
            
            # Track per-strategy best
            if current_best < strategy_best_fitness[strat_idx]:
                strategy_best_fitness[strat_idx] = current_best
            
            strategy_fitness_history[strat_idx].append(current_best)
            strategy_probe_counts[strat_idx] += 1
            
            pop = new_pop
            fitness = new_fit
            
            # Compute improvement rate for adaptation
            n_improved = np.sum(better)
            improvement_rate = n_improved / max(len(better), 1)
            
            # Call the strategy's adaptation method
            self._call_adapt_strategy(strat_idx, improvement_rate, new_pop)
        
        # Compute per-strategy empirical win rate (rank-based)
        # Each strategy gets a score based on its best achieved fitness rank
        # relative to the global best across all strategies
        global_best = min(strategy_best_fitness)
        
        for s in range(len(self.strategy_names)):
            if strategy_probe_counts[s] > 0:
                # Rank-based score: how close did this strategy get to global best?
                # Score = 1 - (best_s - global_best) / (|global_best| + 1e-8)
                # But we need rank-based: compare each generation's best rank
                wins = 0
                total = len(strategy_fitness_history[s])
                if total > 0:
                    for f in strategy_fitness_history[s]:
                        # Compare to median of all strategies at that point
                        other_bests = [strategy_fitness_history[o][min(len(strategy_fitness_history[o])-1, strategy_fitness_history[s].index(f))] 
                                       for o in range(len(self.strategy_names)) if o != s and len(strategy_fitness_history[o]) > 0]
                        if other_bests:
                            median_other = np.median(other_bests)
                            if f <= median_other:
                                wins += 1
                strategy_win_counts[s] = wins / max(total, 1)
        
        # Fallback: if all win counts are 0, use best fitness
        if np.sum(strategy_win_counts) == 0:
            for s in range(len(self.strategy_names)):
                if strategy_best_fitness[s] < np.inf:
                    # Score inversely proportional to best fitness
                    strategy_win_counts[s] = 1.0 / (strategy_best_fitness[s] + 1e-12)
            # Normalize
            total = strategy_win_counts.sum()
            if total > 0:
                strategy_win_counts /= total
        
        # Compute landscape fingerprint from the probe
        fingerprint = self._compute_landscape_fingerprint(pop, fitness_history, probe_iters)
        
        # Winner is the strategy with highest empirical win rate
        winner_idx = int(np.argmax(strategy_win_counts))
        
        # Store fingerprints for commit phase
        self._probe_fingerprints = {
            'winner': winner_idx,
            'win_counts': strategy_win_counts.copy(),
            'probe_counts': strategy_probe_counts.copy(),
            'landscape': fingerprint,
            'best_fitness': best_overall,
        }
        
        return winner_idx, fingerprint
    
    def _probe_mutate(self, pop, strategy_idx):
        """Simplified mutation for probe phase."""
        NP, dim = pop.shape
        mutants = np.empty_like(pop)
        F = 0.6
        
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            r = others[np.random.choice(len(others), 3, replace=False)]
            
            if strategy_idx == 0:  # original-style rand/1
                mutants[i] = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])
            elif strategy_idx == 1:  # catA: best-based
                best_idx = np.argmin(np.sum(pop, axis=1))
                mutants[i] = pop[best_idx] + F * (pop[r[0]] - pop[r[1]])
            elif strategy_idx == 2:  # catE: current-to-pbest
                top_p = max(1, NP // 10)
                pbest = pop[np.argsort(np.sum(pop, axis=1))[:top_p]][np.random.randint(top_p)]
                mutants[i] = pop[i] + F * (pbest - pop[r[0]]) + F * (pop[r[1]] - pop[r[2]])
            else:  # catG: rand/2
                mutants[i] = pop[r[0]] + F * (pop[r[1]] - pop[r[2]]) + F * (pop[r[0]] - pop[r[1]])
        
        return mutants
    
    def _probe_crossover(self, pop, mutants):
        """Simplified crossover for probe phase."""
        NP, dim = pop.shape
        CR = 0.85
        mask = np.random.rand(NP, dim) < CR
        mask[np.arange(NP), np.random.randint(0, dim, NP)] = True
        return np.where(mask, mutants, pop)
    
    def _call_adapt_strategy(self, strategy_idx, improvement_rate, pop):
        """Call the appropriate adaptation method for the given strategy."""
        # Store population for strategies that need it
        self._current_population = pop
        
        if strategy_idx == 0:
            self._adapt_original(improvement_rate)
        elif strategy_idx == 1:
            self._adapt_catA_spread(improvement_rate)
        elif strategy_idx == 2:
            self._adapt_catE_graph(improvement_rate)
        elif strategy_idx == 3:
            self._adapt_catG_bootstrap(improvement_rate)
    
    # ─────────────────────────────────────────────────────────────────
    # Adaptation strategy implementations (from benchmark winners)
    # ─────────────────────────────────────────────────────────────────
    
    def _adapt_original(self, improvement_rate):
        """Adapt F and CR based on improvement rate (original strategy)."""
        if improvement_rate > 0.25:
            target_F = min(self.F * 1.15, 1.5)
        elif improvement_rate < 0.1:
            target_F = max(self.F * 0.85, 0.2)
        else:
            target_F = self.F
        self.F = 0.7 * self.F + 0.3 * target_F
        
        if improvement_rate > 0.2:
            target_CR = min(self.CR * 1.05, 0.98)
        else:
            target_CR = max(self.CR * 0.95, 0.4)
        self.CR = 0.6 * self.CR + 0.4 * target_CR
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    def _adapt_catA_spread(self, improvement_rate):
        """variant_01: Adapt F and CR based on geometric spatial spread of population."""
        pop = getattr(self, '_current_population', None)
        if pop is None or len(pop) < 5:
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        pop_min = pop.min(axis=0)
        pop_max = pop.max(axis=0)
        axis_spread = pop_max - pop_min
        normalized_spread = axis_spread / 200.0
        mean_spread = normalized_spread.mean()
        
        centroid = pop.mean(axis=0)
        centroid_distances = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = centroid_distances.mean()
        expected_dist = 200.0 * np.sqrt(self.dim / max(self.NP, 1))
        normalized_centroid_dist = mean_centroid_dist / max(expected_dist, 1e-6)
        
        spread_factor = np.clip(normalized_centroid_dist, 0.1, 2.0)
        target_F = 0.6 / (spread_factor + 0.3)
        target_F = float(np.clip(target_F, 0.3, 1.5))
        
        target_CR = 0.5 + 0.4 * np.clip(spread_factor - 0.5, 0, 1)
        target_CR = float(np.clip(target_CR, 0.3, 0.95))
        
        if len(self.F_history) > 0:
            alpha = 0.3
            self.F = alpha * target_F + (1 - alpha) * self.F_history[-1]
            self.CR = alpha * target_CR + (1 - alpha) * self.CR_history[-1]
        else:
            self.F = target_F
            self.CR = target_CR
        
        self.F = float(np.clip(self.F, 0.1, 2.0))
        self.CR = float(np.clip(self.CR, 0.1, 0.99))
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        
        if len(self.F_history) > 50:
            self.F_history = self.F_history[-50:]
            self.CR_history = self.CR_history[-50:]
    
    def _adapt_catE_graph(self, improvement_rate):
        """variant_05: Adapt F and CR based on k-NN graph topology of the population."""
        pop = getattr(self, '_current_population', None)
        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
        if pop is None or len(pop) < 5:
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        NP, dim = pop.shape
        k = max(2, min(5, NP // 10))
        
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
        
        avg_edge_length = float(np.mean(np.sqrt(nearest_sq_dists)))
        
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
        
        avg_clustering = float(np.mean(clustering_coeffs))
        
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
        
        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * float(edge_length_trend < -0.01)
        clustering_factor += 0.08 * float(clustering_trend > 0.05)
        self.F = float(np.clip(self.F * clustering_factor, 0.3, 2.0))
        
        median_edge = float(np.median(np.sqrt(sq_dists[sq_dists != np.inf])))
        if avg_edge_length > median_edge * 0.5:
            self.CR = float(np.clip(self.CR * 1.08, 0.1, 0.95))
        else:
            self.CR = float(np.clip(self.CR * 0.93, 0.1, 0.95))
        
        if improvement_rate > 0.15:
            self.F = float(np.clip(self.F * 1.05, 0.3, 2.0))
        elif improvement_rate < 0.03:
            self.F = float(np.clip(self.F * 1.12, 0.3, 2.0))
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    def _adapt_catG_bootstrap(self, improvement_rate):
        """variant_07: Adapt F and CR using bootstrap confidence intervals of improvement rates."""
        if not hasattr(self, '_improvement_history'):
            self._improvement_history = []
        
        self._improvement_history.append(improvement_rate)
        if len(self._improvement_history) > 50:
            self._improvement_history.pop(0)
        
        n_history = len(self._improvement_history)
        if n_history < 5:
            self.F = float(np.clip(self.F + np.random.randn() * 0.05, 0.3, 1.5))
            self.CR = float(np.clip(self.CR + np.random.randn() * 0.03, 0.3, 0.99))
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        history_array = np.array(self._improvement_history)
        n_bootstrap = 100
        bootstrap_indices = np.random.randint(0, n_history, size=(n_bootstrap, n_history))
        bootstrap_means = history_array[bootstrap_indices].mean(axis=1)
        
        ci_lower = float(np.percentile(bootstrap_means, 10))
        ci_upper = float(np.percentile(bootstrap_means, 90))
        ci_width = max(ci_upper - ci_lower, 0.01)
        
        norm_rate = float(np.clip((ci_lower + ci_width * np.random.rand()) / max(ci_upper, 0.1), 0, 1))
        
        target_F = 0.6 * (0.5 + 1.5 * norm_rate)
        target_CR = 0.85 * (1.3 - 0.5 * norm_rate)
        
        alpha = 0.3
        self.F = float(np.clip(alpha * target_F + (1 - alpha) * self.F + np.random.randn() * 0.05, 0.3, 1.5))
        self.CR = float(np.clip(alpha * target_CR + (1 - alpha) * self.CR + np.random.randn() * 0.03, 0.3, 0.99))
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        
        if len(self.F_history) > 100:
            self.F_history = self.F_history[-100:]
            self.CR_history = self.CR_history[-100:]
    
    # ─────────────────────────────────────────────────────────────────
    # Core DE methods (unchanged from original, with strategy dispatch)
    # ─────────────────────────────────────────────────────────────────
    
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
        top_p = max(1, int(np.ceil(p * self.NP)))
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
        probs = np.exp(self._strategy_scores - self._strategy_scores.max())
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
    
    def _crossover_batch(self, population, mutants):
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        for s in range(len(self._strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self._strategy_scores[s] += 0.1 * mask.sum()
        self._strategy_scores *= 0.99
        self._strategy_scores = np.clip(self._strategy_scores, 0.01, 10.0)
    
    def _dispatch_adaptation(self, improvement_rate, population):
        """Dispatch to the committed adaptation strategy.
        This is the PROBE-AND-COMMIT core: no blending, pure dispatch."""
        self._current_population = population
        
        if self._committed_strategy is None:
            # Fallback: use original if no commitment yet
            self._adapt_original(improvement_rate)
        else:
            # Commit to the probe-winning strategy exclusively
            self._call_adapt_strategy(self._committed_strategy, improvement_rate, population)
    
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
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness_val = fitness[best_idx]
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        return new_pop, best_fitness_val, best_solution
    
    # ─────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop with PROBE-AND-COMMIT adaptation."""
        # Estimate remaining budget from stopping_condition
        # We use a conservative estimate: run at least some iterations
        initial_pop = self._initialize_population()
        initial_fitness = func(initial_pop)
        
        if len(initial_fitness) < self.NP:
            initial_fitness = np.full(self.NP, np.nan)
        
        # Quick probe to estimate convergence speed
        probe_test_pop = self._initialize_population()
        probe_test_fitness = func(probe_test_pop)
        if len(probe_test_fitness) < self.NP:
            probe_test_fitness = np.full(self.NP, np.nan)
        
        # Estimate budget: if function is cheap (fast probe), use more iterations
        # We estimate budget based on initial convergence
        try:
            best_init = float(np.nanmin(probe_test_fitness))
            # Run a tiny micro-probe (3 generations) to estimate convergence speed
            micro_pop = probe_test_pop.copy()
            micro_fit = probe_test_fitness.copy()
            for _ in range(3):
                mutants = self._probe_mutate(micro_pop, 0)
                trials = self._probe_crossover(micro_pop, mutants)
                trials = np.clip(trials, -100.0, 100.0)
                tfit = func(trials)
                if len(tfit) < len(trials):
                    tfit = np.full(len(trials), np.nan)
                better = tfit < micro_fit
                micro_pop = np.where(better[:, np.newaxis], trials, micro_pop)
                micro_fit = np.where(better, tfit, micro_fit)
            
            micro_best = float(np.nanmin(micro_fit))
            # If micro-probe improved significantly, function is informative
            # Use full budget; otherwise reduce
            if micro_best >= best_init * 0.99 and best_init > 1e-3:
                estimated_budget = min(self.max_iter, 2000)
            else:
                estimated_budget = self.max_iter
        except Exception:
            estimated_budget = self.max_iter
        
        # PROBE PHASE: run short burst to select best adaptation strategy
        probe_budget = max(10, int(estimated_budget * 0.05))
        winner_idx, fingerprint = self._run_probe_phase(func, probe_budget)
        self._committed_strategy = winner_idx
        self._probe_completed = True
        
        # Reset F/CR to neutral values after probe (adaptation methods may have left them in unusual states)
        self.F = 0.6
        self.CR = 0.85
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Main optimization with committed strategy
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        post_commit_stagnation = 0
        
        while not stopping_condition():
            # Compass mutation
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
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
                post_commit_stagnation = 0
            else:
                post_commit_stagnation += 1
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive
            self._update_archive(population, fitness, improved_mask)
            
            # PROBE-AND-COMMIT: dispatch to the winning adaptation strategy
            improvement_rate = float(np.mean(improved_mask))
            self._dispatch_adaptation(improvement_rate, population)
            
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
            
            # Epsilon-switch contingency: if committed strategy stagnates for too long,
            # allow a small probability of switching to the runner-up strategy
            if post_commit_stagnation > self._post_stagnation_threshold:
                if np.random.rand() < self._epsilon_switch_prob:
                    # Find runner-up from probe results
                    win_counts = self._probe_fingerprints.get('win_counts', np.ones(4))
                    sorted_strategies = np.argsort(win_counts)[::-1]
                    runner_up = sorted_strategies[1] if len(sorted_strategies) > 1 else 0
                    # Override commitment temporarily
                    self._committed_strategy = runner_up
                    post_commit_stagnation = 0
                else:
                    # Reset to committed strategy
                    self._committed_strategy = winner_idx
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
