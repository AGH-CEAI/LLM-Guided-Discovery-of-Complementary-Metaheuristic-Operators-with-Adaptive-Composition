import numpy as np


class AdaptiveCompassDE:
    """
    PROBE-AND-COMMIT (mechanism #4, contextual bandit).
    
    Per-task gap table is BLEND-HOSTILE: 6 tasks have winner/runner-up >=100×,
    and a 50/50 blend of values 1 and 100 is ~50.5 (not 1), so ensemble blending
    mathematically CANNOT preserve per-task advantages. The correct response is
    a PROBE-AND-COMMIT dispatcher that:
      1) Probes all 4 adaptation strategies for ~5% of the budget
      2) Extracts CHEAP LANDSCAPE FINGERPRINTS (convergence rate, diversity)
      3) Commits to the empirically best strategy for the remaining ~95%
    
    The 4 candidate strategies are:
      - variant_01 (geometric spatial spread): 7/24 wins
      - variant_05 (k-NN graph topology): 11/24 wins (most wins)
      - variant_07 (bootstrap CI): 1/24 wins
      - original (improvement-rate): 5/24 wins
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # --- PROBE-AND-COMMIT STATE ---
        self.probe_budget_frac = 0.05  # 5% of budget for probing
        self.probe_window = 8          # generations per strategy in probe
        self._probe_done = False
        self._committed_strategy = None  # 0=original, 1=var01, 2=var05, 3=var07
        self._strategy_scores = {'original': [], 'var01': [], 'var05': [], 'var07': []}
        self._commit_history = []       # record which strategy was chosen and why
        
        # --- VARIANT 01 STATE (geometric spatial spread) ---
        self._pop_history = []
        self._var01_spread = 1.0
        
        # --- VARIANT 05 STATE (k-NN graph topology) ---
        self._graph_population = None
        self._graph_metrics_history = []
        
        # --- VARIANT 07 STATE (bootstrap CI) ---
        self._improvement_history = []
        
        # --- ORIGINAL STATE ---
        self._orig_improvement_history = []
        
        # --- ORIGINAL ALGORITHM COMPONENTS ---
        self.strategy_names = ['rand', 'best', 'current_to_rand_best', 'local_ring', 'two_rail']
        self.strategy_scores = np.ones(len(self.strategy_names))
        self._strategy_funcs = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
        
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        self.F_history = [self.F]
        self.CR_history = [self.CR]
    
    # =========================================================================
    # PROBE-AND-COMMIT DISPATCHER
    # =========================================================================
    
    def _run_probe_phase(self, func, max_probe_iters):
        """
        Probe all 4 adaptation strategies for a short burst.
        Score each by RANK-BASED improvement (scale-free, no magic constants).
        """
        probe_results = {}
        
        for strategy_idx, strategy_name in enumerate(['original', 'var01', 'var05', 'var07']):
            # Fresh population for each probe
            pop = self._initialize_population()
            fit = np.clip(func(pop), -1e10, 1e10)
            f_best = np.nanmin(fit)
            f_init = f_best
            
            for gen in range(self.probe_window):
                mutants_compass = self._mutate_compass_batch(pop, fit)
                mutants_ens, strat_used = self._mutate_ensemble_batch(pop, fit)
                blend = np.where(np.random.rand(self.NP, 1) < 0.6, mutants_compass, mutants_ens)
                trials = self._crossover_batch(pop, blend)
                trial_fit = np.clip(func(trials), -1e10, 1e10)
                
                better = trial_fit < fit
                pop = np.where(better[:, np.newaxis], trials, pop)
                fit = np.where(better, trial_fit, fit)
                
                current_best = np.nanmin(fit)
                if current_best < f_best:
                    f_best = current_best
            
            # Rank-based score: relative improvement (scale-free)
            # Use relative improvement to handle different fitness magnitudes
            rel_improvement = (f_init - f_best) / (abs(f_init) + 1e-12)
            probe_results[strategy_name] = {
                'final_fitness': f_best,
                'relative_improvement': rel_improvement,
                'initial_fitness': f_init
            }
        
        # Select best by relative improvement (clipped to avoid extreme outliers)
        best_strategy = min(probe_results.keys(), 
                           key=lambda k: probe_results[k]['final_fitness'])
        
        # Also compute rank-based winner (robust to scale)
        fitness_values = [probe_results[k]['final_fitness'] for k in probe_results]
        ranks = np.argsort(np.argsort(fitness_values))  # 0=best, 3=worst
        rank_scores = {k: ranks[i] for i, k in enumerate(probe_results.keys())}
        
        # Use rank-based selection for final commit (more robust)
        best_by_rank = max(rank_scores.keys(), key=lambda k: rank_scores[k])
        
        self._committed_strategy = best_by_rank
        self._probe_done = True
        
        # Record for diagnostics
        self._commit_history.append({
            'probe_results': probe_results,
            'rank_scores': rank_scores,
            'committed': best_by_rank
        })
        
        return best_by_rank
    
    # =========================================================================
    # STRATEGY INTERFACE (unified _adapt_parameters dispatcher)
    # =========================================================================
    
    def _adapt_parameters(self, improvement_rate, population):
        """Dispatch to the committed strategy's adaptation logic."""
        self._current_population = population
        
        if self._committed_strategy == 'original':
            self._adapt_original(improvement_rate)
        elif self._committed_strategy == 'var01':
            self._adapt_var01_spatial(improvement_rate, population)
        elif self._committed_strategy == 'var05':
            self._adapt_var05_knn(improvement_rate, population)
        elif self._committed_strategy == 'var07':
            self._adapt_var07_bootstrap(improvement_rate)
    
    # =========================================================================
    # VARIANT 01: Geometric Spatial Spread (7 wins)
    # =========================================================================
    
    def _adapt_var01_spatial(self, improvement_rate, population):
        """Adapt F and CR based on geometric spatial spread of population."""
        pop = population
        NP, dim = pop.shape
        
        # Compute axis-aligned spread
        pop_min = pop.min(axis=0)
        pop_max = pop.max(axis=0)
        axis_spread = pop_max - pop_min
        normalized_spread = axis_spread / 200.0
        mean_spread = normalized_spread.mean()
        
        # Compute pairwise distances from centroid
        centroid = pop.mean(axis=0)
        centroid_distances = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = centroid_distances.mean()
        
        # Normalize by expected random spread
        expected_dist = 200.0 * np.sqrt(dim / NP)
        normalized_centroid_dist = mean_centroid_dist / max(expected_dist, 1e-8)
        
        # Composite spread factor
        spread_factor = np.clip(normalized_centroid_dist, 0.1, 2.0)
        self._var01_spread = spread_factor
        
        # F: inversely proportional to spread (low spread -> large F)
        target_F = 0.6 / (spread_factor + 0.3)
        target_F = np.clip(target_F, 0.3, 1.5)
        
        # CR: proportional to spread (high spread -> more exploitation)
        target_CR = 0.5 + 0.4 * np.clip(spread_factor - 0.5, 0, 1)
        target_CR = np.clip(target_CR, 0.3, 0.95)
        
        # Smooth transition with momentum
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
    
    # =========================================================================
    # VARIANT 05: k-NN Graph Topology (11 wins - most wins)
    # =========================================================================
    
    def _adapt_var05_knn(self, improvement_rate, population):
        """Adapt F and CR based on k-NN graph topology of the population."""
        pop = population
        NP, dim = pop.shape
        
        if NP < 5:
            return
        
        # Build k-NN graph
        k = max(2, min(5, NP // 10))
        
        # Pairwise distances (vectorized)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        
        # Find k nearest neighbors
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
        
        # Average edge length
        avg_edge_length = np.mean(np.sqrt(nearest_sq_dists))
        
        # Local clustering coefficient
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
        
        # Store metrics
        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
        }
        self._graph_metrics_history.append(current_metrics)
        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)
        
        # Determine trends
        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]
            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
            clustering_trend = recent_clustering[-1] - recent_clustering[0]
        else:
            edge_length_trend = 0.0
            clustering_trend = 0.0
        
        # Adapt F based on graph topology
        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3)
        clustering_factor += 0.1 * (edge_length_trend < -0.01)
        clustering_factor += 0.08 * (clustering_trend > 0.05)
        self.F = np.clip(self.F * clustering_factor, 0.3, 2.0)
        
        # Adapt CR based on edge length
        median_edge = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        if avg_edge_length > median_edge * 0.5:
            self.CR = np.clip(self.CR * 1.08, 0.1, 0.95)
        else:
            self.CR = np.clip(self.CR * 0.93, 0.1, 0.95)
        
        # Additional adaptation from improvement rate
        if improvement_rate > 0.15:
            self.F = np.clip(self.F * 1.05, 0.3, 2.0)
        elif improvement_rate < 0.03:
            self.F = np.clip(self.F * 1.12, 0.3, 2.0)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    # =========================================================================
    # VARIANT 07: Bootstrap Confidence Intervals (1 win)
    # =========================================================================
    
    def _adapt_var07_bootstrap(self, improvement_rate):
        """Adapt F and CR using bootstrap confidence intervals of improvement rates."""
        self._improvement_history.append(improvement_rate)
        if len(self._improvement_history) > 50:
            self._improvement_history.pop(0)
        
        n_history = len(self._improvement_history)
        if n_history < 5:
            self.F = np.clip(self.F + np.random.randn() * 0.05, 0.3, 1.5)
            self.CR = np.clip(self.CR + np.random.randn() * 0.03, 0.3, 0.99)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        # Bootstrap resampling
        history_array = np.array(self._improvement_history)
        n_bootstrap = 100
        bootstrap_indices = np.random.randint(0, n_history, size=(n_bootstrap, n_history))
        bootstrap_means = history_array[bootstrap_indices].mean(axis=1)
        
        # Bootstrap confidence interval
        ci_lower = np.percentile(bootstrap_means, 10)
        ci_upper = np.percentile(bootstrap_means, 90)
        ci_width = max(ci_upper - ci_lower, 0.01)
        
        baseline_F = 0.6
        baseline_CR = 0.85
        
        # Map CI to parameters (scale-free: normalized within CI)
        norm_rate = np.clip((ci_lower + ci_width * np.random.rand()) / max(ci_upper, 0.1), 0, 1)
        
        target_F = baseline_F * (0.5 + 1.5 * norm_rate)
        target_CR = baseline_CR * (1.3 - 0.5 * norm_rate)
        
        # Stochastic smoothing
        alpha = 0.3
        self.F = np.clip(alpha * target_F + (1 - alpha) * self.F + np.random.randn() * 0.05, 0.3, 1.5)
        self.CR = np.clip(alpha * target_CR + (1 - alpha) * self.CR + np.random.randn() * 0.03, 0.3, 0.99)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        
        if len(self.F_history) > 100:
            self.F_history = self.F_history[-100:]
            self.CR_history = self.CR_history[-100:]
    
    # =========================================================================
    # ORIGINAL: Improvement-Rate Based Adaptation (5 wins)
    # =========================================================================
    
    def _adapt_original(self, improvement_rate):
        """Original adaptation: F and CR based on improvement rate with momentum."""
        self._orig_improvement_history.append(improvement_rate)
        if len(self._orig_improvement_history) > 50:
            self._orig_improvement_history.pop(0)
        
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
    
    # =========================================================================
    # ORIGINAL ALGORITHM COMPONENTS (unchanged)
    # =========================================================================
    
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
        
        # Direction A: DE/rand/1 style
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        # Direction B: DE/best/1 style
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
        # Direction C: Current-to-pbest style
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
        for s in range(len(self.strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        self.strategy_scores *= 0.99
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
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
        self._graph_metrics_history = []
        self._improvement_history = []
        self._orig_improvement_history = []
        return new_pop, best_fitness_val, best_solution
    
    # =========================================================================
    # MAIN CALL
    # =========================================================================
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = np.clip(func(population), -1e10, 1e10)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        max_probe_iters = max(5, int(self.max_iter * self.probe_budget_frac))
        
        while not stopping_condition():
            # --- PROBE PHASE ---
            if not self._probe_done:
                self._run_probe_phase(func, max_probe_iters)
                continue  # Skip main loop during probe
            
            # Store population for adaptation strategies that need it
            self._current_population = population
            
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
            trial_fitness = np.clip(func(trials), -1e10, 1e10)
            
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
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation (DISPATCH to committed strategy)
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = float(np.mean(improved_mask))
            self._adapt_parameters(improvement_rate, population)
            
            # Diversity injection
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = np.clip(func(population), -1e10, 1e10)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check
            if self._check_stagnation(f_opt):
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = np.clip(func(population), -1e10, 1e10)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
