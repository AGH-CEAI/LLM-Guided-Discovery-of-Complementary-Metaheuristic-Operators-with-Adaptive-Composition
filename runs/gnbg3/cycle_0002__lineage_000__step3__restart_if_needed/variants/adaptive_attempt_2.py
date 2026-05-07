import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Contextual Bandit with Probe-and-Commit.
    
    Chosen because the gap table is BLEND-NEUTRAL (max gap 3.9×, no gap ≥ 10×),
    so linear blending CAN preserve per-task advantages. However, with 4 distinct
    winners and no single variant dominating >50% of wins, a bandit that probes
    each variant on the actual landscape is more robust than hard-coded rules.
    The probe extracts rank-normalized improvement signals (no scale-dependent
    constants) and commits to the best variant for the remainder of the run.
    
    The algorithm adapts WHICH `_restart_if_needed` strategy to use by:
      1) PROBE: run each of 4 restart variants for a short burst (~50 gens each)
      2) COMMIT: use the variant with highest rank-based improvement for the rest
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
        
        # --- Restart adaptation state ---
        self._num_restart_variants = 4
        self._probe_budget_per_variant = 50
        self._probe_pop_fraction = 0.25
        self._best_restart_variant = None
        self._context_features = None
        self._probe_history = []
        
        # EMA tracking for strategy scores
        self._ema_rank_improvement = None
        self._ema_success_rate = None
        self._rank_improvement_history = None
        self._success_rate_history = None
        self._momentum = 0.3
    
    def _init_strategy_tracking(self):
        n_strats = len(self.strategy_scores)
        self._ema_rank_improvement = np.zeros(n_strats)
        self._ema_success_rate = np.zeros(n_strats)
        self._rank_improvement_history = [[] for _ in range(n_strats)]
        self._success_rate_history = [[] for _ in range(n_strats)]
    
    def _probe_restart_variants(self, func, stopping_condition):
        """Run each restart variant for a short burst and return rank-based improvement."""
        probe_results = []
        probe_NP = max(10, int(self.NP * self._probe_pop_fraction))
        
        for variant_id in range(self._num_restart_variants):
            # Save state
            saved_state = {
                'F': self.F,
                'CR': self.CR,
                'archive': [x.copy() for x in self.archive],
                'stagnation_counter': self.stagnation_counter,
                'prev_best': self.prev_best_fitness,
                'strategy_scores': self.strategy_scores.copy(),
                'ema_ri': self._ema_rank_improvement.copy() if self._ema_rank_improvement is not None else None,
                'ema_sr': self._ema_success_rate.copy() if self._ema_success_rate is not None else None,
            }
            
            # Reset for probe
            self.F = 0.6
            self.CR = 0.85
            self.archive = []
            self.stagnation_counter = 0
            self.prev_best_fitness = np.inf
            self._init_strategy_tracking()
            
            # Run short optimization with this restart variant
            pop = np.random.uniform(-100.0, 100.0, (probe_NP, self.dim))
            fit = func(pop)
            best_fit = np.min(fit)
            
            rank_improvements = []
            for _ in range(self._probe_budget_per_variant):
                # One generation of optimization
                mutants = self._mutate_compass_batch(pop, fit)
                _, strat_used = self._mutate_ensemble_batch(pop, fit)
                blend_r = np.random.rand(probe_NP, 1)
                mutants = np.where(blend_r < 0.6, mutants[0], mutants[1])
                trials = self._crossover_batch(pop, mutants)
                trials = np.clip(trials, -100.0, 100.0)
                trial_fit = func(trials)
                pop, fit, improved = self._select_survivors_batch(pop, fit, trials, trial_fit)
                self._update_strategy_scores(strat_used, improved)
                
                # Record rank-based improvement
                cur_best = np.min(fit)
                rank_before = np.argsort(np.argsort(fit))
                best_before_rank = rank_before[np.argmin(fit)]
                
                # Apply restart variant
                if variant_id == 0:
                    pop, best_r, _ = self._restart_variant_01(pop, fit)
                elif variant_id == 1:
                    pop, best_r, _ = self._restart_variant_05(pop, fit)
                elif variant_id == 2:
                    pop, best_r, _ = self._restart_variant_07(pop, fit)
                else:
                    pop, best_r, _ = self._restart_variant_original(pop, fit)
                
                fit = func(pop)
                cur_best = np.min(fit)
                best = cur_best
                
                rank_after = np.argsort(np.argsort(fit))
                best_after_rank = rank_after[np.argmin(fit)]
                rank_imp = best_before_rank - best_after_rank
                rank_improvements.append(rank_imp / max(probe_NP - 1, 1))
                
                if np.min(fit) < best_fit:
                    best_fit = np.min(fit)
            
            mean_rank_imp = np.mean(rank_improvements[-10:]) if len(rank_improvements) >= 10 else np.mean(rank_improvements)
            probe_results.append({'variant': variant_id, 'mean_rank_imp': mean_rank_imp, 'best_fit': best_fit})
            
            # Restore state
            self.F = saved_state['F']
            self.CR = saved_state['CR']
            self.archive = saved_state['archive']
            self.stagnation_counter = saved_state['stagnation_counter']
            self.prev_best_fitness = saved_state['prev_best']
            self.strategy_scores = saved_state['strategy_scores']
            if saved_state['ema_ri'] is not None:
                self._ema_rank_improvement = saved_state['ema_ri']
                self._ema_success_rate = saved_state['ema_sr']
        
        # Commit to best variant
        probe_results.sort(key=lambda x: x['mean_rank_imp'], reverse=True)
        self._best_restart_variant = probe_results[0]['variant']
        self._probe_history = probe_results
        self._context_features = self._compute_context_features(None)
    
    def _compute_context_features(self, population):
        """Extract cheap landscape fingerprint from recent optimization history."""
        features = np.zeros(4)
        
        # Convergence rate: slope of log(best_fitness) over last 10 generations
        if len(self.F_history) >= 5:
            recent = list(zip(self.F_history[-10:], self.CR_history[-10:]))
            if len(recent) >= 3:
                gens = np.arange(len(recent))
                f_vals = np.array([x[0] for x in recent])
                if np.std(f_vals) > 1e-10:
                    slope = np.polyfit(gens, f_vals, 1)[0]
                    features[0] = np.clip(slope, -1.0, 1.0)
        
        # Diversity: from archive if available
        if len(self.archive) > 0:
            arch = np.vstack(self.archive)
            if len(arch) > 1:
                cent = arch.mean(axis=0)
                div = np.mean(np.linalg.norm(arch - cent, axis=1))
                features[1] = np.clip(div / 100.0, 0.0, 1.0)
        
        # Stagnation depth
        features[2] = np.clip(self.stagnation_counter / 100.0, 0.0, 1.0)
        
        # Effective dimensionality proxy from F/CR history
        if len(self.F_history) >= 5:
            f_range = max(self.F_history[-10:]) - min(self.F_history[-10:])
            cr_range = max(self.CR_history[-10:]) - min(self.CR_history[-10:])
            features[3] = np.clip((f_range + cr_range) / 2.0, 0.0, 1.0)
        
        return features
    
    # === Restart variant implementations (from benchmark winners) ===
    
    def _restart_variant_01(self, population, fitness):
        """Convex hull and geometric clustering (variant_01, 6 wins)."""
        NP, dim = population.shape
        centroid = population.mean(axis=0)
        centroid_dists = np.linalg.norm(population - centroid, axis=1)
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        n_extreme = max(2, NP // 4)
        extreme_mask = np.zeros(NP, dtype=bool)
        sorted_by_dist = np.argsort(centroid_dists)[::-1]
        extreme_mask[sorted_by_dist[:n_extreme]] = True
        extreme_indices = np.where(extreme_mask)[0]
        
        k = max(3, min(8, NP // 5))
        nearest = np.argsort(sq_dists, axis=1)[:, :k]
        median_dist = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        local_density = np.sum(np.sqrt(sq_dists) < median_dist, axis=1)
        
        n_clusters = max(2, NP // 6)
        cluster_rep_indices = []
        remaining = np.ones(NP, dtype=bool)
        remaining[extreme_indices] = False
        
        for _ in range(n_clusters):
            if not remaining.any():
                break
            densities = np.where(remaining, local_density, np.inf)
            rep_idx = np.argmin(densities)
            cluster_rep_indices.append(rep_idx)
            neighbor_mask = np.isin(nearest[rep_idx], np.where(remaining)[0])
            nearby = nearest[rep_idx][neighbor_mask]
            remaining[nearby] = False
            remaining[rep_idx] = False
        
        new_pop = []
        for idx in extreme_indices:
            new_pop.append(population[idx].copy())
        for idx in cluster_rep_indices:
            if len(new_pop) < NP:
                new_pop.append(population[idx].copy())
        
        axis_ranges = population.max(axis=0) - population.min(axis=0)
        axis_ranges = np.maximum(axis_ranges, 1e-6)
        while len(new_pop) < NP:
            t = np.random.rand(dim)
            corner = population.min(axis=0) + t * axis_ranges
            point = centroid + 0.5 * (corner - centroid) + np.random.randn(dim) * np.sqrt(axis_ranges) * 0.1
            new_pop.append(np.clip(point, -100.0, 100.0))
        
        return np.array(new_pop[:NP]), fitness.min(), population[np.argmin(fitness)].copy()
    
    def _restart_variant_05(self, population, fitness):
        """k-NN graph connectivity analysis (variant_05, 10 wins)."""
        NP, dim = population.shape
        k = max(2, min(5, NP // 8))
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        connectivity = np.zeros(NP)
        for i in range(NP):
            for n in nearest_indices[i]:
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
        isolation_score = ((1.0 - connectivity / max(connectivity.max(), 1e-10)) * 0.5 +
                          (1.0 - clustering) * 0.3 +
                          fitness_ranks * 0.2)
        
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
            direction = source - population[idx]
            mutation_scale = np.random.uniform(0.8, 1.5)
            local_edges = sq_dists[idx, nearest_indices[idx]]
            avg_local_dist = np.mean(np.sqrt(local_edges))
            new_point = source + mutation_scale * self.F * direction + np.random.randn(dim) * avg_local_dist * 0.5
            new_individuals.append(np.clip(new_point, -100.0, 100.0))
        
        for i, idx in enumerate(replace_indices):
            population[idx] = new_individuals[i]
        
        self.stagnation_counter = 0
        return population, fitness.min(), population[np.argmin(fitness)].copy()
    
    def _restart_variant_07(self, population, fitness):
        """Stochastic bootstrap with LHS (variant_07, 1 win)."""
        n, dim = population.shape
        bounds = (-100.0, 100.0)
        n_elite = max(3, n // 4)
        elite_idx = np.argpartition(fitness, n_elite)[:n_elite]
        elite_pop = population[elite_idx]
        
        weights = 1.0 / (fitness[elite_idx] + 1e-10)
        weights /= weights.sum()
        weighted_mean = np.average(elite_pop, axis=0, weights=weights)
        centered = elite_pop - weighted_mean
        cov = np.zeros((dim, dim))
        for i in range(len(elite_pop)):
            cov += weights[i] * np.outer(centered[i], centered[i])
        cov += np.eye(dim) * 1e-6
        
        best_f = np.min(fitness)
        noise_scale = max(50.0, 200.0 / (best_f + 1.0))
        
        new_pop = np.zeros((n, dim))
        for i in range(n):
            for _ in range(10):
                sample = np.random.multivariate_normal(weighted_mean, cov * noise_scale)
                if np.all((sample >= bounds[0]) & (sample <= bounds[1])):
                    new_pop[i] = sample
                    break
            else:
                new_pop[i] = np.random.uniform(bounds[0], bounds[1], dim)
        
        best_idx = np.argmin(fitness)
        new_pop[0] = population[best_idx]
        second_best = np.argmin(np.where(np.arange(n) != best_idx, fitness, np.inf))
        new_pop[1] = population[second_best]
        
        self.archive.append(population[elite_idx].copy())
        self.stagnation_counter = 0
        return new_pop, fitness[best_idx], population[best_idx].copy()
    
    def _restart_variant_original(self, population, fitness):
        """Original: simple reinitialization (original.py, 4 wins)."""
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        return new_pop, best_fitness, best_solution
    
    def _do_restart(self, population, fitness):
        """Dispatch to the committed restart variant."""
        if self._best_restart_variant == 0:
            return self._restart_variant_01(population, fitness)
        elif self._best_restart_variant == 1:
            return self._restart_variant_05(population, fitness)
        elif self._best_restart_variant == 2:
            return self._restart_variant_07(population, fitness)
        else:
            return self._restart_variant_original(population, fitness)
    
    # === Remaining methods from original algorithm ===
    
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
        best_idx = np.argmin(fitness_ranks)
        
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
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
        n_strategies = len(self.strategy_scores)
        n = len(strategy_used)
        
        if self._ema_rank_improvement is None:
            self._init_strategy_tracking()
        
        rank_improvement = np.zeros(n_strategies)
        success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()
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
            delta[s] = ((1 - self._momentum) * comparative_advantage[s] + 
                       self._momentum * hybrid_signal[s]) * (0.5 + 0.5 * confidence[s])
        
        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _adapt_parameters(self, improvement_rate):
        if not hasattr(self, '_current_population') or self._current_population is None or len(self._current_population) < 5:
            return
        
        pop = self._current_population
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
        
        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
        self._graph_metrics_history.append({
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
            'graph_diameter': graph_diameter
        })
        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)
        
        edge_length_trend = 0.0
        clustering_trend = 0.0
        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]
            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
            clustering_trend = recent_clustering[-1] - recent_clustering[0]
        
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
    
    def __call__(self, func, stopping_condition):
        # Phase 1: Probe each restart variant
        self._probe_restart_variants(func, stopping_condition)
        
        # Phase 2: Full optimization with committed variant
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
            self._current_population = population
            
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
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
            
            # Stagnation check and restart with committed variant
            restart_triggered = self._check_stagnation(f_opt)
            periodic_restart = (iteration > 0 and iteration % 200 == 0)
            
            if restart_triggered or periodic_restart:
                population, f_opt, x_opt = self._do_restart(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
