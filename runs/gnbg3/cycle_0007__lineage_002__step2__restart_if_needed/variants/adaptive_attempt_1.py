import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTIVE RESTART MECHANISM: Thompson Sampling Bandit (mechanism #1)
    
    Two candidate restart strategies:
      - Arm 0: original k-NN graph connectivity restart
      - Arm 1: variant_08 fitness-variance + spatial-clustering restart
    
    Thompson Sampling with Beta posteriors selects the arm each time a
    restart is triggered.  Rewards are RANK-BASED (within a sliding
    window of K=12) so no scale-dependent constants are needed.
    
    Gap-table rationale:
      - Max per-task gap = 2.7× (no blend-hostile tasks).  Linear blending
        is viable but a bandit is equally safe and learns the preference
        online without hard-coding a dispatch table.  Each arm can be
        selected with weight 1.0 (commit), preserving any per-task
        advantage the winner may have.
    
    Calibration:
      - K=12 >= 3 * num_arms (rule b)
      - Rewards are rank-normalized, never raw fitness (rule a)
      - Beta prior = (1,1) uniform (rule e)
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
        
        # ─────────────────────────────────────────────────────────────
        # ADAPTIVE RESTART BANDIT — Thompson Sampling over 2 arms
        # ─────────────────────────────────────────────────────────────
        self._num_arms = 2          # arm 0 = original, arm 1 = variant_08
        self._alpha = np.ones(self._num_arms)   # Beta success pseudocounts
        self._beta  = np.ones(self._num_arms)   # Beta failure pseudocounts
        
        # Sliding window of past restart outcomes
        # Each entry: (arm_used, rank_reward)  where rank_reward ∈ [0,1]
        K = 12                       # >= 3 * num_arms (rule b)
        self._restart_window = []    # will hold at most K entries
        self._window_size = K
        
        # Track which arm was last used so we can attribute the reward
        self._last_arm = -1
        self._last_best_fitness_before_restart = np.inf
    
    # ─────────────────────────────────────────────────────────────────
    #  ARM-SELECTION: Thompson Sampling
    # ─────────────────────────────────────────────────────────────────
    def _sample_arm(self):
        """Thompson Sampling: sample from Beta posteriors and pick the best."""
        samples = np.array([np.random.beta(self._alpha[i], self._beta[i])
                            for i in range(self._num_arms)])
        return int(np.argmax(samples))
    
    def _update_bandit(self, arm, rank_reward):
        """Update Beta posteriors using rank-based reward (no scale magic)."""
        # rank_reward is in [0,1]: 1 = best improvement, 0 = worst in window
        # Map to binary success/failure via a threshold at 0.5
        success = rank_reward >= 0.5
        if success:
            self._alpha[arm] += 1.0
        else:
            self._beta[arm] += 1.0
    
    def _record_restart_outcome(self, arm, best_fitness_after):
        """Record outcome in sliding window and update bandit posterior."""
        # Compute relative improvement (negative = improved)
        prev = self._last_best_fitness_before_restart
        curr = best_fitness_after
        if not np.isfinite(prev) or not np.isfinite(curr):
            delta = 0.0
        else:
            # Use relative improvement to handle different scales
            denom = max(abs(prev), 1e-12)
            delta = (prev - curr) / denom  # positive = improvement
        
        # Store in sliding window
        self._restart_window.append(delta)
        if len(self._restart_window) > self._window_size:
            self._restart_window.pop(0)
        
        # Rank-normalize within the window (rule a: no scale constants)
        if len(self._restart_window) >= 2:
            deltas = np.array(self._restart_window)
            # Rank: highest improvement (largest positive delta) = rank 1.0
            ranks = np.argsort(np.argsort(-deltas))          # descending rank
            rank_reward = ranks[-1] / (len(ranks) - 1)       # winner gets 1.0
        else:
            rank_reward = 1.0  # first entry: optimistic default
        
        self._update_bandit(arm, rank_reward)
    
    # ─────────────────────────────────────────────────────────────────
    #  ORIGINAL _restart_if_needed (arm 0)
    # ─────────────────────────────────────────────────────────────────
    def _restart_original(self, population, fitness):
        """Restart based on k-NN graph connectivity analysis (arm 0)."""
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
        isolation_score = (1.0 - connectivity / (connectivity.max() + 1e-10)) * 0.5 + \
                          (1.0 - clustering) * 0.3 + \
                          fitness_ranks * 0.2
        
        n_replace = max(NP // 4, 3)
        replace_indices = np.argsort(isolation_score)[-n_replace:]
        n_best = min(5, NP)
        best_indices = np.argsort(fitness)[:n_best]
        
        new_individuals = []
        for idx in replace_indices:
            best_source = best_indices[0]
            best_graph_dist = -1
            for b_idx in best_indices:
                n1 = set(nearest_indices[idx])
                n2 = set(nearest_indices[b_idx])
                common = len(n1 & n2)
                graph_dist_approx = 1.0 / (common + 0.1)
                if graph_dist_approx > best_graph_dist:
                    best_graph_dist = graph_dist_approx
                    best_source = b_idx
            
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
        return population, fitness[best_idx], population[best_idx].copy()
    
    # ─────────────────────────────────────────────────────────────────
    #  VARIANT_08 _restart_if_needed (arm 1)
    # ─────────────────────────────────────────────────────────────────
    def _restart_variant08(self, population, fitness):
        """Hybrid restart: fitness-variance + spatial-clustering (arm 1)."""
        if len(population) < 4:
            best_idx = np.nanargmin(fitness)
            return population, fitness[best_idx], population[best_idx].copy()
        
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
        stagnation_signal = fitness_severity * (fitness_severity / total_severity) + \
                            spatial_severity * (spatial_severity / total_severity)
        
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
                    population[idx] = np.clip(donor + np.random.randn(self.dim) * noise_scale,
                                              -100.0, 100.0)
                else:
                    population[idx] = np.random.uniform(-100.0, 100.0, self.dim)
            
            self._fitness_cv_history = self._fitness_cv_history[-3:]
            self._spatial_cv_history = self._spatial_cv_history[-3:]
        
        best_idx = np.nanargmin(fitness)
        return population, fitness[best_idx], population[best_idx].copy()
    
    # ─────────────────────────────────────────────────────────────────
    #  UNIFIED RESTART dispatch: Thompson Sampling → selected arm
    # ─────────────────────────────────────────────────────────────────
    def _restart_if_needed(self, population, fitness):
        """Bandit-selected restart: Thompson Sampling over 2 arms."""
        best_fitness_before = np.nanmin(fitness)
        self._last_best_fitness_before_restart = best_fitness_before
        
        # Sample arm via Thompson Sampling
        arm = self._sample_arm()
        self._last_arm = arm
        
        # Apply the selected arm
        if arm == 0:
            population, f_opt, x_opt = self._restart_original(population, fitness)
        else:
            population, f_opt, x_opt = self._restart_variant08(population, fitness)
        
        # Record outcome for bandit update
        self._record_restart_outcome(arm, f_opt)
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        return population, f_opt, x_opt
    
    # ─────────────────────────────────────────────────────────────────
    #  Original algorithm methods (unchanged)
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
        return ranks / (self.NP - 1)
    
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
            self._current_population = population
            self._adapt_parameters(improvement_rate)
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            if self._check_stagnation(f_opt):
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            iteration += 1
            if stopping_condition():
                break
        
        return f_opt, x_opt
