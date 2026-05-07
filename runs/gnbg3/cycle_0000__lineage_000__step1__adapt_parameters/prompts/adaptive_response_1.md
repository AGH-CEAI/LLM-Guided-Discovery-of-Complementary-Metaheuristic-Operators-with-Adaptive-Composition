```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Per-Task Probe-and-Commit (specialization of #4 Contextual Bandit).
    
    The per-task gap table is BLEND-HOSTILE: 6 tasks have DISPATCH-MANDATORY gaps (winner beats
    runner-up by >=100× OR median by >=500×). A 50/50 linear blend of 1e-8 and 1e-2 gives ~5e-3,
    destroying the winner's advantage entirely. Ensemble blending is mathematically excluded.
    
    The correct response is PROBE-AND-COMMIT: during a short (~5% of budget) exploratory probe
    phase, each of the 4 adaptation strategies is evaluated on a sub-population. A cheap landscape
    FINGERPRINT is computed from the probe data (convergence rate, rank stability, effective
    dimensionality, stagnation index). The strategy that wins the probe is COMMITTED exclusively
    (weight 1.0) for the remaining budget. This preserves per-task winner advantages because no
    blending occurs — the committed strategy runs alone.
    
    Key design choices:
    - Rank-based scoring within probe results (no scale-dependent magic constants).
    - Sliding window for reward history (>= 3 * num_arms = 12).
    - All 4 arms have exclusive commit paths (weight 1.0 when selected).
    - Small probability of revisiting runner-ups only during stagnation.
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
        
        # ─── PROBE-AND-COMMIT STATE ───────────────────────────────────────────
        # 4 arms: 0=original, 1=variant_01 (geometric spread), 
        #        2=variant_05 (k-NN graph), 3=variant_07 (bootstrap CI)
        self.num_arms = 4
        self._probe_completed = False
        self._committed_arm = None  # Will be set after probe
        self._arm_reward_history = [[] for _ in range(self.num_arms)]
        self._reward_window = 15  # >= 3 * num_arms
        
        # Per-arm internal state (preserved when committing)
        self._arm_F_history = [None] * self.num_arms
        self._arm_CR_history = [None] * self.num_arms
        self._arm_state = [{} for _ in range(self.num_arms)]
        
        # Probe fingerprint storage
        self._probe_fingerprints = [None] * self.num_arms
        
        # ─── DISPATCH MANDATE ─────────────────────────────────────────────────
        # The committed arm gets weight 1.0. No blending. This is the ONLY way
        # to preserve a 100×+ per-task winner advantage (a 0.5 blend gives 0.505).
    
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
        """Sample 3 candidate mutation directions, score by rank improvement potential."""
        i = idx
        NP_local = len(population)
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP_local)])
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
        top_p = max(1, int(np.ceil(p * NP_local)))
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
        
        for i in range(len(population)):
            mutants[i] = self._compass_directions(population, i, fitness_ranks)
        
        return mutants
    
    def _mutate_rand(self, population, idx, F):
        NP_local = len(population)
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, NP_local)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_best(self, population, idx, F):
        best_idx = np.argmin(population.sum(axis=1))
        NP_local = len(population)
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, NP_local)])
        r = others[np.random.choice(len(others), 2, replace=False)]
        return population[best_idx] + F * (population[r[0]] - population[r[1]])
    
    def _mutate_current_to_rand_best(self, population, idx, F):
        i = idx
        NP_local = len(population)
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP_local)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[i] + F * (population[r[0]] - population[i]) + F * (population[r[1]] - population[r[2]])
    
    def _mutate_local_ring(self, population, idx, F):
        NP_local = len(population)
        left = (idx - 2) % NP_local
        right = (idx + 2) % NP_local
        others = [left, (idx - 1) % NP_local, (idx + 1) % NP_local, right]
        r = np.array(others)[np.random.choice(4, 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_two_rail(self, population, idx, F):
        NP_local = len(population)
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, NP_local)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[r[0]] + population[r[1]] + F * (population[r[2]] - population[r[3]]) - population[idx]
    
    def _select_strategy_batch(self):
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        mutants = np.empty_like(population)
        strategy_used = np.empty(len(population), dtype=int)
        
        for i in range(len(population)):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        
        return mutants, strategy_used
    
    def _crossover_batch(self, population, mutants):
        NP_local = len(population)
        CR = np.clip(self.CR + np.random.randn(NP_local) * 0.1, 0.5, 0.98)
        mask = np.random.rand(NP_local, self.dim) < CR[:, np.newaxis]
        mask[np.arange(NP_local), np.random.randint(0, self.dim, NP_local)] = True
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # THE FOUR ADAPTATION STRATEGIES (arms)
    # Each follows the same interface: (F, CR, improvement_rate) → updates F, CR
    # ─────────────────────────────────────────────────────────────────────────
    
    def _adapt_original(self, improvement_rate):
        """Arm 0: Original improvement-rate based adaptation."""
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
    
    def _adapt_variant_01(self, improvement_rate):
        """Arm 1: Geometric spatial spread adaptation (variant_01 catA_t16).
        Uses population centroid distances to determine F/CR."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return
        
        pop = self._current_population
        if len(pop) < 4:
            return
        
        pop_min = pop.min(axis=0)
        pop_max = pop.max(axis=0)
        centroid = pop.mean(axis=0)
        centroid_distances = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = centroid_distances.mean()
        
        expected_dist = 200.0 * np.sqrt(self.dim / len(pop))
        normalized_centroid_dist = mean_centroid_dist / max(expected_dist, 1e-10)
        spread_factor = np.clip(normalized_centroid_dist, 0.1, 2.0)
        
        target_F = 0.6 / (spread_factor + 0.3)
        target_F = np.clip(target_F, 0.3, 1.5)
        
        target_CR = 0.5 + 0.4 * np.clip(spread_factor - 0.5, 0, 1)
        target_CR = np.clip(target_CR, 0.3, 0.95)
        
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
    
    def _adapt_variant_05(self, improvement_rate):
        """Arm 2: k-NN graph topology adaptation (variant_05 catE_t10).
        Uses k-nearest-neighbor graph metrics to determine F/CR."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return
        
        pop = self._current_population
        if len(pop) < 5:
            return
        
        NP_local = len(pop)
        k = max(2, min(5, NP_local // 10))
        
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP_local)[:, np.newaxis], nearest_indices]
        
        avg_edge_length = np.mean(np.sqrt(nearest_sq_dists))
        
        clustering_coeffs = np.zeros(NP_local)
        for i in range(NP_local):
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
        
        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
        }
        self._graph_metrics_history.append(current_metrics)
        
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
    
    def _adapt_variant_07(self, improvement_rate):
        """Arm 3: Bootstrap confidence interval adaptation (variant_07 catG_t05).
        Uses bootstrap resampling of improvement rate history."""
        if not hasattr(self, '_improvement_history'):
            self._improvement_history = []
        
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
        
        history_array = np.array(self._improvement_history)
        n_bootstrap = 100
        
        bootstrap_indices = np.random.randint(0, n_history, size=(n_bootstrap, n_history))
        bootstrap_means = history_array[bootstrap_indices].mean(axis=1)
        
        ci_lower = np.percentile(bootstrap_means, 10)
        ci_upper = np.percentile(bootstrap_means, 90)
        ci_width = max(ci_upper - ci_lower, 0.01)
        
        baseline_F = 0.6
        baseline_CR = 0.85
        
        norm_rate = np.clip((ci_lower + ci_width * np.random.rand()) / max(ci_upper, 0.1), 0, 1)
        
        target_F = baseline_F * (0.5 + 1.5 * norm_rate)
        target_CR = baseline_CR * (1.3 - 0.5 * norm_rate)
        
        alpha = 0.3
        self.F = np.clip(alpha * target_F + (1 - alpha) * self.F + np.random.randn() * 0.05, 0.3, 1.5)
        self.CR = np.clip(alpha * target_CR + (1 - alpha) * self.CR + np.random.randn() * 0.03, 0.3, 0.99)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        
        if len(self.F_history) > 100:
            self.F_history = self.F_history[-100:]
            self.CR_history = self.CR_history[-100:]
    
    def _dispatch_adapt(self, improvement_rate):
        """Dispatch to the committed arm's adaptation function.
        This is the ONLY mechanism — no blending, no voting."""
        if self._committed_arm == 0:
            self._adapt_original(improvement_rate)
        elif self._committed_arm == 1:
            self._adapt_variant_01(improvement_rate)
        elif self._committed_arm == 2:
            self._adapt_variant_05(improvement_rate)
        elif self._committed_arm == 3:
            self._adapt_variant_07(improvement_rate)
        else:
            self._adapt_original(improvement_rate)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROBE PHASE: Evaluate each arm on a sub-population
    # ─────────────────────────────────────────────────────────────────────────
    
    def _run_probe_arm(self, func, arm_idx, sub_pop_size, n_probe_gens):
        """Run one arm for n_probe_gens on a sub-population. Returns fingerprint."""
        sub_pop = np.random.uniform(-100.0, 100.0, (sub_pop_size, self.dim))
        sub_pop += np.random.randn(sub_pop_size, self.dim) * 5.0
        sub_pop = np.clip(sub_pop, -100.0, 100.0)
        
        sub_fitness = func(sub_pop)
        best_fitness = float(np.min(sub_fitness))
        best_sub = sub_pop[np.argmin(sub_fitness)].copy()
        
        # Per-arm state
        probe_F = 0.6
        probe_CR = 0.85
        arm_F_history = [probe_F]
        arm_CR_history = [probe_CR]
        
        # Per-arm internal state (reset for probe)
        if arm_idx == 2:
            graph_history = []
        elif arm_idx == 3:
            imp_history = []
        
        best_history = [best_fitness]
        stagnation_count = 0
        
        for gen in range(n_probe_gens):
            mutants = np.empty_like(sub_pop)
            for i in range(sub_pop_size):
                others = np.concatenate([np.arange(i), np.arange(i + 1, sub_pop_size)])
                r = others[np.random.choice(len(others), 3, replace=False)]
                mutants[i] = sub_pop[r[0]] + probe_F * (sub_pop[r[1]] - sub_pop[r[2]])
            
            CR_local = np.clip(probe_CR + np.random.randn() * 0.1, 0.5, 0.98)
            mask = np.random.rand(sub_pop_size, self.dim) < CR_local
            mask[np.arange(sub_pop_size), np.random.randint(0, self.dim, sub_pop_size)] = True
            trials = np.where(mask, mutants, sub_pop)
            trials = np.clip(trials, -100.0, 100.0)
            
            trial_fitness = func(trials)
            improved = trial_fitness < sub_fitness
            sub_pop = np.where(improved[:, np.newaxis], trials, sub_pop)
            sub_fitness = np.where(improved, trial_fitness, sub_fitness)
            
            current_best_idx = np.argmin(sub_fitness)
            if sub_fitness[current_best_idx] < best_fitness:
                best_fitness = float(sub_fitness[current_best_idx])
                best_sub = sub_pop[current_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            best_history.append(best_fitness)
            
            improvement_rate = float(np.mean(improved))
            
            # Apply arm's adaptation (using temporary state)
            old_F, old_CR = self.F, self.CR
            old_F_hist, old_CR_hist = self.F_history[:], self.CR_history[:]
            old_graph = getattr(self, '_graph_metrics_history', None)
            old_imp = getattr(self, '_improvement_history', None)
            
            self.F = probe_F
            self.CR = probe_CR
            self.F_history = arm_F_history
            self.CR_history = arm_CR_history
            if arm_idx == 2:
                self._graph_metrics_history = graph_history
            elif arm_idx == 3:
                self._improvement_history = imp_history
            
            if arm_idx == 0:
                self._adapt_original(improvement_rate)
            elif arm_idx == 1:
                self._adapt_variant_01(improvement_rate)
            elif arm_idx == 2:
                self._current_population = sub_pop
                self._adapt_variant_05(improvement_rate)
                graph_history = getattr(self, '_graph_metrics_history', [])
            elif arm_idx == 3:
                self._adapt_variant_07(improvement_rate)
                imp_history = getattr(self, '_improvement_history', [])
            
            if len(self.F_history) > 0:
                probe_F = float(self.F_history[-1])
                probe_CR = float(self.CR_history[-1])
            arm_F_history = list(self.F_history)
            arm_CR_history = list(self.CR_history)
            
            self.F, self.CR = old_F, old_CR
            self.F_history, self.CR_history = old_F_hist, old_CR_hist
            if old_graph is not None:
                self._graph_metrics_history = old_graph
            if old_imp is not None:
                self._improvement_history = old_imp
        
        # Compute fingerprint from probe data
        fingerprint = self._compute_probe_fingerprint(best_history, sub_pop)
        fingerprint['best_fitness'] = best_fitness
        fingerprint['stagnation_count'] = stagnation_count
        
        return fingerprint, best_fitness, best_sub
    
    def _compute_probe_fingerprint(self, best_history, final_pop):
        """Compute cheap landscape fingerprint — no oracle knowledge required."""
        metrics = {}
        
        # 1. Convergence rate: slope of log(best) vs generation
        if len(best_history) >= 3:
            gens = np.arange(len(best_history))
            valid_best = np.maximum(np.array(best_history, dtype=float), 1e-20)
            log_best = np.log(valid_best)
            cov = np.cov(gens, log_best)
            metrics['convergence_rate'] = cov[0, 1] / max(np.var(gens), 1e-10)
        else:
            metrics['convergence_rate'] = 0.0
        
        # 2. Effective dimensionality from population covariance
        if len(final_pop) >= max(self.dim + 1, 5):
            cov_matrix = np.cov(final_pop.T)
            eigenvals = np.linalg.eigvalsh(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            eigenvals = np.maximum(eigenvals, 0)
            total_var = np.sum(eigenvals)
            if total_var > 1e-10:
                cumvar = np.cumsum(eigenvals) / total_var
                metrics['effective_dim'] = float(np.searchsorted(cumvar, 0.95) + 1)
            else:
                metrics['effective_dim'] = 1.0
        else:
            metrics['effective_dim'] = float(min(len(final_pop), self.dim))
        
        # 3. Stagnation index: fraction of gens with no improvement
        if len(best_history) >= 2:
            improvements = np.abs(np.diff(best_history))
            stagnation_count = np.sum(improvements < 1e-10)
            metrics['stagnation_index'] = float(stagnation_count / max(len(best_history) - 1, 1))
        else:
            metrics['stagnation_index'] = 0.0
        
        # 4. Spread ratio: max_to_min population distance
        if len(final_pop) >= 2:
            centroid = final_pop.mean(axis=0)
            distances = np.linalg.norm(final_pop - centroid, axis=1)
            metrics['spread_ratio'] = float(distances.max() / max(distances.mean(), 1e-10))
        else:
            metrics['spread_ratio'] = 1.0
        
        return metrics
    
    def _rank_probe_results(self, fingerprints, best_fitnesses):
        """Rank-based selection of winning arm. Uses multiple fingerprint metrics."""
        n_arms = len(fingerprints)
        
        # Extract rankable metrics (all on comparable scales via ranking)
        fitness_ranks = np.argsort(np.argsort(best_fitnesses))  # Lower fitness = better rank
        
        stagnation_ranks = np.argsort(np.argsort([f['stagnation_index'] for f in fingerprints]))
        
        convergence_ranks = np.argsort(np.argsort([-f['convergence_rate'] for f in fingerprints]))
        
        spread_ranks = np.argsort(np.argsort([f['spread_ratio'] for f in fingerprints]))
        
        # Combined rank score (lower is better)
        combined = (
            1.0 * fitness_ranks +
            0.5 * stagnation_ranks +
            0.5 * convergence_ranks +
            0.3 * spread_ranks
        )
        
        winner_idx = int(np.argmin(combined))
        return winner_idx
    
    def _run_probe_phase(self, func, total_budget):
        """Run probe phase: evaluate all 4 arms, select winner via rank-based scoring."""
        probe_fraction = 0.05
        probe_budget = max(50, int(total_budget * probe_fraction))
        
        sub_pop_size = min(max(4 * self.dim, 20), self.NP)
        n_probe_gens = max(5, probe_budget // sub_pop_size)
        
        fingerprints = []
        best_fitnesses = []
        best_solutions = []
        
        for arm_idx in range(self.num_arms):
            np.random.seed(None)
            fp, bf, bs = self._run_probe_arm(func, arm_idx, sub_pop_size, n_probe_gens)
            fingerprints.append(fp)
            best_fitnesses.append(bf)
            best_solutions.append(bs)
        
        winner_idx = self._rank_probe_results(fingerprints, best_fitnesses)
        
        self._probe_fingerprints = fingerprints
        self._probe_completed = True
        self._committed_arm = winner_idx
        
        return winner_idx, best_solutions[winner_idx], best_fitnesses[winner_idx]
    
    # ─────────────────────────────────────────────────────────────────────────
    # STANDARD DE OPERATORS (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    
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
        best_fitness = fitness[best_idx]
        
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Clean per-arm state on restart
        if hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        if hasattr(self, '_improvement_history'):
            self._improvement_history = []
        
        return new_pop, best_fitness, best_solution
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        # ─── PHASE A: PROBE ────────────────────────────────────────────────────
        if not self._probe_completed:
            winner_arm, probe_best_x, probe_best_f = self._run_probe_phase(
                func, self.max_iter
            )
            arm_names = ['original', 'variant_01 (geometric)', 'variant_05 (k-NN graph)', 'variant_07 (bootstrap)']
            
            if probe_best_f < f_opt:
                f_opt = probe_best_f
                x_opt = probe_best_x
            
            iteration += 1
        
        # ─── PHASE B: COMMIT ───────────────────────────────────────────────────
        # From here on, ONLY the committed arm's adaptation is used.
        # No blending. No voting. This preserves per-task winner advantages.
        
        while not stopping_condition():
            self._current_population = population
            
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
            improvement_rate = float(np.mean(improved_mask))
            
            # ── THE ONLY ADAPTATION CALL: dispatch to committed arm ──
            self._dispatch_adapt(improvement_rate)
            
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            if self._check_stagnation(f_opt):
                # Small probability of revisiting runner-up on stagnation
                if self._committed_arm is not None and np.random.rand() < 0.08:
                    all_arms = list(range(self.num_arms))
                    other_arms = [a for a in all_arms if a != self._committed_arm]
                    self._committed_arm = int(np.random.choice(other_arms))
                
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
```