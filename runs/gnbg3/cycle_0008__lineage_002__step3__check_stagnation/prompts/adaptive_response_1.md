```python
import numpy as np


class AdaptiveCompassDE:
    """
    Adaptive Compass Differential Evolution with PROBE-AND-COMMIT stagnation detection.
    
    MECHANISM: Probe-and-commit (mechanism #4 contextual bandit with one-shot context).
    
    The per-task winner-gap table shows ONE blend-hostile task (Task 17: 123.1× gap to
    median competitor). Ensemble blending is mathematically excluded — a 50/50 blend of
    the winner (6.57) and runner-up (7.84) yields ~7.20, which is 1.1× worse than the
    winner alone. The gap table also shows 7 distinct winners across 21 non-trivial
    tasks with small runner-up gaps (<2×), indicating that the optimal stagnation
    detector varies by problem characteristics.
    
    The probe phase (~5% of budget, distributed round-robin) measures empirical
    convergence rate of each detector on the ACTUAL landscape. The detector with
    the steepest log-fitness slope is committed for the remaining budget. This
    preserves per-task winner advantages because the committed detector gets weight
    1.0 — no blending that would dilute a 100× advantage.
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
        
        # --- PROBE-AND-COMMIT state ---
        # 6 candidate stagnation detectors (from benchmark winners)
        self._detector_names = [
            'bounding_box',      # variant_09_catA_t17 (6 wins)
            'geometric',         # variant_01_catA_t08 (5 wins)
            'spectral',          # variant_02_catB_t15 (1 win)
            'fitness_rank',      # variant_04_catD_t06 (1 win)
            'graph_theoretic',   # variant_05_catE_t23 (4 wins)
            'hybrid',            # variant_08_catH_t17 (3 wins)
        ]
        self._probe_budget_frac = 0.05  # 5% of budget for probing
        self._committed_detector = None
    
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
    
    def _check_stagnation_original(self, best_fitness):
        """Detect stagnation and low diversity (original method)."""
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
    def _check_stagnation_bounding_box(self, best_fitness):
        """Detect stagnation via axis-aligned bounding box volume analysis (variant_09)."""
        if not hasattr(self, '_bb_history'):
            self._bb_history = []
            self._bb_stagnation_counter = 0

        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        if len(pop) < 2:
            return False

        try:
            min_coords = pop.min(axis=0)
            max_coords = pop.max(axis=0)
            ranges = max_coords - min_coords
            volume = np.prod(ranges + 1e-10)

            self._bb_history.append(volume)
            if len(self._bb_history) > 10:
                self._bb_history.pop(0)

            if len(self._bb_history) >= 3:
                recent = self._bb_history[-3:]
                vol_decreasing = recent[-1] < recent[0] * 0.9

                if vol_decreasing:
                    self._bb_stagnation_counter += 1
                    return self._bb_stagnation_counter >= 3
                else:
                    self._bb_stagnation_counter = 0

            return False
        except Exception:
            return False
    
    def _check_stagnation_geometric(self, best_fitness):
        """Detect stagnation via geometric analysis: centroid drift + pairwise regularity (variant_01)."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        NP, dim = pop.shape

        if NP < 4:
            return False

        centroid = pop.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)

        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        i, j = np.triu_indices(NP, k=1)
        dists = np.sqrt(sq_dists[i, j])

        median_d = np.median(dists)
        std_d = np.std(dists)
        regularity = std_d / (median_d + 1e-10)

        if not hasattr(self, '_centroid_history'):
            self._centroid_history = []
            self._dist_regularity_history = []

        self._centroid_history.append(centroid_norm)
        self._dist_regularity_history.append(regularity)

        max_history = 15
        if len(self._centroid_history) > max_history:
            self._centroid_history.pop(0)
            self._dist_regularity_history.pop(0)

        if len(self._centroid_history) >= 5:
            recent = np.array(self._centroid_history[-5:])
            centroid_range = recent.max() - recent.min()
            search_scale = 200.0 * np.sqrt(dim)
            centroid_stagnant = centroid_range < search_scale * 1e-3
        else:
            centroid_stagnant = False

        regularity_stagnant = regularity < 0.05
        spread_collapsed = median_d < 1e-4 * np.sqrt(dim)
        geometric_stagnant = centroid_stagnant or regularity_stagnant or spread_collapsed

        if geometric_stagnant:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter >= 10:
            if hasattr(self, '_centroid_history'):
                self._centroid_history.clear()
            if hasattr(self, '_dist_regularity_history'):
                self._dist_regularity_history.clear()
            return True

        return False
    
    def _check_stagnation_spectral(self, best_fitness):
        """Detect stagnation via spectral analysis of population covariance (variant_02)."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        NP, dim = pop.shape

        if NP < dim or NP < 5:
            return False

        if not hasattr(self, '_spectral_history'):
            self._spectral_history = []
            self._prev_eigenvalues = None
            self._prev_effective_rank = float(dim)

        centroid = pop.mean(axis=0)
        centered_pop = pop - centroid

        cov = np.cov(centered_pop, rowvar=False)
        eps = 1e-10
        cov_reg = cov + eps * np.eye(dim)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
        except np.linalg.LinAlgError:
            return False

        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > eps]
        if len(eigenvalues) < 2:
            return True

        eig_sum = eigenvalues.sum()
        if eig_sum < eps:
            return True
        eig_normalized = eigenvalues / eig_sum

        p_nonzero = eig_normalized[eig_normalized > eps]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero + eps))
        effective_rank = np.exp(entropy) if entropy > 0 else 1.0

        condition_number = eigenvalues[0] / max(eigenvalues[-1], eps)
        dominance_ratio = eigenvalues[0] / eig_sum

        rank_deficit = 1.0 - (effective_rank / dim)
        condition_issue = np.log1p(condition_number) / 20.0
        stagnation_score = 0.4 * rank_deficit + 0.3 * np.clip(condition_issue, 0, 1) + 0.3 * dominance_ratio

        is_stagnated = (
            effective_rank < 0.15 * dim or
            condition_number > 1e6 or
            dominance_ratio > 0.95 or
            stagnation_score > 0.75
        )

        self._spectral_history.append({
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'dominance_ratio': dominance_ratio,
            'stagnation_score': stagnation_score
        })

        if len(self._spectral_history) > 20:
            self._spectral_history.pop(0)

        return is_stagnated
    
    def _check_stagnation_fitness_rank(self, best_fitness):
        """Detect stagnation using fitness-landscape / rank-based signals (variant_04)."""
        if not hasattr(self, '_stagnation_counter'):
            self._stagnation_counter = 0
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        if not hasattr(self, '_prev_pop_fitness'):
            self._prev_pop_fitness = None

        self._fitness_history.append(float(best_fitness))
        max_history = 20
        if len(self._fitness_history) > max_history:
            self._fitness_history.pop(0)

        stagnation_detected = False

        if hasattr(self, '_current_population_fitness') and self._current_population_fitness is not None:
            pop_fitness = np.asarray(self._current_population_fitness)
            valid_mask = ~np.isnan(pop_fitness)
            if valid_mask.sum() >= 5:
                pop_fitness = pop_fitness[valid_mask]

                if self._prev_pop_fitness is not None and len(self._prev_pop_fitness) == len(pop_fitness):
                    prev_valid = ~np.isnan(self._prev_pop_fitness)
                    if prev_valid.sum() == len(pop_fitness):
                        try:
                            from scipy.stats import spearmanr
                            corr, _ = spearmanr(self._prev_pop_fitness, pop_fitness)
                            if corr is not None and corr > 0.9:
                                stagnation_detected = True
                        except (ValueError, ImportError):
                            pass

                self._prev_pop_fitness = pop_fitness.copy()

                p10, p90 = np.percentile(pop_fitness, [10, 90])
                spread = p90 - p10
                mean_fitness = np.mean(pop_fitness)
                if mean_fitness != 0:
                    rel_spread = spread / (abs(mean_fitness) + 1e-10)
                else:
                    rel_spread = spread

                if rel_spread < 1e-8:
                    stagnation_detected = True
        else:
            if len(self._fitness_history) >= 3:
                recent = self._fitness_history[-3:]
                improvement = max(recent) - min(recent)
                if improvement < 1e-10:
                    stagnation_detected = True

        if stagnation_detected:
            self._stagnation_counter += 1
        else:
            if len(self._fitness_history) >= 2:
                if self._fitness_history[-1] < self._fitness_history[-2] - 1e-12:
                    self._stagnation_counter = 0

        stagnation_threshold = 15
        return self._stagnation_counter >= stagnation_threshold
    
    def _check_stagnation_graph_theoretic(self, best_fitness):
        """Detect stagnation via graph-theoretic properties (variant_05)."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        NP, dim = pop.shape

        if NP < 4:
            return False

        if not hasattr(self, '_topo_stagnation_history'):
            self._topo_stagnation_history = []

        k = max(2, min(5, NP // 8))

        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]

        parent = np.arange(NP)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for i in range(NP):
            for j in nearest_indices[i]:
                union(i, j)

        components = np.unique([find(i) for i in range(NP)])
        n_components = len(components)

        in_mst = np.zeros(NP, dtype=bool)
        in_mst[0] = True
        mst_edges = []
        mst_total_weight = 0.0

        for _ in range(NP - 1):
            min_weight = np.inf
            min_edge = None
            for i in range(NP):
                if not in_mst[i]:
                    continue
                for j in nearest_indices[i]:
                    if not in_mst[j] and sq_dists[i, j] < min_weight:
                        min_weight = sq_dists[i, j]
                        min_edge = (i, j)
            if min_edge is not None:
                mst_edges.append(min_edge)
                mst_total_weight += np.sqrt(min_weight)
                in_mst[min_edge[1]] = True
            else:
                break

        avg_mst_edge_weight = mst_total_weight / max(len(mst_edges), 1)

        edge_lengths = []
        for i in range(NP):
            for j in nearest_indices[i]:
                if i < j:
                    edge_lengths.append(np.sqrt(sq_dists[i, j]))

        avg_knn_edge = np.mean(edge_lengths) if edge_lengths else 0.0

        component_sizes = []
        for c in components:
            size = sum(1 for i in range(NP) if find(i) == c)
            component_sizes.append(size)

        largest_component_ratio = max(component_sizes) / NP if component_sizes else 1.0

        current_metrics = {
            'n_components': n_components,
            'avg_mst_edge': avg_mst_edge_weight,
            'largest_component_ratio': largest_component_ratio,
            'avg_knn_edge': avg_knn_edge
        }
        self._topo_stagnation_history.append(current_metrics)

        if len(self._topo_stagnation_history) > 12:
            self._topo_stagnation_history.pop(0)

        if len(self._topo_stagnation_history) < 4:
            return False

        fragmentation_threshold = NP / 3
        is_fragmented = n_components > fragmentation_threshold

        recent_mst = [m['avg_mst_edge'] for m in self._topo_stagnation_history[-4:]]
        mst_decreasing = all(recent_mst[i] >= recent_mst[i+1] * 0.98 for i in range(len(recent_mst)-1))
        mst_shrinking_rate = (recent_mst[0] - recent_mst[-1]) / max(recent_mst[0], 1e-10)

        recent_lcr = [m['largest_component_ratio'] for m in self._topo_stagnation_history[-4:]]
        lcr_decreasing = recent_lcr[-1] < recent_lcr[0] * 0.85

        recent_knn = [m['avg_knn_edge'] for m in self._topo_stagnation_history[-4:]]
        knn_decreasing = all(recent_knn[i] >= recent_knn[i+1] * 0.97 for i in range(len(recent_knn)-1))

        stagnation_score = 0

        if is_fragmented:
            stagnation_score += 3
        elif n_components > 2:
            stagnation_score += 1

        if mst_decreasing and mst_shrinking_rate > 0.05:
            stagnation_score += 2

        if lcr_decreasing:
            stagnation_score += 2

        if knn_decreasing:
            stagnation_score += 1

        is_stagnated = stagnation_score >= 4

        if is_stagnated:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        extreme_mst_shrink = mst_shrinking_rate > 0.15
        extreme_fragmentation = n_components > NP / 2

        return is_stagnated or extreme_mst_shrink or extreme_fragmentation
    
    def _check_stagnation_hybrid(self, best_fitness):
        """Hybrid stagnation: temporal fitness + spatial dispersion (variant_08)."""
        if not hasattr(self, '_stagn_history_fitness'):
            self._stagn_history_fitness = []
            self._stagn_ema_fitness = float('inf')
            self._stagn_ema_alpha = 0.1

        self._stagn_history_fitness.append(float(best_fitness))
        if len(self._stagn_history_fitness) > 20:
            self._stagn_history_fitness.pop(0)

        if np.isfinite(self._stagn_ema_fitness) and np.isfinite(best_fitness):
            self._stagn_ema_fitness = (1 - self._stagn_ema_alpha) * self._stagn_ema_fitness + self._stagn_ema_alpha * best_fitness
        else:
            self._stagn_ema_fitness = best_fitness

        fitness_improvement = abs(self._stagn_ema_fitness - float(best_fitness)) / (abs(self._stagn_ema_fitness) + 1e-12)

        if not hasattr(self, '_stagn_history_centroid'):
            self._stagn_history_centroid = []

        if hasattr(self, '_current_population') and self._current_population is not None:
            centroid = self._current_population.mean(axis=0)
            self._stagn_history_centroid.append(centroid.copy())
            if len(self._stagn_history_centroid) > 15:
                self._stagn_history_centroid.pop(0)

            if len(self._stagn_history_centroid) >= 3:
                recent = np.array(self._stagn_history_centroid[-3:])
                centroid_drift = np.mean(np.linalg.norm(np.diff(recent, axis=0), axis=1))
            elif len(self._stagn_history_centroid) >= 2:
                centroid_drift = np.linalg.norm(self._stagn_history_centroid[-1] - self._stagn_history_centroid[0])
            else:
                centroid_drift = 0.0

            pop_spread = self._compute_diversity(self._current_population) + 1e-12
            spatial_signal = centroid_drift / (pop_spread * np.sqrt(self.dim) + 1e-12)
        else:
            spatial_signal = 1.0

        if not hasattr(self, '_stagn_fitness_variance'):
            self._stagn_fitness_variance = 1.0
            self._stagn_spatial_variance = 1.0
            self._stagn_fitness_signal_history = []
            self._stagn_spatial_signal_history = []

        self._stagn_fitness_signal_history.append(fitness_improvement)
        self._stagn_spatial_signal_history.append(spatial_signal)

        if len(self._stagn_fitness_signal_history) > 10:
            self._stagn_fitness_signal_history.pop(0)
        if len(self._stagn_spatial_signal_history) > 10:
            self._stagn_spatial_signal_history.pop(0)

        if len(self._stagn_fitness_signal_history) >= 3:
            self._stagn_fitness_variance = max(np.var(self._stagn_fitness_signal_history), 1e-8)
        if len(self._stagn_spatial_signal_history) >= 3:
            self._stagn_spatial_variance = max(np.var(self._stagn_spatial_signal_history), 1e-8)

        eps = 1e-10
        w_fitness = 1.0 / (self._stagn_fitness_variance + eps)
        w_spatial = 1.0 / (self._stagn_spatial_variance + eps)
        total_w = w_fitness + w_spatial + eps
        w_fitness /= total_w
        w_spatial /= total_w

        hybrid_score = w_fitness * fitness_improvement + w_spatial * spatial_signal

        if not hasattr(self, '_generation_count'):
            self._generation_count = 0
        self._generation_count += 1

        gen_factor = min(1.0, self._generation_count / 100.0)
        base_threshold = 1e-4 * gen_factor
        threshold = base_threshold + 1e-6 * (self.dim / 100.0)

        stagnation_detected = (hybrid_score < threshold) and (len(self._stagn_fitness_signal_history) >= 5)

        return stagnation_detected
    
    def _check_stagnation_with_detector(self, detector_idx, best_fitness):
        """Route to the specified stagnation detector."""
        if detector_idx == 0:
            return self._check_stagnation_bounding_box(best_fitness)
        elif detector_idx == 1:
            return self._check_stagnation_geometric(best_fitness)
        elif detector_idx == 2:
            return self._check_stagnation_spectral(best_fitness)
        elif detector_idx == 3:
            return self._check_stagnation_fitness_rank(best_fitness)
        elif detector_idx == 4:
            return self._check_stagnation_graph_theoretic(best_fitness)
        elif detector_idx == 5:
            return self._check_stagnation_hybrid(best_fitness)
        else:
            return self._check_stagnation_original(best_fitness)
    
    def _compute_convergence_rate(self, fitness_history):
        """Compute convergence rate from fitness trajectory using log-linear regression.
        
        Higher (more positive) slope magnitude = faster convergence.
        Returns 0.0 if insufficient data or computation fails.
        """
        if len(fitness_history) < 3:
            return 0.0
        
        # Handle invalid values: replace with large finite value
        valid_fitness = []
        for f in fitness_history:
            if np.isfinite(f) and f > 0:
                valid_fitness.append(f)
            else:
                valid_fitness.append(1e10)
        
        fitness_arr = np.array(valid_fitness)
        
        # Log transform for exponential convergence detection
        log_fitness = np.log(fitness_arr + 1e-10)
        
        # Linear regression: slope of log(fitness) vs iteration
        x = np.arange(len(log_fitness))
        try:
            slope, _ = np.polyfit(x, log_fitness, 1)
            # Negative slope = improving (for minimization). Return positive rate.
            return -slope
        except:
            return 0.0
    
    def _probe_detector(self, func, stopping_condition, detector_idx, probe_iters):
        """Run a probe with a specific stagnation detector and return convergence metrics."""
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        f_opt = fitness[best_idx]
        
        best_fitness_history = []
        stagnation_counter_local = 0
        
        for it in range(probe_iters):
            if stopping_condition():
                break
            
            # Standard DE iteration
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
            current_best = fitness[current_best_idx]
            best_fitness_history.append(current_best)
            
            if current_best < f_opt:
                f_opt = current_best
            
            # Store population for detectors that need it
            self._current_population = population
            self._current_population_fitness = fitness
            
            # Check stagnation with the probe detector
            stagnation_detected = self._check_stagnation_with_detector(detector_idx, f_opt)
            
            # Local stagnation counter for probe (separate from main counter)
            if stagnation_detected:
                stagnation_counter_local += 1
            else:
                stagnation_counter_local = 0
            
            # Restart if stagnated (but not too early)
            if stagnation_counter_local >= 3 and it > 10:
                population = self._restart_population(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
                stagnation_counter_local = 0
        
        return {
            'best_fitness_history': best_fitness_history,
            'final_best': best_fitness_history[-1] if best_fitness_history else np.inf
        }
    
    def _restart_population(self, population, fitness):
        """Restart with diversity injection (called during probing)."""
        NP, dim = population.shape
        
        # Keep best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        # Replace worst half with random + best-based mutations
        n_replace = max(NP // 2, 2)
        worst_indices = np.argsort(fitness)[-n_replace:]
        
        for idx in worst_indices:
            if np.random.rand() < 0.7:
                # Mutation from best
                noise = np.random.randn(dim) * 15.0
                population[idx] = np.clip(best_solution + noise, -100.0, 100.0)
            else:
                # Random
                population[idx] = np.random.uniform(-100.0, 100.0, dim)
        
        # Clear stagnation history for all detectors
        self.stagnation_counter = 0
        if hasattr(self, '_bb_stagnation_counter'):
            self._bb_stagnation_counter = 0
        if hasattr(self, '_centroid_history'):
            self._centroid_history = []
            self._dist_regularity_history = []
        if hasattr(self, '_spectral_history'):
            self._spectral_history = []
        if hasattr(self, '_fitness_history'):
            self._fitness_history = []
        if hasattr(self, '_topo_stagnation_history'):
            self._topo_stagnation_history = []
        if hasattr(self, '_stagn_history_fitness'):
            self._stagn_history_fitness = []
        if hasattr(self, '_stagn_history_centroid'):
            self._stagn_history_centroid = []
        if hasattr(self, '_stagnation_counter'):
            self._stagnation_counter = 0
        
        return population
    
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
        if hasattr(self, '_bb_stagnation_counter'):
            self._bb_stagnation_counter = 0
        if hasattr(self, '_centroid_history'):
            self._centroid_history = []
            self._dist_regularity_history = []
        if hasattr(self, '_spectral_history'):
            self._spectral_history = []
        if hasattr(self, '_fitness_history'):
            self._fitness_history = []
        if hasattr(self, '_topo_stagnation_history'):
            self._topo_stagnation_history = []
        if hasattr(self, '_stagn_history_fitness'):
            self._stagn_history_fitness = []
        if hasattr(self, '_stagn_history_centroid'):
            self._stagn_history_centroid = []
        if hasattr(self, '_stagnation_counter'):
            self._stagnation_counter = 0

        return population, f_opt, x_opt
    
   