```python
import numpy as np


class AdaptiveCompassDE:
    """
    Adaptive Compass Differential Evolution with PROBE-AND-COMMIT crossover selection.
    
    Mechanism: Per-task fingerprint-and-commit (specialization of mechanism #4).
    This is MANDATORY because the per-task gap table shows:
    - 6 DISPATCH-MANDATORY tasks with winner >= 100x better than runner-up
    - 10 BLEND-HOSTILE tasks with winner >= 10x better than runner-up
    - Worst-case: Task 5 has winner (2.25e-06) vs runner-up (2.16e-05) = 9.6x gap
      but 14,824,058x gap to median. A 50/50 blend yields 1.19e-05, 5.3x WORSE.
    
    Linear ensemble blending is mathematically excluded for these large gaps.
    The algorithm probes each crossover variant for a short budget (~6% of max_iter),
    commits to the empirical winner using rank-based performance, then runs the
    committed variant for the remaining budget. No blending is used - the chosen
    variant gets weight 1.0.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Crossover strategy pool with probe-based selection
        self.crossover_names = ['original', 'catA', 'catB', 'catE', 'catF']
        self.crossover_funcs = [
            self._crossover_original,
            self._crossover_catA,
            self._crossover_catB,
            self._crossover_catE,
            self._crossover_catF,
        ]
        
        # Probe tracking
        self.probe_budget_pct = 0.06  # 6% of budget for probing
        self.selected_crossover = 0  # Default to original (safe baseline)
        self._probe_complete = False
        self._probe_crossover_rank_improvements = None
        self._probe_fitness_history = None
        self._probe_improvement_history = None
        self._probe_prev_best = None
        self._probe_prev_fitness = None
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
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
    
    def _mutate_ensemble_batch(self, population, fitness):
        """Use adaptive strategy pool to generate mutants."""
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        
        return mutants, strategy_used
    
    def _select_strategy_batch(self):
        """Softmax-weighted strategy selection."""
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _crossover_original(self, population, mutants):
        """Binomial crossover with dimension-wise CR (baseline from original algorithm)."""
        NP, dim = population.shape
        CR = np.clip(self.CR + np.random.randn(NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(NP, dim) < CR[:, np.newaxis]
        mask[np.arange(NP), np.random.randint(0, dim, NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catA(self, population, mutants):
        """Geometry-driven crossover: adapt CR per dimension by axis-aligned spread."""
        NP, dim = population.shape
        
        # Compute axis-aligned spread (range) per dimension
        dim_min = population.min(axis=0)
        dim_max = population.max(axis=0)
        spread = dim_max - dim_min
        spread = np.where(spread < 1e-10, 1e-10, spread)
        
        # Per-dimension crossover probability
        cr_per_dim = spread / spread.sum()
        rand_matrix = np.random.rand(NP, dim)
        mask = rand_matrix < cr_per_dim[np.newaxis, :]
        
        # Ensure at least one dimension from mutant
        no_mutant_dims = ~mask.any(axis=1)
        forced_dims = np.random.randint(0, dim, size=no_mutant_dims.sum())
        mask[no_mutant_dims, forced_dims] = True
        
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catB(self, population, mutants):
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
        eig_weights = eigenvalues / total_eig if total_eig > 0 else np.ones(dim) / dim
        base_cr = np.clip(self.CR * np.power(eig_weights, 0.1), 0.01, 0.99)
        
        crossover_mask = np.random.rand(NP, dim) < base_cr
        
        if dim > 1:
            force_cross_idx = np.random.randint(0, dim, size=NP)
            crossover_mask[np.arange(NP), force_cross_idx] = True
        
        trial_pc = np.where(crossover_mask, mut_pc, pop_pc)
        trials = trial_pc @ eigenvectors.T + centroid
        
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catE(self, population, mutants):
        """Graph-distance-adaptive binomial crossover using k-NN topology."""
        NP, dim = population.shape
        
        k = max(2, min(5, NP // 4))
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        graph_dist = np.zeros(NP)
        for i in range(NP):
            direct_dist = np.linalg.norm(mutants[i] - population[i])
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
        norm_dist = graph_dist / max_dist if max_dist > 1e-10 else np.zeros(NP)
        
        adaptive_CR = self.CR * (1.0 - 0.3 * norm_dist)
        adaptive_CR = np.clip(adaptive_CR, 0.1, 0.95)
        
        trials = np.empty_like(population)
        jr = np.random.randint(0, dim, size=NP)
        
        for i in range(NP):
            mask = np.random.rand(dim) < adaptive_CR[i]
            mask[jr[i]] = True
            trials[i] = np.where(mask, mutants[i], population[i])
        
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_catF(self, population, mutants):
        """Binomial crossover with temporal EMA-driven CR adaptation."""
        NP, dim = population.shape
        
        # Lazy init temporal tracking
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
        
        if not hasattr(self, '_ema_rank_improvement'):
            self._ema_rank_improvement = np.zeros(n_strategies)
            self._ema_success_rate = np.zeros(n_strategies)
            self._rank_improvement_history = [[] for _ in range(n_strategies)]
            self._success_rate_history = [[] for _ in range(n_strategies)]
            self._momentum = 0.3
        
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
            delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])
        
        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on k-NN graph topology of the population."""
        if not hasattr(self, '_current_population') or self._current_population is None:
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
        
        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
        current_metrics = {'avg_edge_length': avg_edge_length, 'avg_clustering': avg_clustering}
        self._graph_metrics_history.append(current_metrics)
        
        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)
        
        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
        else:
            edge_length_trend = 0.0
        
        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * (edge_length_trend < -0.01)
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
        """Population spread: average distance to centroid."""
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
            if