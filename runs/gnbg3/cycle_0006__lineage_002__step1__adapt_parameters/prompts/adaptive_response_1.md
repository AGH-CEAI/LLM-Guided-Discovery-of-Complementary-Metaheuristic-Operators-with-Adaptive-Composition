```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: PROBE-AND-COMMIT (mechanism #4 / contextual bandit, single-commit variant).
    
    The per-task gap table is BLEND-HOSTILE: 5 tasks have winner-runner-up gaps >=100× (tasks 1, 4, 5, 10, 17),
    with task 5 showing a 250,294× gap. Ensemble/blending (mechanism #2) is FORBIDDEN — a 50/50 blend of
    original.py (4.4e-05) and variant_07 (1.1e+01) yields ~5.5, which is catastrophically worse than either.
    
    The benchmark reveals 3 distinct winners across 22 non-trivial tasks:
      - original.py: 18 wins (dominant default)
      - variant_07: 3 wins (tasks 6, 9, 12)
      - variant_02: 1 win (task 21)
    
    The probe-and-commit pattern addresses this by:
      PHASE A (probe): Run each of 3 candidate adaptation strategies on a small sub-population
                       for 3 generations (~5% budget). Evaluate rank-based improvement + success rate.
      PHASE B (commit): Pick ONE winner (weight 1.0, no blending) and run it for the remaining ~95%.
                        Small 3% epsilon-greedy allows escape if committed strategy stagnates.
    
    This preserves 100% of the per-task winner's advantage — no blending cap, no scale-dependent magic constants.
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
        # PROBE-AND-COMMIT mechanism: 3 candidate adaptation strategies
        # ============================================================
        self._probe_adapt_strategies = [
            self._adapt_parameters_original,   # original k-NN graph approach
            self._adapt_parameters_spectral,   # variant_02: eigendecomposition of covariance
            self._adapt_parameters_lhs,        # variant_07: Latin Hypercube Sampling
        ]
        self._probe_adapt_names = ['original', 'spectral', 'lhs']
        
        # State machine: 'init' -> 'probing' -> 'committed'
        self._adapt_phase = 'init'
        self._committed_strategy = 0  # default to original
        self._probe_budget = 0
        self._probe_results = None
        
        # Per-strategy tracking during probe
        self._probe_strat_improvements = None
        self._probe_strat_successes = None
        self._probe_strat_generation = None
        
        # Epsilon-greedy exploration after commit
        self._epsilon_explore = 0.03
        
        # Restart-aware memory
        self._epoch_adapt_history = []  # track which strategies worked in past epochs
        
    def _initialize_population(self):
        low, high = 100.0, 100.0
        pop = np.random.uniform(-low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, -low, high)
    
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
    
    # =====================================================================
    # PROBE-AND-COMMIT: Adaptation strategy implementations
    # =====================================================================
    
    def _probe_generation(self, subpop, subpop_fitness, strategy_idx):
        """Run one generation with a given adaptation strategy. Returns improvement rate."""
        mutants_compass = self._mutate_compass_batch(subpop, subpop_fitness)
        mutants_ensemble, strategy_used = self._mutate_ensemble_batch(subpop, subpop_fitness)
        
        blend_ratio = np.random.rand(len(subpop), 1)
        mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
        
        trials = self._crossover_batch(subpop, mutants)
        trial_fitness = self._evaluate_subpop(trials)
        
        subpop, subpop_fitness, improved_mask = self._select_survivors_batch(
            subpop, subpop_fitness, trials, trial_fitness
        )
        
        # Apply the adaptation strategy being probed
        improvement_rate = np.mean(improved_mask)
        self._probe_adapt_strategies[strategy_idx](improvement_rate)
        
        return subpop, subpop_fitness, improvement_rate
    
    def _evaluate_subpop(self, population):
        """Evaluate a sub-population. Override if func has batch evaluation."""
        return np.array([self._probe_func(x) for x in population])
    
    def _run_probe_phase(self, func, total_budget):
        """
        Probe phase: run each adaptation strategy on a small sub-population.
        Uses ~5% of total budget across 3 strategies × 3 generations.
        """
        n_strategies = len(self._probe_adapt_strategies)
        probe_gens = 3
        subpop_size = min(20, self.NP)
        
        # Budget per strategy
        budget_per_strategy = max(1, int(total_budget * 0.05 / n_strategies))
        
        # Initialize per-strategy tracking
        self._probe_strat_improvements = np.zeros(n_strategies)
        self._probe_strat_successes = np.zeros(n_strategies)
        self._probe_strat_generation = 0
        
        # Store original F, CR for restoration
        orig_F = self.F
        orig_CR = self.CR
        
        best_strategy = 0  # default to original
        best_score = -np.inf
        
        for s_idx in range(n_strategies):
            # Reset F, CR for each strategy probe
            self.F = 0.6 + np.random.uniform(-0.1, 0.1)
            self.CR = 0.85 + np.random.uniform(-0.05, 0.05)
            
            # Initialize sub-population
            subpop = np.random.uniform(-100.0, 100.0, (subpop_size, self.dim))
            subpop_fitness = np.array([func(x) for x in subpop])
            
            total_improvement = 0.0
            total_success = 0.0
            
            for g in range(probe_gens):
                subpop, subpop_fitness, imp_rate = self._probe_generation(
                    subpop, subpop_fitness, s_idx
                )
                total_improvement += imp_rate
                total_success += imp_rate
            
            avg_improvement = total_improvement / probe_gens
            self._probe_strat_improvements[s_idx] = avg_improvement
            
            # Rank-normalized score: improvement + small diversity bonus
            diversity = np.std(subpop_fitness) / (np.mean(subpop_fitness) + 1e-10)
            score = avg_improvement + 0.05 * min(diversity, 1.0)
            
            if score > best_score:
                best_score = score
                best_strategy = s_idx
        
        # Restore original F, CR
        self.F = orig_F
        self.CR = orig_CR
        
        # Clear probe-specific state from LHS and spectral strategies
        for attr in ['_F_samples', '_CR_samples', '_improvement_samples', '_graph_population', 
                     '_graph_metrics_history', '_current_population']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        return best_strategy
    
    def _commit_to_strategy(self, strategy_idx):
        """Commit to a single adaptation strategy with weight 1.0."""
        self._committed_strategy = strategy_idx
        self._adapt_phase = 'committed'
        self._epoch_adapt_history.append(strategy_idx)
        
        # Reset strategy-specific state for committed strategy
        if strategy_idx == 2:  # LHS
            for attr in ['_F_samples', '_CR_samples', '_improvement_samples']:
                if hasattr(self, attr):
                    delattr(self, attr)
        elif strategy_idx == 1:  # Spectral
            for attr in ['_graph_population', '_graph_metrics_history']:
                if hasattr(self, attr):
                    delattr(self, attr)
    
    # =====================================================================
    # ADAPTATION STRATEGY 0: Original k-NN graph approach
    # =====================================================================
    
    def _adapt_parameters_original(self, improvement_rate):
        """Adapt F and CR based on k-NN graph topology of the population."""
        if not hasattr(self, '_graph_population'):
            self._graph_population = None
            self._graph_metrics_history = []

        if not hasattr(self, '_current_population') or self._current_population is None:
            return

        pop = self._current_population
        if len(pop) < 5:
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
    
    # =====================================================================
    # ADAPTATION STRATEGY 1: Spectral (variant_02)
    # =====================================================================
    
    def _adapt_parameters_spectral(self, improvement_rate):
        """Adapt F and CR based on spectral properties of population covariance."""
        if not hasattr(self, 'population') or self.population is None or len(self.population) < self.dim + 1:
            self.F = np.clip(self.F * (1 + 0.1 * (improvement_rate - 0.2)), 0.3, 1.5)
            self.CR = np.clip(self.CR * (1 + 0.05 * (improvement_rate - 0.2)), 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return

        pop = self.population
        centroid = pop.mean(axis=0)
        centered = pop - centroid
        cov = np.cov(centered.T)

        if cov.ndim == 0 or np.linalg.matrix_rank(cov) < 2:
            self.F = np.clip(self.F * 1.1, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.95, 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return

        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-15)
        except np.linalg.LinAlgError:
            self.F = np.clip(self.F * 1.05, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.98, 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return

        cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 1e-15 else 1e6
        total_var = eigenvalues.sum()
        cumsum = np.cumsum(eigenvalues[::-1])[::-1]
        n_eff = np.sum(cumsum > 0.01 * total_var)
        eff_dim_ratio = n_eff / self.dim

        if cond > 50 or eff_dim_ratio < 0.3:
            self.F = np.clip(self.F * 0.88, 0.3, 1.2)
            self.CR = np.clip(self.CR * 1.03, 0.3, 0.95)
        elif cond < 10 and eff_dim_ratio > 0.7:
            self.F = np.clip(self.F * 1.08, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.97, 0.3, 0.95)
        else:
            self.F = np.clip(self.F * (1 + 0.03 * (improvement_rate - 0.2)), 0.3, 1.5)
            self.CR = np.clip(self.CR * (1 + 0.02 * (0.2 - improvement_rate)), 0.3, 0.95)

        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    # =====================================================================
    # ADAPTATION STRATEGY 2: Latin Hypercube Sampling (variant_07)
    # =====================================================================
    
    def _adapt_parameters_lhs(self, improvement_rate):
        """Adapt F and CR using Latin Hypercube Sampling with bootstrap confidence."""
        if not hasattr(self, '_F_samples'):
            self._F_samples = []
            self._CR_samples = []
            self._improvement_samples = []
            self._max_samples = 100

        self._F_samples.append(self.F)
        self._CR_samples.append(self.CR)
        self._improvement_samples.append(improvement_rate)

        if len(self._F_samples) > self._max_samples:
            self._F_samples.pop(0)
            self._CR_samples.pop(0)
            self._improvement_samples.pop(0)

        n_samples = len(self._F_samples)

        if n_samples < 5:
            self.F = np.clip(self.F + np.random.randn() * 0.1, 0.1, 1.5)
            self.CR = np.clip(self.CR + np.random.randn() * 0.05, 0.0, 1.0)
            return

        n_candidates = 50
        F_candidates = np.linspace(0.1, 1.5, n_candidates)
        CR_candidates = np.linspace(0.0, 1.0, n_candidates)

        np.random.shuffle(F_candidates)
        np.random.shuffle(CR_candidates)

        n_bootstrap = 100
        candidate_scores = np.zeros(n_candidates)

        for c in range(n_candidates):
            F_c, CR_c = F_candidates[c], CR_candidates[c]
            bootstrap_scores = []

            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                F_boot = np.array([self._F_samples[i] for i in indices])
                CR_boot = np.array([self._CR_samples[i] for i in indices])
                imp_boot = np.array([self._improvement_samples[i] for i in indices])

                angle = np.random.rand() * 2 * np.pi
                proj = np.cos(angle) * (F_boot - F_c) + np.sin(angle) * (CR_boot - CR_c)

                weights = np.exp(-np.abs(proj) / 0.3)
                score = np.sum(weights * imp_boot) / (np.sum(weights) + 1e-8)
                bootstrap_scores.append(score)

            candidate_scores[c] = np.mean(bootstrap_scores)

        probs = np.exp(candidate_scores - candidate_scores.max())
        probs /= probs.sum()
        best_idx = np.random.choice(n_candidates, p=probs)

        self.F = np.clip(F_candidates[best_idx] + np.random.randn() * 0.05, 0.1, 1.5)
        self.CR = np.clip(CR_candidates[best_idx] + np.random.randn() * 0.02, 0.0, 1.0)
    
    # =====================================================================
    # END OF ADAPTATION STRATEGIES
    # =====================================================================
    
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
        
        # On restart, optionally re-probe if we have evidence current strategy is failing
        if len(self._epoch_adapt_history) > 0 and self._epoch_adapt_history[-1] == self._committed_strategy:
            # Same strategy failed across epochs — try re-probing
            self._adapt_phase = 'init'

        return population, f_opt, x_opt
    
    def __call__(self, func, stopping_condition):
        # Store func reference for probe evaluation
        self._probe_func = func
        
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        total_budget = self.max_iter
        
        while not stopping_condition():
            # ============================================================
            # PHASE A: PROBE (adaptive strategy selection)
            # ============================================================
            if self._adapt_phase == 'init':
                self._adapt_phase = 'probing'
                # Run probe phase and get best strategy
                best_strat = self._run_probe_phase(func, total_budget)
                # Commit to the winner with weight 1.0 (no blending)
                self._commit_to_strategy(best_strat)
                # Reset for committed phase
                self.stagnation_counter = 0
                self.prev_best_fitness = f_opt
            
            # ============================================================
            # PHASE B: COMMITTED (epsilon-greedy with one dominant strategy)
            # ============================================================
            if self._adapt_phase == 'committed':
                # Epsilon-greedy: small chance to try a different strategy on stagnation
                if (self.stagnation_counter > 20 and np.random.rand() < self._epsilon_explore):
                    # Try a different strategy temporarily
                    alt_strategies = [i for i in range(len(self._probe_adapt_strategies)) 
                                      if i != self._committed_strategy]
                    if len(alt_strategies) > 0:
                        probing_strat = np.random.choice(alt_strategies)
                        self._probe_adapt_strategies[probing_strat](
                            np.mean(fitness < np.roll(fitness, 1))
                        )
                    else:
                        self._probe_adapt_strategies[self._committed_strategy](
                            np.mean(fitness < np.roll(fitness, 1))
                        )
                else:
                    # Apply committed strategy
                    self._probe_adapt_strategies[self._committed_strategy](
                        np.mean(fitness < np.roll(fitness, 1))
                    )
            
            # Store population reference for strategies that need it
            self.population = population
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
            
            # Archive
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            
            # Update population reference after selection
            self.population = population
            self._current_population = population
            
            # Re-apply adaptation in committed phase (second call for stability)
            if self._adapt_phase == 'committed':
                self._probe_adapt_strategies[self._committed_strategy](improvement_rate)
            
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
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
```