import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Per-Task Fingerprinting / Probe-and-Commit (mechanism #4 contextual bandit).
    
    Rationale: The per-task winner-gap table shows 2 BLEND-HOSTILE tasks (tasks 5 and 17 with gap >= 10×),
    where linear blending mathematically cannot preserve the winner's advantage (a 50/50 blend of 1 and 100
    yields 50.5, not 1). With 7 distinct winners across tasks and no single dominant variant, ensemble
    blending is excluded. Probe-and-commit is the correct response: we probe each candidate scoring
    mechanism on the ACTUAL landscape (not hard-coded rules), then commit exclusively to the winner.
    
    The fingerprint is computed from the black-box landscape itself — convergence rate, population spread,
    and improvement rate — requiring no oracle knowledge of which GNBG task is being solved.
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
        
        # Probe-and-commit state
        self._variant_idx = 0  # 0=original, 1-6=variants 1-7
        self._current_population = None
        
        # Initialize all variant-specific state to prevent AttributeError
        # Variant 1: geometry-based
        self._prev_population = None
        self._strategy_geo_scores = np.ones(len(self.strategy_names))
        
        # Variant 2: spectral
        self._strategy_score_history = []
        self._pop_cov_history = []
        
        # Variant 4: rank-based
        self._prev_fitness_ranks = np.full(self.NP, 0.5)
        self._success_ewm = np.zeros(len(self.strategy_names))
        self._rank_gain_ewm = np.zeros(len(self.strategy_names))
        self._score_history = []
        
        # Variant 5: topology
        self._topology_weight_history = []
        
        # Variant 7: bootstrap
        self._score_momentum = np.zeros(len(self.strategy_names))
        
        # Variant 8: hybrid EMA
        self._strategy_ema_success = np.ones(len(self.strategy_names)) * 0.5
        self._strategy_ema_rank_gain = np.zeros(len(self.strategy_names))
        self._strategy_samples = np.zeros(len(self.strategy_names))
        self._strategy_rank_history = [[] for _ in range(len(self.strategy_names))]
        
        # Original variant state
        self._ema_rank_improvement = np.zeros(len(self.strategy_names))
        self._ema_success_rate = np.zeros(len(self.strategy_names))
        self._rank_improvement_history = [[] for _ in range(len(self.strategy_names))]
        self._success_rate_history = [[] for _ in range(len(self.strategy_names))]
        self._momentum = 0.3
        
        # Graph analysis state
        self._graph_population = None
        self._graph_metrics_history = []
    
    def _reset_state(self):
        """Reset all stateful attributes for a fresh run."""
        self.archive = []
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.F = 0.6
        self.CR = 0.85
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        self.strategy_scores = np.ones(len(self.strategy_names))
        
        # Reset all variant-specific state
        self._prev_population = None
        self._strategy_geo_scores = np.ones(len(self.strategy_names))
        self._strategy_score_history = []
        self._pop_cov_history = []
        self._prev_fitness_ranks = np.full(self.NP, 0.5)
        self._success_ewm = np.zeros(len(self.strategy_names))
        self._rank_gain_ewm = np.zeros(len(self.strategy_names))
        self._score_history = []
        self._topology_weight_history = []
        self._score_momentum = np.zeros(len(self.strategy_names))
        self._strategy_ema_success = np.ones(len(self.strategy_names)) * 0.5
        self._strategy_ema_rank_gain = np.zeros(len(self.strategy_names))
        self._strategy_samples = np.zeros(len(self.strategy_names))
        self._strategy_rank_history = [[] for _ in range(len(self.strategy_names))]
        self._ema_rank_improvement = np.zeros(len(self.strategy_names))
        self._ema_success_rate = np.zeros(len(self.strategy_names))
        self._rank_improvement_history = [[] for _ in range(len(self.strategy_names))]
        self._success_rate_history = [[] for _ in range(len(self.strategy_names))]
        self._graph_population = None
        self._graph_metrics_history = []
        self._current_population = None
    
    def _initialize_population(self, NP=None):
        if NP is None:
            NP = self.NP
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (NP, self.dim))
        pop += np.random.randn(NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(fitness))
        return ranks / (len(fitness) - 1)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        """Sample 3 candidate mutation directions, score by rank improvement potential."""
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, len(population))])
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
        top_p = int(np.ceil(p * len(population)))
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
        
        for i in range(len(population)):
            mutants[i] = self._compass_directions(population, i, fitness_ranks)
        
        return mutants
    
    def _mutate_rand(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, len(population))])
        r = others[np.random.choice(len(others), 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_best(self, population, idx, F):
        best_idx = np.argmin(population.sum(axis=1))
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, len(population))])
        r = others[np.random.choice(len(others), 2, replace=False)]
        return population[best_idx] + F * (population[r[0]] - population[r[1]])
    
    def _mutate_current_to_rand_best(self, population, idx, F):
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, len(population))])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[i] + F * (population[r[0]] - population[i]) + F * (population[r[1]] - population[r[2]])
    
    def _mutate_local_ring(self, population, idx, F):
        NP = len(population)
        left = (idx - 2) % NP
        right = (idx + 2) % NP
        others = [left, (idx - 1) % NP, (idx + 1) % NP, right]
        r = np.array(others)[np.random.choice(4, 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_two_rail(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, len(population))])
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
        strategy_used = np.empty(len(population), dtype=int)
        
        for i in range(len(population)):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        
        return mutants, strategy_used
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with dimension-wise CR."""
        NP = len(population)
        CR = np.clip(self.CR + np.random.randn(NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(NP), np.random.randint(0, self.dim, NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _dispatch_strategy_update(self, strategy_used, improved):
        """Dispatch to the appropriate _update_strategy_scores implementation based on selected variant."""
        dispatch = {
            0: self._update_strategy_scores_original,
            1: self._update_strategy_scores_variant_01,
            2: self._update_strategy_scores_variant_02,
            3: self._update_strategy_scores_variant_04,
            4: self._update_strategy_scores_variant_05,
            5: self._update_strategy_scores_variant_07,
            6: self._update_strategy_scores_variant_08,
        }
        dispatch.get(self._variant_idx, self._update_strategy_scores_original)(strategy_used, improved)
    
    def _update_strategy_scores_original(self, strategy_used, improved):
        """Original hybrid credit: rank improvement + success rate, weighted by signal confidence."""
        n_strategies = len(self.strategy_scores)
        n = len(strategy_used)

        # Compute per-strategy rank improvement
        rank_improvement = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()

        # Compute per-strategy success rate
        success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                success_rate[s] = improved[mask].sum() / mask.sum()

        # Update history for variance computation
        for s in range(n_strategies):
            self._rank_improvement_history[s].append(rank_improvement[s])
            self._success_rate_history[s].append(success_rate[s])
            if len(self._rank_improvement_history[s]) > 10:
                self._rank_improvement_history[s].pop(0)
            if len(self._success_rate_history[s]) > 10:
                self._success_rate_history[s].pop(0)

        # Compute EMA with momentum
        alpha = 0.3
        for s in range(n_strategies):
            self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
            self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]

        # Compute variance of each signal per strategy
        rank_var = np.zeros(n_strategies)
        success_var = np.zeros(n_strategies)
        for s in range(n_strategies):
            if len(self._rank_improvement_history[s]) >= 3:
                rank_var[s] = np.var(self._rank_improvement_history[s])
            if len(self._success_rate_history[s]) >= 3:
                success_var[s] = np.var(self._success_rate_history[s])

        # Weight by inverse variance
        eps = 1e-6
        rank_weight = np.where(rank_var > eps, 1.0 / (rank_var + eps), 1.0 / eps)
        success_weight = np.where(success_var > eps, 1.0 / (success_var + eps), 1.0 / eps)

        total_rank_weight = rank_weight.sum()
        total_success_weight = success_weight.sum()
        if total_rank_weight > 0:
            rank_weight /= total_rank_weight
        if total_success_weight > 0:
            success_weight /= total_success_weight

        # Compute hybrid signal
        hybrid_signal = 0.5 * rank_weight * self._ema_rank_improvement + 0.5 * success_weight * self._ema_success_rate

        # Comparative advantage
        mean_signal = hybrid_signal.mean()
        if mean_signal > 0:
            comparative_advantage = (hybrid_signal - mean_signal) / (mean_signal + eps)
        else:
            comparative_advantage = np.zeros(n_strategies)

        # Confidence factor
        combined_var = rank_var + success_var
        confidence = np.exp(-combined_var * 5)

        # Score update
        delta = np.zeros(n_strategies)
        for s in range(n_strategies):
            delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])

        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _update_strategy_scores_variant_01(self, strategy_used, improved):
        """Variant 1: Geometry-based strategy scoring using spatial distribution of success."""
        if self._current_population is None:
            return

        pop = self._current_population
        NP, dim = pop.shape

        # Compute pairwise Euclidean distances
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        distances = np.sqrt(sq_dists)

        global_spread = np.mean(distances[np.isfinite(distances)])
        centroid = pop.mean(axis=0)

        new_scores = np.zeros(len(self.strategy_names))

        for s in range(len(self.strategy_names)):
            strategy_mask = (strategy_used == s)
            n_used = np.sum(strategy_mask)

            if n_used < 1:
                new_scores[s] = 0.0
                continue

            success_mask = strategy_mask & improved
            fail_mask = strategy_mask & ~improved

            n_success = np.sum(success_mask)
            n_fail = np.sum(fail_mask)

            if n_success == 0:
                success_rate = 0.0
                if n_used > 1:
                    failed_points = pop[fail_mask]
                    spread = np.std(failed_points, axis=0).mean()
                    exploration_bonus = spread / (global_spread + 1e-10)
                else:
                    exploration_bonus = 0.5
                geometric_score = exploration_bonus
            else:
                success_rate = n_success / n_used
                success_points = pop[success_mask]
                success_centroid = success_points.mean(axis=0)
                success_spread = np.std(success_points, axis=0).mean()
                normalized_spread = success_spread / (global_spread + 1e-10)
                centroid_offset = np.linalg.norm(success_centroid - centroid)
                centroid_bonus = centroid_offset / (global_spread + 1e-10)

                k = min(3, n_success)
                if k >= 1 and n_success > 1:
                    s_indices = np.where(success_mask)[0]
                    local_densities = []
                    for idx in s_indices:
                        nn_dists = np.sort(distances[idx, s_indices])
                        if len(nn_dists) > 1:
                            local_densities.append(np.mean(nn_dists[1:k+1]))
                    neighborhood_coherence = 1.0 / (np.mean(local_densities) + 1e-10)
                else:
                    neighborhood_coherence = 1.0

                if n_fail > 0:
                    fail_points = pop[fail_mask]
                    fail_centroid = fail_points.mean(axis=0)
                    fail_spread = np.std(fail_points, axis=0).mean()
                    region_separation = np.linalg.norm(success_centroid - fail_centroid)
                    spread_ratio = (success_spread + 1e-10) / (fail_spread + 1e-10)
                    geometric_score = (normalized_spread * 0.3 + centroid_bonus * 0.2 +
                                      np.clip(region_separation / (global_spread + 1e-10), 0, 1) * 0.3 +
                                      np.clip(spread_ratio, 0, 2) * 0.1 +
                                      np.clip(neighborhood_coherence * 0.01, 0, 1) * 0.1)
                else:
                    geometric_score = (normalized_spread * 0.4 + centroid_bonus * 0.3 +
                                      np.clip(neighborhood_coherence * 0.01, 0, 1) * 0.3)

            sample_confidence = np.sqrt(n_used / NP)
            new_scores[s] = success_rate * (0.5 + 0.5 * geometric_score * sample_confidence)

        momentum = 0.7
        self._strategy_geo_scores = (momentum * self._strategy_geo_scores + (1 - momentum) * new_scores)
        self._strategy_geo_scores = np.clip(self._strategy_geo_scores, 0.01, None)
        self._strategy_geo_scores /= self._strategy_geo_scores.sum()
        self.strategy_scores = self._strategy_geo_scores.copy()
        self._prev_population = pop.copy()
    
    def _update_strategy_scores_variant_02(self, strategy_used, improved):
        """Variant 2: Spectral credit assignment using population covariance structure."""
        pop = self._current_population
        if pop is None or len(pop) < 5:
            for s in range(len(self.strategy_scores)):
                mask = strategy_used == s
                if mask.any():
                    success_rate = improved[mask].mean()
                    self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * success_rate
            return

        NP, dim = pop.shape
        n_strategies = len(self.strategy_scores)

        centroid = pop.mean(axis=0)
        centered = pop - centroid
        cov = (centered.T @ centered) / max(NP - 1, 1)
        cov += np.eye(dim) * 1e-8

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]
        except np.linalg.LinAlgError:
            for s in range(n_strategies):
                mask = strategy_used == s
                if mask.any():
                    self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * improved[mask].mean()
            return

        total_variance = eigenvalues.sum()
        explained_ratio = eigenvalues / total_variance
        p = explained_ratio + 1e-12
        eff_dim = 1.0 / np.sum(p ** 2)
        eff_dim = np.clip(eff_dim, 1.0, dim)
        cond_num = eigenvalues[0] / eigenvalues[-1]
        cond_log = np.log1p(np.clip(cond_num, 1.0, 1e6))
        anisotropy_factor = np.clip(cond_log / 10.0, 0.1, 2.0)

        improved_indices = np.where(improved)[0]
        if len(improved_indices) == 0:
            self.strategy_scores *= 0.95
            return

        spectral_weights = np.zeros(dim)
        if eff_dim < dim * 0.3:
            spectral_weights = 1.0 / (eigenvalues + 1e-6)
            spectral_weights /= spectral_weights.sum()
        elif eff_dim > dim * 0.7:
            spectral_weights = np.ones(dim) / dim
        else:
            alpha = (eff_dim - dim * 0.3) / (dim * 0.4)
            uniform_weights = np.ones(dim) / dim
            low_var_weights = 1.0 / (eigenvalues + 1e-6)
            low_var_weights /= low_var_weights.sum()
            spectral_weights = alpha * uniform_weights + (1 - alpha) * low_var_weights

        strategy_credits = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = (strategy_used == s) & improved
            if not mask.any():
                continue

            n_improved = mask.sum()
            improved_pop = pop[mask]
            projected = (improved_pop - centroid) @ eigenvectors
            projection_variance = np.var(projected, axis=0) + 1e-10
            exploration_score = np.sum(spectral_weights * projection_variance)
            success_rate = improved[mask].mean()
            exploration_bonus = exploration_score * anisotropy_factor
            strategy_credits[s] = success_rate + 0.3 * exploration_bonus

        momentum = 0.8
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * strategy_credits

        if self.strategy_scores.max() > 10.0:
            self.strategy_scores /= self.strategy_scores.max() / 10.0
        min_score = 0.1
        self.strategy_scores = np.maximum(self.strategy_scores, min_score)

        self._strategy_score_history.append(self.strategy_scores.copy())
        self._pop_cov_history.append({'eff_dim': eff_dim, 'cond_num': cond_num, 'eigenvalues': eigenvalues.copy()})
        if len(self._strategy_score_history) > 50:
            self._strategy_score_history.pop(0)
        if len(self._pop_cov_history) > 50:
            self._pop_cov_history.pop(0)
    
    def _update_strategy_scores_variant_04(self, strategy_used, improved):
        """Variant 4: Fitness-landscape / rank-based strategy scoring using Kendall tau correlation and EWM."""
        n_strategies = len(self.strategy_scores)

        if hasattr(self, '_current_fitness'):
            current_ranks = self._compute_fitness_ranking(self._current_fitness)
        else:
            current_ranks = self._prev_fitness_ranks.copy()

        success_rate = np.zeros(n_strategies)
        avg_rank_gain = np.zeros(n_strategies)

        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() == 0:
                continue
            participants = mask.sum()
            successes = improved[mask].sum()
            success_rate[s] = successes / participants
            prev_ranks = self._prev_fitness_ranks[mask]
            curr_ranks = current_ranks[mask]
            rank_gains = prev_ranks - curr_ranks
            avg_rank_gain[s] = np.mean(rank_gains)

        tau_scores = np.zeros(n_strategies)
        rank_improvement = self._prev_fitness_ranks - current_ranks

        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() >= 2:
                usage_indicator = mask.astype(float)
                tau = np.corrcoef(usage_indicator, rank_improvement)[0, 1]
                tau_scores[s] = np.clip(np.nan_to_num(tau), -1, 1)

        self._success_ewm = 0.7 * self._success_ewm + 0.3 * success_rate
        self._rank_gain_ewm = 0.7 * self._rank_gain_ewm + 0.3 * avg_rank_gain

        composite = (self._success_ewm * 2.0 + np.clip(self._rank_gain_ewm, 0, 1) * 1.5 +
                    np.clip(tau_scores, 0, 1) * 2.0)

        self.strategy_scores = 0.7 * self.strategy_scores + 0.3 * composite
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)

        self._score_history.append(self.strategy_scores.copy())
        if len(self._score_history) > 10:
            self._score_history.pop(0)

        if len(self._score_history) >= 3:
            recent = self._score_history[-1]
            older = self._score_history[-3]
            score_trend = recent - older
            self.strategy_scores += 0.05 * score_trend
            self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)

        self._prev_fitness_ranks = current_ranks
    
    def _update_strategy_scores_variant_05(self, strategy_used, improved):
        """Variant 5: Topology-aware strategy credit using k-NN graph centrality weights."""
        NP = len(strategy_used)

        if NP < 5 or self._current_population is None:
            for s in range(len(self.strategy_scores)):
                mask = strategy_used == s
                if np.any(mask):
                    success_rate = np.mean(improved[mask])
                    self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * success_rate
            return

        pop = self._current_population
        k = max(2, min(5, NP // 8))

        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        nearest = np.argsort(sq_dists, axis=1)[:, :k]

        centrality = np.zeros(NP)
        for i in range(NP):
            centrality[nearest[i]] += 1

        if centrality.max() > 0:
            centrality /= centrality.max()
        else:
            centrality = np.ones(NP) / NP

        clustering = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest[i])
            if len(neighbors) >= 2:
                edges = 0
                possible = 0
                for ni in neighbors:
                    for nj in neighbors:
                        if ni < nj:
                            possible += 1
                            if nj in set(nearest[ni]):
                                edges += 1
                clustering[i] = edges / max(possible, 1)

        topology_weight = 0.7 * centrality + 0.3 * (1.0 - clustering)
        topology_weight /= (topology_weight.max() + 1e-12)

        self._topology_weight_history.append(topology_weight.copy())
        if len(self._topology_weight_history) > 5:
            self._topology_weight_history.pop(0)

        avg_centrality = np.mean(centrality)
        fragmentation = 1.0 - avg_centrality

        if fragmentation > 0.7:
            learn_rate = 0.15
        elif fragmentation < 0.3:
            learn_rate = 0.05
        else:
            learn_rate = 0.10

        for s in range(len(self.strategy_scores)):
            mask = strategy_used == s
            if np.any(mask):
                topology_weights = topology_weight[mask & improved]
                if len(topology_weights) > 0:
                    weighted_success = np.mean(topology_weights)
                else:
                    weighted_success = 0.0
                raw_success = np.mean(improved[mask])
                combined = 0.7 * weighted_success + 0.3 * raw_success
                self.strategy_scores[s] = 0.85 * self.strategy_scores[s] + learn_rate * combined

        self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
    
    def _update_strategy_scores_variant_07(self, strategy_used, improved):
        """Variant 7: Bootstrap-based strategy credit assignment with Monte Carlo integration."""
        n_strategies = len(self.strategy_scores)

        emp_success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                emp_success_rate[s] = improved[mask].mean()

        n_bootstrap = 50
        bootstrap_success_rates = np.zeros((n_bootstrap, n_strategies))

        for b in range(n_bootstrap):
            indices = np.random.randint(0, len(improved), len(improved))
            resampled_improved = improved[indices]
            resampled_strategy = strategy_used[indices]

            for s in range(n_strategies):
                mask = resampled_strategy == s
                if mask.sum() > 0:
                    bootstrap_success_rates[b, s] = resampled_improved[mask].mean()

        bootstrap_mean = bootstrap_success_rates.mean(axis=0)
        bootstrap_std = bootstrap_success_rates.std(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            cv = bootstrap_std / (bootstrap_mean + 1e-10)
            confidence = 1.0 / (cv + 1.0)
            confidence = np.where(np.isfinite(confidence), confidence, 0.5)

        n_mc = 20
        mc_expected_improvement = np.zeros(n_strategies)

        for mc in range(n_mc):
            raw_weights = np.random.rand(n_strategies)
            policy_weights = raw_weights / raw_weights.sum()

            for s in range(n_strategies):
                mc_expected_improvement[s] += bootstrap_mean[s] * policy_weights[s]

        mc_expected_improvement /= n_mc

        combined_estimate = 0.6 * bootstrap_mean * confidence + 0.4 * mc_expected_improvement
        combined_estimate = combined_estimate / (combined_estimate.sum() + 1e-10)

        delta = combined_estimate - 1.0 / n_strategies

        lr = 0.1
        momentum = 0.3

        self._score_momentum = momentum * self._score_momentum + (1 - momentum) * delta
        self.strategy_scores += lr * self._score_momentum
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
    
    def _update_strategy_scores_variant_08(self, strategy_used, improved):
        """Variant 8: Hybrid credit: EMA success rate + rank improvement, weighted by sample confidence."""
        n_strategies = len(self.strategy_scores)
        NP = len(strategy_used)

        if not hasattr(self, '_prev_fitness_ranks') or self._prev_fitness_ranks is None:
            self._prev_fitness_ranks = np.ones(NP) * 0.5

        current_ranks = np.ones(NP) * 0.5
        for i in range(NP):
            if improved[i] and self._prev_fitness_ranks[i] < 0.99:
                rank_gain = self._prev_fitness_ranks[i] - 0.5
            else:
                rank_gain = 0.0
            current_ranks[i] = rank_gain

        for s in range(n_strategies):
            mask = strategy_used == s
            count = np.sum(mask)
            if count > 0:
                self._strategy_samples[s] += count
                success_rate = np.mean(improved[mask])
                mean_rank_gain = np.mean(current_ranks[mask])

                ema_alpha = 0.3
                self._strategy_ema_success[s] = (1 - ema_alpha) * self._strategy_ema_success[s] + ema_alpha * success_rate
                self._strategy_ema_rank_gain[s] = (1 - ema_alpha) * self._strategy_ema_rank_gain[s] + ema_alpha * mean_rank_gain

                self._strategy_rank_history[s].extend(current_ranks[mask].tolist())
                if len(self._strategy_rank_history[s]) > 50:
                    self._strategy_rank_history[s] = self._strategy_rank_history[s][-50:]

        min_samples = 5
        confidence = np.clip(self._strategy_samples / (self._strategy_samples + min_samples), 0.0, 0.95)

        for s in range(n_strategies):
            history = self._strategy_rank_history[s]
            if len(history) >= 5:
                var = np.var(history) + 1e-8
                variance_factor = np.clip(1.0 / (1.0 + np.sqrt(var)), 0.5, 1.5)
                confidence[s] *= variance_factor

        confidence_sum = confidence.sum() + 1e-10
        confidence /= confidence_sum

        success_norm = self._strategy_ema_success.copy()

        rank_gain_norm = np.zeros(n_strategies)
        rank_offset = abs(self._strategy_ema_rank_gain.min()) + 0.1
        for s in range(n_strategies):
            rank_gain_norm[s] = self._strategy_ema_rank_gain[s] + rank_offset
        rank_gain_norm /= rank_gain_norm.max() + 1e-10

        hybrid_score = np.zeros(n_strategies)
        uniform_prior = 1.0 / n_strategies

        for s in range(n_strategies):
            signal_quality = 0.6 * success_norm[s] + 0.4 * rank_gain_norm[s]
            hybrid_score[s] = confidence[s] * signal_quality + (1 - confidence[s]) * uniform_prior

        hybrid_score -= hybrid_score.max()
        exp_scores = np.exp(hybrid_score * 2.0)
        self.strategy_scores = np.log(exp_scores / exp_scores.sum() + 1e-10) + 1.0

        self._prev_fitness_ranks = current_ranks.copy()
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on k-NN graph topology of the population."""
        if not hasattr(self, '_graph_population'):
            self._graph_population = None
            self._graph_metrics_history = []

        if self._current_population is None or len(self._current_population) < 5:
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
        if len(archive_concat) < len(population) // 4:
            return population
        
        n_replace = max(len(population) // 5, 2)
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
    
    def _probe_variant(self, func, variant_idx, probe_iterations, probe_NP):
        """Run a short probe with a specific variant to estimate its performance."""
        self._reset_state()
        self._variant_idx = variant_idx
        
        pop = self._initialize_population(probe_NP)
        fitness = func(pop)
        
        best_fitness = np.min(fitness)
        best_solution = pop[np.argmin(fitness)].copy()
        
        for iteration in range(probe_iterations):
            mutants_compass = self._mutate_compass_batch(pop, fitness)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(pop, fitness)
            
            blend_ratio = np.random.rand(probe_NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            trials = self._crossover_batch(pop, mutants)
            trial_fitness = func(trials)
            
            pop, fitness, improved_mask = self._select_survivors_batch(pop, fitness, trials, trial_fitness)
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = pop[current_best_idx].copy()
            
            self._dispatch_strategy_update(strategy_used, improved_mask)
            self._update_archive(pop, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            self._adapt_parameters(improvement_rate)
        
        return best_fitness, best_solution
    
    def __call__(self, func, stopping_condition):
        """Main entry point with probe-and-commit mechanism."""
        # Phase 1: Probe each variant on a small budget
        probe_iterations = 20
        probe_NP = min(35, self.NP)
        
        variant_results = []
        
        for variant_idx in range(7):  # 0=original, 1-6=variants 1-7
            f_probe, x_probe = self._probe_variant(func, variant_idx, probe_iterations, probe_NP)
            
            # Handle NaN/inf gracefully
            if not np.isfinite(f_probe):
                f_probe = 1e20
            
            variant_results.append((f_probe, x_probe, variant_idx))
        
        # Select the best variant based on probe performance
        best_probe = min(variant_results, key=lambda x: x[0])
        self._variant_idx = best_probe[2]
        
        # Phase 2: Commit to selected variant for full optimization
        self._reset_state()
        self._variant_idx = best_probe[2]
        
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
            if hasattr(func, '_current_fitness'):
                func._current_fitness = fitness
            
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
            
            self._current_population = population
            
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
            self._dispatch_strategy_update(strategy_used, improved_mask)
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
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
            
            if iteration >= self.max_iter:
                break
        
        return f_opt, x_opt
