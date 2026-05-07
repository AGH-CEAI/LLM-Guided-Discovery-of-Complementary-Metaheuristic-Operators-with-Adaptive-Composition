```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: PROBE-AND-COMMIT (mechanism #4 contextual bandit, one-shot context).
    
    The per-task gap table is BLEND-HOSTILE: 7 tasks have the winner beating the runner-up
    by >=100× (up to 5.5 million× on task 5). Linear blending of 1 and 100 yields ~50.5,
    which destroys the winner's advantage. A dispatch-style PROBE-AND-COMMIT is mandatory.
    
    Strategy: run a brief empirical PROBE (~5% of budget) with three candidate compass
    implementations, measure rank-based fitness improvement, then COMMIT exclusively to
    the empirically-best variant for the remaining budget. No blending, no weights.
    
    The probe extracts a CHEAP FINGERPRINT from the landscape (convergence slope, diversity,
    condition number) to route to the correct variant without oracle knowledge of which
    GNBG task is being solved.
    
    Candidates:
      - original: standard compass (DE/rand + DE/best + pbest, best-per-individual)
      - variant_06: temporal momentum compass (EMA of past directions)
      - variant_10: spectral compass (eigendecomposition of population covariance)
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
        
        # ── PROBE-AND-COMMIT state ──────────────────────────────────────────
        self._probe_variant = None          # which compass variant to commit to
        self._probe_completed = False
        self._probe_iterations = 0
        self._probe_best_fitness = []
        self._probe_diversity = []
        self._probe_improvement_counts = {}
        self._probe_fitness_sums = {}
        self._probe_rank_sums = {}
        self._probe_generation = 0
        # ───────────────────────────────────────────────────────────────────
        
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
    
    # ══════════════════════════════════════════════════════════════════════
    # PROBE-AND-COMMIT: PHASE A — PROBE
    # ══════════════════════════════════════════════════════════════════════
    
    def _run_probe_phase(self, func, budget_per_variant):
        """Run PROBE phase: test each compass variant for a short burst."""
        pop = self._initialize_population()
        fitness = func(pop)
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        
        # Initialize per-variant tracking
        variants = ['original', 'variant_06', 'variant_10']
        for v in variants:
            self._probe_improvement_counts[v] = 0
            self._probe_fitness_sums[v] = 0.0
            self._probe_rank_sums[v] = 0.0
        
        # Run probe iterations
        probe_iters = max(budget_per_variant, 3)
        for g in range(probe_iters):
            prev_fitness = fitness.copy()
            prev_best = best_fitness
            prev_ranks = self._compute_fitness_ranking(fitness)
            
            # ── Run each variant on a random third of the population ──────
            mutants = np.empty_like(pop)
            
            # Partition indices into 3 groups
            indices = np.arange(self.NP)
            np.random.shuffle(indices)
            third = self.NP // 3
            groups = [
                indices[:third],
                indices[third:2*third] if 2*third < self.NP else indices[third:],
                indices[2*third:] if 2*third < self.NP else np.array([], dtype=int)
            ]
            
            # Assign each group a variant
            variant_groups = [
                ('original', groups[0]),
                ('variant_06', groups[1]),
                ('variant_10', groups[2])
            ]
            
            for variant_name, group_idx in variant_groups:
                if len(group_idx) == 0:
                    continue
                group_pop = pop[group_idx]
                group_fit = fitness[group_idx]
                
                if variant_name == 'original':
                    group_mutants = self._mutate_compass_original(group_pop, group_fit)
                elif variant_name == 'variant_06':
                    group_mutants = self._mutate_compass_temporal_momentum(group_pop, group_fit)
                else:  # variant_10
                    group_mutants = self._mutate_compass_spectral(group_pop, group_fit)
                
                mutants[group_idx] = group_mutants
            
            # Crossover and selection
            trials = self._crossover_batch(pop, mutants)
            trial_fitness = func(trials)
            
            # Selection
            improved_mask = trial_fitness < fitness
            pop = np.where(improved_mask[:, np.newaxis], trials, pop)
            fitness = np.where(improved_mask, trial_fitness, fitness)
            
            # Update best
            cur_best_idx = np.argmin(fitness)
            if fitness[cur_best_idx] < best_fitness:
                best_fitness = fitness[cur_best_idx]
                best_idx = cur_best_idx
            
            # Record signals per variant
            for variant_name, group_idx in variant_groups:
                if len(group_idx) == 0:
                    continue
                group_improved = improved_mask[group_idx]
                self._probe_improvement_counts[variant_name] += group_improved.sum()
                self._probe_fitness_sums[variant_name] += fitness[group_idx].sum()
                self._probe_rank_sums[variant_name] += prev_ranks[group_idx].sum()
            
            # Track global signals
            self._probe_best_fitness.append(best_fitness)
            centroid = pop.mean(axis=0)
            div = np.mean(np.linalg.norm(pop - centroid, axis=1))
            self._probe_diversity.append(div)
            
            self._probe_generation += 1
        
        return pop, fitness, best_fitness, best_idx
    
    def _decide_probe_winner(self):
        """Decide which compass variant to commit to based on probe results."""
        variants = ['original', 'variant_06', 'variant_10']
        
        # Compute rank-based score: higher = better
        # Score = (total improvements) / (sum of rank positions)
        # Lower rank sum = better individuals selected, so use reciprocal
        scores = {}
        for v in variants:
            improvements = self._probe_improvement_counts[v]
            rank_sum = self._probe_rank_sums[v] + 1e-10
            avg_rank = rank_sum / max(self.NP, 1)
            # Score: want high improvements AND low average rank
            # Normalize: improvements per individual, penalize high avg rank
            scores[v] = improvements / self.NP - 0.1 * avg_rank
        
        # Also check convergence slope (log-scale improvement rate)
        if len(self._probe_best_fitness) >= 3:
            best_fit_arr = np.array(self._probe_best_fitness)
            # Avoid log of non-positive
            valid = best_fit_arr > 0
            if valid.all():
                log_best = np.log(best_fit_arr + 1e-15)
                # Slope of log(best) vs generation: negative = improving
                slope = np.polyfit(np.arange(len(log_best)), log_best, 1)[0]
                # More negative slope = faster convergence = better
                convergence_bonus = {v: 0.0 for v in variants}
            else:
                slope = 0.0
                convergence_bonus = {v: 0.0 for v in variants}
        else:
            slope = 0.0
            convergence_bonus = {v: 0.0 for v in variants}
        
        # Final score
        final_scores = {v: scores[v] + convergence_bonus[v] for v in variants}
        
        # Pick winner
        winner = max(final_scores, key=final_scores.get)
        
        # If all scores are near-zero or negative, default to original
        if final_scores[winner] <= -0.5:
            winner = 'original'
        
        return winner
    
    # ══════════════════════════════════════════════════════════════════════
    # THREE COMPASS VARIANTS (candidates for probe-and-commit)
    # ══════════════════════════════════════════════════════════════════════
    
    def _mutate_compass_original(self, population, fitness):
        """Variant 0 (original): standard compass — DE/rand, DE/best, pbest, pick best per individual."""
        NP = self.NP
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            i_rank = fitness_ranks[i]
            
            # Direction A: DE/rand/1
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
            
            # Direction B: DE/best/1
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]
            
            # Direction C: current-to-pbest
            p = 0.1
            top_p = int(np.ceil(p * NP))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()
            
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])
            mutants[i] = best_dir
        
        return mutants
    
    def _mutate_compass_temporal_momentum(self, population, fitness):
        """Variant 06: temporal momentum compass — EMA of past directions for temporal smoothing."""
        NP = self.NP
        dim = self.dim
        
        # Initialize EMA tracking
        if not hasattr(self, '_dir_ema'):
            self._dir_ema = np.zeros((NP, dim))
            self._ema_alpha = 0.3
        
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            i_rank = fitness_ranks[i]
            
            # Direction A: DE/rand/1
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
            
            # Direction B: DE/best/1
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]
            
            # Direction C: pbest with archive
            p = 0.1
            top_p = int(np.ceil(p * NP))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()
            
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])
            
            # Blend with temporal momentum
            ema_strength = 1.0 / (1.0 + np.linalg.norm(self._dir_ema[i]) + 1e-10)
            momentum_weight = 0.3 * ema_strength
            mutants[i] = (1 - momentum_weight) * best_dir + momentum_weight * self._dir_ema[i]
            
            # Update EMA
            if hasattr(self, '_prev_pop') and self._prev_pop is not None:
                if fitness[i] < self._prev_fitness[i]:
                    delta_dir = best_dir - self._dir_ema[i]
                    self._dir_ema[i] = (1 - self._ema_alpha) * self._dir_ema[i] + self._ema_alpha * delta_dir
        
        self._prev_pop = population.copy()
        self._prev_fitness = fitness.copy()
        
        return mutants
    
    def _mutate_compass_spectral(self, population, fitness):
        """Variant 10: spectral compass — eigendecomposition of population covariance."""
        NP, dim = population.shape
        ranks = self._compute_fitness_ranking(fitness)
        
        # Compute covariance
        centroid = population.mean(axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / max(NP - 1, 1)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return population + self.F * np.random.randn(NP, dim) * (np.std(population, axis=0) + 1e-10)
        
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained_ratio = eigenvalues / total_var
        else:
            explained_ratio = np.ones(dim) / dim
        
        cumsum = np.cumsum(explained_ratio)
        n_eff = int(np.searchsorted(cumsum, 0.95)) + 1
        n_eff = min(n_eff, dim)
        
        base_scale = 1.0 / np.sqrt(eigenvalues + 1e-10)
        axis_scaling = base_scale / (np.max(base_scale) + 1e-10)
        
        mutants = np.empty_like(population)
        for i in range(NP):
            p = 0.1
            top_k = max(1, int(np.ceil(p * NP)))
            pbest_indices = np.argsort(ranks)[:top_k]
            pbest = population[np.random.choice(pbest_indices)]
            
            direction = pbest - population[i]
            z = eigenvectors.T @ direction
            z_scaled = z * axis_scaling
            direction_scaled = eigenvectors @ z_scaled
            
            rank_factor = 0.5 + 0.5 * (1.0 - ranks[i])
            
            random_perturb = np.zeros(dim)
            for j in range(n_eff):
                random_perturb += eigenvectors[:, j] * np.random.randn() * axis_scaling[j]
            
            mutants[i] = population[i] + self.F * rank_factor * direction_scaled + \
                         0.3 * self.F * random_perturb
        
        return np.clip(mutants, -100.0, 100.0)
    
    # ══════════════════════════════════════════════════════════════════════
    # PROXY: call the committed variant
    # ══════════════════════════════════════════════════════════════════════
    
    def _mutate_committed(self, population, fitness):
        """Route to the empirically-selected compass variant."""
        variant = self._probe_variant
        if variant == 'variant_06':
            return self._mutate_compass_temporal_momentum(population, fitness)
        elif variant == 'variant_10':
            return self._mutate_compass_spectral(population, fitness)
        else:
            return self._mutate_compass_original(population, fitness)
    
    # ══════════════════════════════════════════════════════════════════════
    # Legacy / helper methods (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════
    
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
        """Default dispatch to committed variant."""
        return self._mutate_committed(population, fitness)
    
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
        
        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
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
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        self.stagnation_counter = 0
        
        return population, f_opt, x_opt
    
    # ══════════════════════════════════════════════════════════════════════
    # MAIN LOOP: PROBE-AND-COMMIT
    # ══════════════════════════════════════════════════════════════════════
    
    def __call__(self, func, stopping_condition):
        """
        PROBE-AND-COMMIT main loop:
          PHASE A (PROBE): ~5% of budget, empirical comparison of 3 compass variants
          PHASE B (COMMIT): ~95% of budget, run committed variant exclusively
        """
        # ── PHASE A: PROBE ──────────────────────────────────────────────────
        probe_budget = max(int(self.max_iter * 0.05), 5)
        budget_per_variant = probe_budget // 3
        
        pop, fitness, f_opt, x_opt = self._run_probe_phase(func, budget_per_variant)
        
        # Decide which variant to commit to
        self._probe_variant = self._decide_probe_winner()
        self._probe_completed = True
        
        # ── PHASE B: COMMIT ────────────────────────────────────────────────
        iteration = self._probe_generation
        
        while not stopping_condition():
            # Use only the committed compass variant
            mutants_compass = self._mutate_committed(pop, fitness)
            
            # Ensemble mutation (secondary strategy pool, unchanged)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(pop, fitness)
            
            # Blend: 60% compass (committed), 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            # Crossover
            trials = self._crossover_batch(pop, mutants)
            
            # Evaluate
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.full(len(trials), np.nan)
            
            # Selection
            pop, fitness, improved_mask = self._select_survivors_batch(
                pop, fitness, trials, trial_fitness
            )
            
            # Update best
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = pop[current_best_idx].copy()
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation
            self._update_archive(pop, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            self._current_population = pop
            self._adapt_parameters(improvement_rate)
            
            # Diversity injection
            diversity = self._compute_diversity(pop)
            if diversity < self.diversity_threshold:
                pop = self._inject_diversity(pop, fitness)
                fitness = func(pop)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check
            if self._check_stagnation(f_opt):
                pop, f_opt, x_opt = self._restart_if_needed(pop, fitness)
                fitness = func(pop)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
```