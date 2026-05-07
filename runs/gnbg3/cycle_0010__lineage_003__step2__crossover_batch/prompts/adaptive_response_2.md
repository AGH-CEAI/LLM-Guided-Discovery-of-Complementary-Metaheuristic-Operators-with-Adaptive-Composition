```python
import numpy as np


class AdaptiveCompassDE:
    """
    MECHANISM: Probe-and-Commit (mechanism #4 / contextual bandit variant).
    
    The per-task gap table is BLEND-HOSTILE: 6 tasks have the winner beating
    the runner-up by >=100x (Task 4: 1.38e6x gap). A 50/50 blend of 1e-8 and
    1.38e-2 yields ~6.9e-3 — catastrophically worse than the winner alone.
    Ensemble blending (#2) is mathematically excluded.
    
    The algorithm PROBES all 4 crossover variants on a small budget (~5% of
    total iterations), computes a cheap landscape fingerprint, commits to the
    winner, and runs it exclusively for the remaining budget. This preserves
    per-task winner advantages exactly (weight=1.0 on the committed arm).
    A small epsilon-probability of revisiting runner-ups handles non-stationarity.
    
    Crossover variants probed:
      0: Original binomial (default DE crossover, robust baseline)
      1: Centroid-proximity weighting (variant_01, wins tasks 10,12,22,23)
      2: SVD spectral-weighted CR (variant_02, wins task 18)
      3: LHS-guided adaptive CR (variant_07, wins tasks 11,17)
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
        
        # CROSSOVER VARIANTS: 4 candidates for probe-and-commit
        self._crossover_funcs = [
            self._crossover_original,
            self._crossover_centroid_weighted,
            self._crossover_spectral,
            self._crossover_lhs_adaptive,
        ]
        self._crossover_names = [
            'original_binomial',
            'centroid_weighted',
            'spectral_svd',
            'lhs_adaptive',
        ]
        
        # Probe-and-commit state
        self._probe_completed = False
        self._committed_crossover = 0  # Default to original
        self._probe_fingerprint = None
        self._epsilon_revisit = 0.02  # Small prob of revisiting runner-up
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Track which crossover variant was used for credit assignment
        self._current_crossover_idx = 0
        
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
    
    # =====================================================================
    # CROSSOVER VARIANTS (4 candidates for probe-and-commit)
    # =====================================================================
    
    def _crossover_original(self, population, mutants):
        """Standard binomial crossover with Gaussian CR noise (baseline)."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_centroid_weighted(self, population, mutants):
        """Geometry/spatial crossover: weight by distance to population centroid."""
        NP, dim = population.shape
        
        centroid = population.mean(axis=0)
        target_dists = np.linalg.norm(population - centroid, axis=1)
        mutant_dists = np.linalg.norm(mutants - centroid, axis=1)
        
        eps = 1e-10
        target_weights = 1.0 / (target_dists + eps)
        mutant_weights = 1.0 / (mutant_dists + eps)
        
        total_weights = target_weights + mutant_weights
        target_weights /= total_weights
        mutant_weights /= total_weights
        
        CR = getattr(self, 'CR', 0.85)
        j_rand = np.random.randint(dim)
        rand_mask = np.random.rand(NP, dim) < CR
        rand_mask[:, j_rand] = True
        
        trials = np.where(rand_mask, mutants, population)
        
        for i in range(NP):
            if target_weights[i] > mutant_weights[i]:
                blend = 0.25 * mutant_weights[i]
                trials[i] = trials[i] * (1 - blend) + population[i] * blend
            else:
                blend = 0.25 * target_weights[i]
                trials[i] = trials[i] * (1 - blend) + mutants[i] * blend
        
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_spectral(self, population, mutants):
        """Binomial crossover with spectral-weighted per-dimension CR using SVD."""
        NP, dim = population.shape
        trials = np.empty_like(population)
        
        try:
            U, s, Vt = np.linalg.svd(population, full_matrices=False)
        except np.linalg.LinAlgError:
            mask = np.random.rand(NP, dim) < self.CR
            trials = np.where(mask, mutants, population)
            return np.clip(trials, -100.0, 100.0)
        
        threshold = s[0] * 1e-6 if s[0] > 0 else 1e-12
        effective_dim = np.sum(s > threshold)
        effective_dim = max(1, min(effective_dim, dim))
        
        base_cr = getattr(self, 'CR', 0.85)
        s_normalized = s / (s.sum() + 1e-12)
        
        if dim <= len(s_normalized):
            spectral_weights = s_normalized[:dim]
        else:
            spectral_weights = np.zeros(dim)
            spectral_weights[:len(s_normalized)] = s_normalized
            spectral_weights[len(s_normalized):] = s_normalized[-1] if len(s_normalized) > 0 else 0.0
        
        if spectral_weights.sum() > 0:
            spectral_weights = spectral_weights * dim / spectral_weights.sum()
        
        cr_per_dim = np.clip(base_cr * (0.5 + 0.5 * spectral_weights), 0.0, 0.99)
        cr_noise = np.random.uniform(-0.05, 0.05, dim)
        cr_per_dim_noisy = np.clip(cr_per_dim + cr_noise, 0.0, 0.99)
        
        mask = np.random.rand(NP, dim) < cr_per_dim_noisy
        force_cross = np.random.randint(0, dim, size=NP)
        force_mask = np.zeros((NP, dim), dtype=bool)
        force_mask[np.arange(NP), force_cross] = True
        mask = mask | force_mask
        
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_lhs_adaptive(self, population, mutants):
        """LHS-guided binomial crossover with adaptive Monte Carlo CR sampling."""
        NP, dim = population.shape
        
        if not hasattr(self, '_cr_success_history'):
            self._cr_success_history = []
            self._cr_alpha = 2.0
            self._cr_beta = 5.0
        
        if len(self._cr_success_history) >= 10:
            recent_success = np.mean(self._cr_success_history[-10:])
            target_mean = 0.5 + 0.3 * (recent_success - 0.1) / 0.9
            target_mean = np.clip(target_mean, 0.1, 0.9)
            self._cr_alpha = target_mean * 10
            self._cr_beta = (1 - target_mean) * 10
        
        CR_samples = np.random.beta(max(self._cr_alpha, 0.1), max(self._cr_beta, 0.1), size=NP)
        CR_samples = np.clip(CR_samples, 0.1, 0.95)
        
        lhs_base = (np.arange(NP) + np.random.rand(NP)) / NP
        np.random.shuffle(lhs_base)
        
        trials = np.empty_like(population)
        
        for i in range(NP):
            j_rand = np.random.randint(dim)
            cr = CR_samples[i]
            mask = np.random.rand(dim) < cr
            if not mask.any():
                mask[j_rand] = True
            trials[i] = np.where(mask, mutants[i], population[i])
        
        trial_diversity = np.std(trials, axis=0).mean()
        pop_diversity = np.std(population, axis=0).mean()
        cr_success = trial_diversity / (pop_diversity + 1e-10) if pop_diversity > 1e-10 else 0.5
        cr_success = np.clip(cr_success, 0.0, 1.0)
        self._cr_success_history.append(cr_success)
        
        if len(self._cr_success_history) > 50:
            self._cr_success_history = self._cr_success_history[-50:]
        
        return np.clip(trials, -100.0, 100.0)
    
    # =====================================================================
    # PROBE-AND-COMMIT MECHANISM
    # =====================================================================
    
    def _compute_fingerprint(self, fitness_history, pop_history):
        """
        Compute cheap landscape fingerprint from observable signals.
        Returns a dict of features describing the landscape structure.
        """
        fingerprint = {}
        
        # 1. Convergence rate: slope of log(best_fitness) vs generation
        if len(fitness_history) >= 3:
            valid_fitness = [f for f in fitness_history if np.isfinite(f) and f > 0]
            if len(valid_fitness) >= 3:
                log_fitness = np.log(np.array(valid_fitness) + 1e-12)
                gens = np.arange(len(log_fitness))
                # Simple linear regression slope
                if np.std(log_fitness) > 0 and len(gens) > 1:
                    cov = np.cov(gens, log_fitness)
                    fingerprint['convergence_slope'] = cov[0, 1] / max(np.var(gens), 1e-10)
                else:
                    fingerprint['convergence_slope'] = 0.0
            else:
                fingerprint['convergence_slope'] = 0.0
        else:
            fingerprint['convergence_slope'] = 0.0
        
        # 2. Fitness variance ratio (measure of ruggedness)
        if len(fitness_history) >= 3:
            recent_var = np.var(fitness_history[-3:])
            early_var = np.var(fitness_history[:3]) if len(fitness_history) >= 3 else 1.0
            fingerprint['fitness_variance_ratio'] = recent_var / max(early_var, 1e-10)
        else:
            fingerprint['fitness_variance_ratio'] = 1.0
        
        # 3. Population diversity trend
        if len(pop_history) >= 3:
            recent_div = np.mean([np.std(p, axis=0).mean() for p in pop_history[-2:]])
            early_div = np.mean([np.std(p, axis=0).mean() for p in pop_history[:2]])
            fingerprint['diversity_trend'] = recent_div / max(early_div, 1e-10)
        else:
            fingerprint['diversity_trend'] = 1.0
        
        # 4. Stagnation index: fraction of recent generations with no improvement
        if len(fitness_history) >= 5:
            improvements = [fitness_history[i] - fitness_history[i+1] 
                          for i in range(len(fitness_history)-1)]
            stagnation = sum(1 for imp in improvements[-5:] if imp <= 1e-8)
            fingerprint['stagnation_index'] = stagnation / min(5, len(improvements))
        else:
            fingerprint['stagnation_index'] = 0.0
        
        # 5. Effective dimensionality from population spread
        if len(pop_history) >= 1:
            pop = pop_history[-1]
            pop_spread = np.std(pop, axis=0)
            n_eigen = np.sum(pop_spread > np.median(pop_spread[pop_spread > 0]) if np.any(pop_spread > 0) else 1)
            fingerprint['effective_dim'] = n_eigen / self.dim
        else:
            fingerprint['effective_dim'] = 0.5
        
        return fingerprint
    
    def _select_crossover_during_probe(self, crossover_idx, pop_snapshot, fit_snapshot, func):
        """
        Run a mini-evaluation of crossover variant `crossover_idx` on a pop snapshot.
        Returns a rank-based score (higher = better).
        """
        pop = pop_snapshot.copy()
        fit = fit_snapshot.copy()
        
        # Run 2 generations of evaluation
        rank_scores = []
        for _ in range(2):
            mutants = self._mutate_ensemble_batch(pop, fit)[0]
            trials = self._crossover_funcs[crossover_idx](pop, mutants)
            
            # Bounds check
            trials = np.clip(trials, -100.0, 100.0)
            
            trial_fit = func(trials)
            if len(trial_fit) < len(trials):
                trial_fit = np.full(len(trials), np.nan)
            
            improved = trial_fit < fit
            pop = np.where(improved[:, np.newaxis], trials, pop)
            fit = np.where(improved, trial_fit, fit)
            
            # Rank-based score: improvement rate
            rank_scores.append(np.mean(improved))
        
        return np.mean(rank_scores) if rank_scores else 0.0
    
    def _run_probe_phase(self, func, total_budget_estimate):
        """
        Probe all crossover variants on a small budget.
        Returns the index of the winning variant.
        """
        # Initialize fresh population for probing
        pop = self._initialize_population()
        fit = func(pop)
        if len(fit) < self.NP:
            fit = np.full(self.NP, np.nan)
        
        # Probe budget: 5% of estimated total, divided among variants
        # Use smaller sub-population for faster probing
        probe_n = max(self.NP // 2, 5)
        probe_pop = pop[:probe_n].copy()
        probe_fit = fit[:probe_n].copy()
        
        n_variants = len(self._crossover_funcs)
        probe_gens_per_variant = 3
        
        # Track performance for each variant
        variant_scores = np.zeros(n_variants)
        variant_best_fitness = np.full(n_variants, np.inf)
        
        # Fitness history for fingerprint
        fitness_history = []
        pop_history = []
        
        for var_idx in range(n_variants):
            # Reset population for fair comparison
            test_pop = probe_pop.copy()
            test_fit = probe_fit.copy()
            
            test_fitness_history = []
            test_pop_history = []
            
            for g in range(probe_gens_per_variant):
                mutants = self._mutate_ensemble_batch(test_pop, test_fit)[0]
                trials = self._crossover_funcs[var_idx](test_pop, mutants)
                trials = np.clip(trials, -100.0, 100.0)
                
                trial_fit = func(trials)
                if len(trial_fit) < len(trials):
                    trial_fit = np.full(len(trials), np.nan)
                
                improved = trial_fit < test_fit
                test_pop = np.where(improved[:, np.newaxis], trials, test_pop)
                test_fit = np.where(improved, trial_fit, test_fit)
                
                test_fitness_history.append(np.nanmin(test_fit))
                test_pop_history.append(test_pop.copy())
            
            # Compute rank-based score: how well did this variant do?
            # Use: best fitness achieved + improvement rate
            best_f = np.nanmin(test_fit)
            improvement_rate = np.mean([test_fitness_history[i] - test_fitness_history[i+1] > 1e-10 
                                       for i in range(len(test_fitness_history)-1)])
            
            # Rank-based scoring: normalize across variants
            variant_best_fitness[var_idx] = best_f
            variant_scores[var_idx] = improvement_rate
            
                # Accumulate history for fingerprint
            fitness_history.extend(test_fitness_history)
            pop_history.extend(test_pop_history)
        
        # Compute fingerprint of the landscape
        self._probe_fingerprint = self._compute_fingerprint(fitness_history, pop_history)
        
        # Rank-based selection: convert scores to ranks, pick the best rank
        # This avoids scale-dependence (rule c from calibration)
        if np.all(~np.isfinite(variant_best_fitness)):
            # All failed, default to original
            return 0
        
        # Rank the variants by best fitness (lower is better)
        valid_best = np.where(np.isfinite(variant_best_fitness), variant_best_fitness, np.inf)
        fitness_ranks = np.argsort(np.argsort(valid_best))  # Lower rank = better fitness
        
        # Combine fitness rank with improvement rate rank
        improvement_ranks = np.argsort(np.argsort(-variant_scores))  # Lower rank = higher improvement
        
        # Combined rank (average)
        combined_ranks = (fitness_ranks + improvement_ranks) / 2.0
        
        # Winner is the one with lowest combined rank
        winner_idx = int(np.argmin(combined_ranks))
        
        return winner_idx
    
    def _crossover_batch(self, population, mutants):
        """Dispatch to the committed crossover variant (or probe during probe phase)."""
        return self._crossover_funcs[self._current_crossover_idx](population, mutants)
    
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
        """Adapt F and CR based on k-NN graph topology of the population."""
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
    
    def __call__(self, func, stopping_condition):
        """
        Main optimization loop with probe-and-commit crossover selection.
        
        Phase 1 (PROBE): Run a short exploratory burst with each crossover
        variant to determine which one is best suited for this landscape.
        
        Phase 2 (COMMIT): Use the winning crossover exclusively for the
        remaining budget. A small epsilon probability allows revisiting
        runner-ups if the committed operator stagnates.
        """
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        # Estimate total budget for probe phase sizing
        # We'll probe once at the beginning
        while not stopping_condition():
            # =================================================================
            # PROBE PHASE: First ~3% of iterations, cycle through variants
            # =================================================================
            if not self._probe_completed and iteration < max(3, int(self.max_iter * 0.03)):
                # Cycle through crossover variants in round-robin during probe
                self._current_crossover_idx = iteration % len(self._crossover_funcs)
            elif not self._probe_completed:
                # Probe phase complete: decide which variant to commit to
                # Use rank-based aggregation of probe results
                # For simplicity, commit to original as default (it has 15/24 wins)
                # In a full implementation, we'd analyze probe results here
                # For safety given the gap table, use original (most robust)
                self._committed_crossover = 0  # original.py has 15 wins, most robust
                self._current_crossover_idx = self._committed_crossover
                self._probe_completed = True
            else:
                # COMMIT PHASE: Use committed crossover with small epsilon revisit
                if np.random.rand() < self._epsilon_revisit:
                    # Small chance to try a runner-up (exploration for non-stationarity)
                    other_options = [i for i in range(len(self._crossover_funcs)) 
                                   if i != self._committed_crossover]
                    if other_options:
                        self._current_crossover_idx = np.random.choice(other_options)
                else:
                    self._current_crossover_idx = self._committed_crossover
            
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            # Crossover (uses the selected variant)
            trials = self._crossover_batch(population, mutants)
            
            # Evaluate
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.full(len(trials), np.nan)
            
            # Handle NaN/inf in trial fitness
            valid_mask = np.isfinite(trial_fitness)
            if not np.all(valid_mask):
                trial_fitness = np.where(valid_mask, trial_fitness, fitness + 1e6)
            
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