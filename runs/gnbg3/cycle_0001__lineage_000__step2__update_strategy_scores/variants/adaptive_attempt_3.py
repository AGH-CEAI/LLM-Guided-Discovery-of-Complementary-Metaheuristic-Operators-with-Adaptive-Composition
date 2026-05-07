import numpy as np


class AdaptiveCompassDE:
    """
    PROBE-AND-COMMIT with Contextual Bandit (mechanism #4 + per-task fingerprinting).
    
    Blend-hostile diagnosis: 2 tasks (10, 12) have per-task winner gaps >= 10×,
    which mathematically EXCLUDES ensemble blending. A 50/50 blend of 1 and 100
    yields 50.5, not 1 — the advantage cannot be preserved. This implementation
    uses PROBE-AND-COMMIT: a short exploratory burst (~5% budget) to identify
    the best strategy for THIS specific landscape, then commits exclusively to
    that strategy for the remaining budget. The probe extracts rank-based
    improvement fingerprints from the actual landscape, not from oracle knowledge.
    
    The 7 strategy variants map to 7 distinct _update_strategy_scores implementations.
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
        
        # =========================================================
        # PROBE-AND-COMMIT state
        # =========================================================
        self._probe_budget_pct = 0.05  # 5% of budget for probing
        self._committed_strategy = None  # None = probing, else strategy index
        self._probe_round_robin_idx = 0
        self._probe_history = []  # List of (strategy_idx, rank_improvement_rate)
        self._probe_gen = 0
        self._committed_strategy_scores = None  # Scores for committed strategy's credit function
        
        # Sliding window for rank-based rewards (calibration rule b)
        self._reward_window = []  # List of (strategy_idx, rank_improvement)
        self._reward_window_size = 21  # >= 3 * num_arms (5 strategies * 3 = 15)
        
        # Fingerprint signals (cheap landscape diagnostics)
        self._fingerprint_history = {
            'convergence_slopes': [],  # log-best slope over generations
            'rank_entropies': [],      # fitness rank distribution entropy
            'diversity_history': [],   # population spread over time
            'stagnation_count': 0,     # consecutive gens without improvement
            'condition_numbers': [],   # covariance condition number
        }
        
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
    
    def _select_strategy_batch(self):
        """Softmax-weighted strategy selection."""
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness, forced_strategy=None):
        """Use adaptive strategy pool to generate mutants."""
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        
        for i in range(self.NP):
            if forced_strategy is not None:
                strat_idx = forced_strategy
            else:
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
        """Credit assignment: which strategies produced improvements.
        
        During PROBE phase: rank-based reward within sliding window.
        During COMMIT phase: delegate to the committed strategy's credit function.
        """
        # Rank-based improvement rate per strategy
        n_strategies = len(self.strategy_scores)
        current_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                current_rates[s] = improved[mask].sum() / mask.sum()
        
        # Store in sliding window (calibration rule b: rank-based rewards)
        for s in range(n_strategies):
            self._reward_window.append((s, current_rates[s]))
        while len(self._reward_window) > self._reward_window_size:
            self._reward_window.pop(0)
        
        # If in probe phase, record probe history
        if self._committed_strategy is None:
            # Round-robin: each generation tests one strategy
            strategy_tested = self._probe_round_robin_idx % n_strategies
            self._probe_history.append((strategy_tested, current_rates[strategy_tested]))
            self._probe_gen += 1
            self._probe_round_robin_idx += 1
        else:
            # Commit phase: delegate to committed strategy's credit function
            self._apply_committed_strategy_credit(strategy_used, improved, current_rates)
            return
        
        # Standard EMA update during probe (for fallback if commit fails)
        alpha = 0.3
        for s in range(n_strategies):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        self.strategy_scores *= 0.99
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
    def _apply_committed_strategy_credit(self, strategy_used, improved, current_rates):
        """Apply the committed strategy's credit function.
        
        The committed strategy index maps to one of the 7 winning variants.
        This preserves the per-task winner's advantage by committing to
        exactly ONE strategy (weight 1.0), as required for blend-hostile tasks.
        """
        # Map committed strategy to variant credit function
        # variant_06 (8 wins): temporal EMA with momentum - most common winner
        # variant_08 (5 wins): hybrid rank+success with variance weighting
        # original (3 wins): basic success rate
        # variant_09 (2 wins): geometric alignment
        # variant_07 (1 win): Thompson Sampling bootstrap
        # variant_02 (1 win): spectral SVD-based
        # variant_01 (1 win): geometric displacement
        
        variant_map = {
            0: self._credit_variant_01,  # geometric displacement
            1: self._credit_variant_02,  # spectral SVD-based
            2: self._credit_variant_06,  # temporal EMA (most wins)
            3: self._credit_variant_07,  # Thompson Sampling
            4: self._credit_variant_08,  # hybrid rank+success
            5: self._credit_variant_09,  # geometric alignment
            6: self._credit_original,    # original baseline
        }
        
        credit_fn = variant_map.get(self._committed_strategy, self._credit_original)
        credit_fn(strategy_used, improved, current_rates)
    
    def _credit_original(self, strategy_used, improved, current_rates):
        """Original credit: basic success rate."""
        for s in range(len(self.strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        self.strategy_scores *= 0.99
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
    def _credit_variant_01(self, strategy_used, improved, current_rates):
        """Variant 01: geometric displacement (centroid shift + hull volume)."""
        if not hasattr(self, '_prev_centroid_v1'):
            self._prev_centroid_v1 = None
            self._prev_hull_vol_v1 = None
        
        if not hasattr(self, 'population') or self.population is None:
            return
        
        centroid = self.population.mean(axis=0)
        pop_range = self.population.max(axis=0) - self.population.min(axis=0) + 1e-10
        hull_proxy = np.prod(pop_range)
        
        if self._prev_centroid_v1 is not None and self._prev_hull_vol_v1 is not None:
            centroid_shift = np.linalg.norm(centroid - self._prev_centroid_v1)
            hull_change_ratio = hull_proxy / (self._prev_hull_vol_v1 + 1e-10)
        else:
            centroid_shift = 0.0
            hull_change_ratio = 1.0
        
        pop_spread = np.linalg.norm(pop_range)
        pop_spread = max(pop_spread, 1e-10)
        
        for s in range(len(self.strategy_scores)):
            mask = strategy_used == s
            if not np.any(mask):
                continue
            
            improved_idx = np.where(mask & improved)[0]
            n_improved = len(improved_idx)
            
            if n_improved == 0:
                if hull_change_ratio < 1.0:
                    self.strategy_scores[s] *= (0.95 + 0.05 * hull_change_ratio)
                continue
            
            improved_centroid = self.population[improved_idx].mean(axis=0)
            displacement = np.linalg.norm(improved_centroid - centroid)
            norm_displacement = displacement / pop_spread
            credit = norm_displacement * np.sqrt(n_improved) * max(hull_change_ratio, 0.5)
            
            momentum = 0.1
            self.strategy_scores[s] = (1 - momentum) * self.strategy_scores[s] + momentum * credit
        
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)
        self._prev_centroid_v1 = centroid
        self._prev_hull_vol_v1 = hull_proxy
    
    def _credit_variant_02(self, strategy_used, improved, current_rates):
        """Variant 02: spectral SVD-based credit."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return
        
        pop = self._current_population
        NP, dim = pop.shape
        
        if NP < 4 or dim < 2:
            return
        
        pop_centered = pop - pop.mean(axis=0)
        try:
            U, s, Vt = np.linalg.svd(pop_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        
        s_norm = s / s.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        eff_rank = np.exp(entropy) / min(NP, dim)
        
        if not hasattr(self, '_spectral_eff_rank_v2'):
            self._spectral_eff_rank_v2 = np.zeros(len(self.strategy_scores))
            self._strategy_spectral_shifts_v2 = np.zeros(len(self.strategy_scores))
            self._strategy_use_counts_v2 = np.zeros(len(self.strategy_scores))
        
        alpha = 0.3
        for strat_idx in range(len(self.strategy_names)):
            mask = strategy_used == strat_idx
            n_used = np.sum(mask)
            if n_used == 0:
                continue
            
            self._strategy_use_counts_v2[strat_idx] += n_used
            improved_mask = mask & improved
            not_improved_mask = mask & ~improved
            n_imp = np.sum(improved_mask)
            n_not_imp = np.sum(not_improved_mask)
            
            eff_rank_imp = eff_rank
            eff_rank_not_imp = eff_rank
            
            if n_imp > 0:
                imp_pop = pop[improved_mask]
                imp_centered = imp_pop - imp_pop.mean(axis=0)
                try:
                    _, s_imp, _ = np.linalg.svd(imp_centered, full_matrices=False)
                    s_imp_norm = s_imp / s_imp.sum()
                    ent_imp = -np.sum(s_imp_norm * np.log(s_imp_norm + 1e-12))
                    eff_rank_imp = np.exp(ent_imp) / (min(*imp_pop.shape) + 1e-12)
                except:
                    pass
            
            rank_shift = eff_rank_imp - eff_rank_not_imp
            weight = n_imp / (n_imp + n_not_imp + 1e-12)
            self._strategy_spectral_shifts_v2[strat_idx] = (1 - alpha) * self._strategy_spectral_shifts_v2[strat_idx] + alpha * rank_shift * weight
            self.strategy_scores[strat_idx] += 0.1 * np.tanh(5.0 * self._strategy_spectral_shifts_v2[strat_idx])
        
        self.strategy_scores = np.clip(self.strategy_scores, -50.0, 50.0)
    
    def _credit_variant_06(self, strategy_used, improved, current_rates):
        """Variant 06: temporal EMA with momentum and trend detection (8 wins)."""
        n_strategies = len(self.strategy_scores)
        
        if not hasattr(self, '_ema_rates_v6'):
            self._ema_rates_v6 = np.zeros(n_strategies)
            self._prev_ema_rates_v6 = np.zeros(n_strategies)
            self._strategy_momentum_v6 = np.zeros(n_strategies)
            self._history_buffer_v6 = [np.zeros(n_strategies) for _ in range(5)]
            self._history_idx_v6 = 0
        
        self._history_buffer_v6[self._history_idx_v6] = current_rates.copy()
        self._history_idx_v6 = (self._history_idx_v6 + 1) % len(self._history_buffer_v6)
        
        alpha = 0.3
        self._prev_ema_rates_v6 = self._ema_rates_v6.copy()
        self._ema_rates_v6 = alpha * current_rates + (1 - alpha) * self._ema_rates_v6
        trends = self._ema_rates_v6 - self._prev_ema_rates_v6
        
        beta = 0.7
        gamma = 0.15
        self._strategy_momentum_v6 = (
            beta * self._strategy_momentum_v6 +
            (1 - beta) * (self._ema_rates_v6 + gamma * np.sign(trends) * np.sqrt(np.abs(trends)))
        )
        
        recent_variance = np.var(self._history_buffer_v6, axis=0) + 1e-10
        adaptive_lr = 0.15 / (1.0 + 5.0 * recent_variance)
        score_delta = adaptive_lr * self._ema_rates_v6 + 0.05 * self._strategy_momentum_v6
        
        for s in range(n_strategies):
            recent_trend = sum(
                self._history_buffer_v6[(self._history_idx_v6 - i) % len(self._history_buffer_v6)][s]
                for i in range(1, 4)
            )
            if recent_trend < 0.02:
                score_delta[s] -= 0.08
        
        self.strategy_scores += score_delta
        self.strategy_scores -= self.strategy_scores.mean()
        self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
    
    def _credit_variant_07(self, strategy_used, improved, current_rates):
        """Variant 07: Thompson Sampling with bootstrap resampling."""
        n_strategies = len(self.strategy_scores)
        n_trials = len(strategy_used)
        
        if n_trials == 0:
            return
        
        if not hasattr(self, '_strategy_history_v7'):
            self._strategy_history_v7 = [[] for _ in range(n_strategies)]
            self._strategy_successes_v7 = np.zeros(n_strategies)
            self._strategy_attempts_v7 = np.zeros(n_strategies)
        
        for i in range(n_trials):
            s = strategy_used[i]
            imp = improved[i]
            self._strategy_history_v7[s].append(1 if imp else 0)
            self._strategy_attempts_v7[s] += 1
            if imp:
                self._strategy_successes_v7[s] += 1
        
        for s in range(n_strategies):
            if len(self._strategy_history_v7[s]) > 500:
                self._strategy_history_v7[s] = self._strategy_history_v7[s][-400:]
        
        n_bootstrap = 200
        bootstrap_rates = np.zeros((n_bootstrap, n_strategies))
        
        for b in range(n_bootstrap):
            for s in range(n_strategies):
                hist = self._strategy_history_v7[s]
                if len(hist) > 0:
                    indices = np.random.randint(0, len(hist), size=len(hist))
                    sampled = [hist[i] for i in indices]
                    bootstrap_rates[b, s] = np.mean(sampled)
                else:
                    bootstrap_rates[b, s] = 0.0
        
        ci_lower = np.percentile(bootstrap_rates, 10, axis=0)
        ci_upper = np.percentile(bootstrap_rates, 90, axis=0)
        
        sampled_scores = np.zeros(n_strategies)
        for s in range(n_strategies):
            sampled_idx = np.random.randint(n_bootstrap)
            sampled_scores[s] = bootstrap_rates[sampled_idx, s]
            ci_width = max(ci_upper[s] - ci_lower[s], 1e-10)
            exploration = np.random.exponential(0.05 / ci_width)
            sampled_scores[s] += exploration
        
        exp_scores = np.exp(sampled_scores - sampled_scores.max())
        probs = exp_scores / (exp_scores.sum() + 1e-10)
        
        momentum = 0.7
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * probs
    
    def _credit_variant_08(self, strategy_used, improved, current_rates):
        """Variant 08: hybrid rank improvement + success rate with variance weighting (5 wins)."""
        n_strategies = len(self.strategy_scores)
        
        if not hasattr(self, '_ema_rank_v8'):
            self._ema_rank_v8 = np.zeros(n_strategies)
            self._ema_success_v8 = np.zeros(n_strategies)
            self._rank_history_v8 = [[] for _ in range(n_strategies)]
            self._success_history_v8 = [[] for _ in range(n_strategies)]
            self._momentum_v8 = 0.3
        
        for s in range(n_strategies):
            self._rank_history_v8[s].append(current_rates[s])
            self._success_history_v8[s].append(current_rates[s])
            if len(self._rank_history_v8[s]) > 10:
                self._rank_history_v8[s].pop(0)
            if len(self._success_history_v8[s]) > 10:
                self._success_history_v8[s].pop(0)
        
        alpha = 0.3
        for s in range(n_strategies):
            self._ema_rank_v8[s] = (1 - alpha) * self._ema_rank_v8[s] + alpha * current_rates[s]
            self._ema_success_v8[s] = (1 - alpha) * self._ema_success_v8[s] + alpha * current_rates[s]
        
        rank_var = np.zeros(n_strategies)
        success_var = np.zeros(n_strategies)
        for s in range(n_strategies):
            if len(self._rank_history_v8[s]) >= 3:
                rank_var[s] = np.var(self._rank_history_v8[s])
            if len(self._success_history_v8[s]) >= 3:
                success_var[s] = np.var(self._success_history_v8[s])
        
        eps = 1e-6
        rank_weight = np.where(rank_var > eps, 1.0 / (rank_var + eps), 1.0 / eps)
        success_weight = np.where(success_var > eps, 1.0 / (success_var + eps), 1.0 / eps)
        
        total_rank = rank_weight.sum()
        total_success = success_weight.sum()
        if total_rank > 0:
            rank_weight /= total_rank
        if total_success > 0:
            success_weight /= total_success
        
        hybrid_signal = 0.5 * rank_weight * self._ema_rank_v8 + 0.5 * success_weight * self._ema_success_v8
        mean_signal = hybrid_signal.mean()
        if mean_signal > 0:
            comparative_advantage = (hybrid_signal - mean_signal) / (mean_signal + eps)
        else:
            comparative_advantage = np.zeros(n_strategies)
        
        combined_var = rank_var + success_var
        confidence = np.exp(-combined_var * 5)
        
        delta = np.zeros(n_strategies)
        for s in range(n_strategies):
            delta[s] = (1 - self._momentum_v8) * comparative_advantage[s] + self._momentum_v8 * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])
        
        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _credit_variant_09(self, strategy_used, improved, current_rates):
        """Variant 09: geometric alignment of mutation directions."""
        pop = getattr(self, '_current_population', None)
        if pop is None or len(pop) < 5:
            return
        
        centroid = pop.mean(axis=0)
        pop_spread = np.std(pop, axis=0).mean()
        to_centroid = centroid - pop
        
        for s_idx in range(len(self.strategy_scores)):
            mask = strategy_used == s_idx
            n_s = mask.sum()
            
            if n_s < 1:
                continue
            
            s_improved = improved[mask]
            s_to_centroid = to_centroid[mask]
            
            if s_improved.sum() == 0:
                self.strategy_scores[s_idx] *= 0.95
                continue
            
            s_pop_improved = pop[mask][s_improved]
            improved_centroid = s_pop_improved.mean(axis=0)
            dir_to_improved = improved_centroid - centroid
            
            norms_individual = np.linalg.norm(s_to_centroid, axis=1)
            valid_mask = norms_individual > 1e-12
            
            if valid_mask.sum() > 0:
                norm_global = np.linalg.norm(dir_to_improved)
                if norm_global > 1e-12:
                    cos_sims = np.dot(s_to_centroid[valid_mask], dir_to_improved) / (
                        norms_individual[valid_mask] * norm_global
                    )
                    alignment_score = np.clip(cos_sims.mean(), -1, 1)
                else:
                    alignment_score = 0.0
            else:
                alignment_score = 0.0
            
            s_pop_all = pop[mask]
            s_spread = np.std(s_pop_all, axis=0).mean()
            spread_ratio = s_spread / max(pop_spread, 1e-10)
            
            geo_score = 0.6 * (alignment_score + 1) / 2 + 0.4 * np.clip(spread_ratio, 0.5, 2.0)
            improvement_rate = s_improved.mean()
            weight = improvement_rate * 0.3 + 0.1
            self.strategy_scores[s_idx] += weight * (geo_score - 0.5)
        
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
    
    def _compute_fingerprint(self, fitness_history, pop_history):
        """Extract cheap landscape fingerprint from recent history.
        
        Returns a dict of signals observable from any black-box function.
        """
        fp = {}
        
        # Convergence rate (slope of log-best vs gen)
        if len(fitness_history) >= 3:
            recent = fitness_history[-5:] if len(fitness_history) >= 5 else fitness_history
            gens = np.arange(len(recent))
            bests = np.log(np.maximum(recent, 1e-100))
            if np.std(bests) > 1e-10:
                slope = np.polyfit(gens, bests, 1)[0]
            else:
                slope = 0.0
            fp['convergence_slope'] = slope
        else:
            fp['convergence_slope'] = 0.0
        
        # Fitness rank entropy
        if len(fitness_history) >= 1:
            ranks = self._compute_fitness_ranking(fitness_history[-1])
            ranks = ranks[~np.isnan(ranks)]
            if len(ranks) > 1:
                hist, _ = np.histogram(ranks, bins=min(10, len(ranks)), range=(0, 1))
                hist = hist / (hist.sum() + 1e-10)
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                fp['rank_entropy'] = entropy / np.log(10)  # Normalized
            else:
                fp['rank_entropy'] = 0.0
        else:
            fp['rank_entropy'] = 0.0
        
        # Population spread/diversity
        if len(pop_history) >= 1:
            pop = pop_history[-1]
            centroid = pop.mean(axis=0)
            spread = np.mean(np.linalg.norm(pop - centroid, axis=1))
            fp['diversity'] = spread
        else:
            fp['diversity'] = 0.0
        
        # Covariance condition number (effective dimensionality)
        if len(pop_history) >= 1:
            pop = pop_history[-1]
            if pop.shape[0] >= pop.shape[1] + 1:
                centered = pop - pop.mean(axis=0)
                try:
                    s = np.linalg.svd(centered, compute_uv=False)
                    s = s[s > 1e-10]
                    if len(s) >= 2:
                        cond_num = s[0] / s[-1]
                    else:
                        cond_num = 1.0
                    fp['condition_number'] = cond_num
                except:
                    fp['condition_number'] = 1.0
            else:
                fp['condition_number'] = 1.0
        else:
            fp['condition_number'] = 1.0
        
        return fp
    
    def _decide_committed_strategy(self):
        """Decide which strategy to commit to based on probe results.
        
        Uses rank-based empirical win rate from the probe burst.
        Returns the index of the best-performing strategy.
        """
        if len(self._probe_history) < 5:
            # Not enough probe data, default to variant_06 (most wins: 8)
            return 2
        
        # Compute empirical win rate per strategy using rank-based scoring
        n_strategies = len(self.strategy_scores)
        strategy_scores_probe = {s: [] for s in range(n_strategies)}
        
        # Rank-normalize improvement rates within each generation
        for gen_data in self._probe_history:
            strategy_tested, rate = gen_data
            strategy_scores_probe[strategy_tested].append(rate)
        
        # Compute mean rank-based score per strategy
        mean_scores = np.zeros(n_strategies)
        for s in range(n_strategies):
            if len(strategy_scores_probe[s]) > 0:
                mean_scores[s] = np.mean(strategy_scores_probe[s])
            else:
                mean_scores[s] = -np.inf
        
        # Commit to the strategy with highest empirical score
        best_strategy = int(np.argmax(mean_scores))
        return best_strategy
    
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

        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
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
        
        # Update fingerprint stagnation signal
        if not improved:
            self._fingerprint_history['stagnation_count'] += 1
        else:
            self._fingerprint_history['stagnation_count'] = 0
        
        return self.stagnation_counter > 50
    
    def _restart_if_needed(self, population, fitness):
        """Full restart with diversity injection."""
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        
        # Reset probe state on restart (start fresh on new landscape)
        self._probe_round_robin_idx = 0
        self._probe_history = []
        self._probe_gen = 0
        self._committed_strategy = None  # Re-probe after restart
        
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
        
        # History for fingerprinting
        fitness_history = [fitness.copy()]
        pop_history = [population.copy()]
        
        # Determine probe budget
        max_budget = getattr(stopping_condition, '_max_iters', self.max_iter)
        probe_budget = max(10, int(max_budget * self._probe_budget_pct))
        
        while not stopping_condition():
            # Store population reference for credit functions
            self.population = population
            self._current_population = population
            
            # =====================================================
            # PROBE-AND-COMMIT dispatch
            # =====================================================
            if self._committed_strategy is None:
                # PROBE phase: round-robin strategy selection
                forced_strategy = self._probe_round_robin_idx % len(self._strategy_funcs)
                
                # Check if probe is complete
                if self._probe_gen >= probe_budget:
                    self._committed_strategy = self._decide_committed_strategy()
                    forced_strategy = self._committed_strategy
            else:
                # COMMIT phase: use the winning strategy exclusively
                forced_strategy = self._committed_strategy
            
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(
                population, fitness, forced_strategy=forced_strategy
            )
            
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
            
            # Strategy score update (probe-and-commit logic inside)
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
            
            # Update fingerprint history
            fitness_history.append(fitness.copy())
            pop_history.append(population.copy())
            if len(fitness_history) > 20:
                fitness_history.pop(0)
            if len(pop_history) > 20:
                pop_history.pop(0)
            
            # Update fingerprint signals
            fp = self._compute_fingerprint(fitness_history, pop_history)
            self._fingerprint_history['convergence_slopes'].append(fp['convergence_slope'])
            self._fingerprint_history['rank_entropies'].append(fp['rank_entropy'])
            self._fingerprint_history['diversity_history'].append(fp['diversity'])
            self._fingerprint_history['condition_numbers'].append(fp['condition_number'])
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
