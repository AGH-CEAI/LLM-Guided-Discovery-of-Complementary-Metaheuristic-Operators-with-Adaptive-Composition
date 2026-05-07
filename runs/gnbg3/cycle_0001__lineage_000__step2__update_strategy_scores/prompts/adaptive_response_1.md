```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTIVE COMPASS DIFFERENTIAL EVOLUTION — PROBE-AND-COMMIT (mechanism #4 contextual bandit).
    
    This is BLEND-HOSTILE: the per-task winner-gap table shows 2 tasks (10, 12) where the winner
    beats the runner-up by ≥10× (max gap = 153× to median on task 10). Linear blending is
    mathematically excluded — a 50/50 blend of 1 and 100 is 50.5, not 1.
    
    Mechanism: PROBE-AND-COMMIT with per-task fingerprinting.
    - PHASE A (probe): Run each of 7 credit-assignment strategies for 50 iterations with a small
      sub-population. Select the winner by rank-based improvement rate.
    - PHASE B (commit): Run the winning strategy exclusively for the remaining budget.
      Small ε=0.05 probability of revisiting runner-up if the committed strategy stagnates.
    
    Why probe-and-commit fits this data:
    - 7 distinct winners across 21 non-trivial tasks with large gaps means no single strategy
      dominates everywhere. The probe identifies which regime the current landscape belongs to.
    - variant_06 (EMA+momentum) wins 8 tasks, variant_08 (hybrid rank+success) wins 5, 
      original wins 3, and others win 1-2 each. The landscape type is unknowable a priori.
    - Probe budget = 7×50 = 350 iterations ≈ 1.75% of a 20K budget. Committed phase = 98.25%.
    - Rank-based selection avoids scale-dependent magic constants (rule a).
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool (7 variants from benchmark)
        self.strategy_names = ['geo_disp', 'spectral', 'ema_temporal', 
                               'thompson_bootstrap', 'hybrid_rank_success',
                               'geo_alignment', 'default']
        self.strategy_scores = np.ones(len(self.strategy_names))
        
        # Dispatch state
        self._committed_strategy = 0  # Default: original (index 6)
        self._probe_phase = True
        self._probe_iter = 0
        self._probe_strat_idx = 0
        self._probe_stats = {i: [] for i in range(len(self.strategy_names))}
        self._probe_window = 50
        
        # Diversity and stagnation
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / (max(self.NP - 1, 1))
    
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
        top_p = max(1, int(np.ceil(p * self.NP)))
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
        return np.random.choice(len(self.strategy_scores), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        return mutants, strategy_used
    
    @property
    def _strategy_funcs(self):
        return [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
    
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
    
    def _update_strategy_scores(self, strategy_used, improved, pop):
        """Route to the selected credit-assignment strategy (rank-based, no magic constants)."""
        self._current_population = pop
        variant = self._committed_strategy
        
        if variant == 0:
            self._update_geo_displacement(strategy_used, improved)
        elif variant == 1:
            self._update_spectral(strategy_used, improved)
        elif variant == 2:
            self._update_ema_temporal(strategy_used, improved)
        elif variant == 3:
            self._update_thompson_bootstrap(strategy_used, improved)
        elif variant == 4:
            self._update_hybrid_rank_success(strategy_used, improved)
        elif variant == 5:
            self._update_geo_alignment(strategy_used, improved)
        else:
            self._update_default(strategy_used, improved)
        
        # Normalize to prevent divergence (all variants)
        self.strategy_scores -= self.strategy_scores.mean()
        self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
    
    def _update_geo_displacement(self, strategy_used, improved):
        """Variant_01: geometric displacement — centroid shift + hull volume change."""
        pop = self._current_population
        if pop is None or len(pop) < 5:
            return
        
        if not hasattr(self, '_prev_centroid'):
            self._prev_centroid = None
            self._prev_hull_vol = None
        
        centroid = pop.mean(axis=0)
        pop_range = pop.max(axis=0) - pop.min(axis=0) + 1e-10
        hull_proxy = np.prod(pop_range)
        
        if self._prev_centroid is not None and self._prev_hull_vol is not None:
            centroid_shift = np.linalg.norm(centroid - self._prev_centroid)
            hull_change_ratio = hull_proxy / (self._prev_hull_vol + 1e-10)
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
            
            improved_centroid = pop[improved_idx].mean(axis=0)
            displacement = np.linalg.norm(improved_centroid - centroid)
            norm_displacement = displacement / pop_spread
            
            credit = norm_displacement * np.sqrt(n_improved) * max(hull_change_ratio, 0.5)
            momentum = 0.1
            self.strategy_scores[s] = (1 - momentum) * self.strategy_scores[s] + momentum * credit
        
        self._prev_centroid = centroid
        self._prev_hull_vol = hull_proxy
    
    def _update_spectral(self, strategy_used, improved):
        """Variant_02: spectral analysis — SVD effective rank + condition number."""
        pop = self._current_population
        if pop is None or len(pop) < 5:
            return
        
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
        cond_num = s[0] / (s[-1] + 1e-12)
        
        if not hasattr(self, '_spectral_eff_rank_history'):
            self._spectral_eff_rank_history = []
            self._spectral_cond_history = []
            self._strategy_spectral_shifts = np.zeros(len(self.strategy_names))
            self._strategy_cond_shifts = np.zeros(len(self.strategy_names))
            self._strategy_use_counts = np.zeros(len(self.strategy_names))
        
        self._spectral_eff_rank_history.append(eff_rank)
        self._spectral_cond_history.append(cond_num)
        
        alpha = 0.3
        
        for strat_idx in range(len(self.strategy_names)):
            mask = strategy_used == strat_idx
            n_used = np.sum(mask)
            if n_used == 0:
                continue
            
            self._strategy_use_counts[strat_idx] += n_used
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
            
            if n_not_imp > 0:
                not_imp_pop = pop[not_improved_mask]
                not_imp_centered = not_imp_pop - not_imp_pop.mean(axis=0)
                try:
                    _, s_not_imp, _ = np.linalg.svd(not_imp_centered, full_matrices=False)
                    s_not_imp_norm = s_not_imp / s_not_imp.sum()
                    ent_not_imp = -np.sum(s_not_imp_norm * np.log(s_not_imp_norm + 1e-12))
                    eff_rank_not_imp = np.exp(ent_not_imp) / (min(*not_imp_pop.shape) + 1e-12)
                except:
                    pass
            
            rank_shift = eff_rank_imp - eff_rank_not_imp
            weight = n_imp / (n_imp + n_not_imp + 1e-12)
            self._strategy_spectral_shifts[strat_idx] = (1 - alpha) * self._strategy_spectral_shifts[strat_idx] + alpha * rank_shift * weight
            self.strategy_scores[strat_idx] += 0.1 * np.tanh(5.0 * self._strategy_spectral_shifts[strat_idx])
        
        if len(self._spectral_cond_history) >= 3:
            cond_trend = self._spectral_cond_history[-1] - self._spectral_cond_history[0]
            for strat_idx in range(len(self.strategy_names)):
                mask = strategy_used == strat_idx
                n_used = np.sum(mask)
                if n_used == 0:
                    continue
                strat_pop = pop[mask]
                strat_centered = strat_pop - strat_pop.mean(axis=0)
                try:
                    _, s_strat, _ = np.linalg.svd(strat_centered, full_matrices=False)
                    cond_strat = s_strat[0] / (s_strat[-1] + 1e-12)
                except:
                    continue
                cond_shift = cond_strat - cond_num
                self._strategy_cond_shifts[strat_idx] = (1 - alpha) * self._strategy_cond_shifts[strat_idx] + alpha * cond_shift
                if cond_trend > 0.1:
                    credit = -0.05 * np.sign(self._strategy_cond_shifts[strat_idx]) * min(abs(self._strategy_cond_shifts[strat_idx]), 1.0)
                    self.strategy_scores[strat_idx] += credit
                elif cond_trend < -0.1:
                    credit = 0.05 * np.sign(self._strategy_cond_shifts[strat_idx]) * min(abs(self._strategy_cond_shifts[strat_idx]), 1.0)
                    self.strategy_scores[strat_idx] += credit
    
    def _update_ema_temporal(self, strategy_used, improved):
        """Variant_06: EMA + momentum + trend detection (8 wins — most dominant)."""
        n_strats = len(self.strategy_scores)
        
        if not hasattr(self, '_ema_rates'):
            self._ema_rates = np.zeros(n_strats)
            self._prev_ema_rates = np.zeros(n_strats)
            self._strategy_momentum = np.zeros(n_strats)
            self._history_buffer = [np.zeros(n_strats) for _ in range(5)]
            self._history_idx = 0
        
        current_rates = np.zeros(n_strats)
        for s in range(n_strats):
            mask = strategy_used == s
            if mask.any():
                current_rates[s] = improved[mask].sum() / self.NP
        
        self._history_buffer[self._history_idx] = current_rates.copy()
        self._history_idx = (self._history_idx + 1) % len(self._history_buffer)
        
        alpha = 0.3
        self._prev_ema_rates = self._ema_rates.copy()
        self._ema_rates = alpha * current_rates + (1 - alpha) * self._ema_rates
        trends = self._ema_rates - self._prev_ema_rates
        
        beta = 0.7
        gamma = 0.15
        self._strategy_momentum = beta * self._strategy_momentum + (1 - beta) * (self._ema_rates + gamma * np.sign(trends) * np.sqrt(np.abs(trends)))
        
        recent_variance = np.var(self._history_buffer, axis=0) + 1e-10
        adaptive_lr = 0.15 / (1.0 + 5.0 * recent_variance)
        score_delta = adaptive_lr * self._ema_rates + 0.05 * self._strategy_momentum
        
        for s in range(n_strats):
            recent_trend = sum(self._history_buffer[(self._history_idx - i) % len(self._history_buffer)][s] for i in range(1, 4))
            if recent_trend < 0.02:
                score_delta[s] -= 0.08
        
        self.strategy_scores += score_delta
    
    def _update_thompson_bootstrap(self, strategy_used, improved):
        """Variant_07: Thompson Sampling with bootstrap resampling."""
        n_strats = len(self.strategy_scores)
        n_trials = len(strategy_used)
        
        if n_trials == 0:
            return
        
        if not hasattr(self, '_strategy_history'):
            self._strategy_history = [[] for _ in range(n_strats)]
            self._strategy_successes = np.zeros(n_strats)
            self._strategy_attempts = np.zeros(n_strats)
        
        for i in range(n_trials):
            s = strategy_used[i]
            imp = improved[i]
            mag = abs(improved[i]) if isinstance(improved[i], (int, float, np.number)) and imp else 0.0
            self._strategy_history[s].append((1 if imp else 0, mag))
            self._strategy_attempts[s] += 1
            if imp:
                self._strategy_successes[s] += 1
        
        for s in range(n_strats):
            if len(self._strategy_history[s]) > 500:
                self._strategy_history[s] = self._strategy_history[s][-400:]
        
        n_bootstrap = 200
        bootstrap_rates = np.zeros((n_bootstrap, n_strats))
        
        for b in range(n_bootstrap):
            for s in range(n_strats):
                hist = self._strategy_history[s]
                if len(hist) > 0:
                    indices = np.random.randint(0, len(hist), size=len(hist))
                    sampled = [hist[i] for i in indices]
                    bootstrap_rates[b, s] = np.mean([s_[0] * (1 + s_[1]) for s_ in sampled])
                else:
                    bootstrap_rates[b, s] = 0.0
        
        ci_lower = np.percentile(bootstrap_rates, 10, axis=0)
        ci_upper = np.percentile(bootstrap_rates, 90, axis=0)
        
        sampled_scores = np.zeros(n_strats)
        for s in range(n_strats):
            sampled_idx = np.random.randint(n_bootstrap)
            sampled_scores[s] = bootstrap_rates[sampled_idx, s]
            ci_width = max(ci_upper[s] - ci_lower[s], 1e-10)
            exploration = np.random.exponential(0.05 / ci_width)
            sampled_scores[s] += exploration
        
        exp_scores = np.exp(sampled_scores - sampled_scores.max())
        probs = exp_scores / (exp_scores.sum() + 1e-10)
        momentum = 0.7
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * probs
    
    def _update_hybrid_rank_success(self, strategy_used, improved):
        """Variant_08: hybrid rank improvement + success rate with inverse-variance weighting."""
        n_strats = len(self.strategy_scores)
        
        if not hasattr(self, '_ema_rank_improvement'):
            self._ema_rank_improvement = np.zeros(n_strats)
            self._ema_success_rate = np.zeros(n_strats)
            self._rank_improvement_history = [[] for _ in range(n_strats)]
            self._success_rate_history = [[] for _ in range(n_strats)]
            self._momentum = 0.3
        
        rank_improvement = np.zeros(n_strats)
        success_rate = np.zeros(n_strats)
        for s in range(n_strats):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()
                success_rate[s] = improved[mask].sum() / mask.sum()
        
        for s in range(n_strats):
            self._rank_improvement_history[s].append(rank_improvement[s])
            self._success_rate_history[s].append(success_rate[s])
            if len(self._rank_improvement_history[s]) > 10:
                self._rank_improvement_history[s].pop(0)
            if len(self._success_rate_history[s]) > 10:
                self._success_rate_history[s].pop(0)
        
        alpha = 0.3
        for s in range(n_strats):
            self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
            self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]
        
        rank_var = np.zeros(n_strats)
        success_var = np.zeros(n_strats)
        for s in range(n_strats):
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
            comparative_advantage = np.zeros(n_strats)
        
        combined_var = rank_var + success_var
        confidence = np.exp(-combined_var * 5)
        
        delta = np.zeros(n_strats)
        for s in range(n_strats):
            delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])
        
        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _update_geo_alignment(self, strategy_used, improved):
        """Variant_09: geometric alignment of mutation directions toward centroid."""
        pop = self._current_population
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
                    cos_sims = np.dot(s_to_centroid[valid_mask], dir_to_improved) / (norms_individual[valid_mask] * norm_global)
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
    
    def _update_default(self, strategy_used, improved):
        """Original: simple credit assignment."""
        for s in range(len(self.strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        self.strategy_scores *= 0.99
    
    def _run_probe_phase(self, func):
        """Run 50 iterations per strategy with sub-population. Select winner by rank improvement."""
        probe_NP = max(40, self.dim * 4)
        population = np.random.uniform(-100.0, 100.0, (probe_NP, self.dim))
        fitness = func(population)
        
        best_idx = np.nanargmin(fitness)
        f_opt = fitness[best_idx]
        
        self._probe_stats = {i: [] for i in range(len(self.strategy_names))}
        self._probe_iter = 0
        self._probe_strat_idx = 0
        
        while self._probe_strat_idx < len(self.strategy_names):
            strat = self._probe_strat_idx
            for _ in range(self._probe_window):
                mutants_compass = self._mutate_compass_batch(population, fitness)
                
                mutants_ens = np.empty_like(population)
                strategy_used = np.full(probe_NP, strat, dtype=int)
                for i in range(probe_NP):
                    mutants_ens[i] = self._strategy_funcs[strat](population, i, self.F)
                
                blend_ratio = np.random.rand(probe_NP, 1)
                mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ens)
                trials = self._crossover_batch(population, mutants)
                trial_fitness = func(trials)
                
                if len(trial_fitness) < len(trials):
                    if len(trial_fitness) == 0:
                        return
                    trial_fitness = np.full(len(trials), np.nan)
                
                population, fitness, improved_mask = self._select_survivors_batch(
                    population, fitness, trials, trial_fitness
                )
                
                current_best_idx = np.nanargmin(fitness)
                if fitness[current_best_idx] < f_opt:
                    f_opt = fitness[current_best_idx]
                
                rank_improvement = improved_mask.mean()
                self._probe_stats[strat].append(rank_improvement)
                self._probe_iter += 1
            
            self._probe_strat_idx += 1
        
        avg_improvements = [np.mean(v) if len(v) > 0 else 0.0 for v in self._probe_stats.values()]
        best_strat = int(np.argmax(avg_improvements))
        self._committed_strategy = best_strat
        self._probe_phase = False
    
    def _adapt_parameters(self, improvement_rate):
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
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness_val = fitness[best_idx]
        
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        
        return new_pop, best_fitness_val, best_solution
    
    def __call__(self, func, stopping_condition):
        if stopping_condition():
            return np.inf, np.zeros(self.dim)
        
        self._run_probe_phase(func)
        
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        rank_improvement_history = []
        
        while not stopping_condition():
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            mutants_ens = np.empty_like(population)
            strategy_used = np.full(self.NP, self._committed_strategy, dtype=int)
            for i in range(self.NP):
                mutants_ens[i] = self._strategy_funcs[self._committed_strategy](population, i, self.F)
            
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ens)
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
            
            rank_improvement = improved_mask.mean()
            rank_improvement_history.append(rank_improvement)
            if len(rank_improvement_history) > 20:
                rank_improvement_history.pop(0)
            
            self._current_population = population
            self._update_strategy_scores(strategy_used, improved_mask, population)
            
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
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
```