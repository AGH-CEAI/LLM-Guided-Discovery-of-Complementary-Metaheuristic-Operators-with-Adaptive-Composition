```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: PROBE-AND-COMMIT (mechanism #4 contextual bandit variant)
    
    The per-task winner-gap table shows 2 BLEND-HOSTILE tasks (Task 10: 12.4× gap,
    Task 12: 8.8× gap to runner-up). Linear blending CANNOT preserve these advantages
    (a 50/50 blend of 1 and 100 is 50.5, not 1). PROBE-AND-COMMIT is chosen because:
    1. Different tasks have different winners; no single strategy dominates everywhere.
    2. Blend-hostile gaps >=10× make ensemble blending mathematically invalid.
    3. A cheap fingerprint of the landscape (convergence rate, diversity, effective rank)
       can be computed from any black-box func without oracle knowledge.
    4. After committing, we keep epsilon-greedy exploration to handle non-stationarity.
    
    The algorithm probes each strategy on a small sub-population (~5% of budget),
    computes landscape fingerprints, commits to the winner, then runs to completion.
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
        # Candidates: the 4 winning update strategies from benchmark table
        self._probe_update_funcs = [
            self._update_strategy_scores_original,   # original.py (3 wins)
            self._update_strategy_scores_catF_t04,   # variant_06 (8 wins) - TEMPORAL
            self._update_strategy_scores_catH_t12,   # variant_08 (5 wins) - HYBRID RANK
            self._update_strategy_scores_catA_t10,   # variant_01 (1 win)  - GEOMETRIC DISP
            self._update_strategy_scores_catA_t10_v2, # variant_09 (2 wins) - GEOMETRIC ALIGN
        ]
        self._probe_names = [
            'original', 'temporal_EMA', 'hybrid_rank', 'geo_displacement', 'geo_alignment'
        ]
        
        # Probe results
        self._committed_strategy = None  # index into _probe_update_funcs
        self._probe_results = None  # dict with fingerprint + empirical win rate
        self._probe_budget_frac = 0.06  # 6% of total budget for probing
        
        # Epsilon-greedy commit: small probability of revisiting other strategies
        self._epsilon_commit = 0.08
        
        # --- Fingerprint tracking (for probe phase) ---
        self._fingerprint_history = []
        
    def _compute_landscape_fingerprint(self, fitness_history, pop_history, generation):
        """Compute cheap landscape fingerprint from observable signals.
        
        Returns dict with:
        - conv_rate: convergence rate (slope of log-best vs gen)
        - diversity_ratio: current diversity / initial diversity
        - eff_rank: effective rank from population covariance (0-1)
        - stagnation_frac: fraction of recent gens with no improvement
        - improv_rate: recent improvement rate
        """
        fp = {}
        
        # Convergence rate: slope of log(best_fitness) vs generation
        if len(fitness_history) >= 3:
            gens = np.arange(len(fitness_history))
            # Use log of positive fitness, handle zeros/negatives
            log_fitness = np.log(np.maximum(fitness_history, 1e-15))
            # Linear regression slope
            if np.std(log_fitness) > 0:
                cov = np.cov(gens, log_fitness)[0, 1]
                var = np.var(gens)
                fp['conv_rate'] = cov / var if var > 1e-10 else 0.0
            else:
                fp['conv_rate'] = 0.0
        else:
            fp['conv_rate'] = 0.0
        
        # Diversity ratio
        if pop_history is not None and len(pop_history) >= 2:
            initial_div = np.mean(np.std(pop_history[0], axis=0)) if len(pop_history[0]) > 1 else 1.0
            current_div = np.mean(np.std(pop_history[-1], axis=0)) if len(pop_history[-1]) > 1 else 1.0
            fp['diversity_ratio'] = current_div / (initial_div + 1e-10)
        else:
            fp['diversity_ratio'] = 1.0
        
        # Effective rank from population covariance
        if pop_history is not None and len(pop_history) >= 1:
            pop = pop_history[-1]
            if len(pop) >= 4 and self.dim >= 2:
                centered = pop - pop.mean(axis=0)
                try:
                    _, s, _ = np.linalg.svd(centered, full_matrices=False)
                    s_norm = s / s.sum()
                    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
                    eff_rank = np.exp(entropy) / min(len(pop), self.dim)
                    fp['eff_rank'] = eff_rank
                except:
                    fp['eff_rank'] = 1.0
            else:
                fp['eff_rank'] = 1.0
        else:
            fp['eff_rank'] = 1.0
        
        # Stagnation fraction (recent generations without improvement)
        if len(fitness_history) >= 3:
            recent = fitness_history[-3:]
            n_improving = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1] - 1e-10)
            fp['stagnation_frac'] = 1.0 - n_improving / max(len(recent) - 1, 1)
        else:
            fp['stagnation_frac'] = 0.0
        
        # Recent improvement rate (if we have history)
        if len(fitness_history) >= 2:
            fp['improv_rate'] = max(0.0, (fitness_history[-2] - fitness_history[-1]) / 
                                    (abs(fitness_history[-2]) + 1e-10))
        else:
            fp['improv_rate'] = 0.0
        
        return fp
    
    def _run_probe_phase(self, func, total_budget):
        """PROBE phase: test each update strategy on a small sub-population.
        
        Returns: (winner_idx, fingerprint_dict)
        """
        probe_budget = max(5, int(total_budget * self._probe_budget_frac))
        sub_NP = max(10, self.NP // 3)  # Small sub-population for speed
        
        # Initialize sub-population
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (sub_NP, self.dim))
        pop += np.random.randn(sub_NP, self.dim) * 5.0
        pop = np.clip(pop, low, high)
        
        fitness = func(pop)
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        
        # Storage for probe results
        # Each strategy: list of (rank_improvement, success_rate) per generation
        n_strategies = len(self._probe_update_funcs)
        strategy_fitness_history = [[] for _ in range(n_strategies)]
        strategy_improved_counts = [0] * n_strategies
        strategy_total_counts = [0] * n_strategies
        
        # Round-robin probe: each strategy gets a few generations
        probe_gens_per_strategy = max(2, probe_budget // n_strategies)
        
        global_fitness_history = [best_fitness]
        pop_history = [pop.copy()]
        
        for probe_iter in range(probe_budget):
            strategy_idx = probe_iter % n_strategies
            
            # Use the current update strategy
            update_func = self._probe_update_funcs[strategy_idx]
            
            # Simple DE mutation (rand/1/bin) for probe - neutral to update strategy
            mutants = np.empty_like(pop)
            strategy_used = np.empty(sub_NP, dtype=int)
            
            for i in range(sub_NP):
                others = np.concatenate([np.arange(i), np.arange(i + 1, sub_NP)])
                r = others[np.random.choice(len(others), 3, replace=False)]
                mutants[i] = pop[r[0]] + self.F * (pop[r[1]] - pop[r[2]])
                strategy_used[i] = strategy_idx
            
            # Crossover
            CR = np.clip(self.CR + np.random.randn(sub_NP) * 0.1, 0.5, 0.98)
            mask = np.random.rand(sub_NP, self.dim) < CR[:, np.newaxis]
            mask[np.arange(sub_NP), np.random.randint(0, self.dim, sub_NP)] = True
            trials = np.clip(np.where(mask, mutants, pop), low, high)
            
            # Evaluate
            trial_fitness = func(trials)
            
            # Selection
            improved = trial_fitness < fitness
            new_fitness = np.where(improved, trial_fitness, fitness)
            new_pop = np.where(improved[:, np.newaxis], trials, pop)
            
            # Track results for this strategy
            strategy_improved_counts[strategy_idx] += improved.sum()
            strategy_total_counts[strategy_idx] += sub_NP
            
            # Rank-based improvement: compare to population median
            median_fit = np.median(fitness)
            rank_improvement = np.mean((median_fit - new_fitness) / (abs(median_fit) + 1e-10))
            strategy_fitness_history[strategy_idx].append(rank_improvement)
            
            pop = new_pop
            fitness = new_fitness
            
            current_best = np.min(fitness)
            global_fitness_history.append(current_best)
            pop_history.append(pop.copy())
            
            if current_best < best_fitness:
                best_fitness = current_best
        
        # Compute empirical win rate for each strategy (rank-based, not raw fitness)
        # Normalize: how often did this strategy produce top-tier improvements?
        all_improvements = []
        for s in range(n_strategies):
            for imp in strategy_fitness_history[s]:
                all_improvements.append((s, imp))
        
        # Rank normalize across all improvements
        if len(all_improvements) > 0:
            sorted_improvements = sorted(all_improvements, key=lambda x: x[1], reverse=True)
            n = len(sorted_improvements)
            rank_scores = {}
            for rank, (s, _) in enumerate(sorted_improvements):
                rank_scores[s] = rank_scores.get(s, 0.0) + (n - rank) / n
            
            # Normalize rank scores to [0, 1]
            max_score = max(rank_scores.values()) if rank_scores else 1.0
            win_rates = {s: rank_scores.get(s, 0.0) / max_score for s in range(n_strategies)}
        else:
            win_rates = {s: 1.0 / n_strategies for s in range(n_strategies)}
        
        # Compute landscape fingerprint
        fingerprint = self._compute_landscape_fingerprint(
            global_fitness_history, pop_history, probe_budget
        )
        fingerprint['win_rates'] = win_rates
        fingerprint['probe_budget_used'] = probe_budget
        
        # Select winner: highest rank-normalized win rate
        winner_idx = max(range(n_strategies), key=lambda s: win_rates.get(s, 0.0))
        
        self._probe_results = fingerprint
        return winner_idx, fingerprint
    
    # =====================================================================
    # The 5 candidate _update_strategy_scores implementations from winners
    # =====================================================================
    
    def _update_strategy_scores_original(self, strategy_used, improved):
        """Baseline: count-based credit (from original.py, 3 wins)."""
        for s in range(len(self.strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        self.strategy_scores *= 0.99
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
    def _update_strategy_scores_catF_t04(self, strategy_used, improved):
        """Temporal credit with EMA, momentum, and trend detection.
        From variant_06_catF_t04_idea_0.py (8 wins - most dominant)."""
        n_strategies = len(self.strategy_scores)
        
        if not hasattr(self, '_ema_rates'):
            self._ema_rates = np.zeros(n_strategies)
            self._prev_ema_rates = np.zeros(n_strategies)
            self._strategy_momentum = np.zeros(n_strategies)
            self._history_buffer = [np.zeros(n_strategies) for _ in range(5)]
            self._history_idx = 0
        
        current_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
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
        self._strategy_momentum = (
            beta * self._strategy_momentum +
            (1 - beta) * (self._ema_rates + gamma * np.sign(trends) * np.sqrt(np.abs(trends)))
        )
        
        recent_variance = np.var(self._history_buffer, axis=0) + 1e-10
        adaptive_lr = 0.15 / (1.0 + 5.0 * recent_variance)
        score_delta = adaptive_lr * self._ema_rates + 0.05 * self._strategy_momentum
        
        for s in range(n_strategies):
            recent_trend = sum(
                self._history_buffer[(self._history_idx - i) % len(self._history_buffer)][s]
                for i in range(1, 4)
            )
            if recent_trend < 0.02:
                score_delta[s] -= 0.08
        
        self.strategy_scores += score_delta
        self.strategy_scores -= self.strategy_scores.mean()
        self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
    
    def _update_strategy_scores_catH_t12(self, strategy_used, improved):
        """Hybrid rank improvement + success rate, weighted by signal confidence.
        From variant_08_catH_t12_idea_0.py (5 wins)."""
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
                success_rate[s] = improved[mask].sum() / mask.sum()
                rank_improvement[s] = success_rate[s]  # Same for binary improvement
        
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
    
    def _update_strategy_scores_catA_t10(self, strategy_used, improved):
        """Credit assignment using geometric displacement: centroid shift and hull volume change.
        From variant_01_catA_t10_idea_0.py (1 win)."""
        if not hasattr(self, 'population') or self.population is None:
            return
        
        if not hasattr(self, '_prev_centroid'):
            self._prev_centroid = None
            self._prev_hull_vol = None
        
        centroid = self.population.mean(axis=0)
        pop_range = self.population.max(axis=0) - self.population.min(axis=0) + 1e-10
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
            
            improved_centroid = self.population[improved_idx].mean(axis=0)
            displacement = np.linalg.norm(improved_centroid - centroid)
            norm_displacement = displacement / pop_spread
            credit = norm_displacement * np.sqrt(n_improved) * max(hull_change_ratio, 0.5)
            
            momentum = 0.1
            self.strategy_scores[s] = (1 - momentum) * self.strategy_scores[s] + momentum * credit
        
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)
        self._prev_centroid = centroid
        self._prev_hull_vol = hull_proxy
    
    def _update_strategy_scores_catA_t10_v2(self, strategy_used, improved):
        """Credit assignment based on geometric alignment of mutation directions.
        From variant_09_catA_t10_idea_0.py (2 wins)."""
        if len(improved) == 0 or self.NP < 2:
            return
        
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
    
    # =====================================================================
    # Active update dispatcher: uses committed strategy or epsilon-greedy
    # =====================================================================
    
    def _update_strategy_scores_active(self, strategy_used, improved):
        """Dispatch to the committed strategy, with epsilon-greedy exploration.
        
        This preserves per-task winner advantages because:
        - The committed strategy gets weight ~1.0 (minus epsilon)
        - No blending means the winner's 12× advantage on Task 10 is preserved
        - Epsilon allows recovery if the landscape changes mid-run
        """
        if self._committed_strategy is None:
            # Fallback: use original if not yet probed
            self._update_strategy_scores_original(strategy_used, improved)
            return
        
        # Epsilon-greedy: small chance of trying a different strategy
        if np.random.rand() < self._epsilon_commit:
            # Pick uniformly from non-committed strategies
            alternatives = [i for i in range(len(self._probe_update_funcs)) 
                          if i != self._committed_strategy]
            if alternatives:
                alt_idx = np.random.choice(alternatives)
                self._probe_update_funcs[alt_idx](strategy_used, improved)
                return
        
        # Commit to the winner
        self._probe_update_funcs[self._committed_strategy](strategy_used, improved)
    
    # =====================================================================
    # Original DE implementation (unchanged methods)
    # =====================================================================
    
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
        
        return new_pop, best_fitness, best_solution
    
    def __call__(self, func, stopping_condition):
        # =================================================================
        # PHASE A: PROBE (6% of budget)
        # =================================================================
        # Estimate total budget from stopping_condition behavior
        # We'll probe for a fixed small number of iterations
        probe_iterations = max(5, int(self.max_iter * self._probe_budget_frac))
        
        # Run probe phase
        self._committed_strategy, fingerprint = self._run_probe_phase(func, self.max_iter)
        
        # Reset state for commit phase
        self.archive = []
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.F = 0.6
        self.CR = 0.85
        self.strategy_scores = np.ones(len(self.strategy_names))
        
        # Reset temporal state for whichever strategy we committed to
        for attr in ['_ema_rates', '_prev_ema_rates', '_strategy_momentum', 
                     '_history_buffer', '_ema_rank_improvement', '_ema_success_rate',
                     '_rank_improvement_history', '_success_rate_history',
                     '_prev_centroid', '_prev_hull_vol']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # =================================================================
        # PHASE B: COMMIT (94% of budget)
        # =================================================================
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        # Store committed strategy name for logging
        committed_name = self._probe_names[self._committed_strategy] if self._committed_strategy is not None else "none"
        
        while not stopping_condition():
            # Store population reference for update strategies that need it
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
            
            # Strategy score update: use PROBE-AND-COMMIT dispatcher
            self._update_strategy_scores_active(strategy_used, improved_mask)
            
            # Archive and parameter adaptation
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.nanmean(improved_mask)
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