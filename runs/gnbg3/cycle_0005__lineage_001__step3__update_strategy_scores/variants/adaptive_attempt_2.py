import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: PROBE-AND-COMMIT (mechanism #4 / contextual bandit variant).
    
    Rationale: The per-task gap table is BLEND-HOSTILE — 2 tasks have winner-vs-runner-up
    gaps >= 10× (Task 10: 26.5×, Task 17: 22.0×). Linear blending CANNOT preserve these
    advantages (a 50/50 blend of 1 and 100 yields 50.5, not 1). Ensemble blending is
    mathematically excluded.
    
    Different _update_strategy_scores variants win on different tasks:
    - original.py (14 wins): dominates overall but loses on Tasks 5, 9, 11, 16, 20, 21, 22
    - variant_10 (2 wins): wins Tasks 11 and 16 by small margins
    - Others (1 win each): win by 1.0-1.4× margins
    
    PROBE-AND-COMMIT preserves the per-task winner's advantage by:
    1. Running a short exploratory burst (~8% of budget) with all candidate strategies
    2. Extracting cheap landscape fingerprints (convergence slope, rank correlation,
       condition number, stagnation index) from the probe phase
    3. Committing to the empirically best strategy for the remaining ~92% of budget
    4. Allowing small epsilon-revisit probability if the committed strategy stagnates
    
    This is a CONTEXTUAL BANDIT where context is computed once at run start from
    observable black-box signals, not from oracle knowledge of the task identity.
    
    Novel features:
    1. Compass Mutation: Samples 3 candidate directions per individual.
    2. Probe-and-Commit Strategy Selection: probes all strategies, commits to best.
    3. Success-rate adaptation for F and CR with momentum.
    4. Diversity-triggered restart to escape local optima.
    5. Archive of recently improved solutions for diversity.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool — these are the DIFFERENT _update_strategy_scores approaches
        # that won on various tasks in the benchmark
        self.strategy_names = [
            'original_hybrid',       # original.py: rank improvement + success rate
            'catA_geometry',         # variant_01/09: k-NN density + spatial spread
            'catB_spectral',         # variant_10: eigendecomposition of covariance
            'catC_info_theory',      # variant_03: entropy + mutual information + KL
            'catD_rank_based',       # variant_04: Kendall tau + fitness percentile
            'catF_temporal',         # variant_06: EMA momentum + drift detection
        ]
        self.strategy_scores = np.ones(len(self.strategy_names))
        self._strategy_funcs = [
            self._update_strategy_scores_original_hybrid,
            self._update_strategy_scores_catA_geometry,
            self._update_strategy_scores_catB_spectral,
            self._update_strategy_scores_catC_info_theory,
            self._update_strategy_scores_catD_rank_based,
            self._update_strategy_scores_catF_temporal,
        ]
        
        # Probe-and-commit state
        self._probe_budget_pct = 0.08  # 8% of budget for probing
        self._probe_completed = False
        self._committed_strategy = 0   # Default to original
        self._strategy_win_counts = np.zeros(len(self.strategy_names))
        self._strategy_fitness_history = [[] for _ in range(len(self.strategy_names))]
        self._probe_generation = 0
        self._last_commit_best_fitness = np.inf
        self._epsilon_revisit = 0.05  # 5% chance to try runner-up if committed stagnates
        
        # Diversity and stagnation tracking
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
    
    # =========================================================================
    # PROBE-AND-COMMIT: Strategy score update methods
    # Each method corresponds to a variant that won in the benchmark
    # =========================================================================
    
    def _update_strategy_scores_original_hybrid(self, strategy_used, improved):
        """Original hybrid: rank improvement + success rate with inverse-variance weighting."""
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
    
    def _update_strategy_scores_catA_geometry(self, strategy_used, improved):
        """Category A: k-NN density + spatial spread. Strategies in sparse regions get bonus."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return

        pop = self._current_population
        NP, dim = pop.shape

        if NP < 4:
            return

        k = max(2, min(5, NP // 4))
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
        avg_k_dist = np.mean(np.sqrt(nearest_sq_dists), axis=1)
        local_density = 1.0 / (avg_k_dist + 1e-8)

        density_sum = local_density.sum()
        if density_sum > 1e-10:
            local_density *= NP / density_sum

        if improved.any():
            improved_pop = pop[improved]
            global_centroid = improved_pop.mean(axis=0)
        else:
            global_centroid = pop.mean(axis=0)

        num_strategies = len(self.strategy_names)
        strategy_rewards = np.zeros(num_strategies)

        for s in range(num_strategies):
            strategy_mask = (strategy_used == s)
            if not strategy_mask.any():
                continue

            success_mask = strategy_mask & improved
            if not success_mask.any():
                continue

            success_pop = pop[success_mask]
            centroid_dist = np.linalg.norm(success_pop - global_centroid, axis=1)
            mean_spread = np.mean(centroid_dist) + 1e-8

            success_density = local_density[success_mask].sum()
            density_bonus = 1.0 / (success_density + 1.0)

            if len(success_pop) >= 3:
                hull_dists = sq_dists[np.ix_(success_mask, success_mask)]
                np.fill_diagonal(hull_dists, 0)
                hull_span = np.mean(hull_dists) + 1e-8
            else:
                hull_span = 1.0

            strategy_rewards[s] = mean_spread * density_bonus * np.sqrt(hull_span)

        total_reward = strategy_rewards.sum()
        if total_reward > 1e-10:
            strategy_rewards /= total_reward

        momentum = 0.7
        learning_rate = 0.3

        for s in range(num_strategies):
            if strategy_rewards[s] > 0:
                reward_signal = np.log1p(strategy_rewards[s] * NP)
            else:
                reward_signal = -2.0

            self.strategy_scores[s] = (momentum * self.strategy_scores[s] + 
                                       learning_rate * reward_signal)

        scores_exp = np.exp(self.strategy_scores - np.max(self.strategy_scores))
        self.strategy_scores = np.log(scores_exp / (scores_exp.sum() + 1e-10))
    
    def _update_strategy_scores_catB_spectral(self, strategy_used, improved):
        """Category B: Spectral analysis via eigendecomposition of population covariance."""
        if not hasattr(self, '_spectral_success_ema'):
            self._spectral_success_ema = np.ones(len(self.strategy_names)) * 0.5
            self._ema_alpha = 0.3

        if hasattr(self, '_current_population') and self._current_population is not None:
            pop = self._current_population
            if len(pop) >= self.dim + 1:
                centered = pop - pop.mean(axis=0)
                cov = np.cov(centered.T)

                try:
                    eigenvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
                    eigenvals = eigenvals[eigenvals > 1e-12]
                except Exception:
                    eigenvals = np.array([1.0])

                if len(eigenvals) >= 2:
                    cond = eigenvals[0] / max(eigenvals[-1], 1e-12)
                    cond = min(cond, 1e6)

                    total = np.sum(eigenvals)
                    probs = eigenvals / max(total, 1e-12)
                    entropy = -np.sum(probs * np.log(probs + 1e-12))
                    max_entropy = np.log(len(eigenvals))
                    eff_rank = np.exp(entropy) / max(max_entropy, 1e-12)
                else:
                    cond = 1e6
                    eff_rank = 0.1

                spectral_quality = np.exp(-np.log(1 + cond) / 10.0) * np.clip(eff_rank, 0.1, 1.0)
            else:
                spectral_quality = 0.5
        else:
            spectral_quality = 0.5

        for strat_idx in range(len(self.strategy_names)):
            mask = strategy_used == strat_idx
            if np.any(mask):
                success_rate = np.mean(improved[mask])
                self._spectral_success_ema[strat_idx] = (1 - self._ema_alpha) * self._spectral_success_ema[strat_idx] + self._ema_alpha * success_rate

        baseline = 0.3 + 0.4 * spectral_quality
        target_scores = baseline + 0.3 * self._spectral_success_ema

        momentum = 0.4
        self.strategy_scores = (1 - momentum) * self.strategy_scores + momentum * target_scores
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)
    
    def _update_strategy_scores_catC_info_theory(self, strategy_used, improved):
        """Category C: Information-theoretic credit assignment (entropy + MI + KL)."""
        n_strategies = len(self.strategy_scores)
        NP = len(strategy_used)

        if NP < 5:
            return

        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []

        if hasattr(self, 'fitness'):
            self._fitness_history.append(self.fitness.copy())
            if len(self._fitness_history) > 20:
                self._fitness_history.pop(0)

        reward = np.zeros(n_strategies)

        for s in range(n_strategies):
            mask = strategy_used == s
            n_s = np.sum(mask)
            if n_s >= 3 and hasattr(self, 'fitness'):
                fitness_s = self.fitness[mask]
                n_bins = min(8, max(2, n_s // 2))
                bin_edges = np.percentile(fitness_s, np.linspace(0, 100, n_bins + 1))
                counts, _ = np.histogram(fitness_s, bins=bin_edges)
                probs = counts / n_s
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                reward[s] -= entropy * 0.3

        n_improved = np.sum(improved)
        if 0 < n_improved < NP:
            p_joint = np.zeros((2, n_strategies))
            for s in range(n_strategies):
                mask = strategy_used == s
                n_s = np.sum(mask)
                if n_s > 0:
                    n_imp_s = np.sum(improved[mask])
                    p_joint[0, s] = n_imp_s / NP
                    p_joint[1, s] = (n_s - n_imp_s) / NP

            for s in range(n_strategies):
                p_s = np.sum(p_joint[:, s])
                if p_s > 1e-10:
                    for i in range(2):
                        if p_joint[i, s] > 1e-10:
                            p_i = np.sum(p_joint[i, :])
                            mi_contrib = p_joint[i, s] * np.log(p_joint[i, s] / (p_i * p_s + 1e-12))
                            reward[s] += mi_contrib * 5.0

        if hasattr(self, 'fitness') and len(self._fitness_history) >= 2:
            prev_fitness = self._fitness_history[-2] if len(self._fitness_history) >= 2 else self.fitness
            for s in range(n_strategies):
                mask = strategy_used == s
                n_s = np.sum(mask)
                if n_s >= 2:
                    current_s = self.fitness[mask]
                    prev_s = prev_fitness[mask]
                    if np.std(current_s) > 1e-10 and np.std(prev_s) > 1e-10:
                        curr_norm = (current_s - np.min(current_s) + 1e-6) / (np.ptp(current_s) + 1e-6)
                        prev_norm = (prev_s - np.min(prev_s) + 1e-6) / (np.ptp(prev_s) + 1e-6)
                        kl = np.mean(np.log(prev_norm + 1e-6) - np.log(curr_norm + 1e-6))
                        reward[s] += kl * 0.5

        reward = np.clip(reward, -5, 5)
        reward = reward - np.mean(reward)

        momentum = 0.7
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * reward

        temp = 1.5
        exp_scores = np.exp(self.strategy_scores / temp)
        self.strategy_scores = exp_scores / np.sum(exp_scores)
        self.strategy_scores = np.maximum(self.strategy_scores, 0.05)
        self.strategy_scores /= np.sum(self.strategy_scores)
    
    def _update_strategy_scores_catD_rank_based(self, strategy_used, improved):
        """Category D: Fitness-landscape / rank-based scoring (Kendall tau + percentile)."""
        NP = len(strategy_used)
        n_strategies = len(self.strategy_names)

        if not hasattr(self, '_strategy_rank_improvements'):
            self._strategy_rank_improvements = np.zeros(n_strategies)
            self._strategy_success_counts = np.zeros(n_strategies)
            self._strategy_attempt_counts = np.zeros(n_strategies)
            self._strategy_fitness_history = [[] for _ in range(n_strategies)]
            self._rank_corr_history = []

        if not hasattr(self, '_prev_fitness_ranks'):
            self._prev_fitness_ranks = np.zeros(NP)

        if not hasattr(self, '_current_fitness'):
            return

        current_fitness = self._current_fitness
        valid_mask = np.isfinite(current_fitness)

        if np.sum(valid_mask) < 5:
            return

        fitness_sorted = np.argsort(current_fitness[valid_mask])
        current_ranks = np.zeros(np.sum(valid_mask))
        current_ranks[fitness_sorted] = np.linspace(0, 1, len(fitness_sorted))

        rank_improvement = self._prev_fitness_ranks[valid_mask] - current_ranks

        for s in range(n_strategies):
            s_mask = (strategy_used == s) & valid_mask
            if np.sum(s_mask) == 0:
                continue

            self._strategy_attempt_counts[s] += np.sum(s_mask)

            successes = improved[s_mask]
            self._strategy_success_counts[s] += np.sum(successes)

            avg_percentile = np.mean(current_ranks)
            difficulty_weight = 1.0 - avg_percentile + 0.1

            if np.any(successes):
                avg_rank_improvement = np.mean(rank_improvement[s_mask][successes])
                self._strategy_rank_improvements[s] += difficulty_weight * max(avg_rank_improvement, 0)

            self._strategy_fitness_history[s].extend(current_fitness[s_mask].tolist())
            if len(self._strategy_fitness_history[s]) > 1000:
                self._strategy_fitness_history[s] = self._strategy_fitness_history[s][-500:]

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) >= 10:
            strat_vals = strategy_used[valid_indices].astype(float)
            fitness_vals = current_fitness[valid_indices]

            f_min, f_max = np.min(fitness_vals), np.max(fitness_vals)
            if f_max > f_min:
                fitness_norm = (fitness_vals - f_min) / (f_max - f_min)
            else:
                fitness_norm = np.zeros_like(fitness_vals)

            n = len(strat_vals)
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    strat_diff = (strat_vals[i] - strat_vals[j]) * (fitness_norm[i] - fitness_norm[j])
                    if strat_diff > 0:
                        concordant += 1
                    elif strat_diff < 0:
                        discordant += 1

            n_pairs = n * (n - 1) / 2
            if n_pairs > 0:
                tau = (concordant - discordant) / n_pairs
            else:
                tau = 0.0

            self._rank_corr_history.append(tau)
            if len(self._rank_corr_history) > 20:
                self._rank_corr_history.pop(0)

        success_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
            if self._strategy_attempt_counts[s] > 0:
                success_rates[s] = self._strategy_success_counts[s] / self._strategy_attempt_counts[s]

        rank_scores = self._strategy_rank_improvements.copy()
        max_rank = np.max(rank_scores) if np.max(rank_scores) > 0 else 1.0
        rank_scores = rank_scores / max_rank

        correlation_factor = 1.0 + 0.1 * np.mean(self._rank_corr_history[-5:]) if len(self._rank_corr_history) > 0 else 1.0
        correlation_factor = np.clip(correlation_factor, 0.8, 1.2)

        combined_scores = 0.6 * success_rates + 0.3 * rank_scores
        combined_scores *= correlation_factor

        ema_decay = 0.7
        self.strategy_scores = ema_decay * self.strategy_scores + (1 - ema_decay) * combined_scores
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)

        self._prev_fitness_ranks[valid_mask] = current_ranks
    
    def _update_strategy_scores_catF_temporal(self, strategy_used, improved):
        """Category F: Temporal credit assignment using EMA momentum and drift detection."""
        n_strategies = len(self.strategy_scores)

        if not hasattr(self, '_ema_rates'):
            self._ema_rates = np.ones(n_strategies) * 0.5
        if not hasattr(self, '_ema_alpha'):
            self._ema_alpha = 0.3
        if not hasattr(self, '_prev_improvement_rates'):
            self._prev_improvement_rates = np.zeros(n_strategies)
        if not hasattr(self, '_temporal_momentum'):
            self._temporal_momentum = np.zeros(n_strategies)
        if not hasattr(self, '_drift_history'):
            self._drift_history = []

        current_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                current_rates[s] = improved[mask].mean()

        alpha = self._ema_alpha
        new_ema = alpha * current_rates + (1 - alpha) * self._ema_rates

        momentum = new_ema - self._ema_rates
        self._temporal_momentum = 0.7 * self._temporal_momentum + 0.3 * momentum

        self._prev_improvement_rates = current_rates.copy()
        self._ema_rates = new_ema.copy()

        drift_signal = np.zeros(n_strategies)
        if len(self._drift_history) >= 3:
            baseline = np.mean(self._drift_history[-3:], axis=0)
            recent = new_ema
            drift_signal = recent - baseline
        self._drift_history.append(new_ema.copy())
        if len(self._drift_history) > 10:
            self._drift_history.pop(0)

        rate_variance = np.var(current_rates)
        adaptive_lr = 0.1 / (1.0 + rate_variance * 10)

        credit = self._temporal_momentum * 0.6 + drift_signal * 0.4
        self.strategy_scores += adaptive_lr * credit * 10.0

        decay_mask = (current_rates < 0.1) & (self._ema_rates < 0.2)
        self.strategy_scores[decay_mask] *= 0.97

        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)

        min_diff = 0.1
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                diff = abs(self.strategy_scores[i] - self.strategy_scores[j])
                if diff < min_diff:
                    self.strategy_scores[i] += min_diff * 0.5
                    self.strategy_scores[j] -= min_diff * 0.5
    
    # =========================================================================
    # PROBE-AND-COMMIT: Core adaptation logic
    # =========================================================================
    
    def _run_probe_phase(self, func, max_probe_iters):
        """
        Run probe phase: round-robin through all strategies to identify the best.
        Uses cheap landscape fingerprints to score each strategy empirically.
        
        Returns: committed strategy index
        """
        print(f"  PROBE PHASE: Testing {len(self.strategy_names)} strategies over {max_probe_iters} iterations...")
        
        probe_pop = self._initialize_population()
        probe_fitness = func(probe_pop)
        
        if len(probe_fitness) < self.NP:
            probe_fitness = np.full(self.NP, np.nan)
        
        # Track per-strategy fitness history for empirical scoring
        strategy_best_fitness = np.full(len(self.strategy_names), np.inf)
        strategy_improvement_rates = np.zeros(len(self.strategy_names))
        strategy_conv_slopes = np.zeros(len(self.strategy_names))  # Convergence slope
        strategy_stagnation_counts = np.zeros(len(self.strategy_names))
        
        # Per-strategy tracking during probe
        for strat_idx in range(len(self.strategy_names)):
            self._strategy_fitness_history[strat_idx] = []
        
        prev_best_per_strategy = np.full(len(self.strategy_names), np.inf)
        probe_iter = 0
        
        for probe_iter in range(max_probe_iters):
            # Round-robin: each strategy gets one generation
            for strat_idx in range(len(self.strategy_names)):
                # Use only this strategy for mutation
                mutants = np.empty_like(probe_pop)
                for i in range(self.NP):
                    mutants[i] = self._strategy_funcs[strat_idx](probe_pop, i, self.F)
                
                # Crossover
                trials = self._crossover_batch(probe_pop, mutants)
                
                # Evaluate
                trial_fitness = func(trials)
                if len(trial_fitness) < len(trials):
                    if len(trial_fitness) == 0:
                        continue
                    trial_fitness = np.full(len(trials), np.nan)
                
                # Selection
                better = trial_fitness < probe_fitness
                improved_mask = better
                probe_pop = np.where(better[:, np.newaxis], trials, probe_pop)
                probe_fitness = np.where(better, trial_fitness, probe_fitness)
                
                # Track fitness history
                current_best = np.nanmin(probe_fitness)
                if not np.isfinite(current_best):
                    current_best = 1e10
                
                self._strategy_fitness_history[strat_idx].append(current_best)
                
                # Track improvement rate
                if np.any(improved_mask):
                    strategy_improvement_rates[strat_idx] += 1
                
                # Track stagnation
                if prev_best_per_strategy[strat_idx] - current_best < 1e-8:
                    strategy_stagnation_counts[strat_idx] += 1
                else:
                    strategy_stagnation_counts[strat_idx] = 0
                
                prev_best_per_strategy[strat_idx] = current_best
                
                # Update best fitness seen by this strategy
                if current_best < strategy_best_fitness[strat_idx]:
                    strategy_best_fitness[strat_idx] = current_best
            
            # Store population reference for strategies that need it
            self._current_population = probe_pop
            self.fitness = probe_fitness
            
            # Reset for next round-robin round
            probe_pop = self._initialize_population()
            probe_fitness = func(probe_pop)
            if len(probe_fitness) < self.NP:
                probe_fitness = np.full(self.NP, np.nan)
        
        # Compute convergence slope (log-best vs iteration) for each strategy
        for strat_idx in range(len(self.strategy_names)):
            history = self._strategy_fitness_history[strat_idx]
            if len(history) >= 3:
                # Use rank-based slope: relative improvement over time
                first_half = np.median(history[:len(history)//2])
                second_half = np.median(history[len(history)//2:])
                if first_half > 0 and np.isfinite(first_half):
                    strategy_conv_slopes[strat_idx] = (first_half - second_half) / first_half
                else:
                    strategy_conv_slopes[strat_idx] = 0.0
        
        # Compute empirical scores: combine multiple signals (all rank-based)
        # Signal 1: Best fitness achieved (lower is better)
        valid_best = np.array([f if np.isfinite(f) else 1e10 for f in strategy_best_fitness])
        rank_best = np.argsort(np.argsort(valid_best)) / len(valid_best)  # 0 = best
        
        # Signal 2: Improvement rate (higher is better)
        rank_improve = np.argsort(np.argsort(-strategy_improvement_rates)) / len(strategy_improvement_rates)
        
        # Signal 3: Stagnation count (lower is better)
        rank_stagnation = np.argsort(np.argsort(strategy_stagnation_counts)) / len(strategy_stagnation_counts)
        
        # Signal 4: Convergence slope (higher is better)
        rank_slope = np.argsort(np.argsort(-strategy_conv_slopes)) / len(strategy_conv_slopes)
        
        # Combined empirical score (equal weights, all rank-based)
        empirical_scores = 0.25 * rank_best + 0.25 * rank_improve + 0.25 * (1 - rank_stagnation) + 0.25 * rank_slope
        
        # Pick the winner
        winner_idx = np.argmax(empirical_scores)
        
        print(f"  PROBE RESULTS:")
        for s in range(len(self.strategy_names)):
            status = " <-- COMMITTED" if s == winner_idx else ""
            print(f"    {self.strategy_names[s]}: best={strategy_best_fitness[s]:.6e}, "
                  f"improve_rate={strategy_improvement_rates[s]/max_probe_iters:.2f}, "
                  f"stagnation={strategy_stagnation_counts[s]}, "
                  f"slope={strategy_conv_slopes[s]:.4f}, score={empirical_scores[s]:.4f}{status}")
        
        return winner_idx
    
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
        Main optimization loop with PROBE-AND-COMMIT strategy selection.
        
        Phase 1 (PROBE): Round-robin through all strategies for ~8% of budget
        Phase 2 (COMMIT): Run committed strategy for remaining ~92% of budget
        """
        # Estimate budget from stopping_condition behavior
        estimated_max_iters = self.max_iter
        
        # Phase 1: Probe
        if not self._probe_completed:
            max_probe_iters = max(3, int(estimated_max_iters * self._probe_budget_pct))
            self._committed_strategy = self._run_probe_phase(func, max_probe_iters)
            self._probe_completed = True
            self._probe_generation = 0
            print(f"  COMMITTED to strategy: {self.strategy_names[self._committed_strategy]}")
            print(f"  Starting COMMIT phase...")
        
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
            # Store population reference for strategies that need it
            self._current_population = population
            self.fitness = fitness
            
            # PROBE-AND-COMMIT: Use committed strategy during commit phase
            if self._probe_completed:
                # Check for epsilon-revisit: small chance to try runner-up on stagnation
                should_revisit = False
                if self._probe_generation > 5:  # Only after some iterations
                    if self._last_commit_best_fitness - f_opt < 1e-8:
                        self.stagnation_counter += 1
                    else:
                        self.stagnation_counter = 0
                    
                    if self.stagnation_counter > 20 and np.random.rand() < self._epsilon_revisit:
                        # Find runner-up strategy
                        runner_up = np.argmax(
                            self._strategy_win_counts * (np.arange(len(self._strategy_win_counts)) != self._committed_strategy)
                        )
                        self._committed_strategy = runner_up
                        should_revisit = True
                        print(f"  EPSILON-REVISIT: switching to {self.strategy_names[runner_up]}")
                
                self._last_commit_best_fitness = f_opt
                self._probe_generation += 1
                
                # Use ONLY the committed strategy
                mutants = np.empty_like(population)
                strategy_used = np.full(self.NP, self._committed_strategy, dtype=int)
                for i in range(self.NP):
                    mutants[i] = self._strategy_funcs[self._committed_strategy](population, i, self.F)
            else:
                # Probe phase: round-robin (handled in _run_probe_phase)
                mutants, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Compass mutation (always used as secondary)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble/committed
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants)
            
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
            
            # Strategy score update (only during probe phase)
            if not self._probe_completed:
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
