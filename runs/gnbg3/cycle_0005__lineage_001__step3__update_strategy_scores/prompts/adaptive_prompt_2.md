Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_update_strategy_scores` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t06_id  variant_02_catB_t15_id  variant_03_catC_t22_id  variant_04_catD_t19_id  variant_05_catE_t15_id  variant_06_catF_t17_id  variant_08_catH_t17_id  variant_09_catA_t17_id  variant_10_catB_t17_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
1      4.732816e-06            4.323565e-05            -inf                    3.295949e-05            4.371933e-05            -inf                    3.213060e-05            4.509795e-05            4.234378e-05            3.530282e-05            
2      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
3      9.212033e-01            1.842041e+00            -inf                    1.948516e+00            1.161156e+00            -inf                    1.545200e+00            1.509141e+00            1.280207e+00            1.350939e+00            
4      1.000000e-08            1.000000e-08            -inf                    3.999990e-08            1.000000e-08            -inf                    1.333333e-08            1.000000e-08            1.000000e-08            3.999990e-08            
5      2.156517e-05            1.561187e-05            -inf                    9.876322e-05            9.670818e-05            -inf                    4.774350e-05            8.039878e-05            2.832314e-04            3.019611e-04            
6      3.269602e+01            3.439150e+01            -inf                    3.957920e+01            3.617205e+01            -inf                    4.354883e+01            3.533664e+01            3.844693e+01            4.553316e+01            
7      6.172597e-02            6.487378e-02            -inf                    7.714911e-02            7.722338e-02            -inf                    7.793770e-02            1.465069e-01            7.356746e-02            7.503305e-02            
8      3.056784e-02            4.520034e-02            -inf                    8.040515e-02            5.829030e-02            -inf                    7.283762e-02            6.515150e-02            6.509093e-02            3.984343e-02            
9      9.312889e+00            9.491058e+00            -inf                    9.399944e+00            9.441988e+00            -inf                    9.379914e+00            9.745877e+00            9.277506e+00            9.585185e+00            
10     2.735470e-06            1.334598e-04            -inf                    1.068173e-04            9.622721e-05            -inf                    9.423605e-05            9.278573e-05            8.512388e-05            7.255405e-05            
11     1.507611e+01            2.453878e+01            -inf                    2.816860e+01            2.501879e+01            -inf                    1.495591e+01            1.733604e+01            2.208147e+01            1.448884e+01            
12     8.126950e-03            1.337336e-01            -inf                    8.101863e-02            1.583413e-01            -inf                    9.745572e-02            1.564393e-01            1.367113e-01            1.049227e-01            
13     2.527304e+00            2.816236e+00            -inf                    2.685377e+00            2.742331e+00            -inf                    2.543225e+00            4.524482e+00            3.150655e+00            5.124195e+00            
14     3.509648e+00            4.143887e+00            -inf                    4.197434e+00            3.624775e+00            -inf                    3.731990e+00            3.911199e+00            4.545356e+00            4.482234e+00            
15     1.721943e+00            2.160269e+00            -inf                    2.292751e+00            2.384411e+00            -inf                    1.787147e+00            2.339884e+00            2.274865e+00            2.330683e+00            
16     8.319955e+02            5.243407e+02            -inf                    5.872078e+02            6.486323e+02            -inf                    8.811582e+02            1.016842e+03            9.982894e+02            5.078369e+02            
17     1.543570e+01            4.551410e+02            -inf                    3.373083e+03            3.401972e+02            -inf                    2.293015e+03            3.879353e+02            2.235172e+03            1.325175e+03            
18     2.038357e+01            2.199720e+01            -inf                    2.174425e+01            2.293351e+01            -inf                    2.074450e+01            2.398865e+01            2.255105e+01            2.848451e+01            
19     3.342850e+01            3.530202e+01            -inf                    3.621355e+01            4.825505e+01            -inf                    3.929886e+01            4.040200e+01            4.601333e+01            4.432899e+01            
20     1.658605e+01            1.505458e+01            -inf                    1.431568e+01            1.345260e+01            -inf                    1.540741e+01            1.846012e+01            1.582526e+01            1.477794e+01            
21     4.632922e+00            4.602915e+00            -inf                    4.537415e+00            4.546276e+00            -inf                    4.531265e+00            4.560532e+00            4.581743e+00            4.616378e+00            
22     4.804777e+00            5.227954e+00            -inf                    4.529856e+00            5.495418e+00            -inf                    5.261086e+00            6.758187e+00            5.681863e+00            6.055939e+00            
23     2.946017e+01            3.638180e+01            -inf                    3.687467e+01            3.552612e+01            -inf                    3.599941e+01            3.852546e+01            3.270472e+01            3.387658e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: original.py  (error=4.732816e-06)
Task  2: SKIPPED (trivial)
Task  3: original.py  (error=9.212033e-01)
Task  4: SKIPPED (trivial)
Task  5: variant_01_catA_t06_idea_0.py  (error=1.561187e-05)
Task  6: original.py  (error=3.269602e+01)
Task  7: original.py  (error=6.172597e-02)
Task  8: original.py  (error=3.056784e-02)
Task  9: variant_09_catA_t17_idea_0.py  (error=9.277506e+00)
Task 10: original.py  (error=2.735470e-06)
Task 11: variant_10_catB_t17_idea_0.py  (error=1.448884e+01)
Task 12: original.py  (error=8.126950e-03)
Task 13: original.py  (error=2.527304e+00)
Task 14: original.py  (error=3.509648e+00)
Task 15: original.py  (error=1.721943e+00)
Task 16: variant_10_catB_t17_idea_0.py  (error=5.078369e+02)
Task 17: original.py  (error=1.543570e+01)
Task 18: original.py  (error=2.038357e+01)
Task 19: original.py  (error=3.342850e+01)
Task 20: variant_04_catD_t19_idea_0.py  (error=1.345260e+01)
Task 21: variant_06_catF_t17_idea_0.py  (error=4.531265e+00)
Task 22: variant_03_catC_t22_idea_0.py  (error=4.529856e+00)
Task 23: original.py  (error=2.946017e+01)

WIN COUNTS:
  original.py: 14 wins
  variant_10_catB_t17_idea_0.py: 2 wins
  variant_01_catA_t06_idea_0.py: 1 wins
  variant_09_catA_t17_idea_0.py: 1 wins
  variant_04_catD_t19_idea_0.py: 1 wins
  variant_06_catF_t17_idea_0.py: 1 wins
  variant_03_catC_t22_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     1   original.py                     4.7328e-06       variant_06_catF_t17_idea_0.py    3.2131e-05         6.8×    8.9×
     3   original.py                     9.2120e-01       variant_04_catD_t19_idea_0.py    1.1612e+00         1.3×    1.6×
     5   variant_01_catA_t06_idea_0.py   1.5612e-05       original.py                      2.1565e-05         1.4×    6.2×
     6   original.py                     3.2696e+01       variant_01_catA_t06_idea_0.py    3.4391e+01         1.1×    1.2×
     7   original.py                     6.1726e-02       variant_01_catA_t06_idea_0.py    6.4874e-02         1.1×    1.2×
     8   original.py                     3.0568e-02       variant_10_catB_t17_idea_0.py    3.9843e-02         1.3×    2.1×
     9   variant_09_catA_t17_idea_0.py   9.2775e+00       original.py                      9.3129e+00         1.0×    1.0×
    10   original.py                     2.7355e-06       variant_10_catB_t17_idea_0.py    7.2554e-05        26.5×   34.4×
    11   variant_10_catB_t17_idea_0.py   1.4489e+01       variant_06_catF_t17_idea_0.py    1.4956e+01         1.0×    1.5×
    12   original.py                     8.1269e-03       variant_03_catC_t22_idea_0.py    8.1019e-02        10.0×   16.5×
    13   original.py                     2.5273e+00       variant_06_catF_t17_idea_0.py    2.5432e+00         1.0×    1.1×
    14   original.py                     3.5096e+00       variant_04_catD_t19_idea_0.py    3.6248e+00         1.0×    1.2×
    15   original.py                     1.7219e+00       variant_06_catF_t17_idea_0.py    1.7871e+00         1.0×    1.3×
    16   variant_10_catB_t17_idea_0.py   5.0784e+02       variant_01_catA_t06_idea_0.py    5.2434e+02         1.0×    1.6×
    17   original.py                     1.5436e+01       variant_04_catD_t19_idea_0.py    3.4020e+02        22.0×   85.9×
    18   original.py                     2.0384e+01       variant_06_catF_t17_idea_0.py    2.0744e+01         1.0×    1.1×
    19   original.py                     3.3429e+01       variant_01_catA_t06_idea_0.py    3.5302e+01         1.1×    1.2×
    20   variant_04_catD_t19_idea_0.py   1.3453e+01       variant_03_catC_t22_idea_0.py    1.4316e+01         1.1×    1.1×
    21   variant_06_catF_t17_idea_0.py   4.5313e+00       variant_03_catC_t22_idea_0.py    4.5374e+00         1.0×    1.0×
    22   variant_03_catC_t22_idea_0.py   4.5299e+00       original.py                      4.8048e+00         1.1×    1.2×
    23   original.py                     2.9460e+01       variant_09_catA_t17_idea_0.py    3.2705e+01         1.1×    1.2×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 21
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_t06_idea_0.py, variant_03_catC_t22_idea_0.py, variant_04_catD_t19_idea_0.py, variant_06_catF_t17_idea_0.py, variant_09_catA_t17_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 2  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 17 — original.py (1.54e+01) vs variant_04_catD_t19_idea_0.py (3.40e+02) = 22.0× gap to runner-up, 85.9× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.78e+02, which is ~11.5× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_t06_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Geometry-based strategy credit: spatial analysis of where strategies operate."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return

        pop = self._current_population
        NP, dim = pop.shape

        # Global population metrics
        global_centroid = pop.mean(axis=0)

        # Pairwise distances (vectorized)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # Global spread
        global_spread = np.mean(np.sqrt(sq_dists))
        global_spread = max(global_spread, 1e-10)

        n_strategies = len(self.strategy_scores)

        for s in range(n_strategies):
            mask = (strategy_used == s)
            count = np.sum(mask)

            if count < 2:
                continue

            # Strategy centroid
            centroid = pop[mask].mean(axis=0)

            # Strategy spread: mean distance from centroid
            centroid_dists = np.linalg.norm(pop[mask] - centroid, axis=1)
            spread = np.mean(centroid_dists)
            spread_ratio = spread / global_spread

            # Distance from strategy centroid to global centroid
            centroid_to_global = np.linalg.norm(centroid - global_centroid)
            normalized_dist = centroid_to_global / global_spread

            # k-NN local density within strategy group
            k = min(3, count - 1)
            if k > 0:
                group_sq_dists = sq_dists[np.ix_(mask, mask)]
                np.fill_diagonal(group_sq_dists, np.inf)
                knn_dists = np.sort(group_sq_dists, axis=1)[:, :k]
                avg_knn = np.mean(np.sqrt(knn_dists))
                density_score = 1.0 / (avg_knn + 1e-10)
            else:
                density_score = 0.0

            # Geometric score: reward strategies in underexplored sparse regions
            # Components: (1) spread ratio, (2) inverse density, (3) distance from global centroid
            geometric_score = spread_ratio * 0.35 + density_score * 0.35 + normalized_dist * 0.30

            # Clamp to prevent extreme values
            geometric_score = np.clip(geometric_score, 0.01, 10.0)

            # Exponential moving average update
            self.strategy_scores[s] = 0.85 * self.strategy_scores[s] + 0.15 * geometric_score
```

# --- From variant_03_catC_t22_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """
        Information-theoretic credit assignment using:
        - Entropy of fitness distributions per strategy
        - Mutual information between strategy and improvement signal
        - KL divergence of strategy fitness from population baseline
        """
        n_strategies = len(self.strategy_scores)
        NP = len(strategy_used)

        if NP < 5:
            return

        # Initialize history tracking for distributional analysis
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []

        # Store current fitness for temporal analysis
        if hasattr(self, 'fitness'):
            self._fitness_history.append(self.fitness.copy())
            if len(self._fitness_history) > 20:
                self._fitness_history.pop(0)

        reward = np.zeros(n_strategies)

        # 1. Entropy-based reward: strategies that reduce fitness variance get positive credit
        for s in range(n_strategies):
            mask = strategy_used == s
            n_s = np.sum(mask)
            if n_s >= 3 and hasattr(self, 'fitness'):
                fitness_s = self.fitness[mask]
                # Compute entropy of binned fitness distribution
                n_bins = min(8, max(2, n_s // 2))
                bin_edges = np.percentile(fitness_s, np.linspace(0, 100, n_bins + 1))
                counts, _ = np.histogram(fitness_s, bins=bin_edges)
                probs = counts / n_s
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                # Lower entropy = more concentrated = potentially better exploitation
                reward[s] -= entropy * 0.3

        # 2. Mutual information: strategies with high MI with improvement signal
        n_improved = np.sum(improved)
        if 0 < n_improved < NP:
            # Build joint distribution P(strategy, improvement)
            p_joint = np.zeros((2, n_strategies))
            for s in range(n_strategies):
                mask = strategy_used == s
                n_s = np.sum(mask)
                if n_s > 0:
                    n_imp_s = np.sum(improved[mask])
                    p_joint[0, s] = n_imp_s / NP  # improved
                    p_joint[1, s] = (n_s - n_imp_s) / NP  # not improved

            # Compute MI = sum_{i,s} P(i,s) * log(P(i,s) / (P(i)*P(s)))
            for s in range(n_strategies):
                p_s = np.sum(p_joint[:, s])
                if p_s > 1e-10:
                    for i in range(2):
                        if p_joint[i, s] > 1e-10:
                            p_i = np.sum(p_joint[i, :])
                            mi_contrib = p_joint[i, s] * np.log(p_joint[i, s] / (p_i * p_s + 1e-12))
                            reward[s] += mi_contrib * 5.0  # Scale for numerical stability

        # 3. KL divergence: strategies shifting fitness distribution toward lower values
        if hasattr(self, 'fitness') and len(self._fitness_history) >= 2:
            prev_fitness = self._fitness_history[-2] if len(self._fitness_history) >= 2 else self.fitness
            for s in range(n_strategies):
                mask = strategy_used == s
                n_s = np.sum(mask)
                if n_s >= 2:
                    current_s = self.fitness[mask]
                    prev_s = prev_fitness[mask]
                    # Normalize for distribution comparison
                    if np.std(current_s) > 1e-10 and np.std(prev_s) > 1e-10:
                        curr_norm = (current_s - np.min(current_s) + 1e-6) / (np.ptp(current_s) + 1e-6)
                        prev_norm = (prev_s - np.min(prev_s) + 1e-6) / (np.ptp(prev_s) + 1e-6)
                        # KL(current || prev): lower current values shift distribution left
                        kl = np.mean(np.log(prev_norm + 1e-6) - np.log(curr_norm + 1e-6))
                        reward[s] += kl * 0.5

        # Apply momentum-based update with softmax normalization
        reward = np.clip(reward, -5, 5)
        reward = reward - np.mean(reward)  # Center rewards

        momentum = 0.7
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * reward

        # Softmax normalization with temperature
        temp = 1.5
        exp_scores = np.exp(self.strategy_scores / temp)
        self.strategy_scores = exp_scores / np.sum(exp_scores)

        # Ensure minimum probability to prevent strategy starvation
        self.strategy_scores = np.maximum(self.strategy_scores, 0.05)
        self.strategy_scores /= np.sum(self.strategy_scores)
```

# --- From variant_04_catD_t19_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """
        Category D: Fitness-landscape / rank-based strategy scoring.
        Uses Kendall Tau rank correlation + fitness percentile-weighted success rates.
        No distances or geometry — pure fitness signal analysis.
        """
        NP = len(strategy_used)
        n_strategies = len(self.strategy_names)

        # Initialize per-strategy tracking if first call
        if not hasattr(self, '_strategy_rank_improvements'):
            self._strategy_rank_improvements = np.zeros(n_strategies)
            self._strategy_success_counts = np.zeros(n_strategies)
            self._strategy_attempt_counts = np.zeros(n_strategies)
            self._strategy_fitness_history = [[] for _ in range(n_strategies)]
            self._rank_corr_history = []

        # Compute fitness percentile for each individual (0 = worst, 1 = best)
        if not hasattr(self, '_prev_fitness_ranks'):
            self._prev_fitness_ranks = np.zeros(NP)

        # Get current fitness (need to access from optimizer state)
        if not hasattr(self, '_current_fitness'):
            return  # No fitness available yet

        current_fitness = self._current_fitness
        valid_mask = np.isfinite(current_fitness)

        if np.sum(valid_mask) < 5:
            return

        # Compute current fitness ranks (normalized 0-1)
        fitness_sorted = np.argsort(current_fitness[valid_mask])
        current_ranks = np.zeros(np.sum(valid_mask))
        current_ranks[fitness_sorted] = np.linspace(0, 1, len(fitness_sorted))

        # Compute rank improvement per individual
        rank_improvement = self._prev_fitness_ranks[valid_mask] - current_ranks

        # Update per-strategy metrics
        for s in range(n_strategies):
            s_mask = (strategy_used == s) & valid_mask
            if np.sum(s_mask) == 0:
                continue

            # Track attempts
            self._strategy_attempt_counts[s] += np.sum(s_mask)

            # Track successes
            successes = improved[s_mask]
            self._strategy_success_counts[s] += np.sum(successes)

            # Compute fitness percentile of individuals this strategy acted on
            avg_percentile = np.mean(current_ranks)  # percentile of individuals this strategy worked on

            # Weight success by how "difficult" the individuals were (lower percentile = harder)
            difficulty_weight = 1.0 - avg_percentile + 0.1  # Add 0.1 to avoid div by zero issues

            # Rank improvement for this strategy's successes
            if np.any(successes):
                avg_rank_improvement = np.mean(rank_improvement[s_mask][successes])
                self._strategy_rank_improvements[s] += difficulty_weight * max(avg_rank_improvement, 0)

            # Store fitness history for correlation analysis
            self._strategy_fitness_history[s].extend(current_fitness[s_mask].tolist())
            # Keep bounded
            if len(self._strategy_fitness_history[s]) > 1000:
                self._strategy_fitness_history[s] = self._strategy_fitness_history[s][-500:]

        # Compute Kendall Tau rank correlation between strategy indices and fitness
        # Higher fitness (lower value for minimization) should correlate with better strategies
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) >= 10:
            strat_vals = strategy_used[valid_indices].astype(float)
            fitness_vals = current_fitness[valid_indices]

            # Normalize fitness to 0-1 scale for correlation
            f_min, f_max = np.min(fitness_vals), np.max(fitness_vals)
            if f_max > f_min:
                fitness_norm = (fitness_vals - f_min) / (f_max - f_min)
            else:
                fitness_norm = np.zeros_like(fitness_vals)

            # Compute Kendall tau manually (avoid scipy dependency)
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

        # Compute success rates per strategy
        success_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
            if self._strategy_attempt_counts[s] > 0:
                success_rates[s] = self._strategy_success_counts[s] / self._strategy_attempt_counts[s]

        # Compute rank improvement scores (normalized)
        rank_scores = self._strategy_rank_improvements.copy()
        max_rank = np.max(rank_scores) if np.max(rank_scores) > 0 else 1.0
        rank_scores = rank_scores / max_rank

        # Combine: 60% success rate, 30% rank improvement, 10% correlation adjustment
        correlation_factor = 1.0 + 0.1 * np.mean(self._rank_corr_history[-5:]) if len(self._rank_corr_history) > 0 else 1.0
        correlation_factor = np.clip(correlation_factor, 0.8, 1.2)

        combined_scores = 0.6 * success_rates + 0.3 * rank_scores
        combined_scores *= correlation_factor

        # Apply exponential moving average for temporal smoothing
        ema_decay = 0.7
        self.strategy_scores = ema_decay * self.strategy_scores + (1 - ema_decay) * combined_scores

        # Ensure minimum scores for exploration
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)

        # Update previous fitness ranks for next iteration
        self._prev_fitness_ranks[valid_mask] = current_ranks
```

# --- From variant_06_catF_t17_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Temporal credit assignment using EMA momentum and drift detection."""
        n_strategies = len(self.strategy_scores)

        # Initialize temporal tracking attributes
        if not hasattr(self, '_ema_rates'):
            self._ema_rates = np.ones(n_strategies) * 0.5
        if not hasattr(self, '_ema_alpha'):
            self._ema_alpha = 0.3  # EMA smoothing factor
        if not hasattr(self, '_prev_improvement_rates'):
            self._prev_improvement_rates = np.zeros(n_strategies)
        if not hasattr(self, '_temporal_momentum'):
            self._temporal_momentum = np.zeros(n_strategies)
        if not hasattr(self, '_drift_history'):
            self._drift_history = []

        # Compute per-strategy improvement rates for this generation
        current_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                current_rates[s] = improved[mask].mean()

        # Update exponential moving averages with momentum
        alpha = self._ema_alpha
        new_ema = alpha * current_rates + (1 - alpha) * self._ema_rates

        # Compute momentum: rate of change in EMA
        momentum = new_ema - self._ema_rates

        # Detect temporal drift: sustained directional change
        self._temporal_momentum = 0.7 * self._temporal_momentum + 0.3 * momentum

        # Store for next iteration
        self._prev_improvement_rates = current_rates.copy()
        self._ema_rates = new_ema.copy()

        # Drift detection: compare recent trend to historical baseline
        drift_signal = np.zeros(n_strategies)
        if len(self._drift_history) >= 3:
            baseline = np.mean(self._drift_history[-3:], axis=0)
            recent = new_ema
            drift_signal = recent - baseline
        self._drift_history.append(new_ema.copy())
        if len(self._drift_history) > 10:
            self._drift_history.pop(0)

        # Compute adaptive learning rate based on signal confidence
        rate_variance = np.var(current_rates)
        adaptive_lr = 0.1 / (1.0 + rate_variance * 10)

        # Credit assignment: combine momentum and drift
        # Positive momentum + positive drift = accelerating strategy
        # Negative momentum + negative drift = declining strategy
        credit = self._temporal_momentum * 0.6 + drift_signal * 0.4

        # Apply momentum-boosted credit to scores
        self.strategy_scores += adaptive_lr * credit * 10.0

        # Temporal decay: slowly reduce scores of underperforming strategies
        decay_mask = (current_rates < 0.1) & (self._ema_rates < 0.2)
        self.strategy_scores[decay_mask] *= 0.97

        # Soft normalization to prevent extreme divergence
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)

        # Ensure minimum score diversity
        min_diff = 0.1
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                diff = abs(self.strategy_scores[i] - self.strategy_scores[j])
                if diff < min_diff:
                    self.strategy_scores[i] += min_diff * 0.5
                    self.strategy_scores[j] -= min_diff * 0.5
```

# --- From variant_09_catA_t17_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """
        Geometry-based strategy scoring using k-NN density and spatial spread.

        Strategies that generate improvements in underexplored (low-density) regions
        OR in geometrically diverse locations receive higher rewards.
        """
        if not hasattr(self, '_current_population') or self._current_population is None:
            return

        pop = self._current_population
        NP, dim = pop.shape

        if NP < 4:
            return

        # Compute k-NN distances for local density estimation
        k = max(2, min(5, NP // 4))

        # Pairwise squared distances (vectorized)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # Find k nearest neighbors for each individual
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]

        # Local density: inverse of average k-NN distance (higher = denser)
        avg_k_dist = np.mean(np.sqrt(nearest_sq_dists), axis=1)
        local_density = 1.0 / (avg_k_dist + 1e-8)

        # Normalize density to sum to NP
        density_sum = local_density.sum()
        if density_sum > 1e-10:
            local_density *= NP / density_sum

        # Compute centroid of all improving individuals
        if improved.any():
            improved_pop = pop[improved]
            global_centroid = improved_pop.mean(axis=0)
        else:
            global_centroid = pop.mean(axis=0)

        # Score each strategy based on geometric properties of its successes
        num_strategies = len(self.strategy_names)
        strategy_rewards = np.zeros(num_strategies)

        for s in range(num_strategies):
            # Individuals that used this strategy
            strategy_mask = (strategy_used == s)
            if not strategy_mask.any():
                continue

            # Among those, which improved?
            success_mask = strategy_mask & improved
            if not success_mask.any():
                continue

            # Geographic spread: mean distance from global centroid of improvements
            success_pop = pop[success_mask]
            centroid_dist = np.linalg.norm(success_pop - global_centroid, axis=1)
            mean_spread = np.mean(centroid_dist) + 1e-8

            # Density of successful region (lower density = more novel exploration)
            success_density = local_density[success_mask].sum()
            density_bonus = 1.0 / (success_density + 1.0)

            # Convex hull area proxy (for spread in multiple dimensions)
            if len(success_pop) >= 3:
                # Use pairwise distance span as hull proxy
                hull_dists = sq_dists[np.ix_(success_mask, success_mask)]
                np.fill_diagonal(hull_dists, 0)
                hull_span = np.mean(hull_dists) + 1e-8
            else:
                hull_span = 1.0

            # Combined geometric score: spread * novelty * hull coverage
            strategy_rewards[s] = mean_spread * density_bonus * np.sqrt(hull_span)

        # Normalize rewards
        total_reward = strategy_rewards.sum()
        if total_reward > 1e-10:
            strategy_rewards /= total_reward

        # Update scores using exponential moving average with momentum
        momentum = 0.7
        learning_rate = 0.3

        for s in range(num_strategies):
            if strategy_rewards[s] > 0:
                reward_signal = np.log1p(strategy_rewards[s] * NP)
            else:
                reward_signal = -2.0

            self.strategy_scores[s] = (momentum * self.strategy_scores[s] + 
                                       learning_rate * reward_signal)

        # Softmax normalization for final probabilities
        scores_exp = np.exp(self.strategy_scores - np.max(self.strategy_scores))
        self.strategy_scores = np.log(scores_exp / (scores_exp.sum() + 1e-10))
```

# --- From variant_10_catB_t17_idea_0.py (2 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Category B: Spectral analysis via eigendecomposition of population covariance."""
        # Initialize EMA tracking
        if not hasattr(self, '_spectral_success_ema'):
            self._spectral_success_ema = np.ones(len(self.strategy_names)) * 0.5
            self._ema_alpha = 0.3

        # Compute spectral metrics from population covariance
        if hasattr(self, '_current_population') and self._current_population is not None:
            pop = self._current_population
            if len(pop) >= self.dim + 1:
                # Center and compute covariance
                centered = pop - pop.mean(axis=0)
                cov = np.cov(centered.T)

                # Eigendecomposition: extract eigenvalues
                try:
                    eigenvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
                    eigenvals = eigenvals[eigenvals > 1e-12]
                except Exception:
                    eigenvals = np.array([1.0])

                if len(eigenvals) >= 2:
                    # Condition number: ratio of largest to smallest eigenvalue
                    cond = eigenvals[0] / max(eigenvals[-1], 1e-12)
                    cond = min(cond, 1e6)

                    # Effective rank via eigenvalue entropy
                    total = np.sum(eigenvals)
                    probs = eigenvals / max(total, 1e-12)
                    entropy = -np.sum(probs * np.log(probs + 1e-12))
                    max_entropy = np.log(len(eigenvals))
                    eff_rank = np.exp(entropy) / max(max_entropy, 1e-12)
                else:
                    cond = 1e6
                    eff_rank = 0.1

                # Spectral quality: 1.0 = well-conditioned (favor exploitation), 0.0 = ill-conditioned (favor exploration)
                spectral_quality = np.exp(-np.log(1 + cond) / 10.0) * np.clip(eff_rank, 0.1, 1.0)
            else:
                spectral_quality = 0.5
        else:
            spectral_quality = 0.5

        # Update strategy success rates via EMA
        for strat_idx in range(len(self.strategy_names)):
            mask = strategy_used == strat_idx
            if np.any(mask):
                success_rate = np.mean(improved[mask])
                self._spectral_success_ema[strat_idx] = (1 - self._ema_alpha) * self._spectral_success_ema[strat_idx] + self._ema_alpha * success_rate

        # Compute target scores: blend spectral quality with success rate
        # High spectral_quality -> reward strategies with high success (exploitation)
        # Low spectral_quality -> favor balanced exploration
        baseline = 0.3 + 0.4 * spectral_quality
        target_scores = baseline + 0.3 * self._spectral_success_ema

        # Smooth update: prevent drastic changes
        momentum = 0.4
        self.strategy_scores = (1 - momentum) * self.strategy_scores + momentum * target_scores

        # Ensure minimum score to avoid starvation
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)
```

────────────────────────────────────────────────────────────────────────
ADAPTATION MECHANISM MENU — pick ONE primary mechanism and motivate it.
Different mechanisms are appropriate to different operator types. A bandit
over six near-equivalent metrics is wasted; an ensemble that VOTES gives
a useful new signal even when the candidates are similar.

  1) DISCRETE ARM SELECTION (multi-armed bandit family)
     - Thompson Sampling, UCB1, EXP3, ε-greedy
     - Best when: operators are mutually exclusive, easy to credit-assign,
       reward is on a stable scale.
     - REQUIRED if you use this: rank-normalize rewards within a sliding
       window so magnitudes don't dominate. NEVER hard-code multipliers
       like *1e5 or *1e10 — those are scale-dependent magic constants.

  2) ENSEMBLE / VOTING / BLENDING
     - Run all candidates every generation and combine their outputs:
       weighted average, median, soft-vote, rank-aggregation.
     - Weights can be learned online (e.g. exponentially weighted majority).
     - Best when: candidates are correlated but each captures a different
       failure mode. Robust on its own — does not need credit assignment.

  3) CONDITIONAL / RULE-BASED DISPATCH
     - Hand-coded rules that pick the operator from a feature of the
       current population state: stagnation count, condition number of C,
       fitness-rank entropy, generation index, restart count, etc.
     - Best when: the failure modes are diagnosable. Fast convergence
       because there is no exploration cost.

  4) CONTEXTUAL BANDIT / FEATURE-CONDITIONED SELECTION
     - Like (1) but the arm-distribution depends on a feature vector
       describing the current state of the search (population shape,
       improvement rate, diversity, generation, ...).
     - Best when: different operators are best in different regimes and
       you can measure the regime cheaply.

  5) HYPER-HEURISTIC / SCHEDULE LEARNING
     - Maintain a *sequence* of operators rather than a single per-step
       choice (e.g. operator A for k_1 generations, then B for k_2).
     - Schedule itself is adapted (e.g. via genetic encoding of the
       sequence, or adapting k_i with success-history).
     - Best when: operators benefit from being applied in bursts.

  6) SELF-ADAPTIVE PARAMETERS (continuous, not discrete)
     - Instead of picking among discrete operators, evolve continuous
       parameters that morph the operator (e.g. CR, F, σ, mutation
       radius). Parameters can be encoded per-individual and inherited.
     - Best when: the operator family is a continuous family.

  7) CO-EVOLUTION
     - Run TWO populations: one of solutions, one of operators / operator
       configurations. Operators evolve based on their measured fitness.
     - Best when: the operator design space is large and you want
       genuine novelty.

  8) RESTART-AWARE / EPOCHED ADAPTATION
     - Different operators on different restart epochs. Memory of which
       operators worked in past epochs informs the current epoch's
       distribution.
     - Best when: the algorithm restarts often (multimodal landscapes).

  9) META-CONTROLLED ADAPTATION
     - A small online controller (e.g. a learned linear policy or a
       hand-tuned PID-like loop) maps measured signals to operator
       weights. The controller's gains are themselves adapted slowly.
     - Best when: the dynamics are reasonably smooth.

────────────────────────────────────────────────────────────────────────
PER-TASK FINGERPRINTING / PROBE-AND-COMMIT (a specialisation of mechanism #4)

When the per-task winner table shows that DIFFERENT TASKS have DIFFERENT
winners with LARGE GAPS (>=10×), no single operator wins everywhere and
ensemble blending mathematically cannot preserve the per-task advantages
(a 50/50 blend of 1 and 100 is 50.5, not 1). The right pattern is:

  PHASE A (PROBE, ~3-8% of total budget):
    - Run a SHORT exploratory burst with each candidate operator in turn,
      OR with a deterministic round-robin across operators each generation,
      using a small sub-population.
    - From the probe burst, extract a CHEAP FINGERPRINT of the landscape:
        * convergence rate of the best-so-far (slope of log-best vs gen)
        * fitness rank-correlation across consecutive generations (proxy
          for landscape ruggedness / smoothness)
        * effective dimensionality from the early covariance (number of
          eigenvalues that explain >95% of variance)
        * separability proxy: ratio of axis-aligned vs diagonal step
          improvement counts
        * basin-of-attraction estimate: variance of best-so-far across
          parallel mini-restarts
        * stagnation index: fraction of probe gens with no improvement

  PHASE B (COMMIT, ~92-97% of budget):
    - Compute a fingerprint VECTOR (concatenate the cheap signals above).
    - For each candidate operator, compute its EMPIRICAL win rate on the
      probe burst (rank-based, not raw fitness).
    - COMMIT to the operator that won the probe burst, and run it for the
      remaining budget. Optionally keep a small probability ε of revisiting
      a runner-up if the committed operator stagnates.

This pattern is the correct response when the per-task gap table is
blend-hostile. It is a CONTEXTUAL BANDIT (mechanism #4) where the context
is computed once at the start of the run rather than every generation.

IMPORTANT: the fingerprint is computed FROM THE LANDSCAPE THE OPTIMIZER IS
ON, not from `func` identity (the optimizer never knows which GNBG task it
is solving). The cheap signals listed above are observable from any
black-box `func` and require no oracle knowledge.

────────────────────────────────────────────────────────────────────────
CALIBRATION RULES (mandatory — these are the most common failure modes):

a) NEVER use scale-dependent magic constants in reward shaping.
   FORBIDDEN: `np.log1p(diversity_improvement * 1e5)`,
              `np.log1p(func_improvement * 1e10)`,
              any hard-coded multiplier > 100.
   ALLOWED:   rank-based credit (was this generation in the top tercile of
              the last K generations?), z-score normalization, relative
              improvement (delta / |incumbent|), success/failure binary.

b) If you use a sliding window of size K, declare K as a parameter and
   make K >= 3 * num_arms so each arm is sampled enough times to be
   estimable. If num_arms is large, prefer ensemble (mechanism #2) over
   discrete selection.

c) Mixed reward signals (e.g. 0.6*diversity + 0.4*fitness) MUST be on
   the same scale. Either both rank-based, both z-scored, or both
   relative-improvement. Mixing log1p(x*1e5) with log1p(y*1e10) is wrong.

d) The mechanism must be VISIBLY DIFFERENT from a plain bandit-over-
   operators if the variants are highly correlated. If you can't credit-
   assign reliably, switch mechanism — don't try to fix a broken
   reward signal with more clipping/decay.

e) Initial state matters. Don't start with `alpha = beta = 1` (uniform
   prior) if you have no reason to think the prior is uninformative —
   the original method may already be a strong choice.


ADDITIONAL CALIBRATION RULES (apply when the per-task gap table is
blend-hostile, i.e. ≥1 task with gap ≥ 10×):

f) HARD-CODED REGIME→WEIGHT TABLES ARE FORBIDDEN. Lines like
       weights = {'converging': [0.50, 0.20, 0.15, 0.15], ...}
   are a known failure mode: they cap the dominant variant at the
   chosen constant (e.g. 0.50) regardless of how much it actually wins
   by. If task X has variant_07 winning by 100×, a 0.50 weight on
   variant_07 still loses ~50× of its advantage. Either:
     - DISPATCH (mechanism #3): pick ONE variant from a rule and use
       only that one (weight 1.0). No blending.
     - LEARN WEIGHTS ONLINE: initialize weights uniformly and update
       them from rank-based fitness improvement. The hand-coded prior
       table is the bug.

g) IF YOU IMPLEMENT A REGIME DETECTOR: every regime label it can return
   MUST appear in the dispatch / weighting logic. Dead branches like
   `regime_to_variant = {'exploring': 3, ...}` where `'exploring'` is
   never returned from the detector are a silent bug — the
   credit-assignment update never reaches that arm.

h) FOR EACH ARM/VARIANT YOU INCLUDE: there must exist at least one
   reachable code path that COMMITS TO THAT ARM ALONE (weight 1.0 or
   exclusive selection). If every branch of your dispatcher mixes the
   arm with at least one other arm, the arm's per-task advantage cannot
   be recovered. Test mentally: "if variant_X is 100× better on task T,
   does my dispatcher have a state in which it picks variant_X with
   weight ≥ 0.95?" If no, redesign.
────────────────────────────────────────────────────────────────────────

DECISION GUIDE — choose your mechanism by looking at the data above:

- If the per-task gap table shows ANY task with gap >=100× and at least
  two distinct winners → MUST use PROBE-AND-COMMIT or mechanism #3
  (rule-based dispatch). Ensemble blending is mathematically excluded.
- If the per-task gap table shows >=3 tasks with gap >=10× and at least
  two distinct winners → strongly prefer PROBE-AND-COMMIT or mechanism
  #4 (contextual bandit). Ensemble blending will cap your achievable
  per-task accuracy.
- If ONE variant clearly dominates (>50% of wins AND no other variant
  has a gap >=10× win on its own tasks) → use mechanism #3 with that
  variant as default.
- If wins are spread roughly evenly across many variants AND all gaps
  are small (<5×) → mechanism #1 (bandit) or #4 (contextual bandit).
- If variants compute correlated quantities and ALL gaps are < 3× →
  mechanism #2 (ensemble/voting) is acceptable.
- If there is a clear regime split (variant X wins early, variant Y
  wins late) → mechanism #5 (schedule) or #4 (contextual).
- If the operator is a continuous family → mechanism #6 (self-adaptive
  parameters) is more powerful than picking among discrete snapshots.

REQUIREMENTS:
- Respond with the COMPLETE class code inside a single ```python``` block.
- Include `import numpy as np` at the top.
- At the START of the class docstring, write a SHORT comment naming the
  adaptation mechanism you chose (from the menu above) and 1–2 lines
  explaining why it fits the per-task winner-gap table — not just the
  win-count distribution. If the gap table is blend-hostile, the
  docstring must explicitly state this and explain how your dispatcher
  preserves the per-task winner's advantage.
- You may add new attributes in __init__ and new private helper methods.
- Do NOT change signatures of existing public methods (__call__, etc.).
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt).
- Handle ALL edge cases: no -inf, no NaN, no crashes, clip to bounds.
- If you use any of mechanisms #1, #4, #5, #8: rewards MUST be rank-based
  or z-scored within a sliding window. NO scale-dependent constants.
- If the per-task gap table contains any task with gap ≥ 10×, you MUST
  use a dispatch-style mechanism (PROBE-AND-COMMIT, #3, or #4) — NOT
  ensemble blending. Linear blending mathematically cannot preserve
  large per-task gaps.

Original algorithm:
```python
import numpy as np


class AdaptiveCompassDE:
    """
    Adaptive Compass Differential Evolution.
    
    Novel features:
    1. Compass Mutation: Samples 3 candidate directions per individual,
       evaluates them cheaply via fitness ranking proximity, picks the best.
    2. Adaptive Strategy Pool: 5 mutation strategies with softmax-weighted selection.
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

        # Initialize tracking attributes
        if not hasattr(self, '_ema_rank_improvement'):
            self._ema_rank_improvement = np.zeros(n_strategies)
            self._ema_success_rate = np.zeros(n_strategies)
            self._rank_improvement_history = [[] for _ in range(n_strategies)]
            self._success_rate_history = [[] for _ in range(n_strategies)]
            self._momentum = 0.3

        # Compute per-strategy rank improvement (D: fitness-landscape based)
        # Higher rank improvement = better credit
        rank_improvement = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()

        # Compute per-strategy success rate (D: fitness-landscape based)
        success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                success_rate[s] = improved[mask].sum() / mask.sum()

        # Update history for variance computation
        for s in range(n_strategies):
            self._rank_improvement_history[s].append(rank_improvement[s])
            self._success_rate_history[s].append(success_rate[s])
            # Keep bounded history
            if len(self._rank_improvement_history[s]) > 10:
                self._rank_improvement_history[s].pop(0)
            if len(self._success_rate_history[s]) > 10:
                self._success_rate_history[s].pop(0)

        # Compute EMA with momentum for temporal stability (F: temporal/dynamical)
        alpha = 0.3
        for s in range(n_strategies):
            self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
            self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]

        # Compute variance of each signal per strategy (principled weighting signal)
        rank_var = np.zeros(n_strategies)
        success_var = np.zeros(n_strategies)
        for s in range(n_strategies):
            if len(self._rank_improvement_history[s]) >= 3:
                rank_var[s] = np.var(self._rank_improvement_history[s])
            if len(self._success_rate_history[s]) >= 3:
                success_var[s] = np.var(self._success_rate_history[s])

        # Weight by inverse variance (data-driven): consistent signals get more weight
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        rank_weight = np.where(rank_var > eps, 1.0 / (rank_var + eps), 1.0 / eps)
        success_weight = np.where(success_var > eps, 1.0 / (success_var + eps), 1.0 / eps)

        # Normalize weights
        total_rank_weight = rank_weight.sum()
        total_success_weight = success_weight.sum()
        if total_rank_weight > 0:
            rank_weight /= total_rank_weight
        if total_success_weight > 0:
            success_weight /= total_success_weight

        # Compute hybrid signal: weighted combination of rank improvement and success rate
        hybrid_signal = 0.5 * rank_weight * self._ema_rank_improvement + 0.5 * success_weight * self._ema_success_rate

        # Comparative advantage: how much better is this strategy vs population mean
        mean_signal = hybrid_signal.mean()
        if mean_signal > 0:
            comparative_advantage = (hybrid_signal - mean_signal) / (mean_signal + eps)
        else:
            comparative_advantage = np.zeros(n_strategies)

        # Compute confidence factor based on total effective sample size
        # Higher confidence = lower variance = more consistent performance
        combined_var = rank_var + success_var
        confidence = np.exp(-combined_var * 5)  # Exponential decay with variance

        # Score update with momentum and confidence weighting
        delta = np.zeros(n_strategies)
        for s in range(n_strategies):
            # Combine comparative advantage with absolute performance
            delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])  # Scale by confidence

        # Apply update
        self.strategy_scores += delta

        # Ensure scores stay positive (softmax needs positive values)
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on k-NN graph topology of the population."""
        # Store population reference for graph analysis
        if not hasattr(self, '_graph_population'):
            self._graph_population = None
            self._graph_metrics_history = []

        # Get current population from the optimizer's state
        # The population is accessible via the caller's frame or stored attribute
        if not hasattr(self, '_current_population'):
            return  # No population available yet

        pop = self._current_population
        if pop is None or len(pop) < 5:
            return

        NP, dim = pop.shape

        # Build k-NN graph (k based on population size)
        k = max(2, min(5, NP // 10))

        # Compute pairwise Euclidean distances (vectorized)
        # Shape: (NP, NP)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)

        # Find k nearest neighbors for each point (exclude self)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]

        # Compute average edge length of k-NN graph
        avg_edge_length = np.mean(np.sqrt(nearest_sq_dists))

        # Compute local clustering coefficient for each node
        clustering_coeffs = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest_indices[i])
            if len(neighbors) < 2:
                clustering_coeffs[i] = 0.0
                continue
            # Count edges among neighbors
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

        # Compute graph diameter proxy (max distance to k-th nearest neighbor)
        graph_diameter = np.max(np.sqrt(nearest_sq_dists))

        # Store current metrics
        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
            'graph_diameter': graph_diameter
        }
        self._graph_metrics_history.append(current_metrics)

        # Keep history bounded
        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)

        # Determine trend if we have history
        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]

            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
            clustering_trend = recent_clustering[-1] - recent_clustering[0]
        else:
            edge_length_trend = 0.0
            clustering_trend = 0.0

        # Adapt F based on graph topology
        # High clustering = population fragmented into niches -> increase F for exploration
        # Decreasing edge length = population converging -> increase F to escape
        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * (edge_length_trend < -0.01)
        clustering_factor += 0.08 * (clustering_trend > 0.05)  # Increasing clustering -> niche formation
        self.F = np.clip(self.F * clustering_factor, 0.3, 2.0)

        # Adapt CR based on graph topology
        # Sparse graph (large edge lengths) -> increase CR for more exploitation of good regions
        # Dense graph (small edge lengths) -> decrease CR to maintain diversity
        median_edge = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        if avg_edge_length > median_edge * 0.5:
            # Population well-spread -> encourage exploitation
            self.CR = np.clip(self.CR * 1.08, 0.1, 0.95)
        else:
            # Population clustered -> encourage exploration via crossover
            self.CR = np.clip(self.CR * 0.93, 0.1, 0.95)

        # Additional adaptation based on improvement rate (blended with topology)
        if improvement_rate > 0.15:
            self.F = np.clip(self.F * 1.05, 0.3, 2.0)
        elif improvement_rate < 0.03:
            self.F = np.clip(self.F * 1.12, 0.3, 2.0)  # Boost exploration on stagnation

        # Record history
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

        # Build k-NN graph to analyze population topology
        k = max(2, min(5, NP // 8))

        # Compute pairwise distances
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # Find k nearest neighbors for each individual
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Compute connectivity: how many times each individual appears in others' neighborhoods
        connectivity = np.zeros(NP)
        for i in range(NP):
            neighbors = nearest_indices[i]
            for n in neighbors:
                connectivity[n] += 1

        # Compute local clustering coefficient for each node
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

        # Identify isolated individuals: low connectivity OR low clustering AND low fitness rank
        fitness_ranks = self._compute_fitness_ranking(fitness)
        isolation_score = (1.0 - connectivity / connectivity.max()) * 0.5 + \
                          (1.0 - clustering) * 0.3 + \
                          fitness_ranks * 0.2

        # Find indices to replace (most isolated)
        n_replace = max(NP // 4, 3)
        replace_indices = np.argsort(isolation_score)[-n_replace:]

        # Get best individuals for mutation sources
        n_best = min(5, NP)
        best_indices = np.argsort(fitness)[:n_best]

        # Create replacement individuals via topology-guided mutation
        new_individuals = []
        for idx in replace_indices:
            # Select source based on graph distance: prefer individuals far in graph from idx
            best_source = None
            best_graph_dist = -1
            for b_idx in best_indices:
                # Approximate graph distance via common neighbors
                n1 = set(nearest_indices[idx])
                n2 = set(nearest_indices[b_idx])
                common = len(n1 & n2)
                graph_dist_approx = 1.0 / (common + 0.1)  # Lower common neighbors = further
                if graph_dist_approx > best_graph_dist:
                    best_graph_dist = graph_dist_approx
                    best_source = b_idx

            if best_source is None:
                best_source = best_indices[0]

            # Mutation: directional jump toward unexplored regions
            source = population[best_source]
            direction = population[best_source] - population[idx]
            mutation_scale = np.random.uniform(0.8, 1.5)

            # Add noise proportional to local edge length
            local_edges = sq_dists[idx, nearest_indices[idx]]
            avg_local_dist = np.mean(np.sqrt(local_edges))

            new_point = source + mutation_scale * self.F * direction + \
                        np.random.randn(dim) * avg_local_dist * 0.5
            new_individuals.append(np.clip(new_point, -100.0, 100.0))

        # Inject new individuals into population
        for i, idx in enumerate(replace_indices):
            population[idx] = new_individuals[i]

        # Update best solution
        best_idx = np.argmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]

        # Reset stagnation counter
        self.stagnation_counter = 0

        return population, f_opt, x_opt
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
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