Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_update_strategy_scores` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t10_id  variant_02_catB_t09_id  variant_03_catC_t19_id  variant_04_catD_t21_id  variant_05_catE_t14_id  variant_06_catF_t04_id  variant_07_catG_t12_id  variant_08_catH_t12_id  variant_09_catA_t10_id  variant_10_catB_t10_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      4.259914e-08            4.378463e-08            4.696067e-08            -inf                    -inf                    4.618062e-08            4.247145e-08            4.103001e-08            4.388407e-08            4.622550e-08            -inf                    
1      1.786280e-05            3.863778e-05            3.990669e-05            -inf                    -inf                    5.182391e-05            3.321315e-06            2.766272e-05            5.530550e-06            3.563289e-05            -inf                    
2      1.000000e-08            1.000000e-08            1.000000e-08            -inf                    -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
3      1.378117e+00            1.965793e+00            1.692076e+00            -inf                    -inf                    1.577421e+00            1.077880e+00            1.831244e+00            1.152365e+00            2.126471e+00            -inf                    
4      1.999998e-08            1.687626e-02            3.999991e-08            -inf                    -inf                    1.632639e-02            1.999998e-08            1.999999e-08            1.999998e-08            1.734497e-02            -inf                    
5      4.452708e-05            3.312927e-04            2.273889e-04            -inf                    -inf                    2.264357e-04            7.958341e-05            3.986645e-04            1.191883e-05            1.037008e-04            -inf                    
6      4.173817e+01            3.909579e+01            4.231173e+01            -inf                    -inf                    3.882764e+01            3.974093e+01            4.228261e+01            4.358166e+01            3.858292e+01            -inf                    
7      5.894093e-02            8.887189e-02            7.617608e-02            -inf                    -inf                    1.283490e-01            4.092578e-02            6.989026e-02            7.088215e-02            7.340728e-02            -inf                    
8      3.545024e-02            7.304891e-02            5.843355e-02            -inf                    -inf                    4.961727e-02            2.796747e-02            5.330856e-02            2.449280e-02            6.439787e-02            -inf                    
9      9.404178e+00            9.202650e+00            9.458778e+00            -inf                    -inf                    9.274912e+00            8.869823e+00            9.520867e+00            9.492886e+00            9.254817e+00            -inf                    
10     1.870306e-05            1.136242e-04            2.160315e-04            -inf                    -inf                    2.363108e-04            1.031811e-05            2.603935e-04            8.349811e-07            1.278741e-04            -inf                    
11     2.729658e+01            1.631887e+02            1.864661e+02            -inf                    -inf                    1.752609e+02            6.783730e+01            1.179647e+02            7.764800e+01            1.173697e+02            -inf                    
12     1.188863e-01            2.670494e+01            4.382402e+01            -inf                    -inf                    1.591996e+01            9.034829e+00            1.113262e+01            1.046585e+00            1.105654e+01            -inf                    
13     2.449321e+00            3.141570e+00            4.122282e+00            -inf                    -inf                    3.544424e+00            2.706263e+00            4.103388e+00            3.518920e+00            3.172991e+00            -inf                    
14     8.226955e+00            8.838532e+00            8.894506e+00            -inf                    -inf                    7.689910e+00            7.845887e+00            8.718582e+00            8.096381e+00            6.541267e+00            -inf                    
15     3.055214e+00            3.216238e+00            3.257215e+00            -inf                    -inf                    3.171934e+00            2.999507e+00            3.248321e+00            3.001914e+00            3.097638e+00            -inf                    
16     1.018866e+03            7.481225e+02            7.225381e+02            -inf                    -inf                    8.233910e+02            6.610267e+02            8.953838e+02            5.060266e+02            9.809777e+02            -inf                    
17     4.467852e+03            6.736897e+03            3.851536e+03            -inf                    -inf                    7.158030e+03            1.609455e+03            3.528688e+03            6.083630e+03            1.163127e+04            -inf                    
18     3.212187e+01            3.079874e+01            3.078671e+01            -inf                    -inf                    3.078999e+01            2.850791e+01            3.227384e+01            3.293457e+01            3.562046e+01            -inf                    
19     1.356591e+02            1.546561e+02            1.381714e+02            -inf                    -inf                    1.491214e+02            1.570916e+02            1.271641e+02            1.484078e+02            1.431592e+02            -inf                    
20     2.391089e+01            2.361698e+01            2.136937e+01            -inf                    -inf                    2.283857e+01            2.313108e+01            2.447027e+01            2.520810e+01            2.372378e+01            -inf                    
21     5.288898e+00            5.176453e+00            5.274411e+00            -inf                    -inf                    5.292242e+00            5.144343e+00            5.178482e+00            5.206457e+00            5.212423e+00            -inf                    
22     1.348507e+01            1.208999e+01            1.246482e+01            -inf                    -inf                    1.385562e+01            1.282307e+01            1.329894e+01            1.312866e+01            1.398436e+01            -inf                    
23     3.652570e+01            4.254232e+01            3.937194e+01            -inf                    -inf                    3.732278e+01            4.085552e+01            3.915555e+01            3.570663e+01            4.329923e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_06_catF_t04_idea_0.py  (error=3.321315e-06)
Task  2: SKIPPED (trivial)
Task  3: variant_06_catF_t04_idea_0.py  (error=1.077880e+00)
Task  4: SKIPPED (trivial)
Task  5: variant_08_catH_t12_idea_0.py  (error=1.191883e-05)
Task  6: variant_09_catA_t10_idea_0.py  (error=3.858292e+01)
Task  7: variant_06_catF_t04_idea_0.py  (error=4.092578e-02)
Task  8: variant_08_catH_t12_idea_0.py  (error=2.449280e-02)
Task  9: variant_06_catF_t04_idea_0.py  (error=8.869823e+00)
Task 10: variant_08_catH_t12_idea_0.py  (error=8.349811e-07)
Task 11: original.py  (error=2.729658e+01)
Task 12: original.py  (error=1.188863e-01)
Task 13: original.py  (error=2.449321e+00)
Task 14: variant_09_catA_t10_idea_0.py  (error=6.541267e+00)
Task 15: variant_06_catF_t04_idea_0.py  (error=2.999507e+00)
Task 16: variant_08_catH_t12_idea_0.py  (error=5.060266e+02)
Task 17: variant_06_catF_t04_idea_0.py  (error=1.609455e+03)
Task 18: variant_06_catF_t04_idea_0.py  (error=2.850791e+01)
Task 19: variant_07_catG_t12_idea_0.py  (error=1.271641e+02)
Task 20: variant_02_catB_t09_idea_0.py  (error=2.136937e+01)
Task 21: variant_06_catF_t04_idea_0.py  (error=5.144343e+00)
Task 22: variant_01_catA_t10_idea_0.py  (error=1.208999e+01)
Task 23: variant_08_catH_t12_idea_0.py  (error=3.570663e+01)

WIN COUNTS:
  variant_06_catF_t04_idea_0.py: 8 wins
  variant_08_catH_t12_idea_0.py: 5 wins
  original.py: 3 wins
  variant_09_catA_t10_idea_0.py: 2 wins
  variant_07_catG_t12_idea_0.py: 1 wins
  variant_02_catB_t09_idea_0.py: 1 wins
  variant_01_catA_t10_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     1   variant_06_catF_t04_idea_0.py   3.3213e-06       variant_08_catH_t12_idea_0.py    5.5306e-06         1.7×   10.7×
     3   variant_06_catF_t04_idea_0.py   1.0779e+00       variant_08_catH_t12_idea_0.py    1.1524e+00         1.1×    1.6×
     5   variant_08_catH_t12_idea_0.py   1.1919e-05       original.py                      4.4527e-05         3.7×   19.0×
     6   variant_09_catA_t10_idea_0.py   3.8583e+01       variant_05_catE_t14_idea_0.py    3.8828e+01         1.0×    1.1×
     7   variant_06_catF_t04_idea_0.py   4.0926e-02       original.py                      5.8941e-02         1.4×    1.8×
     8   variant_08_catH_t12_idea_0.py   2.4493e-02       variant_06_catF_t04_idea_0.py    2.7967e-02         1.1×    2.2×
     9   variant_06_catF_t04_idea_0.py   8.8698e+00       variant_01_catA_t10_idea_0.py    9.2026e+00         1.0×    1.1×
    10   variant_08_catH_t12_idea_0.py   8.3498e-07       variant_06_catF_t04_idea_0.py    1.0318e-05        12.4×  153.1×
    11   original.py                     2.7297e+01       variant_06_catF_t04_idea_0.py    6.7837e+01         2.5×    4.3×
    12   original.py                     1.1889e-01       variant_08_catH_t12_idea_0.py    1.0466e+00         8.8×   93.6×
    13   original.py                     2.4493e+00       variant_06_catF_t04_idea_0.py    2.7063e+00         1.1×    1.4×
    14   variant_09_catA_t10_idea_0.py   6.5413e+00       variant_05_catE_t14_idea_0.py    7.6899e+00         1.2×    1.3×
    15   variant_06_catF_t04_idea_0.py   2.9995e+00       variant_08_catH_t12_idea_0.py    3.0019e+00         1.0×    1.1×
    16   variant_08_catH_t12_idea_0.py   5.0603e+02       variant_06_catF_t04_idea_0.py    6.6103e+02         1.3×    1.6×
    17   variant_06_catF_t04_idea_0.py   1.6095e+03       variant_07_catG_t12_idea_0.py    3.5287e+03         2.2×    3.8×
    18   variant_06_catF_t04_idea_0.py   2.8508e+01       variant_02_catB_t09_idea_0.py    3.0787e+01         1.1×    1.1×
    19   variant_07_catG_t12_idea_0.py   1.2716e+02       original.py                      1.3566e+02         1.1×    1.2×
    20   variant_02_catB_t09_idea_0.py   2.1369e+01       variant_05_catE_t14_idea_0.py    2.2839e+01         1.1×    1.1×
    21   variant_06_catF_t04_idea_0.py   5.1443e+00       variant_01_catA_t10_idea_0.py    5.1765e+00         1.0×    1.0×
    22   variant_01_catA_t10_idea_0.py   1.2090e+01       variant_02_catB_t09_idea_0.py    1.2465e+01         1.0×    1.1×
    23   variant_08_catH_t12_idea_0.py   3.5707e+01       original.py                      3.6526e+01         1.0×    1.1×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 21
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_t10_idea_0.py, variant_02_catB_t09_idea_0.py, variant_06_catF_t04_idea_0.py, variant_07_catG_t12_idea_0.py, variant_08_catH_t12_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 2  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 10 — variant_08_catH_t12_idea_0.py (8.35e-07) vs variant_06_catF_t04_idea_0.py (1.03e-05) = 12.4× gap to runner-up, 153.1× gap to median competitor. A 50/50 blend with the runner-up would yield ~5.58e-06, which is ~6.7× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_t10_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Credit assignment using geometric displacement: centroid shift and hull volume change."""
        # Access population from instance (set by __call__ before this method)
        if not hasattr(self, 'population') or self.population is None:
            return

        # Initialize geometric state tracking on first call
        if not hasattr(self, '_prev_centroid'):
            self._prev_centroid = None
            self._prev_hull_vol = None

        # Compute current population geometry
        centroid = self.population.mean(axis=0)
        pop_range = self.population.max(axis=0) - self.population.min(axis=0) + 1e-10
        hull_proxy = np.prod(pop_range)

        # Compute geometric change from previous generation
        if self._prev_centroid is not None and self._prev_hull_vol is not None:
            centroid_shift = np.linalg.norm(centroid - self._prev_centroid)
            hull_change_ratio = hull_proxy / (self._prev_hull_vol + 1e-10)
        else:
            centroid_shift = 0.0
            hull_change_ratio = 1.0

        # Population spread for normalization
        pop_spread = np.linalg.norm(pop_range)
        pop_spread = max(pop_spread, 1e-10)

        # Credit assignment per strategy using geometric signals
        for s in range(len(self.strategy_scores)):
            mask = strategy_used == s
            if not np.any(mask):
                continue

            improved_idx = np.where(mask & improved)[0]
            n_improved = len(improved_idx)

            if n_improved == 0:
                # No improvements: geometric penalty based on hull contraction
                if hull_change_ratio < 1.0:
                    self.strategy_scores[s] *= (0.95 + 0.05 * hull_change_ratio)
                continue

            # Geometric displacement of improved solutions from centroid
            improved_centroid = self.population[improved_idx].mean(axis=0)
            displacement = np.linalg.norm(improved_centroid - centroid)
            norm_displacement = displacement / pop_spread

            # Credit = normalized displacement * sqrt(success count) * hull expansion factor
            # sqrt(n_improved) balances between having many small wins vs few large jumps
            credit = norm_displacement * np.sqrt(n_improved) * max(hull_change_ratio, 0.5)

            # Exponential moving average update with momentum
            momentum = 0.1
            self.strategy_scores[s] = (1 - momentum) * self.strategy_scores[s] + momentum * credit

        # Prevent pathological score values
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)

        # Store current geometric state for next iteration
        self._prev_centroid = centroid
        self._prev_hull_vol = hull_proxy
```

# --- From variant_02_catB_t09_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Credit assignment using spectral analysis of population structure.

        Category B: Spectral / linear-algebraic
        Uses SVD-based metrics (effective rank, condition number) to assess
        how each strategy affects population subspace structure and diversity.
        """
        import numpy as np

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

        self.strategy_scores = np.clip(self.strategy_scores, -50.0, 50.0)
        scores_exp = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs = scores_exp / (scores_exp.sum() + 1e-12)
```

# --- From variant_06_catF_t04_idea_0.py (8 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Temporal strategy credit with EMA, momentum, and trend detection."""
        n_strategies = len(self.strategy_scores)

        # Initialize temporal state if first call
        if not hasattr(self, '_ema_rates'):
            self._ema_rates = np.zeros(n_strategies)
            self._prev_ema_rates = np.zeros(n_strategies)
            self._strategy_momentum = np.zeros(n_strategies)
            self._history_buffer = [np.zeros(n_strategies) for _ in range(5)]
            self._history_idx = 0

        # Compute per-strategy improvement rates for this generation
        current_rates = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.any():
                current_rates[s] = improved[mask].sum() / self.NP

        # Store in circular buffer for trend analysis
        self._history_buffer[self._history_idx] = current_rates.copy()
        self._history_idx = (self._history_idx + 1) % len(self._history_buffer)

        # Exponential moving average (alpha=0.3, favor recent data)
        alpha = 0.3
        self._prev_ema_rates = self._ema_rates.copy()
        self._ema_rates = alpha * current_rates + (1 - alpha) * self._ema_rates

        # Trend detection: compare current EMA to previous
        # Positive trend = improving, negative = declining
        trends = self._ema_rates - self._prev_ema_rates

        # Per-strategy momentum: weighted combination of current and trend
        # Momentum decays if strategy not used recently
        beta = 0.7  # momentum retention
        gamma = 0.15  # trend weight
        self._strategy_momentum = (
            beta * self._strategy_momentum +
            (1 - beta) * (self._ema_rates + gamma * np.sign(trends) * np.sqrt(np.abs(trends)))
        )

        # Compute adaptive learning rate per strategy (slower for consistent, faster for volatile)
        recent_variance = np.var(self._history_buffer, axis=0) + 1e-10
        adaptive_lr = 0.15 / (1.0 + 5.0 * recent_variance)

        # Update scores: EMA rate drives credit, momentum adds stability bonus
        score_delta = adaptive_lr * self._ema_rates + 0.05 * self._strategy_momentum

        # Penalize strategies showing sustained decline (negative trend)
        for s in range(n_strategies):
            recent_trend = sum(
                self._history_buffer[(self._history_idx - i) % len(self._history_buffer)][s]
                for i in range(1, 4)
            )
            if recent_trend < 0.02:  # Consistently low performance
                score_delta[s] -= 0.08

        # Apply update with momentum
        self.strategy_scores += score_delta

        # Normalize to prevent score divergence
        self.strategy_scores -= self.strategy_scores.mean()
        self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
```

# --- From variant_07_catG_t12_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Credit assignment using Monte Carlo bootstrap + Thompson Sampling.

        Category G: Fundamentally stochastic approach using bootstrap resampling
        to estimate confidence intervals, with Thompson Sampling for exploration.
        Credit is weighted by improvement magnitude for task 12's difficulty.
        """
        n_strategies = len(self.strategy_scores)
        n_trials = len(strategy_used)

        if n_trials == 0:
            return

        # Initialize history tracking
        if not hasattr(self, '_strategy_history'):
            self._strategy_history = [[] for _ in range(n_strategies)]
            self._strategy_successes = np.zeros(n_strategies)
            self._strategy_attempts = np.zeros(n_strategies)

        # Record history and update counts
        for i in range(n_trials):
            s = strategy_used[i]
            imp = improved[i]
            # Magnitude weighting: track both success and improvement amount
            mag = abs(improved[i]) if isinstance(improved[i], (int, float, np.number)) and imp else 0.0
            self._strategy_history[s].append((1 if imp else 0, mag))
            self._strategy_attempts[s] += 1
            if imp:
                self._strategy_successes[s] += 1

        # Rotate history to prevent unbounded growth
        for s in range(n_strategies):
            if len(self._strategy_history[s]) > 500:
                self._strategy_history[s] = self._strategy_history[s][-400:]

        # Bootstrap resampling for confidence intervals
        n_bootstrap = 200
        bootstrap_rates = np.zeros((n_bootstrap, n_strategies))

        for b in range(n_bootstrap):
            for s in range(n_strategies):
                hist = self._strategy_history[s]
                if len(hist) > 0:
                    # Sample with replacement
                    indices = np.random.randint(0, len(hist), size=len(hist))
                    sampled = [hist[i] for i in indices]
                    # Weighted success rate: success * magnitude
                    bootstrap_rates[b, s] = np.mean([s_[0] * (1 + s_[1]) for s_ in sampled])
                else:
                    bootstrap_rates[b, s] = 0.0

        # Compute confidence intervals
        ci_lower = np.percentile(bootstrap_rates, 10, axis=0)
        ci_upper = np.percentile(bootstrap_rates, 90, axis=0)
        mean_rates = np.mean(bootstrap_rates, axis=0)

        # Thompson Sampling: sample from bootstrap, add confidence-weighted exploration
        sampled_scores = np.zeros(n_strategies)
        for s in range(n_strategies):
            sampled_idx = np.random.randint(n_bootstrap)
            sampled_scores[s] = bootstrap_rates[sampled_idx, s]
            # Exploration bonus inversely proportional to confidence (CI width)
            ci_width = max(ci_upper[s] - ci_lower[s], 1e-10)
            exploration = np.random.exponential(0.05 / ci_width)
            sampled_scores[s] += exploration

        # Softmax normalization
        exp_scores = np.exp(sampled_scores - sampled_scores.max())
        probs = exp_scores / (exp_scores.sum() + 1e-10)

        # Update strategy scores with momentum
        momentum = 0.7
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * probs
```

# --- From variant_08_catH_t12_idea_0.py (5 wins) ---
```python
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
```

# --- From variant_09_catA_t10_idea_0.py (2 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Credit assignment based on geometric alignment of mutation directions."""
        if len(improved) == 0 or self.NP < 2:
            return

        NP = self.NP

        # Get current population from optimizer state
        pop = getattr(self, '_current_population', None)
        if pop is None or len(pop) < 5:
            return

        # Compute population geometric properties
        centroid = pop.mean(axis=0)
        pop_spread = np.std(pop, axis=0).mean()

        # Compute mutation direction toward centroid
        to_centroid = centroid - pop

        # For each strategy, compute directional alignment and spread contribution
        strategy_counts = np.bincount(strategy_used, minlength=len(self.strategy_scores))

        for s_idx in range(len(self.strategy_scores)):
            mask = strategy_used == s_idx
            n_s = mask.sum()

            if n_s < 1:
                continue

            # Get individuals using this strategy
            s_improved = improved[mask]
            s_to_centroid = to_centroid[mask]

            if s_improved.sum() == 0:
                # No improvements: small penalty
                self.strategy_scores[s_idx] *= 0.95
                continue

            # Compute centroid of improved solutions using this strategy
            s_pop_improved = pop[mask][s_improved]
            improved_centroid = s_pop_improved.mean(axis=0)

            # Direction from population to improved centroid
            dir_to_improved = improved_centroid - centroid

            # Directional alignment: cosine similarity between individual directions and global direction
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

            # Spatial spread contribution: how much does this strategy expand population coverage
            s_pop_all = pop[mask]
            s_spread = np.std(s_pop_all, axis=0).mean()
            spread_ratio = s_spread / max(pop_spread, 1e-10)

            # Combined geometric score
            geo_score = 0.6 * (alignment_score + 1) / 2 + 0.4 * np.clip(spread_ratio, 0.5, 2.0)

            # Improvement rate for this strategy
            improvement_rate = s_improved.mean()

            # Weighted update
            weight = improvement_rate * 0.3 + 0.1
            self.strategy_scores[s_idx] += weight * (geo_score - 0.5)

        # Dampen scores to prevent runaway
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
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
        """Credit assignment: which strategies produced improvements."""
        for s in range(len(self.strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        
        # Decay all scores slightly
        self.strategy_scores *= 0.99
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
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
        """Full restart with diversity injection."""
        # Keep best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Reinitialize rest
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        
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