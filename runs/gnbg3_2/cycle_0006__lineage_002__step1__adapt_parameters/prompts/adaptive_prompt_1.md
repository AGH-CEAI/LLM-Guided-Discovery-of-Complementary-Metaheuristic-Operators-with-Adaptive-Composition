Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_adapt_parameters` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.153200e-07            5.582592e-06            -inf                    -inf                    4.892638e-06            3.506478e+00            1.537245e-04            2.268194e-04            -inf                    -inf                    
1      2.012327e+00            4.031369e+00            -inf                    -inf                    4.471933e+00            3.237445e+01            2.690612e+00            1.642116e+00            -inf                    -inf                    
2      3.758710e-05            5.870203e-03            -inf                    -inf                    4.474758e+00            1.802780e+01            2.298161e-02            3.507873e-02            -inf                    -inf                    
3      2.382116e+00            1.321359e+01            -inf                    -inf                    1.595064e+01            1.333375e+02            1.111003e+01            2.922439e+00            -inf                    -inf                    
4      9.237054e-01            1.201751e+00            -inf                    -inf                    1.216059e+00            2.829618e+00            1.096592e+00            1.302817e+00            -inf                    -inf                    
5      4.955770e+01            2.350817e+03            -inf                    -inf                    4.721569e+03            7.179865e+06            1.932290e+03            1.091726e+01            -inf                    -inf                    
6      2.139748e+02            5.600715e+02            -inf                    -inf                    6.734299e+02            9.264515e+02            6.354242e+02            1.537124e+02            -inf                    -inf                    
7      4.472982e+00            1.156905e+01            -inf                    -inf                    1.572835e+01            1.149325e+02            1.067999e+01            6.311482e+00            -inf                    -inf                    
8      1.799692e+00            3.387906e+00            -inf                    -inf                    4.132959e+00            2.029072e+01            3.540871e+00            1.945829e+00            -inf                    -inf                    
9      3.990353e+01            5.424652e+01            -inf                    -inf                    5.034741e+01            1.183439e+02            5.058423e+01            3.125713e+01            -inf                    -inf                    
10     1.075207e+00            5.298913e+00            -inf                    -inf                    2.626772e+01            2.525642e+02            2.530333e-02            2.643819e-01            -inf                    -inf                    
11     4.010905e+02            3.936512e+02            -inf                    -inf                    5.015824e+02            1.050959e+03            3.357837e+02            3.744852e+02            -inf                    -inf                    
12     2.653441e+00            6.676994e+00            -inf                    -inf                    4.210585e+01            1.873835e+02            1.681565e-01            2.385813e+00            -inf                    -inf                    
13     2.995945e+01            3.127339e+01            -inf                    -inf                    3.654480e+01            5.047443e+01            2.828905e+01            2.830730e+01            -inf                    -inf                    
14     7.457330e+00            3.530668e+00            -inf                    -inf                    4.063595e+00            1.516082e+01            4.244799e+00            3.836813e+00            -inf                    -inf                    
15     3.575172e+00            3.585162e+00            -inf                    -inf                    3.685033e+00            4.040244e+00            3.519463e+00            3.611162e+00            -inf                    -inf                    
16     5.025285e+03            5.872409e+03            -inf                    -inf                    7.707006e+03            1.563736e+04            4.506987e+03            4.633503e+03            -inf                    -inf                    
17     7.983097e+04            7.851891e+04            -inf                    -inf                    1.136559e+05            2.556699e+05            7.358159e+04            9.906005e+04            -inf                    -inf                    
18     5.797739e+01            5.610458e+01            -inf                    -inf                    6.530177e+01            8.977133e+01            5.437528e+01            5.728956e+01            -inf                    -inf                    
19     1.475811e+02            1.242922e+02            -inf                    -inf                    1.430154e+02            3.397254e+02            1.002841e+02            8.887347e+01            -inf                    -inf                    
20     2.066916e+01            2.054784e+01            -inf                    -inf                    2.331358e+01            5.567496e+01            1.974894e+01            2.086585e+01            -inf                    -inf                    
21     5.748109e+00            5.666342e+00            -inf                    -inf                    5.817497e+00            5.985251e+00            5.540064e+00            5.680201e+00            -inf                    -inf                    
22     1.325451e+01            1.282177e+01            -inf                    -inf                    1.361418e+01            1.971701e+01            1.267905e+01            1.318818e+01            -inf                    -inf                    
23     5.172734e+01            4.119517e+01            -inf                    -inf                    4.060983e+01            9.488272e+01            4.469148e+01            4.060961e+01            -inf                    -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.153200e-07)
Task  1: variant_07_catG_idea_0.py  (error=1.642116e+00)
Task  2: original.py  (error=3.758710e-05)
Task  3: original.py  (error=2.382116e+00)
Task  4: original.py  (error=9.237054e-01)
Task  5: variant_07_catG_idea_0.py  (error=1.091726e+01)
Task  6: variant_07_catG_idea_0.py  (error=1.537124e+02)
Task  7: original.py  (error=4.472982e+00)
Task  8: original.py  (error=1.799692e+00)
Task  9: variant_07_catG_idea_0.py  (error=3.125713e+01)
Task 10: variant_06_catF_idea_0.py  (error=2.530333e-02)
Task 11: variant_06_catF_idea_0.py  (error=3.357837e+02)
Task 12: variant_06_catF_idea_0.py  (error=1.681565e-01)
Task 13: variant_06_catF_idea_0.py  (error=2.828905e+01)
Task 14: variant_01_catA_idea_0.py  (error=3.530668e+00)
Task 15: variant_06_catF_idea_0.py  (error=3.519463e+00)
Task 16: variant_06_catF_idea_0.py  (error=4.506987e+03)
Task 17: variant_06_catF_idea_0.py  (error=7.358159e+04)
Task 18: variant_06_catF_idea_0.py  (error=5.437528e+01)
Task 19: variant_07_catG_idea_0.py  (error=8.887347e+01)
Task 20: variant_06_catF_idea_0.py  (error=1.974894e+01)
Task 21: variant_06_catF_idea_0.py  (error=5.540064e+00)
Task 22: variant_06_catF_idea_0.py  (error=1.267905e+01)
Task 23: variant_07_catG_idea_0.py  (error=4.060961e+01)

WIN COUNTS:
  variant_06_catF_idea_0.py: 11 wins
  original.py: 6 wins
  variant_07_catG_idea_0.py: 6 wins
  variant_01_catA_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (1 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using population geometry and spatial structure."""
        # Store population geometry for adaptation
        if not hasattr(self, 'pop_geometry_history'):
            self.pop_geometry_history = {
                'pairwise_means': [],
                'pairwise_stds': [],
                'axis_spreads': [],
                'aspect_ratios': [],
                'centroid_dists': []
            }

        # Get current population from instance attribute (set by _optimize)
        pop = getattr(self, '_current_population', None)
        if pop is None:
            # Fallback: gentle correction if no population available
            self.F = np.clip(self.F * 1.0, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * 1.0, 0.1, 0.9)
            return

        # === Compute geometric features ===
        # 1. Axis-aligned spread (range per dimension)
        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        axis_spreads = pop_max - pop_min
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads) + 1e-10
        aspect_ratio = max_spread / min_spread  # Elongation measure

        # 2. Pairwise distance statistics (subsample for efficiency)
        n_geo = min(20, len(pop))
        indices = np.random.choice(len(pop), n_geo, replace=False)
        subpop = pop[indices]
        diffs = subpop[:, np.newaxis, :] - subpop[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        triu_indices = np.triu_indices(n_geo, k=1)
        pairwise_mean = np.mean(pairwise_dists[triu_indices])
        pairwise_std = np.std(pairwise_dists[triu_indices])

        # 3. Centroid-relative geometry
        centroid = np.mean(pop, axis=0)
        centroid_dists = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = np.mean(centroid_dists)

        # 4. Store history (window of 10)
        history = self.pop_geometry_history
        history['pairwise_means'].append(pairwise_mean)
        history['pairwise_stds'].append(pairwise_std)
        history['axis_spreads'].append(max_spread)
        history['aspect_ratios'].append(aspect_ratio)
        history['centroid_dists'].append(mean_centroid_dist)

        for key in history:
            if len(history[key]) > 10:
                history[key].pop(0)

        # === Compute geometric regime indicators ===
        # Compare current geometry to historical average
        hist_pm = np.mean(history['pairwise_means'])
        hist_as = np.mean(history['axis_spreads'])

        diversity_ratio = pairwise_mean / (hist_pm + 1e-10)
        spread_ratio = max_spread / (hist_as + 1e-10)

        # Detect geometric states
        is_expanded = diversity_ratio > 1.1 or spread_ratio > 1.15
        is_contracted = diversity_ratio < 0.9 or spread_ratio < 0.85
        is_highly_elongated = aspect_ratio > 5.0
        is_near_spherical = aspect_ratio < 2.0

        # === Compute success rate for weighting ===
        success_rate = np.mean(improved_mask)

        # === Adapt F based on geometric diversity ===
        # High diversity → reduce F (focused exploitation)
        # Low diversity → increase F (wider exploration)
        if is_expanded:
            F_adjustment = 0.88  # Contract: reduce step size
        elif is_contracted:
            F_adjustment = 1.12  # Expand: increase step size
        else:
            # Gentle correction based on diversity trend
            delta = diversity_ratio - 1.0
            F_adjustment = 1.0 - 0.15 * delta

        # Success-rate weighting: successful adaptation should stabilize
        if success_rate > 0.3:
            F_adjustment = F_adjustment * 0.95 + 1.0 * 0.05  # Stabilize on success
        elif success_rate < 0.1:
            F_adjustment *= 1.08  # Boost adjustment on failure

        # === Adapt Cr based on geometric shape ===
        # Elongated population → higher Cr (recombine across axes)
        # Spherical population → lower Cr (preserve structure)
        if is_highly_elongated:
            Cr_adjustment = 1.10  # Increase recombination
        elif is_near_spherical:
            Cr_adjustment = 0.92  # Decrease recombination
        else:
            # Modulate by aspect ratio deviation from spherical
            aspect_deviation = (aspect_ratio - 2.0) / 8.0  # Normalize: 2→0, 10→1
            aspect_deviation = np.clip(aspect_deviation, -0.2, 0.2)
            Cr_adjustment = 1.0 + 0.15 * aspect_deviation

        # Success-rate weighting for Cr
        if success_rate > 0.35:
            Cr_adjustment *= 0.97  # Reduce on high success
        elif success_rate < 0.15:
            Cr_adjustment *= 1.05  # Increase exploration on low success

        # === Apply adjustments with clipping ===
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```

# --- From variant_06_catF_idea_0.py (11 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal autocorrelation, rate-of-change, and drift."""
        success_rate = np.mean(improved_mask)

        # Initialize temporal state on first call
        if not hasattr(self, 'success_history'):
            self.success_history = []
            self.F_ewma = self.F
            self.Cr_ewma = self.Cr
            self.stagnation_window = []
            self.stagnation_counter = 0
            self.last_best = np.inf

        # Append to rolling time series
        self.success_history.append(success_rate)
        if len(self.success_history) > 20:
            self.success_history.pop(0)

        # Track improvement for stagnation window
        current_best = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.inf
        improved_this_gen = current_best < self.last_best
        self.last_best = current_best

        self.stagnation_window.append(1 if improved_this_gen else 0)
        if len(self.stagnation_window) > 12:
            self.stagnation_window.pop(0)

        # Compute rate-of-change: derivative-like signal from EMA difference
        alpha_fast = 0.4
        alpha_slow = 0.15
        if not hasattr(self, 'ewma_fast'):
            self.ewma_fast = success_rate
            self.ewma_slow = success_rate
        self.ewma_fast = alpha_fast * success_rate + (1 - alpha_fast) * self.ewma_fast
        self.ewma_slow = alpha_slow * success_rate + (1 - alpha_slow) * self.ewma_slow
        rate_of_change = self.ewma_fast - self.ewma_slow  # Positive = improving, negative = declining

        # Compute autocorrelation at lag-1 using time series
        n_hist = len(self.success_history)
        autocorr = 0.0
        if n_hist >= 4:
            series = np.array(self.success_history)
            mean_series = np.mean(series)
            var_series = np.var(series)
            if var_series > 1e-12:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr = np.clip(autocorr, -1.0, 1.0)

        # Compute temporal drift: EMA-weighted parameter history
        self.F_ewma = 0.7 * self.F_ewma + 0.3 * F_used
        self.Cr_ewma = 0.7 * self.Cr_ewma + 0.3 * Cr_used
        F_drift_ratio = self.F / (self.F_ewma + 1e-12)
        Cr_drift_ratio = self.Cr / (self.Cr_ewma + 1e-12)

        # Stagnation severity from time-window
        stagnation_ratio = np.mean(self.stagnation_window)  # Fraction of generations with improvement

        # Regime detection via temporal signals
        is_oscillating = autocorr > 0.4 and abs(rate_of_change) < 0.05
        is_declining = rate_of_change < -0.02
        is_stagnant = stagnation_ratio < 0.2
        is_converging = rate_of_change > 0.02 and stagnation_ratio > 0.5

        # Temporal adjustment logic
        if is_oscillating:
            # Cyclic behavior detected: inject diversity via F boost
            F_adjustment = 1.25 * np.clip(F_drift_ratio, 0.8, 1.3)
            Cr_adjustment = 0.85
            self.stagnation_counter += 1
        elif is_declining and is_stagnant:
            # Declining with stagnation: strong exploration push
            F_adjustment = 1.15 * (1.1 - stagnation_ratio) / np.clip(F_drift_ratio, 0.8, 1.4)
            Cr_adjustment = 0.88
            self.stagnation_counter += 1
        elif is_stagnant and stagnation_ratio < 0.1:
            # Deep stagnation: large perturbation
            F_adjustment = 0.7
            Cr_adjustment = 1.2
            self.stagnation_counter += 2
        elif is_converging:
            # Steady improvement: fine-tune toward exploitation
            F_adjustment = 0.95 * np.clip(F_drift_ratio, 0.9, 1.1)
            Cr_adjustment = 1.05
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        else:
            # Neutral: gentle correction based on rate
            F_adjustment = 1.0 + 0.1 * rate_of_change
            Cr_adjustment = 1.0 - 0.05 * rate_of_change

        # Apply adjustments
        self.F = np.clip(self.F * F_adjustment, 0.1, 2.0)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.05, 0.95)

        # Decay stagnation counter slowly
        self.stagnation_counter = max(0, self.stagnation_counter - 0.1)
```

# --- From variant_07_catG_idea_0.py (6 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using bootstrap confidence estimation of success rate."""
        n = len(improved_mask)
        success_rate = np.mean(improved_mask)
        n_success = int(success_rate * n)

        # Initialize bootstrap state
        if not hasattr(self, 'bootstrap_means'):
            self.bootstrap_means = []
            self.bootstrap_ci_width = []
            self.sample_buffer = improved_mask.copy().astype(float)
            self.sample_buffer[improved_mask] = 1.0
            self.sample_buffer[~improved_mask] = 0.0

        # Bootstrap resampling to estimate uncertainty in success rate
        n_bootstrap = max(50, 3 * n)  # More bootstrap samples for stability
        bootstrap_success_rates = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Resample with replacement (Monte Carlo sampling)
            indices = np.random.randint(0, n, size=n)
            bootstrap_success_rates[b] = np.mean(self.sample_buffer[indices])

        # Compute statistics from bootstrap distribution
        bootstrap_mean = np.mean(bootstrap_success_rates)
        bootstrap_std = np.std(bootstrap_success_rates)

        # Confidence interval width (proxy for estimation uncertainty)
        ci_lower = np.percentile(bootstrap_success_rates, 5)
        ci_upper = np.percentile(bootstrap_success_rates, 95)
        ci_width = ci_upper - ci_lower

        # Store for adaptive tuning
        self.bootstrap_means.append(bootstrap_mean)
        self.bootstrap_ci_width.append(ci_width)
        if len(self.bootstrap_means) > 20:
            self.bootstrap_means.pop(0)
            self.bootstrap_ci_width.pop(0)

        # Compute uncertainty trend (rising uncertainty = more exploration needed)
        if len(self.bootstrap_means) >= 5:
            recent_uncertainty = np.mean(self.bootstrap_ci_width[-5:])
            long_term_uncertainty = np.mean(self.bootstrap_ci_width)
            uncertainty_trend = recent_uncertainty / (long_term_uncertainty + 1e-10)
        else:
            uncertainty_trend = 1.0

        # Base success rate from bootstrap (more robust than raw)
        robust_success = bootstrap_mean

        # Stochastic perturbation magnitude proportional to uncertainty
        # High CI width → high uncertainty → large perturbations for exploration
        exploration_noise_scale = np.clip(ci_width * 2.0, 0.05, 0.4)

        # Regime detection via bootstrap statistics
        high_success = robust_success > 0.3
        low_success = robust_success < 0.15
        high_uncertainty = ci_width > 0.25
        rising_uncertainty = uncertainty_trend > 1.2

        # Compute stochastic adjustment with uncertainty-driven scaling
        if high_success and not high_uncertainty:
            # Exploitation: high success, low uncertainty → fine-tune toward convergence
            F_adjustment = 1.08
            Cr_adjustment = 1.05
            # Add small stochastic nudge (Monte Carlo perturbation)
            F_noise = np.random.normal(0, 0.02)
            Cr_noise = np.random.normal(0, 0.02)
        elif low_success or high_uncertainty or rising_uncertainty:
            # Exploration: low success OR high uncertainty → inject diversity stochastically
            # Perturbation magnitude scales with bootstrap uncertainty
            F_adjustment = 0.82 - 0.1 * bootstrap_std
            Cr_adjustment = 0.88
            # Large stochastic perturbations proportional to estimation uncertainty
            F_noise = np.random.normal(0, exploration_noise_scale)
            Cr_noise = np.random.normal(0, exploration_noise_scale * 0.5)
        else:
            # Neutral regime: gentle correction with moderate stochasticity
            delta = robust_success - 0.2
            F_adjustment = 1.0 + 0.1 * delta
            Cr_adjustment = 1.0 - 0.05 * delta
            F_noise = np.random.normal(0, exploration_noise_scale * 0.5)
            Cr_noise = np.random.normal(0, exploration_noise_scale * 0.3)

        # Apply adjustments with stochastic perturbation
        self.F = np.clip(self.F * F_adjustment + F_noise, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment + Cr_noise, 0.1, 0.9)

        # Update sample buffer for next iteration
        self.sample_buffer = improved_mask.copy().astype(float)
        self.sample_buffer[improved_mask] = 1.0
        self.sample_buffer[~improved_mask] = 0.0
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

────────────────────────────────────────────────────────────────────────

DECISION GUIDE — choose your mechanism by looking at the data above:

- If ONE variant clearly dominates (>50% of wins) → use mechanism #3
  (rule-based dispatch with that variant as default + fallback).
- If wins are spread roughly evenly across many variants → mechanism #1
  (bandit) or #4 (contextual bandit) is appropriate.
- If variants compute correlated quantities and the differences look
  like rescalings → mechanism #2 (ensemble/voting), NOT a bandit. A
  bandit cannot learn between near-identical arms.
- If there is a clear regime split (variant X wins early, variant Y
  wins late) → mechanism #5 (schedule) or #4 (contextual).
- If the operator is a continuous family → mechanism #6 (self-adaptive
  parameters) is more powerful than picking among discrete snapshots.

REQUIREMENTS:
- Respond with the COMPLETE class code inside a single ```python``` block.
- Include `import numpy as np` at the top.
- At the START of the class docstring, write a SHORT comment naming the
  adaptation mechanism you chose (from the menu above) and 1–2 lines
  explaining why it fits the benchmark results.
- You may add new attributes in __init__ and new private helper methods.
- Do NOT change signatures of existing public methods (__call__, etc.).
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt).
- Handle ALL edge cases: no -inf, no NaN, no crashes, clip to bounds.
- If you use any of mechanisms #1, #4, #5, #8: rewards MUST be rank-based
  or z-scored within a sliding window. NO scale-dependent constants.
- If the variants are highly correlated (the typical case for diversity-
  metric variants), mechanism #2 (ensemble) is strongly preferred.

Original algorithm:
```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    A DE variant with:
    - Ring topology for mutation parent selection
    - Directional memory that biases mutation toward successful directions
    - Adaptive F and Cr based on historical success rates
    - Composite mutation: directional + rand/1 blended adaptively
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Control parameters
        self.F_base = 0.5
        self.Cr_base = 0.3
        self.F = self.F_base
        self.Cr = self.Cr_base
        
        # Directional memory
        self.direction_memory_size = 5
        self.successful_directions = []
        self.direction_decay = 0.95
        
        # Adaptation tracking
        self.adaptation_window = 10
        self.F_success_history = []
        self.Cr_success_history = []
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling for better spread."""
        sampler = np.linspace(self.lower, self.upper, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            indices = np.random.permutation(self.np)
            offsets = np.random.uniform(0, 1, self.np)
            population[:, d] = sampler[indices] + offsets * (sampler[1] - sampler[0])
        
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[np.isnan(fitness)] = np.inf
        
        best_idx = np.argmin(fitness)
        return population, fitness, fitness[best_idx], population[best_idx].copy()
    
    def _clip_to_bounds(self, population):
        """Hard clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        """Create ring neighbor indices for each population member."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid for directional bias."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions for adaptive biasing."""
        if not np.any(successful_mask):
            self.successful_directions = [
                d * self.direction_decay for d in self.successful_directions
            ]
            return
        
        for mutation_vec in successful_mutations[successful_mask]:
            self.successful_directions.append(mutation_vec.copy())
        
        if len(self.successful_directions) > self.direction_memory_size:
            self.successful_directions = self.successful_directions[-self.direction_memory_size:]
    
    def _compute_directional_bias(self, target_indices):
        """Compute directional bias from historical successful mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using composite strategy:
        - Ring-based rand/1 component
        - Directional component from historical memory
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next
        
        # Standard DE rand/1 mutation
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with perturbation based on diversity
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Directional component: bias toward successful regions
        directional_component = directional_bias
        
        # Blend based on adaptation success
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Per-individual Cr with bounded distribution
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        """Evaluate batch, handle budget exhaustion gracefully."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
            """Greedy selection with spectral diversity modulation."""
            improved_mask = trial_fitness < fitness
            improvement = fitness - trial_fitness

            # Compute population covariance eigenvalue spread
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
            # High spread → anisotropic population → reduce selection pressure
            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0

            # Modulation: in ill-conditioned populations, accept smaller improvements
            # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
            diversity_bonus = np.log1p(condition)

            # Scaled improvement relative to population fitness range
            fit_range = np.ptp(fitness)
            fit_range = max(fit_range, 1e-10)
            scaled_improvement = improvement / fit_range

            # Accept if improved OR if scaled improvement exceeds diversity threshold
            accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
            accept_mask = improved_mask | (scaled_improvement > accept_threshold)

            new_population = population.copy()
            new_fitness = fitness.copy()

            new_population[accept_mask] = trials[accept_mask]
            new_fitness[accept_mask] = trial_fitness[accept_mask]

            return new_population, new_fitness, accept_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics and parameter drift detection."""
        success_rate = np.mean(improved_mask)

        # Initialize EMA state on first call
        if not hasattr(self, 'success_ewma_short'):
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        # Update EMAs with different smoothing constants
        alpha_short = 0.3   # Fast EMA for short-term dynamics
        alpha_long = 0.1    # Slow EMA for trend detection
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        # Track parameter history for drift detection
        self.F_history.append(F_used)
        self.Cr_history.append(F_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)

        # Compute parameter drift from historical mean
        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)

        # Regime detection via EMA divergence
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15

        # Compute momentum-adjusted base adjustments
        momentum_factor = np.clip(F_drift, 0.7, 1.4)

        if short_trending_up and self.success_ewma_short > 0.25:
            # Exploitation phase: success accelerating, increase convergence pressure
            F_adjustment = 1.05 * momentum_factor
            Cr_adjustment = 1.08
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            # Exploration phase: success declining, inject diversity
            F_adjustment = 0.88 / momentum_factor
            Cr_adjustment = 0.92
            self.stagnation_counter += 1
        elif both_stagnant:
            # Deep stagnation: large perturbation to escape local traps
            F_adjustment = 0.75
            Cr_adjustment = 1.15
            self.stagnation_counter += 2
        else:
            # Neutral regime: gentle correction toward baseline
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.08 * delta
            Cr_adjustment = 1.0 - 0.05 * delta

        # Apply adjustments with drift correction
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter > self.stagnation_threshold or 
            diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Adapt control parameters
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution

```