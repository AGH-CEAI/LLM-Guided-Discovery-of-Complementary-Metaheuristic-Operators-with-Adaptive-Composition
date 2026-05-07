Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.151470e+01            8.116406e+01            7.824869e+01            3.076110e+01            -inf                    2.239777e+01            1.106250e+01            1.344053e+02            5.729600e+00            9.991673e+00            9.639481e+00            
1      3.929690e+01            2.026058e+02            2.109309e+02            7.398702e+01            -inf                    5.876665e+01            4.182899e+01            2.625984e+02            2.979164e+01            3.794950e+01            2.917541e+00            
2      1.253804e+02            4.831875e+02            5.740794e+02            3.149743e+02            -inf                    2.368895e+02            1.448840e+02            1.055873e+03            8.052139e+01            1.330787e+02            1.426079e+02            
3      1.138889e+02            2.702158e+02            3.238332e+02            1.765316e+02            -inf                    1.696555e+02            1.211613e+02            4.162328e+02            1.117107e+02            1.118297e+02            6.843786e+00            
4      3.617389e+00            5.886524e+00            5.781934e+00            4.408708e+00            -inf                    4.128672e+00            3.679493e+00            6.356908e+00            3.324825e+00            3.559994e+00            2.804566e+00            
5      3.947106e+07            6.052457e+08            7.262474e+09            2.558893e+08            -inf                    1.356933e+08            5.512215e+07            1.607854e+10            1.899190e+07            4.224190e+07            9.844950e+01            
6      2.349123e+03            1.128270e+03            5.742430e+03            3.909540e+03            -inf                    3.026733e+03            2.398983e+03            1.505652e+04            1.690764e+03            2.251765e+03            1.317773e+03            
7      1.195920e+02            2.863144e+02            3.768255e+02            1.968799e+02            -inf                    1.776261e+02            1.251018e+02            5.983964e+02            9.972201e+01            1.221924e+02            8.312018e+00            
8      2.423080e+01            4.419238e+01            5.186872e+01            3.380739e+01            -inf                    3.073009e+01            2.468075e+01            6.448222e+01            2.065120e+01            2.371444e+01            1.176810e+01            
9      2.039170e+02            1.209072e+02            2.754378e+02            2.555656e+02            -inf                    2.376543e+02            2.080534e+02            5.477397e+02            2.019871e+02            2.044679e+02            1.567469e+02            
10     1.389075e+02            4.456150e+02            5.611291e+02            1.999154e+02            -inf                    1.525376e+02            1.411093e+02            5.386157e+02            1.418304e+02            1.320135e+02            7.889535e+01            
11     6.704208e+02            1.722198e+03            2.232642e+03            8.370596e+02            -inf                    6.381209e+02            6.545767e+02            2.006884e+03            6.619365e+02            6.640069e+02            2.596490e+02            
12     1.196735e+02            3.023776e+02            3.634566e+02            1.501166e+02            -inf                    1.345454e+02            1.280221e+02            3.314052e+02            1.258824e+02            1.171030e+02            1.095714e+02            
13     3.257742e+01            6.209190e+01            6.733524e+01            3.847791e+01            -inf                    3.449608e+01            3.314440e+01            6.446052e+01            3.275134e+01            3.294125e+01            2.382198e+01            
14     1.426576e+01            2.087984e+01            1.828381e+01            1.564269e+01            -inf                    1.532340e+01            1.450488e+01            2.171154e+01            1.467444e+01            1.417381e+01            1.273579e+01            
15     3.749129e+00            4.320389e+00            4.527882e+00            3.892203e+00            -inf                    3.747861e+00            3.764032e+00            4.461469e+00            3.722434e+00            3.782856e+00            3.860743e+00            
16     7.857622e+03            4.399434e+04            2.780196e+04            1.213926e+04            -inf                    8.517281e+03            7.608117e+03            4.504738e+04            8.112142e+03            7.497499e+03            5.401070e+03            
17     8.931846e+04            4.253581e+05            4.760870e+05            1.277549e+05            -inf                    9.660855e+04            8.653014e+04            4.076480e+05            8.740932e+04            8.540669e+04            6.764446e+04            
18     6.168288e+01            1.073695e+02            1.111429e+02            6.869812e+01            -inf                    6.238206e+01            6.060494e+01            1.057871e+02            5.942158e+01            6.082028e+01            5.351889e+01            
19     2.842327e+02            5.486981e+02            5.296277e+02            3.202940e+02            -inf                    3.189413e+02            2.780852e+02            6.226903e+02            2.980826e+02            2.824181e+02            2.309333e+02            
20     4.028307e+01            7.121166e+01            7.736562e+01            4.759933e+01            -inf                    4.517510e+01            4.191714e+01            7.873370e+01            4.142171e+01            4.022921e+01            3.724459e+01            
21     5.649571e+00            6.132470e+00            6.058979e+00            5.725181e+00            -inf                    5.689563e+00            5.650321e+00            6.064833e+00            5.654350e+00            5.655403e+00            5.667701e+00            
22     1.677934e+01            2.307136e+01            2.355150e+01            1.777533e+01            -inf                    1.785613e+01            1.690338e+01            2.443262e+01            1.725791e+01            1.693957e+01            1.536783e+01            
23     9.836847e+01            1.576446e+02            1.472637e+02            1.047104e+02            -inf                    1.024639e+02            9.298738e+01            1.703795e+02            9.944839e+01            9.848150e+01            7.888192e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_08_catH_idea_0.py  (error=5.729600e+00)
Task  1: variant_10_catB_idea_0.py  (error=2.917541e+00)
Task  2: variant_08_catH_idea_0.py  (error=8.052139e+01)
Task  3: variant_10_catB_idea_0.py  (error=6.843786e+00)
Task  4: variant_10_catB_idea_0.py  (error=2.804566e+00)
Task  5: variant_10_catB_idea_0.py  (error=9.844950e+01)
Task  6: variant_01_catA_idea_0.py  (error=1.128270e+03)
Task  7: variant_10_catB_idea_0.py  (error=8.312018e+00)
Task  8: variant_10_catB_idea_0.py  (error=1.176810e+01)
Task  9: variant_01_catA_idea_0.py  (error=1.209072e+02)
Task 10: variant_10_catB_idea_0.py  (error=7.889535e+01)
Task 11: variant_10_catB_idea_0.py  (error=2.596490e+02)
Task 12: variant_10_catB_idea_0.py  (error=1.095714e+02)
Task 13: variant_10_catB_idea_0.py  (error=2.382198e+01)
Task 14: variant_10_catB_idea_0.py  (error=1.273579e+01)
Task 15: variant_08_catH_idea_0.py  (error=3.722434e+00)
Task 16: variant_10_catB_idea_0.py  (error=5.401070e+03)
Task 17: variant_10_catB_idea_0.py  (error=6.764446e+04)
Task 18: variant_10_catB_idea_0.py  (error=5.351889e+01)
Task 19: variant_10_catB_idea_0.py  (error=2.309333e+02)
Task 20: variant_10_catB_idea_0.py  (error=3.724459e+01)
Task 21: original.py  (error=5.649571e+00)
Task 22: variant_10_catB_idea_0.py  (error=1.536783e+01)
Task 23: variant_10_catB_idea_0.py  (error=7.888192e+01)

WIN COUNTS:
  variant_10_catB_idea_0.py: 18 wins
  variant_08_catH_idea_0.py: 3 wins
  variant_01_catA_idea_0.py: 2 wins
  original.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_08_catH_idea_0.py       5.7296e+00       variant_10_catB_idea_0.py        9.6395e+00         1.7×    3.9×
     1   variant_10_catB_idea_0.py       2.9175e+00       variant_08_catH_idea_0.py        2.9792e+01        10.2×   20.1×
     2   variant_08_catH_idea_0.py       8.0521e+01       original.py                      1.2538e+02         1.6×    2.9×
     3   variant_10_catB_idea_0.py       6.8438e+00       variant_08_catH_idea_0.py        1.1171e+02        16.3×   24.8×
     4   variant_10_catB_idea_0.py       2.8046e+00       variant_08_catH_idea_0.py        3.3248e+00         1.2×    1.5×
     5   variant_10_catB_idea_0.py       9.8449e+01       variant_08_catH_idea_0.py        1.8992e+07       192910×    >1e6×
     6   variant_01_catA_idea_0.py       1.1283e+03       variant_10_catB_idea_0.py        1.3178e+03         1.2×    2.1×
     7   variant_10_catB_idea_0.py       8.3120e+00       variant_08_catH_idea_0.py        9.9722e+01        12.0×   21.4×
     8   variant_10_catB_idea_0.py       1.1768e+01       variant_08_catH_idea_0.py        2.0651e+01         1.8×    2.6×
     9   variant_01_catA_idea_0.py       1.2091e+02       variant_10_catB_idea_0.py        1.5675e+02         1.3×    1.7×
    10   variant_10_catB_idea_0.py       7.8895e+01       variant_09_catA_idea_0.py        1.3201e+02         1.7×    1.9×
    11   variant_10_catB_idea_0.py       2.5965e+02       variant_05_catE_idea_0.py        6.3812e+02         2.5×    2.6×
    12   variant_10_catB_idea_0.py       1.0957e+02       variant_09_catA_idea_0.py        1.1710e+02         1.1×    1.2×
    13   variant_10_catB_idea_0.py       2.3822e+01       original.py                      3.2577e+01         1.4×    1.4×
    14   variant_10_catB_idea_0.py       1.2736e+01       variant_09_catA_idea_0.py        1.4174e+01         1.1×    1.2×
    15   variant_08_catH_idea_0.py       3.7224e+00       variant_05_catE_idea_0.py        3.7479e+00         1.0×    1.0×
    16   variant_10_catB_idea_0.py       5.4011e+03       variant_09_catA_idea_0.py        7.4975e+03         1.4×    1.6×
    17   variant_10_catB_idea_0.py       6.7644e+04       variant_09_catA_idea_0.py        8.5407e+04         1.3×    1.4×
    18   variant_10_catB_idea_0.py       5.3519e+01       variant_08_catH_idea_0.py        5.9422e+01         1.1×    1.2×
    19   variant_10_catB_idea_0.py       2.3093e+02       variant_06_catF_idea_0.py        2.7809e+02         1.2×    1.4×
    20   variant_10_catB_idea_0.py       3.7245e+01       variant_09_catA_idea_0.py        4.0229e+01         1.1×    1.2×
    21   original.py                     5.6496e+00       variant_06_catF_idea_0.py        5.6503e+00         1.0×    1.0×
    22   variant_10_catB_idea_0.py       1.5368e+01       original.py                      1.6779e+01         1.1×    1.2×
    23   variant_10_catB_idea_0.py       7.8882e+01       variant_06_catF_idea_0.py        9.2987e+01         1.2×    1.3×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 4 (original.py, variant_01_catA_idea_0.py, variant_08_catH_idea_0.py, variant_10_catB_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 4  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 1  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — variant_10_catB_idea_0.py (9.84e+01) vs variant_08_catH_idea_0.py (1.90e+07) = 192910.1× gap to runner-up, 1378303.2× gap to median competitor. A 50/50 blend with the runner-up would yield ~9.50e+06, which is ~96455.5× WORSE than the winner alone.

STRONGLY RECOMMENDED MECHANISM: per-task fingerprint-and-commit (PROBE-AND-COMMIT pattern below) or mechanism #4 (contextual bandit). Mechanism #2 (ensemble/blending) is DISCOURAGED: there are 4 tasks where the winner beats the runner-up by >=10× and ensemble blending will degrade those wins.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (2 wins) ---
```python
def _position_update_batch(self):
        """Update positions using geometric normalization and centroid guidance."""
        # Compute population centroid
        centroid = np.mean(self.population, axis=0)

        # Compute axis-aligned spread (range per dimension)
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        spread = pop_max - pop_min

        # Normalize velocity by spread (prevent extreme steps in narrow dimensions)
        epsilon = 1e-10
        spread_safe = np.where(spread > epsilon, spread, 1.0)
        normalized_velocity = self.velocity / spread_safe

        # Compute centroid-relative distance for each particle
        centroid_dist = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        global_dist = np.linalg.norm(centroid) + epsilon

        # Direction toward centroid (negative of position offset)
        direction_to_centroid = centroid - self.population

        # Geometric correction strength: stronger when spread is large (encourage convergence)
        correction_strength = 0.02 * np.clip(spread.mean() / (global_dist + epsilon), 0.01, 0.1)

        # Apply geometric correction toward centroid
        correction = correction_strength * direction_to_centroid

        # Update positions with normalized velocity and geometric correction
        new_population = self.population + normalized_velocity + correction
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_08_catH_idea_0.py (3 wins) ---
```python
def _position_update_batch(self):
        """Hybrid position update: velocity + stagnation-weighted global-best attraction."""
        # Compute stagnation factor (0 = no stagnation, 1 = severe stagnation)
        stagnation_threshold = 50 + self.dim // 2
        stagnation_factor = min(1.0, self.stagnation_counter / stagnation_threshold)

        # Velocity-based movement (primary mechanism)
        velocity_component = self.velocity

        # Global-best attraction (secondary mechanism, activates when stagnant)
        if self.global_best is not None and stagnation_factor > 0.1:
            # Direction toward global best, scaled by stagnation severity
            direction_to_best = self.global_best - self.population
            best_pull = stagnation_factor * 0.5 * direction_to_best
        else:
            best_pull = 0.0

        # Hybrid update: blend velocity with global-best pull
        new_population = self.population + velocity_component + best_pull
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_10_catB_idea_0.py (18 wins) ---
```python
def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
        # Compute population covariance for spectral analysis
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)

        # Eigendecomposition for spectral scaling
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute condition-number-based scaling (stuck populations have high condition number)
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)

            # Project velocity onto principal axes and scale
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale

            # Transform back to original space
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            # Fallback to standard update on decomposition failure
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
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


class AdaptiveTopologyParticleSwarm:
    """
    Adaptive Topology Particle Swarm Optimizer with DE Mutation (ATPSO-DM).
    
    Combines:
    - Ring topology PSO with adaptive neighborhood size
    - Adaptive inertia weight based on swarm diversity
    - DE/rand/1 mutation injection for exploration
    - Adaptive cognitive/social coefficients
    - Diversity-guided restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        range_width = self.upper_bound - self.lower_bound
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
        
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        # Handle budget exhaustion gracefully
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            # Ring neighborhood indices
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            # Include self in neighborhood comparison
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on swarm diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Low diversity: increase exploration
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            # High diversity: increase exploitation
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            # Normal range: gradual convergence
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Increase neighborhood size for more diversity
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            # Decrease neighborhood size for faster convergence
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        # Adapt cognitive coefficient based on personal best improvement
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        
        # Adapt social coefficient inversely to cognitive
        social = self.social_base * (2.0 - improvement_ratio)
        
        return cognitive, social
    
    def _velocity_update_batch(self):
        """Update velocities with adaptive components and DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)
        
        # DE/rand/1 mutation injection for enhanced exploration
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)  # Decreases over time
        mutation_active = mutation_mask < mutation_threshold
        
        # Generate mutation vectors from random distinct indices
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        # Apply mutation where active
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _position_update_batch(self):
        """Update positions based on velocities."""
        new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            # Reinitialize worst 30% of particles
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            # Generate new random particles
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            # Reset fitness for replaced particles
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            # Perturb global best slightly
            if self.global_best is not None:
                perturbation = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + perturbation)
                self.personal_best[0] = self.population[0]
                self.personal_best_fitness[0] = np.inf
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        # Handle truncated initial evaluation
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        # Check initial stopping condition
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Main optimization loop
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            # Check for budget exhaustion
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Update bests
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            
            # Restart check
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```