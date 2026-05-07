Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_mutate_compass_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t13_id  variant_02_catB_t16_id  variant_03_catC_t19_id  variant_04_catD_t11_id  variant_05_catE_t16_id  variant_06_catF_t05_id  variant_07_catG_t05_id  variant_08_catH_t05_id  variant_09_catA_t05_id  variant_10_catB_t05_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.513706e-03            1.824411e-05            1.324143e-05            -inf                    1.810387e-04            4.718273e-06            -inf                    -inf                    1.067081e-05            2.971967e-02            
1      4.732816e-06            7.910662e+00            1.373260e+00            1.468391e+00            -inf                    1.117160e+01            7.719253e-01            -inf                    -inf                    2.064013e+00            1.127638e+01            
2      1.000000e-08            1.996612e-03            4.946954e-04            2.750328e-04            -inf                    2.994881e-04            1.731593e-04            -inf                    -inf                    2.239978e-04            3.079765e+00            
3      9.212033e-01            6.023597e+00            2.144948e+01            1.711945e+01            -inf                    5.001342e+00            4.304966e+00            -inf                    -inf                    3.868880e+00            1.649817e+01            
4      1.000000e-08            4.626413e-01            4.605669e-01            4.389833e-01            -inf                    1.995875e+00            2.851760e-01            -inf                    -inf                    4.431185e-01            1.632240e+00            
5      2.156517e-05            5.161905e+02            1.095909e+03            1.310725e+04            -inf                    2.456701e+03            1.187755e+02            -inf                    -inf                    2.013285e+02            2.027210e+03            
6      3.269602e+01            1.199246e+02            3.284591e+02            3.767929e+02            -inf                    9.450731e+01            1.069551e+02            -inf                    -inf                    3.729769e+02            3.236648e+02            
7      6.172597e-02            9.807237e+00            1.272913e+01            1.351524e+01            -inf                    6.054249e+00            5.651086e+00            -inf                    -inf                    6.171749e+00            1.209039e+01            
8      3.056784e-02            4.640817e+00            3.933044e+00            3.397988e+00            -inf                    3.596609e+00            2.652946e+00            -inf                    -inf                    1.979842e+00            1.031138e+01            
9      9.312889e+00            4.483496e+01            6.156546e+01            4.349923e+01            -inf                    5.233449e+01            1.919459e+01            -inf                    -inf                    1.351664e+01            1.317987e+01            
10     2.735470e-06            1.981575e+00            9.111830e+00            1.192468e+01            -inf                    1.566425e+00            6.061167e-01            -inf                    -inf                    1.198375e+00            2.638323e+00            
11     1.507611e+01            3.698397e+02            2.923216e+02            2.146987e+02            -inf                    4.657458e+02            1.869927e+01            -inf                    -inf                    1.010884e+02            1.182260e+02            
12     8.126950e-03            1.857861e+01            3.164918e+01            3.920987e+01            -inf                    1.044056e+01            4.555230e+00            -inf                    -inf                    5.461437e+00            7.300628e+00            
13     2.527304e+00            2.557728e+01            1.415179e+01            1.284273e+01            -inf                    3.304660e+01            3.773866e+00            -inf                    -inf                    1.193957e+01            9.645403e+00            
14     3.509648e+00            3.783968e+00            9.223293e+00            1.232912e+01            -inf                    4.144673e+00            5.517624e+00            -inf                    -inf                    9.505528e+00            5.355694e+00            
15     1.721943e+00            3.360297e+00            3.394215e+00            3.400745e+00            -inf                    3.658961e+00            2.084738e+00            -inf                    -inf                    3.203590e+00            3.009057e+00            
16     8.319955e+02            2.001277e+03            1.490353e+03            2.097441e+03            -inf                    7.406114e+03            5.750686e+02            -inf                    -inf                    1.953279e+03            3.746257e+02            
17     1.543570e+01            4.708410e+04            1.456415e+04            1.957054e+03            -inf                    7.425536e+04            5.189781e+02            -inf                    -inf                    4.645528e+03            5.912633e+03            
18     2.038357e+01            4.454582e+01            3.706016e+01            2.735256e+01            -inf                    5.005617e+01            1.870506e+01            -inf                    -inf                    2.025557e+01            3.132818e+01            
19     3.342850e+01            7.795851e+01            1.717409e+02            2.125112e+02            -inf                    1.545499e+02            4.549862e+01            -inf                    -inf                    2.072576e+02            6.248174e+01            
20     1.658605e+01            2.021652e+01            2.760499e+01            2.845072e+01            -inf                    2.733212e+01            1.515712e+01            -inf                    -inf                    2.854976e+01            2.145242e+01            
21     4.632922e+00            4.725961e+00            5.241559e+00            5.345710e+00            -inf                    4.678001e+00            4.885117e+00            -inf                    -inf                    5.362110e+00            4.932290e+00            
22     4.804777e+00            1.095617e+01            1.463859e+01            1.560452e+01            -inf                    1.316275e+01            6.923362e+00            -inf                    -inf                    1.570571e+01            1.098077e+01            
23     2.946017e+01            4.543075e+01            8.058833e+01            8.706786e+01            -inf                    4.887563e+01            3.205117e+01            -inf                    -inf                    6.213529e+01            3.738020e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.000000e-08)
Task  1: original.py  (error=4.732816e-06)
Task  2: original.py  (error=1.000000e-08)
Task  3: original.py  (error=9.212033e-01)
Task  4: original.py  (error=1.000000e-08)
Task  5: original.py  (error=2.156517e-05)
Task  6: original.py  (error=3.269602e+01)
Task  7: original.py  (error=6.172597e-02)
Task  8: original.py  (error=3.056784e-02)
Task  9: original.py  (error=9.312889e+00)
Task 10: original.py  (error=2.735470e-06)
Task 11: original.py  (error=1.507611e+01)
Task 12: original.py  (error=8.126950e-03)
Task 13: original.py  (error=2.527304e+00)
Task 14: original.py  (error=3.509648e+00)
Task 15: original.py  (error=1.721943e+00)
Task 16: variant_10_catB_t05_idea_0.py  (error=3.746257e+02)
Task 17: original.py  (error=1.543570e+01)
Task 18: variant_06_catF_t05_idea_0.py  (error=1.870506e+01)
Task 19: original.py  (error=3.342850e+01)
Task 20: variant_06_catF_t05_idea_0.py  (error=1.515712e+01)
Task 21: original.py  (error=4.632922e+00)
Task 22: original.py  (error=4.804777e+00)
Task 23: original.py  (error=2.946017e+01)

WIN COUNTS:
  original.py: 21 wins
  variant_06_catF_t05_idea_0.py: 2 wins
  variant_10_catB_t05_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   original.py                     1.0000e-08       variant_06_catF_t05_idea_0.py    4.7183e-06       471.8×   1824×
     1   original.py                     4.7328e-06       variant_06_catF_t05_idea_0.py    7.7193e-01       163101×  436107×
     2   original.py                     1.0000e-08       variant_06_catF_t05_idea_0.py    1.7316e-04       17316×  29949×
     3   original.py                     9.2120e-01       variant_09_catA_t05_idea_0.py    3.8689e+00         4.2×    6.5×
     4   original.py                     1.0000e-08       variant_06_catF_t05_idea_0.py    2.8518e-01         >1e6×    >1e6×
     5   original.py                     2.1565e-05       variant_06_catF_t05_idea_0.py    1.1878e+02         >1e6×    >1e6×
     6   original.py                     3.2696e+01       variant_05_catE_t16_idea_0.py    9.4507e+01         2.9×    9.9×
     7   original.py                     6.1726e-02       variant_06_catF_t05_idea_0.py    5.6511e+00        91.6×  158.9×
     8   original.py                     3.0568e-02       variant_09_catA_t05_idea_0.py    1.9798e+00        64.8×  117.7×
     9   original.py                     9.3129e+00       variant_10_catB_t05_idea_0.py    1.3180e+01         1.4×    4.7×
    10   original.py                     2.7355e-06       variant_06_catF_t05_idea_0.py    6.0612e-01       221577×  724400×
    11   original.py                     1.5076e+01       variant_06_catF_t05_idea_0.py    1.8699e+01         1.2×   14.2×
    12   original.py                     8.1269e-03       variant_06_catF_t05_idea_0.py    4.5552e+00       560.5×   1285×
    13   original.py                     2.5273e+00       variant_06_catF_t05_idea_0.py    3.7739e+00         1.5×    5.1×
    14   original.py                     3.5096e+00       variant_01_catA_t13_idea_0.py    3.7840e+00         1.1×    1.6×
    15   original.py                     1.7219e+00       variant_06_catF_t05_idea_0.py    2.0847e+00         1.2×    2.0×
    16   variant_10_catB_t05_idea_0.py   3.7463e+02       variant_06_catF_t05_idea_0.py    5.7507e+02         1.5×    5.2×
    17   original.py                     1.5436e+01       variant_06_catF_t05_idea_0.py    5.1898e+02        33.6×  383.0×
    18   variant_06_catF_t05_idea_0.py   1.8705e+01       variant_09_catA_t05_idea_0.py    2.0256e+01         1.1×    1.7×
    19   original.py                     3.3429e+01       variant_06_catF_t05_idea_0.py    4.5499e+01         1.4×    4.6×
    20   variant_06_catF_t05_idea_0.py   1.5157e+01       original.py                      1.6586e+01         1.1×    1.8×
    21   original.py                     4.6329e+00       variant_05_catE_t16_idea_0.py    4.6780e+00         1.0×    1.1×
    22   original.py                     4.8048e+00       variant_06_catF_t05_idea_0.py    6.9234e+00         1.4×    2.7×
    23   original.py                     2.9460e+01       variant_06_catF_t05_idea_0.py    3.2051e+01         1.1×    1.7×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 3 (original.py, variant_06_catF_t05_idea_0.py, variant_10_catB_t05_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 10  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 7  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — original.py (2.16e-05) vs variant_06_catF_t05_idea_0.py (1.19e+02) = 5507748.2× gap to runner-up, 50818461.0× gap to median competitor. A 50/50 blend with the runner-up would yield ~5.94e+01, which is ~2753874.6× WORSE than the winner alone.

MANDATORY MECHANISM: per-task fingerprint-and-commit (the PROBE-AND-COMMIT pattern below — a specialisation of mechanism #4). Mechanism #2 (ensemble/blending) is EXPLICITLY FORBIDDEN: the per-task gap table shows multiple tasks where the winner beats the runner-up by >=100×, and a linear blend of the two cannot recover the winner's value. You may also use mechanism #3 (rule-based dispatch) if you can derive a CHEAP, OBSERVABLE rule that maps a landscape fingerprint to a winner.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_06_catF_t05_idea_0.py (2 wins) ---
```python
def _mutate_compass_batch(self, population, fitness):
        """Temporal momentum compass: blend current direction with EMA of past successes."""
        NP = self.NP

        # Initialize temporal momentum tracking
        if not hasattr(self, '_dir_ema'):
            self._dir_ema = np.zeros((NP, self.dim))
            self._ema_alpha = 0.3

        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)

        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]

            i_rank = fitness_ranks[i]

            # Direction A: DE/rand/1 style
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)

            # Direction B: DE/best/1 style
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

            # Select best direction for this individual
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])

            # Blend with temporal momentum (EMA of past directions)
            # Higher momentum weight when EMA is stronger (more consistent direction)
            ema_strength = 1.0 / (1.0 + np.linalg.norm(self._dir_ema[i]))
            momentum_weight = 0.3 * ema_strength
            mutants[i] = (1 - momentum_weight) * best_dir + momentum_weight * self._dir_ema[i]

            # Update EMA: track direction toward better fitness
            if hasattr(self, '_prev_population') and self._prev_population is not None:
                if fitness[i] < getattr(self, '_prev_fitness', fitness)[i]:
                    delta_dir = best_dir - self._dir_ema[i]
                    self._dir_ema[i] = (1 - self._ema_alpha) * self._dir_ema[i] + self._ema_alpha * delta_dir

        # Store for next generation comparison
        self._prev_population = population.copy()
        self._prev_fitness = fitness.copy()

        return mutants
```

# --- From variant_10_catB_t05_idea_0.py (1 wins) ---
```python
def _mutate_compass_batch(self, population, fitness):
        """
        Category B: Spectral/linear-algebraic compass mutation.
        Uses eigendecomposition of population covariance matrix to identify
        principal axes and scale mutation inversely to eigenvalue magnitude.
        Directions with low variance (small eigenvalues) are explored more aggressively.
        """
        NP, dim = population.shape

        # Compute fitness ranks for selection pressure
        ranks = self._compute_fitness_ranking(fitness)
        best_idx = np.argmin(fitness)
        i_best = np.argmin(ranks)

        # Compute population covariance matrix (centered)
        centroid = population.mean(axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / max(NP - 1, 1)

        # Eigendecomposition: V @ diag(λ) @ V.T
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            # Fallback: use standard deviation scaling
            std_pop = np.std(population, axis=0) + 1e-10
            return population + self.F * np.random.randn(NP, dim) * std_pop

        # Sort by descending eigenvalue (descending importance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Clamp eigenvalues to avoid numerical issues
        eigenvalues = np.maximum(eigenvalues, 1e-12)

        # Compute total variance for normalization
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained_ratio = eigenvalues / total_var
        else:
            explained_ratio = np.ones(dim) / dim

        # Effective dimensionality: number of PCs needed to explain 95% variance
        cumsum = np.cumsum(explained_ratio)
        n_effective = int(np.searchsorted(cumsum, 0.95)) + 1
        n_effective = min(n_effective, dim)

        # Compute mutation scaling per principal axis
        # Low eigenvalue (low variance) -> high scaling (more exploration)
        # High eigenvalue (high variance) -> low scaling (conservative)
        base_scale = 1.0 / np.sqrt(eigenvalues + 1e-10)
        axis_scaling = base_scale / (np.max(base_scale) + 1e-10)

        # Generate mutants per individual
        mutants = np.empty_like(population)

        for i in range(NP):
            # Direction toward p-best (top 10% by fitness)
            p = 0.1
            top_k = max(1, int(np.ceil(p * NP)))
            pbest_indices = np.argsort(ranks)[:top_k]
            pbest = population[np.random.choice(pbest_indices)]

            # Base direction: current to pbest
            direction = pbest - population[i]

            # Transform direction to eigenspace: z = V.T @ direction
            z = eigenvectors.T @ direction

            # Scale each component inversely to eigenvalue
            z_scaled = z * axis_scaling

            # Transform back to original space
            direction_scaled = eigenvectors @ z_scaled

            # Normalize scale based on fitness rank (lower rank = more exploration)
            rank_factor = 0.5 + 0.5 * (1.0 - ranks[i])

            # Add random perturbation scaled by inverse eigenvalue
            random_perturb = np.zeros(dim)
            for j in range(n_effective):
                # More perturbation along low-variance axes
                random_perturb += eigenvectors[:, j] * np.random.randn() * axis_scaling[j]

            # Compute mutant
            mutants[i] = population[i] + self.F * rank_factor * direction_scaled + \
                         0.3 * self.F * random_perturb

        return np.clip(mutants, -100.0, 100.0)
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