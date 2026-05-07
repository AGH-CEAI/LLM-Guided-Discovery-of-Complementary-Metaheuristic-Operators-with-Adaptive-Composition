Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_adapt_inertia_weight` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.138718e+01            1.238761e+01            1.393436e+01            3.250941e+01            1.370206e+01            4.492664e+01            1.144576e+01            1.197131e+01            1.574028e+01            1.452530e+01            1.466158e+01            
1      4.263920e+00            1.089324e+00            1.265024e+00            7.219897e+01            1.705734e+00            9.473542e+01            4.122594e+00            3.325630e+00            1.628223e+00            1.235168e+00            1.161638e+00            
2      1.170097e+02            1.334752e+02            1.485766e+02            3.158648e+02            1.230694e+02            3.869838e+02            1.115301e+02            1.296528e+02            1.535086e+02            1.535440e+02            1.559109e+02            
3      7.995690e+00            2.293293e+00            2.706820e+00            1.724352e+02            3.270462e+00            1.935804e+02            7.524048e+00            9.920949e+00            3.525100e+00            2.516226e+00            2.822029e+00            
4      1.787386e+00            2.207944e+00            2.677887e+00            4.397251e+00            2.865480e+00            4.763052e+00            1.832834e+00            1.991464e+00            3.104148e+00            2.531007e+00            2.662380e+00            
5      2.441198e+03            3.531085e+01            1.461592e+06            1.803295e+07            6.800137e+02            3.310581e+08            1.641548e+03            3.122308e+03            2.135488e+01            8.826611e+02            1.103316e+02            
6      9.126561e+02            9.003834e+02            1.317748e+03            3.301630e+03            1.343922e+03            4.208867e+03            1.044344e+03            1.001595e+03            1.636459e+03            1.275117e+03            1.330678e+03            
7      9.302883e+00            7.619867e+00            7.911639e+00            1.885346e+02            8.023688e+00            2.288884e+02            8.864395e+00            8.488694e+00            7.464840e+00            7.709834e+00            8.076861e+00            
8      5.957475e+00            6.225298e+00            1.132191e+01            3.241161e+01            3.411010e+00            3.679433e+01            6.081776e+00            5.644898e+00            1.432239e+01            1.120104e+01            1.056811e+01            
9      1.260085e+02            1.324493e+02            1.678287e+02            2.416043e+02            1.307912e+02            2.759146e+02            1.139238e+02            1.202922e+02            1.835431e+02            1.452414e+02            1.377852e+02            
10     5.639272e+01            6.479277e+01            9.247289e+01            1.384113e+02            6.968309e+01            1.922903e+02            5.354929e+01            5.377950e+01            7.660759e+01            7.663018e+01            9.602114e+01            
11     1.584534e+02            2.921609e+02            2.120752e+02            5.801144e+02            2.090504e+02            7.948525e+02            1.332252e+02            2.496977e+02            1.673626e+02            1.762366e+02            1.590757e+02            
12     7.063918e+01            6.790572e+01            6.611587e+01            1.163255e+02            6.198528e+01            1.472793e+02            6.084305e+01            6.702239e+01            7.240723e+01            5.849630e+01            7.117113e+01            
13     2.335513e+01            2.519374e+01            2.563062e+01            3.194147e+01            2.647629e+01            3.930183e+01            2.520077e+01            2.295286e+01            2.458777e+01            2.556804e+01            2.507989e+01            
14     1.060822e+01            1.077922e+01            1.182721e+01            1.493984e+01            1.154342e+01            1.557832e+01            9.722852e+00            1.023671e+01            1.342339e+01            1.241088e+01            1.292315e+01            
15     4.027568e+00            3.465573e+00            3.667977e+00            3.663569e+00            3.723965e+00            3.783512e+00            3.444533e+00            3.930398e+00            3.362650e+00            3.227951e+00            3.538073e+00            
16     3.423104e+03            3.874816e+03            7.246884e+03            8.891287e+03            4.570539e+03            1.202911e+04            4.000525e+03            3.457178e+03            5.890916e+03            5.086409e+03            5.414391e+03            
17     2.607505e+04            9.296221e+04            6.871504e+04            7.284655e+04            9.451465e+04            1.035025e+05            4.128307e+04            1.076795e+05            6.231589e+04            6.838595e+04            7.526292e+04            
18     4.773294e+01            4.842410e+01            4.736040e+01            6.118193e+01            4.939925e+01            6.709511e+01            5.060382e+01            4.549852e+01            4.921301e+01            5.144529e+01            5.693756e+01            
19     1.902733e+02            2.107115e+02            2.373990e+02            3.053495e+02            2.118909e+02            3.114618e+02            1.920296e+02            2.000518e+02            2.384719e+02            2.355124e+02            2.154064e+02            
20     3.241445e+01            3.395699e+01            3.701697e+01            4.395218e+01            3.405052e+01            4.742209e+01            3.308889e+01            3.296323e+01            3.541590e+01            3.606776e+01            3.676573e+01            
21     5.869834e+00            5.679724e+00            5.696505e+00            5.717649e+00            5.647775e+00            5.748910e+00            5.756342e+00            5.748230e+00            5.741452e+00            5.716363e+00            5.695338e+00            
22     1.500838e+01            1.502248e+01            1.470788e+01            1.699945e+01            1.435876e+01            1.755782e+01            1.491273e+01            1.441714e+01            1.504124e+01            1.424903e+01            1.443481e+01            
23     5.459156e+01            6.276260e+01            6.827633e+01            9.377180e+01            6.907428e+01            9.934104e+01            5.616668e+01            5.901517e+01            7.326236e+01            6.675124e+01            6.926054e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.138718e+01)
Task  1: variant_01_catA_idea_0.py  (error=1.089324e+00)
Task  2: variant_06_catF_idea_0.py  (error=1.115301e+02)
Task  3: variant_01_catA_idea_0.py  (error=2.293293e+00)
Task  4: original.py  (error=1.787386e+00)
Task  5: variant_08_catH_idea_0.py  (error=2.135488e+01)
Task  6: variant_01_catA_idea_0.py  (error=9.003834e+02)
Task  7: variant_08_catH_idea_0.py  (error=7.464840e+00)
Task  8: variant_04_catD_idea_0.py  (error=3.411010e+00)
Task  9: variant_06_catF_idea_0.py  (error=1.139238e+02)
Task 10: variant_06_catF_idea_0.py  (error=5.354929e+01)
Task 11: variant_06_catF_idea_0.py  (error=1.332252e+02)
Task 12: variant_09_catA_idea_0.py  (error=5.849630e+01)
Task 13: variant_07_catG_idea_0.py  (error=2.295286e+01)
Task 14: variant_06_catF_idea_0.py  (error=9.722852e+00)
Task 15: variant_09_catA_idea_0.py  (error=3.227951e+00)
Task 16: original.py  (error=3.423104e+03)
Task 17: original.py  (error=2.607505e+04)
Task 18: variant_07_catG_idea_0.py  (error=4.549852e+01)
Task 19: original.py  (error=1.902733e+02)
Task 20: original.py  (error=3.241445e+01)
Task 21: variant_04_catD_idea_0.py  (error=5.647775e+00)
Task 22: variant_09_catA_idea_0.py  (error=1.424903e+01)
Task 23: original.py  (error=5.459156e+01)

WIN COUNTS:
  original.py: 7 wins
  variant_06_catF_idea_0.py: 5 wins
  variant_01_catA_idea_0.py: 3 wins
  variant_09_catA_idea_0.py: 3 wins
  variant_08_catH_idea_0.py: 2 wins
  variant_04_catD_idea_0.py: 2 wins
  variant_07_catG_idea_0.py: 2 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   original.py                     1.1387e+01       variant_06_catF_idea_0.py        1.1446e+01         1.0×    1.2×
     1   variant_01_catA_idea_0.py       1.0893e+00       variant_10_catB_idea_0.py        1.1616e+00         1.1×    2.3×
     2   variant_06_catF_idea_0.py       1.1153e+02       original.py                      1.1701e+02         1.0×    1.4×
     3   variant_01_catA_idea_0.py       2.2933e+00       variant_09_catA_idea_0.py        2.5162e+00         1.1×    2.4×
     4   original.py                     1.7874e+00       variant_06_catF_idea_0.py        1.8328e+00         1.0×    1.5×
     5   variant_08_catH_idea_0.py       2.1355e+01       variant_01_catA_idea_0.py        3.5311e+01         1.7×   95.6×
     6   variant_01_catA_idea_0.py       9.0038e+02       original.py                      9.1266e+02         1.0×    1.5×
     7   variant_08_catH_idea_0.py       7.4648e+00       variant_01_catA_idea_0.py        7.6199e+00         1.0×    1.1×
     8   variant_04_catD_idea_0.py       3.4110e+00       variant_07_catG_idea_0.py        5.6449e+00         1.7×    3.2×
     9   variant_06_catF_idea_0.py       1.1392e+02       variant_07_catG_idea_0.py        1.2029e+02         1.1×    1.2×
    10   variant_06_catF_idea_0.py       5.3549e+01       variant_07_catG_idea_0.py        5.3780e+01         1.0×    1.4×
    11   variant_06_catF_idea_0.py       1.3323e+02       original.py                      1.5845e+02         1.2×    1.6×
    12   variant_09_catA_idea_0.py       5.8496e+01       variant_06_catF_idea_0.py        6.0843e+01         1.0×    1.2×
    13   variant_07_catG_idea_0.py       2.2953e+01       original.py                      2.3355e+01         1.0×    1.1×
    14   variant_06_catF_idea_0.py       9.7229e+00       variant_07_catG_idea_0.py        1.0237e+01         1.1×    1.2×
    15   variant_09_catA_idea_0.py       3.2280e+00       variant_08_catH_idea_0.py        3.3626e+00         1.0×    1.1×
    16   original.py                     3.4231e+03       variant_07_catG_idea_0.py        3.4572e+03         1.0×    1.5×
    17   original.py                     2.6075e+04       variant_06_catF_idea_0.py        4.1283e+04         1.6×    2.8×
    18   variant_07_catG_idea_0.py       4.5499e+01       variant_02_catB_idea_0.py        4.7360e+01         1.0×    1.1×
    19   original.py                     1.9027e+02       variant_06_catF_idea_0.py        1.9203e+02         1.0×    1.2×
    20   original.py                     3.2414e+01       variant_07_catG_idea_0.py        3.2963e+01         1.0×    1.1×
    21   variant_04_catD_idea_0.py       5.6478e+00       variant_01_catA_idea_0.py        5.6797e+00         1.0×    1.0×
    22   variant_09_catA_idea_0.py       1.4249e+01       variant_04_catD_idea_0.py        1.4359e+01         1.0×    1.0×
    23   original.py                     5.4592e+01       variant_06_catF_idea_0.py        5.6167e+01         1.0×    1.3×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_idea_0.py, variant_04_catD_idea_0.py, variant_06_catF_idea_0.py, variant_07_catG_idea_0.py, variant_08_catH_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 1  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — variant_08_catH_idea_0.py (2.14e+01) vs variant_01_catA_idea_0.py (3.53e+01) = 1.7× gap to runner-up, 95.6× gap to median competitor. A 50/50 blend with the runner-up would yield ~2.83e+01, which is ~1.3× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (3 wins) ---
```python
def _adapt_inertia_weight(self):
        """Adapt inertia weight based on axis-aligned geometric spread ratio."""
        # Compute axis-aligned bounding box
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10

        # Geometric mean vs arithmetic mean ratio of spreads
        # Low ratio -> anisotropic (elongated) swarm -> increase exploration (higher inertia)
        # High ratio -> isotropic (spherical) swarm -> increase exploitation (lower inertia)
        geo_mean = np.exp(np.mean(np.log(spread)))
        arith_mean = np.mean(spread)
        spread_ratio = geo_mean / (arith_mean + 1e-10)

        # Normalize ratio: for dim=10, random uniform spread_ratio ~0.9
        # Tight clustering gives spread_ratio ~0.01
        spread_ratio_normalized = np.clip(spread_ratio / 0.9, 0.0, 1.0)

        # Invert: low ratio -> high inertia (exploration)
        # Generational decay pushes toward exploitation over time
        target_inertia = (1.0 - spread_ratio_normalized) * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

        # Smooth convergence toward target
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```

# --- From variant_04_catD_idea_0.py (2 wins) ---
```python
def _adapt_inertia_weight(self):
        """Adapt inertia weight using fitness-rank landscape analysis.

        Uses:
        1. Fitness-rank spread: detects when population clusters at similar fitness (local optimum trap)
        2. Fitness-distance correlation (FDC): detects deceptive landscapes where population
           clusters around local optima despite being far from global best
        3. Rank quartile analysis: identifies population compression state

        High inertia when ranks are diverse (exploring basin), low inertia when
        ranks compress (stuck in local optimum) or FDC is low (deceptive landscape).
        """
        # Compute fitness percentile ranks (0=worst, 1=best)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, len(self.current_fitness) - 1)

        # Fitness-rank spread: low spread = all individuals at similar fitness = trapped
        rank_spread = np.std(fitness_ranks) + 1e-10

        # Rank interquartile range: compressed IQR = population stuck
        q75, q25 = np.percentile(fitness_ranks, [75, 25])
        rank_iqr = q75 - q25 + 1e-10

        # Fitness-distance correlation (FDC): measures landscape structure
        # High FDC = fitness correlates with distance to global best (good exploration)
        # Low/negative FDC = deceptive landscape, population clustered away from global optimum
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            distance_ranks = np.argsort(np.argsort(distances)) / max(1, len(distances) - 1)

            # Pearson correlation on ranks (approximates Spearman)
            if np.std(distance_ranks) > 1e-10 and np.std(fitness_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, distance_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0

        # Clamp FDC to reasonable range
        fdc = np.clip(fdc, -0.5, 1.0)

        # Determine landscape state from rank analysis
        # State 1: Rank compression (stuck in local optimum) -> aggressive exploration
        if rank_spread < 0.12 or rank_iqr < 0.15:
            # Population clustered at similar fitness: inject exploration
            self.inertia_weight = max(0.3, self.inertia_weight * 0.75)
        # State 2: Deceptive landscape (low FDC) -> moderate exploration boost
        elif fdc < 0.2:
            # Low fitness-distance correlation: population scattered in fitness space
            self.inertia_weight = max(0.4, self.inertia_weight * 0.85)
        # State 3: Rank diversity + good FDC -> increase exploitation
        elif rank_spread > 0.30 and rank_iqr > 0.35 and fdc > 0.5:
            # Population well-spread in fitness with good landscape structure
            self.inertia_weight = min(0.90, self.inertia_weight * 1.08)
        # State 4: Normal convergence
        else:
            # Gradual convergence schedule
            self.inertia_weight = 0.729 - 0.08 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

        # Apply FDC modulation as multiplicative factor
        # Low FDC -> reduce inertia further (extra exploration pressure)
        fdc_modulation = 0.6 + 0.4 * np.clip(fdc + 0.5, 0.0, 1.0)
        self.inertia_weight *= fdc_modulation

        # Final bounds
        self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
```

# --- From variant_06_catF_idea_0.py (5 wins) ---
```python
def _adapt_inertia_weight(self):
        """Adapt inertia weight using temporal dynamics: diversity momentum, rate-of-change, and stagnation."""
        current_diversity = self._compute_diversity()

        # Initialize temporal state on first call
        if not hasattr(self, '_ema_diversity'):
            self._ema_diversity = current_diversity
            self._prev_diversity = current_diversity
            self._diversity_history = [current_diversity]
            self._fitness_stagnation_window = []
            self._inertia_adjustment_momentum = 0.0

        # Exponential moving average of diversity (smoothing across generations)
        alpha = 0.2  # EMA smoothing factor
        self._ema_diversity = alpha * current_diversity + (1 - alpha) * self._ema_diversity

        # Rate of change of diversity (temporal derivative)
        diversity_delta = current_diversity - self._prev_diversity
        self._prev_diversity = current_diversity

        # Accumulate diversity history for window-based analysis
        self._diversity_history.append(current_diversity)
        if len(self._diversity_history) > 50:
            self._diversity_history.pop(0)

        # Stagnation detection: track if fitness is improving
        if len(self._fitness_stagnation_window) < 20:
            self._fitness_stagnation_window.append(self.global_best_fitness)
        else:
            self._fitness_stagnation_window.pop(0)
            self._fitness_stagnation_window.append(self.global_best_fitness)

        # Detect fitness stagnation over window
        fitness_improving = False
        if len(self._fitness_stagnation_window) >= 5:
            recent_improvement = self._fitness_stagnation_window[0] - self._fitness_stagnation_window[-1]
            fitness_improving = recent_improvement > 1e-6

        # Compute trend: linear regression slope over diversity history
        trend_slope = 0.0
        if len(self._diversity_history) >= 5:
            n = len(self._diversity_history)
            x = np.arange(n)
            x_mean = np.mean(x)
            y = np.array(self._diversity_history)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2) + 1e-10
            trend_slope = np.sum((x - x_mean) * (y - y_mean)) / denom

        # Momentum-based adjustment: track direction consistency
        momentum_decay = 0.8
        self._inertia_adjustment_momentum = (
            momentum_decay * self._inertia_adjustment_momentum + 
            (1 - momentum_decay) * np.sign(diversity_delta)
        )

        # Determine inertia adjustment based on temporal signals
        if self._ema_diversity < self.diversity_threshold_low:
            # Low diversity: increase exploration
            if trend_slope < -1e-6 and not fitness_improving:
                # Diversity declining AND fitness stagnating: aggressive exploration
                adjustment = 1.15
            elif abs(self._inertia_adjustment_momentum) > 0.5:
                # Momentum indicates sustained direction: amplify response
                adjustment = 1.1
            else:
                adjustment = 1.05
            self.inertia_weight = min(0.95, self.inertia_weight * adjustment)

        elif self._ema_diversity > self.diversity_threshold_high:
            # High diversity: increase exploitation
            if trend_slope > 1e-6 and fitness_improving:
                # Diversity growing AND fitness improving: moderate reduction
                adjustment = 0.9
            else:
                adjustment = 0.95
            self.inertia_weight = max(0.4, self.inertia_weight * adjustment)

        else:
            # Normal range: use trend to modulate schedule
            base_inertia = 0.729 - 0.1 * (self.generation / 1000)

            # If diversity has been steadily declining (convergence pressure), 
            # counteract with slightly higher inertia
            if trend_slope < -0.5:
                base_inertia += 0.1
            # If diversity has been steadily increasing (dispersion pressure),
            # allow slightly lower inertia for faster convergence
            elif trend_slope > 0.5:
                base_inertia -= 0.05

            self.inertia_weight = np.clip(base_inertia, 0.4, 0.95)
```

# --- From variant_07_catG_idea_0.py (2 wins) ---
```python
def _adapt_inertia_weight(self):
        """Adapt inertia weight using bootstrap confidence intervals on diversity.

        Category G: Stochastic / sampling-based
        Uses bootstrap resampling to estimate diversity uncertainty, then makes
        probabilistic decisions about exploration vs exploitation.
        """
        pop = self.population
        np_pop = self.np

        # Bootstrap resampling for diversity confidence interval
        n_bootstrap = 50
        bootstrap_diversities = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Sample with replacement (stratified: sample indices, use those rows)
            indices = np.random.randint(0, np_pop, size=np_pop)
            sample = pop[indices]
            centroid = np.mean(sample, axis=0)
            distances = np.linalg.norm(sample - centroid, axis=1)
            bootstrap_diversities[b] = np.mean(distances)

        # Compute confidence interval statistics
        ci_low = np.percentile(bootstrap_diversities, 10)
        ci_high = np.percentile(bootstrap_diversities, 90)
        ci_width = ci_high - ci_low
        median_diversity = np.median(bootstrap_diversities)

        # Use Latin hypercube sampling to probe inertia-diversity relationship
        # Sample 20 random inertia candidates and estimate their effect
        n_probes = 20
        probe_inertias = np.linspace(0.4, 0.95, n_probes)
        # Shuffle to ensure coverage (Latin hypercube property)
        np.random.shuffle(probe_inertias)

        # Monte Carlo estimate: expected diversity change per inertia setting
        # Model: higher inertia -> higher diversity (for fixed velocities)
        expected_diversity = np.zeros(n_probes)
        for i, w in enumerate(probe_inertias):
            # Simulate: diversity_next ≈ current_diversity + w * velocity_effect
            # Sample velocity magnitudes to get stochastic estimate
            vel_mags = np.linalg.norm(self.velocity, axis=1)
            vel_effect = np.mean(vel_mags) * w
            # Add noise from bootstrap uncertainty
            noise_std = ci_width / 2
            expected_diversity[i] = median_diversity + vel_effect + np.random.normal(0, noise_std)

        # Select inertia that maximizes expected diversity within safe bounds
        # (Monte Carlo selection based on sampled outcomes)
        best_idx = np.argmax(expected_diversity)
        candidate_inertia = probe_inertias[best_idx]

        # Stochastic decision based on CI overlap with thresholds
        rng = np.random.random()

        if ci_high < self.diversity_threshold_low:
            # Low diversity regime (high confidence): increase inertia
            self.inertia_weight = min(0.95, self.inertia_weight * 1.1)
        elif ci_low > self.diversity_threshold_high:
            # High diversity regime (high confidence): decrease inertia
            self.inertia_weight = max(0.4, self.inertia_weight * 0.9)
        elif ci_width > 5.0:
            # High uncertainty (wide CI): favor exploration
            if rng < 0.7:
                self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
            else:
                self.inertia_weight = candidate_inertia
        else:
            # Normal regime: use Monte Carlo-selected candidate with decay
            decay = 0.1 * (self.generation / 1000)
            self.inertia_weight = candidate_inertia - decay
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```

# --- From variant_08_catH_idea_0.py (2 wins) ---
```python
def _adapt_inertia_weight(self):
        """Hybrid inertia adaptation: combine stagnation signal with spectral condition."""
        diversity = self._compute_diversity()

        # Mechanism 1: stagnation-based signal (temporal)
        stagnation_penalty = np.exp(-self.stagnation_counter / 30.0)

        # Mechanism 2: spectral condition number (linear-algebraic)
        centered = self.population - np.mean(self.population, axis=0)
        try:
            _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            spectral_stress = np.clip(cond / 50.0, 0.0, 1.0)
        except np.linalg.LinAlgError:
            spectral_stress = 0.5

        # Principled switching: stagnation dominates when stuck, spectral guides otherwise
        if self.stagnation_counter > 15:
            # Stagnation regime: use stagnation signal as primary driver
            if diversity < self.diversity_threshold_low:
                self.inertia_weight = max(0.3, self.inertia_weight * 0.85)
            else:
                self.inertia_weight = min(0.9, self.inertia_weight * 1.02)
        else:
            # Normal regime: blend stagnation + spectral signals
            combined_exploration_pressure = 0.6 * (1.0 - stagnation_penalty) + 0.4 * spectral_stress

            if combined_exploration_pressure > 0.5:
                self.inertia_weight = max(0.35, 0.729 - 0.15 * combined_exploration_pressure)
            else:
                self.inertia_weight = min(0.95, 0.729 + 0.10 * (0.5 - combined_exploration_pressure))

        self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
```

# --- From variant_09_catA_idea_0.py (3 wins) ---
```python
def _adapt_inertia_weight(self):
        """Adapt inertia weight based on axis-aligned spread ratio (Category A: Geometry/Spatial).

        Detects dimensional collapse: when the swarm collapses non-uniformly (e.g., spread
        along one axis >> others), the swarm may miss optima in under-explored dimensions.
        The axis-aligned spread ratio measures this anisotropy geometrically.
        """
        # Compute axis-aligned bounding box
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        search_width = self.upper_bound - self.lower_bound + 1e-10

        # Per-dimension spread (geometric: how much of each axis is covered)
        spreads = max_pos - min_pos
        normalized_spreads = spreads / search_width

        max_spread = np.max(normalized_spreads)
        min_spread = np.min(normalized_spreads)

        # Ratio of max to min spread: high ratio = anisotropic collapse
        axis_ratio = max_spread / (min_spread + 1e-10)

        # Log-scaled response to ratio (geometric, not spectral)
        log_ratio = np.log1p(axis_ratio)

        # Inertia weight: higher when collapsed (more exploration needed)
        if log_ratio > 5.0:  # extreme collapse (ratio > 148)
            self.inertia_weight = 0.92
        elif log_ratio > 3.5:  # severe collapse (ratio > 33)
            self.inertia_weight = 0.85
        elif log_ratio > 2.0:  # moderate collapse (ratio > 7.4)
            self.inertia_weight = 0.78
        elif log_ratio > 1.0:  # mild anisotropy (ratio > 2.7)
            self.inertia_weight = 0.72
        else:  # roughly isotropic spread
            # Gradual convergence schedule
            self.inertia_weight = 0.729 - 0.08 * (self.generation / 1000)

        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
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
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE: Task 5 has the winner (variant_10)
    beating the median competitor by 13917×. A 50/50 ensemble blend of
    31.17 and 60.30 yields ~45.7, which is 1.5× WORSE than the winner
    alone. Linear blending cannot preserve per-task advantages.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): short round-robin burst with each operator,
        tracking rank-based improvement scores on the ACTUAL landscape.
      - PHASE B (commit): run ONLY the winning operator for the
        remaining budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if the committed operator stagnates, allow a
        small probability of switching to the runner-up.
    
    Win-count distribution (10/7/2/2/2/1 across 6 winners) confirms
    that different operators win on different tasks, making a single
    default operator suboptimal without probing.
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
        
        # --- PROBE-AND-COMMIT STATE ---
        # Map operator ID -> (score_sum, count) for rank-based UCB scoring
        self.operator_scores = {}   # {op_id: [score1, score2, ...]}
        self.operator_total_score = {}
        self.operator_sample_count = {}
        self._committed_operator = None  # Set after probe
        
        # Operators available for selection
        self._operators = [
            'original',
            'variant_01_catA',
            'variant_03_catC',
            'variant_04_catD',
            'variant_05_catE',
            'variant_10_catB',
        ]
        for op in self._operators:
            self.operator_scores[op] = []
            self.operator_total_score[op] = 0.0
            self.operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator, min 20 total
        self._probe_gens_per_op = 3
        self._probe_total_budget = max(len(self._operators) * self._probe_gens_per_op, 20)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Track if we're in probe phase
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        # Running rank of best fitness in current probe window (for scoring)
        self._probe_fitness_history = []  # best fitness per gen in probe
        self._probe_op_history = []       # which operator per gen
        
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
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE VARIANTS (from benchmark winners)
    # -------------------------------------------------------------------------
    
    def _velocity_update_original(self):
        """Original velocity update with DE/rand/1 mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_01_catA(self):
        """Update velocities using geometric layout: centroid, k-NN repulsion, spread."""
        centroid = np.mean(self.population, axis=0)
        
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10
        
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.cognitive_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        social = self.social_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        social_component = social * r2 * (self.local_best - self.population)
        
        centroid_direction = centroid - self.population
        centroid_strength = np.exp(-dist_to_centroid / 100.0)
        centroid_component = 0.3 * centroid_strength * centroid_direction
        
        k = min(5, self.np - 1)
        knn_repulsion = np.zeros((self.np, self.dim))
        for i in range(self.np):
            diffs = self.population - self.population[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nn_indices = np.argpartition(dists, k)[:k]
            nn_dists = dists[nn_indices]
            for j, nn_idx in enumerate(nn_indices):
                if nn_dists[j] > 1e-10:
                    diff = self.population[i] - self.population[nn_idx]
                    knn_repulsion[i] += diff / (nn_dists[j] ** 2 + 1e-10)
        
        knn_component = 0.1 * knn_repulsion
        
        spread_modulation = np.mean(spread) / (spread + 1e-10)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            centroid_component +
            knn_component
        )
        new_velocity = new_velocity * spread_modulation
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_03_catC(self):
        """Update velocities with entropy-modulated distribution-guided exploration."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mean_pop = np.mean(self.population, axis=0)
        centered = self.population - mean_pop
        
        cov_pop = np.cov(centered.T)
        cov_pop += np.eye(self.dim) * 1e-8
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_pop)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            
            log_det = np.sum(np.log(eigenvalues))
            entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + log_det)
            
            max_entropy = self.dim * np.log(self.upper_bound - self.lower_bound + 1e-10)
            min_entropy = self.dim * np.log(1e-6)
            entropy_normalized = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0, 1)
            
            dispersive_strength = 0.5 * (1.0 - entropy_normalized)
            
            inv_sqrt_eigen = 1.0 / np.sqrt(eigenvalues + 1e-10)
            scaled_diff = centered * inv_sqrt_eigen
            mahal_dist = np.linalg.norm(scaled_diff, axis=1, keepdims=True)
            
            max_mahal = np.max(mahal_dist) + 1e-10
            mahal_normalized = mahal_dist / max_mahal
            
            to_mean = mean_pop - self.population
            to_mean_norm = np.linalg.norm(to_mean, axis=1, keepdims=True) + 1e-10
            to_mean_dir = to_mean / to_mean_norm
            
            dispersive_component = (
                dispersive_strength *
                mahal_normalized *
                to_mean_dir *
                np.random.uniform(0, 1, (self.np, self.dim))
            )
            
            random_explore = 0.3 * (1.0 - entropy_normalized) * np.random.uniform(-1, 1, (self.np, self.dim))
            
            entropy_velocity = dispersive_component + random_explore
            
        except np.linalg.LinAlgError:
            entropy_velocity = 0.1 * np.random.uniform(-1, 1, (self.np, self.dim))
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            entropy_velocity
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_04_catD(self):
        """Update velocities with fitness-rank-adaptive components."""
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            
            if np.std(distances) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                corr = np.corrcoef(
                    np.argsort(np.argsort(self.current_fitness)),
                    np.argsort(np.argsort(distances))
                )[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0
        
        social = self.social_base * np.clip(1.0 + corr, 0.2, 1.5)
        cognitive = self.cognitive_base
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_strength = 0.3 * np.clip(1.0 - corr, 0.1, 1.0)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            mutation_strength * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_05_catE(self):
        """Update velocities using k-NN graph topology and betweenness-based social routing."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        k = min(max(3, self.neighborhood_size), self.np - 1)
        
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        adjacency = np.zeros((self.np, self.np), dtype=np.float64)
        for i in range(self.np):
            adjacency[i, knn_indices[i]] = 1.0
            adjacency[knn_indices[i], i] = 1.0
        
        degree = np.sum(adjacency, axis=1)
        degree_norm = degree / (degree.max() + 1e-10)
        
        d_sqrt_inv = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
        laplacian = np.eye(self.np) - d_sqrt_inv @ adjacency @ d_sqrt_inv
        
        np.random.seed(42)
        v = np.random.randn(self.np)
        v = v / (np.linalg.norm(v) + 1e-10)
        for _ in range(20):
            v = laplacian @ v
            v = v / (np.linalg.norm(v) + 1e-10)
        
        fiedler = np.abs(v @ laplacian @ v) / (np.dot(v, v) + 1e-10)
        connectivity_strength = np.clip(fiedler * 5.0, 0.1, 2.0)
        
        betweenness = np.zeros(self.np)
        n_samples = min(50, self.np)
        sample_nodes = np.random.choice(self.np, n_samples, replace=False)
        
        for src in sample_nodes:
            dist = np.full(self.np, np.inf)
            pred = [[] for _ in range(self.np)]
            dist[src] = 0
            queue = [src]
            
            while queue:
                curr = queue.pop(0)
                for nb in knn_indices[curr]:
                    if dist[nb] == np.inf:
                        dist[nb] = dist[curr] + 1
                        queue.append(nb)
                    if dist[nb] == dist[curr] + 1:
                        pred[nb].append(curr)
            
            sigma = np.zeros(self.np)
            sigma[src] = 1
            for d in range(int(dist.max()) + 1) if dist.max() < np.inf else []:
                for node in np.where(dist == d)[0]:
                    for p in pred[node]:
                        sigma[node] += sigma[p]
            
            for node in range(self.np):
                if node != src and dist[node] < np.inf:
                    for p in pred[node]:
                        betweenness[node] += sigma[p] / (sigma[node] + 1e-10)
        
        betweenness_norm = betweenness / (betweenness.max() + 1e-10)
        
        social_component = np.zeros((self.np, self.dim))
        
        for i in range(self.np):
            neighbor_best_idx = knn_indices[i][np.argmin(self.personal_best_fitness[knn_indices[i]])]
            neighbor_best_pos = self.personal_best[neighbor_best_idx]
            
            influence_weight = 0.5 * (1.0 + degree_norm[i]) * (1.0 + betweenness_norm[i])
            social_scaled = social * connectivity_strength
            
            social_component[i] = social_scaled * r2[i] * influence_weight * (neighbor_best_pos - self.population[i])
        
        if fiedler < 0.2:
            rescue_strength = np.clip((0.2 - fiedler) * 3.0, 0.0, 0.8)
            if self.global_best is not None:
                rescue = rescue_strength * (self.global_best - self.population)
            else:
                best_third = np.argsort(self.personal_best_fitness)[:max(1, self.np // 3)]
                centroid = np.mean(self.population[best_third], axis=0)
                rescue = rescue_strength * (centroid - self.population)
            social_component += rescue
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.05 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(mutation_active, 0.2 * (mutation_vectors - self.population), 0.0)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_10_catB(self):
        """Update velocities with spectral-condition-guided anisotropic mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        centered = self.population - np.mean(self.population, axis=0)
        
        try:
            _, singular_values, right_sv = np.linalg.svd(centered, full_matrices=False)
            
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            mutation_strength = np.clip(0.5 / (1.0 + 0.1 * np.log1p(cond)), 0.1, 0.5)
            
            total_variance = np.sum(singular_values ** 2) + 1e-10
            cumvar = np.cumsum(singular_values ** 2) / total_variance
            effective_dim = np.searchsorted(cumvar, 0.95) + 1
            
            if effective_dim < self.dim * 0.5:
                principal_axes = right_sv[:effective_dim].T
                parallel_component = self.population @ principal_axes @ principal_axes.T
                orthogonal_residual = self.population - parallel_component
                orthogonal_strength = 0.3 * (1.0 - effective_dim / self.dim)
                mutation_component = orthogonal_strength * orthogonal_residual
            else:
                mutation_component = np.zeros((self.np, self.dim))
            
            mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
            mutation_threshold = mutation_strength * (1.0 - self.generation / 5000)
            mutation_active = mutation_mask < mutation_threshold
            
            mutation_vectors = np.zeros((self.np, self.dim))
            for i in range(self.np):
                indices = np.random.choice(
                    [j for j in range(self.np) if j != i], 3, replace=False
                )
                mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                    self.population[indices[1]] - self.population[indices[2]]
                )
            
            de_component = np.where(
                mutation_active,
                mutation_strength * (mutation_vectors - self.population),
                0.0
            )
            
            total_mutation = mutation_component + de_component
            
        except np.linalg.LinAlgError:
            mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
            mutation_threshold = 0.3 * (1.0 - self.generation / 5000)
            mutation_active = mutation_mask < mutation_threshold
            
            mutation_vectors = np.zeros((self.np, self.dim))
            for i in range(self.np):
                indices = np.random.choice(
                    [j for j in range(self.np) if j != i], 3, replace=False
                )
                mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                    self.population[indices[1]] - self.population[indices[2]]
                )
            
            total_mutation = np.where(
                mutation_active,
                0.3 * (mutation_vectors - self.population),
                0.0
            )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            total_mutation
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> velocity update method
    # -------------------------------------------------------------------------
    
    def _dispatch_velocity_update(self, operator):
        """Route to the appropriate velocity update implementation."""
        if operator == 'original':
            self._velocity_update_original()
        elif operator == 'variant_01_catA':
            self._velocity_update_variant_01_catA()
        elif operator == 'variant_03_catC':
            self._velocity_update_variant_03_catC()
        elif operator == 'variant_04_catD':
            self._velocity_update_variant_04_catD()
        elif operator == 'variant_05_catE':
            self._velocity_update_variant_05_catE()
        elif operator == 'variant_10_catB':
            self._velocity_update_variant_10_catB()
        else:
            # Fallback to original
            self._velocity_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score for rank-based evaluation.
        
        Uses rank-based scoring: a generation's score is its percentile
        rank among all probe generations seen so far (lower fitness = better
        = higher score). This is scale-independent.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        # Compute rank-based score: fraction of probe gens with WORSE fitness
        if len(self._probe_fitness_history) < 2:
            score = 1.0  # First sample gets best score
        else:
            current_best = min(self._probe_fitness_history)
            current_worst = max(self._probe_fitness_history)
            if current_worst > current_best:
                # Normalize: higher score = better (lower fitness)
                normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
                score = np.clip(normalized, 0.0, 1.0)
            else:
                score = 1.0
        
        self.operator_scores[operator].append(score)
        self.operator_total_score[operator] += score
        self.operator_sample_count[operator] += 1
    
    def _compute_ucb_score(self, operator):
        """Compute UCB1-style score for operator selection."""
        n = self.operator_sample_count[operator]
        if n == 0:
            return float('inf')  # Unexplored operators get priority
        
        total_n = sum(self.operator_sample_count.values())
        avg_score = self.operator_total_score[operator] / n
        
        # UCB1 exploration bonus
        if total_n > 0:
            exploration = np.sqrt(2.0 * np.log(total_n) / n)
        else:
            exploration = float('inf')
        
        return avg_score + exploration
    
    def _select_best_operator(self):
        """Select operator with highest UCB score from probe data."""
        best_op = self._operators[0]
        best_score = -float('inf')
        
        for op in self._operators:
            ucb = self._compute_ucb_score(op)
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
    def _select_runner_up_operator(self):
        """Select the second-best operator by UCB score."""
        scores = [(op, self._compute_ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2:
            return scores[1][0]
        return scores[0][0]
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
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
        Run the optimizer with PROBE-AND-COMMIT operator selection.
        
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
        
        # Reset probe state
        for op in self._operators:
            self.operator_scores[op] = []
            self.operator_total_score[op] = 0.0
            self.operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation phase
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Use the current probe operator
                self._dispatch_velocity_update(current_op)
                self._position_update_batch()
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < self.np:
                    self.np = actual_n
                    self.population = self.population[:actual_n]
                    self.velocity = self.velocity[:actual_n]
                    self.current_fitness = self.current_fitness[:actual_n]
                    break
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                
                # Record probe score
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_score(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= len(self._operators):
                    self._in_probe_phase = False
                    self._committed_operator = self._select_best_operator()
                    self._runner_up_operator = self._select_runner_up_operator()
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._dispatch_velocity_update(self._runner_up_operator)
                else:
                    self._dispatch_velocity_update(self._committed_operator)
                
                self._position_update_batch()
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < self.np:
                    self.np = actual_n
                    self.population = self.population[:actual_n]
                    self.velocity = self.velocity[:actual_n]
                    self.current_fitness = self.current_fitness[:actual_n]
                    break
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```