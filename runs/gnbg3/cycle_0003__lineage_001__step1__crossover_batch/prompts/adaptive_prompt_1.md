Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_crossover_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t16_id  variant_02_catB_t21_id  variant_03_catC_t14_id  variant_04_catD_t18_id  variant_05_catE_t09_id  variant_06_catF_t05_id  variant_08_catH_t05_id  variant_09_catA_t05_id  variant_10_catB_t05_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.766356e+00            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
1      4.732816e-06            1.026423e+01            8.737018e-01            3.850649e+00            -inf                    1.468958e-01            2.792008e-03            1.254104e-02            3.893156e+00            -inf                    
2      1.000000e-08            2.865196e-07            2.084080e+00            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
3      9.212033e-01            4.799945e+01            6.578761e-02            4.705012e+00            -inf                    2.646840e+00            1.452347e+00            1.817215e+00            2.549506e+00            -inf                    
4      1.000000e-08            1.835045e+00            1.583529e+00            1.691086e+00            -inf                    1.395894e-02            2.792492e-02            1.333333e-08            2.086111e-02            -inf                    
5      2.156517e-05            1.390052e+06            5.998301e+00            1.310559e+03            -inf                    3.336418e+01            2.250678e-06            9.173972e+01            3.267686e+00            -inf                    
6      3.269602e+01            9.241850e+02            2.554083e-03            9.136537e+02            -inf                    7.977418e+01            3.331239e+01            4.610717e+01            3.845525e+01            -inf                    
7      6.172597e-02            4.566255e+01            5.598024e-01            5.768123e+00            -inf                    2.986838e+00            2.938773e-02            1.775170e+00            2.156829e+00            -inf                    
8      3.056784e-02            1.123353e+01            2.392887e+00            4.022657e+00            -inf                    1.281925e+00            1.040713e-01            5.894477e-01            2.403133e+00            -inf                    
9      9.312889e+00            1.478715e+02            1.712873e+00            1.127854e+02            -inf                    3.491189e+01            9.314833e+00            1.124212e+01            2.035028e+01            -inf                    
10     2.735470e-06            1.307370e-06            5.082715e+00            3.856708e-05            -inf                    1.233529e-05            2.709751e-07            5.053579e-06            9.134916e+00            -inf                    
11     1.507611e+01            1.480536e+03            1.569802e+01            5.464267e+02            -inf                    1.340594e+01            1.722141e+01            2.356683e+01            4.457587e+01            -inf                    
12     8.126950e-03            6.021668e-05            1.685884e+01            5.641564e-03            -inf                    6.449540e-02            1.652780e-03            4.147148e-02            9.999625e+00            -inf                    
13     2.527304e+00            5.085691e+01            4.007171e+00            1.238292e+01            -inf                    2.033563e+00            3.589753e+00            2.151301e+00            1.425946e+01            -inf                    
14     3.509648e+00            1.485287e+01            6.027406e+00            1.472364e+01            -inf                    7.297028e+00            4.208032e+00            5.345217e+00            5.725912e+00            -inf                    
15     1.721943e+00            4.239013e+00            1.831008e+00            3.763728e+00            -inf                    2.443341e+00            2.642996e+00            1.818198e+00            2.519666e+00            -inf                    
16     8.319955e+02            9.892783e+03            4.289900e+01            6.292564e+03            -inf                    1.330420e+03            6.582313e+02            6.617952e+02            8.403613e+02            -inf                    
17     1.543570e+01            2.629134e+05            7.588478e+03            1.428648e+04            -inf                    1.374169e+01            6.267745e+02            8.090010e+01            9.180430e+03            -inf                    
18     2.038357e+01            8.818961e+01            2.415299e+01            4.504345e+01            -inf                    2.050105e+01            2.258685e+01            2.566742e+01            2.984985e+01            -inf                    
19     3.342850e+01            3.946463e+02            4.360410e+01            3.299119e+02            -inf                    9.417110e+01            2.986357e+01            5.984073e+01            5.973706e+01            -inf                    
20     1.658605e+01            2.436887e+01            9.450789e+00            2.455509e+01            -inf                    1.651118e+01            1.463184e+01            1.692193e+01            1.649096e+01            -inf                    
21     4.632922e+00            5.912514e+00            4.795656e+00            5.639974e+00            -inf                    5.085641e+00            4.620264e+00            4.724390e+00            4.808686e+00            -inf                    
22     4.804777e+00            2.151725e+01            5.185989e+00            1.769564e+01            -inf                    1.067350e+01            4.137000e+00            7.599743e+00            6.455106e+00            -inf                    
23     2.946017e+01            9.684143e+01            7.815352e+00            8.553679e+01            -inf                    5.819805e+01            3.078768e+01            4.103299e+01            3.315557e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: original.py  (error=4.732816e-06)
Task  2: SKIPPED (trivial)
Task  3: variant_02_catB_t21_idea_0.py  (error=6.578761e-02)
Task  4: original.py  (error=1.000000e-08)
Task  5: variant_06_catF_t05_idea_0.py  (error=2.250678e-06)
Task  6: variant_02_catB_t21_idea_0.py  (error=2.554083e-03)
Task  7: variant_06_catF_t05_idea_0.py  (error=2.938773e-02)
Task  8: original.py  (error=3.056784e-02)
Task  9: variant_02_catB_t21_idea_0.py  (error=1.712873e+00)
Task 10: variant_06_catF_t05_idea_0.py  (error=2.709751e-07)
Task 11: variant_05_catE_t09_idea_0.py  (error=1.340594e+01)
Task 12: variant_01_catA_t16_idea_0.py  (error=6.021668e-05)
Task 13: variant_05_catE_t09_idea_0.py  (error=2.033563e+00)
Task 14: original.py  (error=3.509648e+00)
Task 15: original.py  (error=1.721943e+00)
Task 16: variant_02_catB_t21_idea_0.py  (error=4.289900e+01)
Task 17: variant_05_catE_t09_idea_0.py  (error=1.374169e+01)
Task 18: original.py  (error=2.038357e+01)
Task 19: variant_06_catF_t05_idea_0.py  (error=2.986357e+01)
Task 20: variant_02_catB_t21_idea_0.py  (error=9.450789e+00)
Task 21: variant_06_catF_t05_idea_0.py  (error=4.620264e+00)
Task 22: variant_06_catF_t05_idea_0.py  (error=4.137000e+00)
Task 23: variant_02_catB_t21_idea_0.py  (error=7.815352e+00)

WIN COUNTS:
  original.py: 6 wins
  variant_02_catB_t21_idea_0.py: 6 wins
  variant_06_catF_t05_idea_0.py: 6 wins
  variant_05_catE_t09_idea_0.py: 3 wins
  variant_01_catA_t16_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     1   original.py                     4.7328e-06       variant_06_catF_t05_idea_0.py    2.7920e-03       589.9×  184605×
     3   variant_02_catB_t21_idea_0.py   6.5788e-02       original.py                      9.2120e-01        14.0×   38.8×
     4   original.py                     1.0000e-08       variant_08_catH_t05_idea_0.py    1.3333e-08         1.3×    >1e6×
     5   variant_06_catF_t05_idea_0.py   2.2507e-06       original.py                      2.1565e-05         9.6×    >1e6×
     6   variant_02_catB_t21_idea_0.py   2.5541e-03       original.py                      3.2696e+01       12801×  18052×
     7   variant_06_catF_t05_idea_0.py   2.9388e-02       original.py                      6.1726e-02         2.1×   73.4×
     8   original.py                     3.0568e-02       variant_06_catF_t05_idea_0.py    1.0407e-01         3.4×   78.3×
     9   variant_02_catB_t21_idea_0.py   1.7129e+00       original.py                      9.3129e+00         5.4×   11.9×
    10   variant_06_catF_t05_idea_0.py   2.7098e-07       variant_01_catA_t16_idea_0.py    1.3074e-06         4.8×   45.5×
    11   variant_05_catE_t09_idea_0.py   1.3406e+01       original.py                      1.5076e+01         1.1×    1.8×
    12   variant_01_catA_t16_idea_0.py   6.0217e-05       variant_06_catF_t05_idea_0.py    1.6528e-03        27.4×  688.7×
    13   variant_05_catE_t09_idea_0.py   2.0336e+00       variant_08_catH_t05_idea_0.py    2.1513e+00         1.1×    2.0×
    14   original.py                     3.5096e+00       variant_06_catF_t05_idea_0.py    4.2080e+00         1.2×    1.7×
    15   original.py                     1.7219e+00       variant_08_catH_t05_idea_0.py    1.8182e+00         1.1×    1.5×
    16   variant_02_catB_t21_idea_0.py   4.2899e+01       variant_06_catF_t05_idea_0.py    6.5823e+02        15.3×   19.6×
    17   variant_05_catE_t09_idea_0.py   1.3742e+01       original.py                      1.5436e+01         1.1×  552.2×
    18   original.py                     2.0384e+01       variant_05_catE_t09_idea_0.py    2.0501e+01         1.0×    1.3×
    19   variant_06_catF_t05_idea_0.py   2.9864e+01       original.py                      3.3429e+01         1.1×    2.0×
    20   variant_02_catB_t21_idea_0.py   9.4508e+00       variant_06_catF_t05_idea_0.py    1.4632e+01         1.5×    1.8×
    21   variant_06_catF_t05_idea_0.py   4.6203e+00       original.py                      4.6329e+00         1.0×    1.0×
    22   variant_06_catF_t05_idea_0.py   4.1370e+00       original.py                      4.8048e+00         1.2×    1.8×
    23   variant_02_catB_t21_idea_0.py   7.8154e+00       original.py                      2.9460e+01         3.8×    5.3×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 22
  - Distinct winners across tasks: 5 (original.py, variant_01_catA_t16_idea_0.py, variant_02_catB_t21_idea_0.py, variant_05_catE_t09_idea_0.py, variant_06_catF_t05_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 10  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 6  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — variant_06_catF_t05_idea_0.py (2.25e-06) vs original.py (2.16e-05) = 9.6× gap to runner-up, 14824058.2× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.19e-05, which is ~5.3× WORSE than the winner alone.

MANDATORY MECHANISM: per-task fingerprint-and-commit (the PROBE-AND-COMMIT pattern below — a specialisation of mechanism #4). Mechanism #2 (ensemble/blending) is EXPLICITLY FORBIDDEN: the per-task gap table shows multiple tasks where the winner beats the runner-up by >=100×, and a linear blend of the two cannot recover the winner's value. You may also use mechanism #3 (rule-based dispatch) if you can derive a CHEAP, OBSERVABLE rule that maps a landscape fingerprint to a winner.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_t16_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants):
        """Geometry-driven crossover: adapt CR per dimension by axis-aligned spread."""
        NP, dim = population.shape

        # Compute axis-aligned spread (range) per dimension — purely geometric
        dim_min = population.min(axis=0)
        dim_max = population.max(axis=0)
        spread = dim_max - dim_min

        # Prevent division by zero: dimensions with zero spread get minimal mutation
        spread = np.where(spread < 1e-10, 1e-10, spread)

        # Normalize to get per-dimension crossover probability
        # Higher spread → higher probability of taking from mutant (more exploration)
        cr_per_dim = spread / spread.sum()

        # Generate crossover masks: each dimension has independent probability
        rand_matrix = np.random.rand(NP, dim)

        # CR_i for dimension j = cr_per_dim[j]
        # If rand < CR_j, take from mutant; otherwise keep target
        mask = rand_matrix < cr_per_dim[np.newaxis, :]

        # Ensure at least one dimension comes from mutant (classic DE guarantee)
        no_mutant_dims = ~mask.any(axis=1)
        forced_dims = np.random.randint(0, dim, size=no_mutant_dims.sum())
        mask[no_mutant_dims, forced_dims] = True

        # Build trial vectors
        trials = np.where(mask, mutants, population)

        return trials
```

# --- From variant_02_catB_t21_idea_0.py (6 wins) ---
```python
def _crossover_batch(self, population, mutants):
        """Spectral crossover using eigenvalue-weighted PCA decomposition."""
        NP, dim = population.shape

        # Center the population
        centroid = population.mean(axis=0)
        centered = population - centroid

        # Compute covariance and its eigendecomposition
        # Cov = (1/N) * centered^T * centered
        cov = (centered.T @ centered) / NP

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            # Fallback if eigendecomposition fails
            return np.where(
                np.random.rand(NP, dim) < self.CR,
                mutants,
                population
            )

        # Sort eigenvectors by descending eigenvalue magnitude
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Clip small/negative eigenvalues for numerical stability
        eigenvalues = np.clip(eigenvalues, 1e-12, None)

        # Project population and mutants into eigenvector space
        # This gives coordinates along principal axes
        pop_pc = centered @ eigenvectors  # (NP, dim)
        mut_pc = (mutants - centroid) @ eigenvectors  # (NP, dim)

        # Compute eigenvalue-weighted crossover probabilities
        # Higher eigenvalues = higher probability of mixing in that dimension
        total_eig = eigenvalues.sum()
        if total_eig > 0:
            eig_weights = eigenvalues / total_eig
        else:
            eig_weights = np.ones(dim) / dim

        # Adaptive base CR: higher for high-eig dims, lower for low-eig dims
        # This biases exploration toward high-variance directions
        base_cr = np.clip(self.CR * np.power(eig_weights, 0.1), 0.01, 0.99)

        # Binomial crossover in PC space with eigenvalue-adaptive probabilities
        crossover_mask = np.random.rand(NP, dim) < base_cr

        # Ensure at least one dimension is crossed over per individual
        if dim > 1:
            force_cross_idx = np.random.randint(0, dim, size=NP)
            crossover_mask[np.arange(NP), force_cross_idx] = True

        # Mix population and mutant in PC space
        trial_pc = np.where(crossover_mask, mut_pc, pop_pc)

        # Transform back to original space
        trials = trial_pc @ eigenvectors.T + centroid

        return trials
```

# --- From variant_05_catE_t09_idea_0.py (3 wins) ---
```python
def _crossover_batch(self, population, mutants):
        """Graph-distance-adaptive binomial crossover using k-NN topology."""
        NP, dim = population.shape

        # Build k-NN graph over population
        k = max(2, min(5, NP // 4))

        # Compute pairwise squared distances (vectorized)
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # Find k nearest neighbors for each individual
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Compute graph-based distance from each mutant to its target
        # Graph distance considers indirect paths through shared neighbors
        graph_dist = np.zeros(NP)

        for i in range(NP):
            target = population[i]
            mutant = mutants[i]

            # Direct Euclidean distance
            direct_dist = np.linalg.norm(mutant - target)

            # Graph-based distance: consider shared neighbors
            mutant_neighbors = set(knn_indices[i])

            if len(mutant_neighbors) == 0:
                graph_dist[i] = direct_dist
                continue

            # Check second-order connections (neighbors of neighbors)
            second_order = set()
            for neighbor in mutant_neighbors:
                second_order.update(knn_indices[neighbor])
            second_order.discard(i)
            second_order = second_order - mutant_neighbors

            # Graph distance approximation: direct + penalty for lack of graph connectivity
            if len(second_order) == 0:
                graph_penalty = 1.5
            else:
                # More second-order neighbors = better graph connectivity = lower distance
                connectivity_factor = len(second_order) / float(k * k)
                graph_penalty = 1.0 / (1.0 + connectivity_factor)

            graph_dist[i] = direct_dist * graph_penalty

        # Normalize graph distances
        max_dist = graph_dist.max()
        if max_dist > 1e-10:
            norm_dist = graph_dist / max_dist
        else:
            norm_dist = np.zeros(NP)

        # Compute adaptive CR per individual based on graph topology
        # Short graph distance -> high CR (trust mutant more)
        # Long graph distance -> low CR (preserve target features)
        adaptive_CR = self.CR * (1.0 - 0.3 * norm_dist)
        adaptive_CR = np.clip(adaptive_CR, 0.1, 0.95)

        # Generate trial population via binomial crossover
        trials = np.empty_like(population)

        # Random dimensions to always inherit from mutant
        jr = np.random.randint(0, dim, size=NP)

        for i in range(NP):
            cr = adaptive_CR[i]
            mask = np.random.rand(dim) < cr
            mask[jr[i]] = True
            trials[i] = np.where(mask, mutants[i], population[i])

        return trials
```

# --- From variant_06_catF_t05_idea_0.py (6 wins) ---
```python
def _crossover_batch(self, population, mutants):
        """Binomial crossover with temporal EMA-driven CR adaptation.

        Category F (Temporal / dynamical): Tracks crossover success rate across
        generations using exponential moving averages, adjusts CR based on
        temporal trends rather than single-generation snapshots.
        """
        NP, dim = population.shape

        # Initialize temporal tracking attributes (lazy init)
        if not hasattr(self, '_ema_crossover_success'):
            self._ema_crossover_success = 0.5  # Initial success rate guess
            self._ema_fitness_change = 0.0
            self._prev_mean_fitness = None
            self._crossover_success_history = []
            self._CR_ema = self.CR  # Smoothed CR for temporal stability
            self._momentum_buffer = 0.0

        # Compute per-individual crossover success signal
        # We'll use fitness ranks as proxy: trial rank improvement indicates success
        # This will be updated externally via tracking in the main loop

        # Track population-level fitness dynamics (temporal signal)
        current_mean_fitness = population.mean(axis=0).sum()  # Scalar proxy for mean fitness
        if self._prev_mean_fitness is not None:
            fitness_drift = current_mean_fitness - self._prev_mean_fitness
            # EMA of fitness drift rate (positive = population improving/moving)
            alpha_drift = 0.2
            self._ema_fitness_change = (1 - alpha_drift) * self._ema_fitness_change + alpha_drift * fitness_drift

        self._prev_mean_fitness = current_mean_fitness

        # Temporal CR adaptation based on EMA of crossover success
        # Higher EMA success -> population responding well to crossover -> reduce exploration
        # Lower EMA success -> crossover not helping -> increase exploration via higher CR

        # Compute target CR based on temporal signals
        target_correction = 0.0

        # Signal 1: EMA crossover success rate (smoothed over generations)
        success_rate = self._ema_crossover_success
        if success_rate > 0.4:
            # Good success: exploit more, reduce CR
            target_correction -= 0.08 * (success_rate - 0.4)
        else:
            # Low success: explore more, increase CR
            target_correction += 0.12 * (0.4 - success_rate)

        # Signal 2: Fitness drift velocity (convergence detection)
        # Negative drift = population moving toward better region (converging)
        # Strong convergence -> increase CR to maintain diversity
        drift_magnitude = abs(self._ema_fitness_change)
        if drift_magnitude > 1e-6:
            # Population is drifting (converging or moving)
            target_correction += 0.05 * np.sign(drift_magnitude)  # Encourage diversity during drift

        # Signal 3: Momentum from previous corrections (temporal smoothing)
        self._momentum_buffer = 0.7 * self._momentum_buffer + 0.3 * target_correction

        # Apply correction to smoothed CR
        self._CR_ema = np.clip(self._CR_ema + self._momentum_buffer, 0.1, 0.98)

        # Individual-level CR: use base CR with temporal modulation
        # Some diversity in CR per individual for robustness
        individual_CR = np.random.uniform(
            max(0.1, self._CR_ema - 0.15),
            min(0.98, self._CR_ema + 0.15),
            size=NP
        )

        # Binomial crossover
        # Determine which dimensions get mutated for each individual
        r_dim = np.random.randint(0, dim, size=NP)

        mask = np.zeros((NP, dim), dtype=bool)
        for i in range(NP):
            mask[i, r_dim[i]] = True
            n_mutate = int(np.ceil(individual_CR[i] * dim))
            other_dims = np.random.choice(dim, size=n_mutate, replace=False)
            mask[i, other_dims] = True

        trials = np.where(mask, mutants, population)

        # Store current CR for external tracking
        self.CR = self._CR_ema

        return trials
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