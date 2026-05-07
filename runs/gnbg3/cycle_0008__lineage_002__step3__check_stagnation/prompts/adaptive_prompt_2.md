Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_check_stagnation` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t08_id  variant_02_catB_t15_id  variant_03_catC_t07_id  variant_04_catD_t06_id  variant_05_catE_t23_id  variant_06_catF_t17_id  variant_07_catG_t17_id  variant_08_catH_t17_id  variant_09_catA_t17_id  variant_10_catB_t17_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
1      8.168989e-06            7.978583e-06            4.708358e-06            -inf                    5.569528e-06            4.575326e-06            1.436729e-05            1.031175e-04            6.167383e-06            6.070655e-06            7.309608e-04            
2      1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
3      1.223575e+00            1.419491e+00            1.269068e+00            -inf                    1.082938e+00            1.569544e+00            1.235845e+00            2.769436e+00            1.104162e+00            7.375106e-01            3.115779e+00            
4      1.333333e-08            1.333333e-08            1.999998e-08            -inf                    1.168993e-02            1.000000e-08            3.999990e-08            1.674080e-02            1.000000e-08            1.000000e-08            1.874338e-02            
5      4.418369e-05            1.335265e-05            3.154282e-05            -inf                    5.195369e-05            1.062639e-04            8.212731e-05            1.467354e-01            1.937852e-05            9.928790e-06            3.605271e+00            
6      4.445228e+01            3.597318e+01            4.026642e+01            -inf                    3.574456e+01            3.436438e+01            3.435295e+01            4.643833e+01            4.673203e+01            3.314805e+01            4.111651e+01            
7      6.746527e-02            7.073341e-02            5.549239e-02            -inf                    7.069603e-02            6.236719e-02            1.206000e-01            7.274981e-01            4.868565e-02            5.168188e-02            1.507910e+00            
8      3.004360e-02            3.943265e-02            2.835567e-02            -inf                    3.892520e-02            2.833435e-02            3.842283e-02            2.776624e-01            3.473951e-02            3.175484e-02            6.622156e-01            
9      9.579154e+00            8.859108e+00            9.506901e+00            -inf                    9.392406e+00            9.719549e+00            9.270673e+00            9.302713e+00            9.307070e+00            9.375354e+00            9.462122e+00            
10     1.107942e-06            9.276676e-07            1.382683e-06            -inf                    1.293597e-05            2.567797e-06            2.214503e-05            1.381972e-03            4.375498e-06            7.828857e-07            5.122572e-02            
11     1.538928e+01            1.488548e+01            1.148260e+01            -inf                    2.164703e+01            1.601335e+01            1.384573e+02            1.454031e+02            1.121427e+01            1.153109e+01            1.431144e+02            
12     2.001051e-02            1.709205e-02            3.558762e-02            -inf                    2.141876e-01            3.194396e-02            3.690166e+00            3.041394e+00            1.565010e-02            2.850957e-02            4.074176e+00            
13     2.242993e+00            3.253061e+00            2.121683e+00            -inf                    3.366475e+00            2.541149e+00            3.497306e+00            4.261932e+00            2.889796e+00            2.576136e+00            7.421122e+00            
14     3.764646e+00            3.750763e+00            4.388945e+00            -inf                    4.826967e+00            3.808625e+00            5.028773e+00            4.298495e+00            4.344433e+00            4.019595e+00            4.603870e+00            
15     1.660662e+00            2.336556e+00            1.579325e+00            -inf                    2.795162e+00            1.467865e+00            2.571900e+00            2.939473e+00            1.698155e+00            1.552484e+00            2.956743e+00            
16     8.620800e+02            8.529037e+02            6.406180e+02            -inf                    8.722988e+02            4.830890e+02            7.000243e+02            2.011437e+03            5.999693e+02            7.162043e+02            8.922345e+02            
17     7.835861e+00            6.570366e+00            4.092333e+02            -inf                    4.807235e+03            8.235710e+00            6.630955e+03            7.135465e+03            8.088107e+02            1.526734e+02            1.030787e+04            
18     2.505887e+01            3.061471e+01            2.536863e+01            -inf                    2.412415e+01            2.502195e+01            3.248555e+01            3.522781e+01            2.454927e+01            2.642475e+01            3.198576e+01            
19     4.541112e+01            5.672916e+01            3.314335e+01            -inf                    4.660785e+01            3.893754e+01            5.486158e+01            6.422262e+01            3.756741e+01            3.262041e+01            7.895529e+01            
20     1.792894e+01            1.372509e+01            1.473472e+01            -inf                    1.770726e+01            1.628157e+01            1.644725e+01            1.873335e+01            1.644222e+01            1.650059e+01            2.001677e+01            
21     4.574629e+00            4.630380e+00            4.573615e+00            -inf                    4.680128e+00            4.545232e+00            4.722086e+00            4.619498e+00            4.632998e+00            4.541739e+00            4.678303e+00            
22     5.186078e+00            5.175732e+00            7.311615e+00            -inf                    5.576183e+00            5.659992e+00            7.954322e+00            8.143058e+00            5.964629e+00            5.822000e+00            8.526376e+00            
23     3.085307e+01            3.311250e+01            3.540520e+01            -inf                    3.715536e+01            3.838787e+01            4.022974e+01            3.551773e+01            3.601434e+01            3.449935e+01            4.188036e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_05_catE_t23_idea_0.py  (error=4.575326e-06)
Task  2: SKIPPED (trivial)
Task  3: variant_09_catA_t17_idea_0.py  (error=7.375106e-01)
Task  4: SKIPPED (trivial)
Task  5: variant_09_catA_t17_idea_0.py  (error=9.928790e-06)
Task  6: variant_09_catA_t17_idea_0.py  (error=3.314805e+01)
Task  7: variant_08_catH_t17_idea_0.py  (error=4.868565e-02)
Task  8: variant_05_catE_t23_idea_0.py  (error=2.833435e-02)
Task  9: variant_01_catA_t08_idea_0.py  (error=8.859108e+00)
Task 10: variant_09_catA_t17_idea_0.py  (error=7.828857e-07)
Task 11: variant_08_catH_t17_idea_0.py  (error=1.121427e+01)
Task 12: variant_08_catH_t17_idea_0.py  (error=1.565010e-02)
Task 13: variant_02_catB_t15_idea_0.py  (error=2.121683e+00)
Task 14: variant_01_catA_t08_idea_0.py  (error=3.750763e+00)
Task 15: variant_05_catE_t23_idea_0.py  (error=1.467865e+00)
Task 16: variant_05_catE_t23_idea_0.py  (error=4.830890e+02)
Task 17: variant_01_catA_t08_idea_0.py  (error=6.570366e+00)
Task 18: variant_04_catD_t06_idea_0.py  (error=2.412415e+01)
Task 19: variant_09_catA_t17_idea_0.py  (error=3.262041e+01)
Task 20: variant_01_catA_t08_idea_0.py  (error=1.372509e+01)
Task 21: variant_09_catA_t17_idea_0.py  (error=4.541739e+00)
Task 22: variant_01_catA_t08_idea_0.py  (error=5.175732e+00)
Task 23: original.py  (error=3.085307e+01)

WIN COUNTS:
  variant_09_catA_t17_idea_0.py: 6 wins
  variant_01_catA_t08_idea_0.py: 5 wins
  variant_05_catE_t23_idea_0.py: 4 wins
  variant_08_catH_t17_idea_0.py: 3 wins
  variant_02_catB_t15_idea_0.py: 1 wins
  variant_04_catD_t06_idea_0.py: 1 wins
  original.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     1   variant_05_catE_t23_idea_0.py   4.5753e-06       variant_02_catB_t15_idea_0.py    4.7084e-06         1.0×    1.7×
     3   variant_09_catA_t17_idea_0.py   7.3751e-01       variant_04_catD_t06_idea_0.py    1.0829e+00         1.5×    1.7×
     5   variant_09_catA_t17_idea_0.py   9.9288e-06       variant_01_catA_t08_idea_0.py    1.3353e-05         1.3×    5.2×
     6   variant_09_catA_t17_idea_0.py   3.3148e+01       variant_06_catF_t17_idea_0.py    3.4353e+01         1.0×    1.2×
     7   variant_08_catH_t17_idea_0.py   4.8686e-02       variant_09_catA_t17_idea_0.py    5.1682e-02         1.1×    1.5×
     8   variant_05_catE_t23_idea_0.py   2.8334e-02       variant_02_catB_t15_idea_0.py    2.8356e-02         1.0×    1.4×
     9   variant_01_catA_t08_idea_0.py   8.8591e+00       variant_06_catF_t17_idea_0.py    9.2707e+00         1.0×    1.1×
    10   variant_09_catA_t17_idea_0.py   7.8289e-07       variant_01_catA_t08_idea_0.py    9.2767e-07         1.2×    5.6×
    11   variant_08_catH_t17_idea_0.py   1.1214e+01       variant_02_catB_t15_idea_0.py    1.1483e+01         1.0×    1.4×
    12   variant_08_catH_t17_idea_0.py   1.5650e-02       variant_01_catA_t08_idea_0.py    1.7092e-02         1.1×    2.3×
    13   variant_02_catB_t15_idea_0.py   2.1217e+00       original.py                      2.2430e+00         1.1×    1.5×
    14   variant_01_catA_t08_idea_0.py   3.7508e+00       original.py                      3.7646e+00         1.0×    1.2×
    15   variant_05_catE_t23_idea_0.py   1.4679e+00       variant_09_catA_t17_idea_0.py    1.5525e+00         1.1×    1.6×
    16   variant_05_catE_t23_idea_0.py   4.8309e+02       variant_08_catH_t17_idea_0.py    5.9997e+02         1.2×    1.8×
    17   variant_01_catA_t08_idea_0.py   6.5704e+00       original.py                      7.8359e+00         1.2×  123.1×
    18   variant_04_catD_t06_idea_0.py   2.4124e+01       variant_08_catH_t17_idea_0.py    2.4549e+01         1.0×    1.1×
    19   variant_09_catA_t17_idea_0.py   3.2620e+01       variant_02_catB_t15_idea_0.py    3.3143e+01         1.0×    1.4×
    20   variant_01_catA_t08_idea_0.py   1.3725e+01       variant_02_catB_t15_idea_0.py    1.4735e+01         1.1×    1.2×
    21   variant_09_catA_t17_idea_0.py   4.5417e+00       variant_05_catE_t23_idea_0.py    4.5452e+00         1.0×    1.0×
    22   variant_01_catA_t08_idea_0.py   5.1757e+00       original.py                      5.1861e+00         1.0×    1.2×
    23   original.py                     3.0853e+01       variant_01_catA_t08_idea_0.py    3.3112e+01         1.1×    1.2×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 21
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_t08_idea_0.py, variant_02_catB_t15_idea_0.py, variant_04_catD_t06_idea_0.py, variant_05_catE_t23_idea_0.py, variant_08_catH_t17_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 1  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 17 — variant_01_catA_t08_idea_0.py (6.57e+00) vs original.py (7.84e+00) = 1.2× gap to runner-up, 123.1× gap to median competitor. A 50/50 blend with the runner-up would yield ~7.20e+00, which is ~1.1× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_t08_idea_0.py (5 wins) ---
```python
def _check_stagnation(self, best_fitness):
        """Detect stagnation via geometric analysis: centroid drift + pairwise regularity."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        NP, dim = pop.shape

        if NP < 4:
            return False

        # --- A: Geometry / spatial analysis ---

        # 1. Compute current centroid and pairwise distance statistics
        centroid = pop.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)

        # Pairwise distances (upper triangle)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        i, j = np.triu_indices(NP, k=1)
        dists = np.sqrt(sq_dists[i, j])

        median_d = np.median(dists)
        std_d = np.std(dists)

        # Regularity ratio: std/median (regular simplex → 0, random → higher)
        regularity = std_d / (median_d + 1e-10)

        # 2. Initialize or update centroid history (F: temporal tracking)
        if not hasattr(self, '_centroid_history'):
            self._centroid_history = []
            self._dist_regularity_history = []

        self._centroid_history.append(centroid_norm)
        self._dist_regularity_history.append(regularity)

        # Keep bounded history
        max_history = 15
        if len(self._centroid_history) > max_history:
            self._centroid_history.pop(0)
            self._dist_regularity_history.pop(0)

        # 3. Stagnation detection using geometric signals

        # Signal A: Centroid drift (temporal)
        if len(self._centroid_history) >= 5:
            recent = np.array(self._centroid_history[-5:])
            centroid_range = recent.max() - recent.min()
            # Population stagnant if centroid barely moved relative to search space
            search_scale = 200.0 * np.sqrt(dim)  # approximate search space scale
            centroid_stagnant = centroid_range < search_scale * 1e-3
        else:
            centroid_stagnant = False

        # Signal B: Pairwise regularity (spatial) — population too uniform
        regularity_stagnant = regularity < 0.05

        # Signal C: Pairwise spread collapsed (spatial) — population too tight
        spread_collapsed = median_d < 1e-4 * np.sqrt(dim)

        # Combined geometric stagnation
        geometric_stagnant = centroid_stagnant or regularity_stagnant or spread_collapsed

        if geometric_stagnant:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        # Clear history on restart trigger (prevents stale signals)
        if self.stagnation_counter >= 10:
            if hasattr(self, '_centroid_history'):
                self._centroid_history.clear()
            if hasattr(self, '_dist_regularity_history'):
                self._dist_regularity_history.clear()
            return True

        return False
```

# --- From variant_02_catB_t15_idea_0.py (1 wins) ---
```python
def _check_stagnation(self, best_fitness):
        """Detect stagnation via spectral analysis of population covariance.

        Uses eigenvalue decomposition to measure effective dimensionality,
        condition number, and spectral concentration of the population.
        Stagnation is detected when the population has collapsed to a 
        low-rank subspace or when the covariance structure indicates
        loss of exploration capability (Category B: Spectral/linear-algebraic).
        """
        # Need population from optimizer state
        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        NP, dim = pop.shape

        # Need sufficient samples for reliable covariance estimate
        if NP < dim or NP < 5:
            return False

        # Initialize spectral tracking
        if not hasattr(self, '_spectral_history'):
            self._spectral_history = []
            self._prev_eigenvalues = None
            self._prev_effective_rank = float(dim)

        # Center the population
        centroid = pop.mean(axis=0)
        centered_pop = pop - centroid

        # Compute covariance matrix with regularization
        cov = np.cov(centered_pop, rowvar=False)

        # Add small regularization for numerical stability
        eps = 1e-10
        cov_reg = cov + eps * np.eye(dim)

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
        except np.linalg.LinAlgError:
            return False

        # Sort eigenvalues descending
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Remove near-zero eigenvalues (numerical artifacts)
        eigenvalues = eigenvalues[eigenvalues > eps]
        if len(eigenvalues) < 2:
            return True  # Severely collapsed

        # Normalize eigenvalues to probability distribution
        eig_sum = eigenvalues.sum()
        if eig_sum < eps:
            return True
        eig_normalized = eigenvalues / eig_sum

        # --- Spectral Metrics ---

        # 1. Effective rank via eigenvalue entropy
        # H = -sum(p * log(p)), normalized by max entropy log(dim)
        p_nonzero = eig_normalized[eig_normalized > eps]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero + eps))
        max_entropy = np.log(len(eigenvalues))
        effective_rank = (np.exp(entropy) if entropy > 0 else 1.0)

        # 2. Condition number (ratio of max to min eigenvalue)
        condition_number = eigenvalues[0] / max(eigenvalues[-1], eps)

        # 3. Dominance ratio: what fraction of variance is in top eigenvalue
        dominance_ratio = eigenvalues[0] / eig_sum

        # 4. Cumulative variance in top k dimensions
        k = max(1, int(0.1 * dim))  # Top 10% of dimensions
        top_k_eig = eigenvalues[:k]
        cumulative_variance_topk = top_k_eig.sum() / eig_sum

        # 5. Spectral spread (variance of normalized eigenvalues)
        spectral_spread = np.var(eig_normalized)

        # --- Stagnation Detection ---

        # Track changes
        rank_change = self._prev_effective_rank - effective_rank
        self._prev_effective_rank = effective_rank

        # Compute stagnation indicators
        rank_deficit = 1.0 - (effective_rank / dim)  # How much rank is lost
        condition_issue = np.log1p(condition_number) / 20.0  # Log-scaled condition
        dominance_issue = dominance_ratio  # Single eigenvalue dominance

        # Combined stagnation score
        stagnation_score = (
            0.4 * rank_deficit +
            0.3 * np.clip(condition_issue, 0, 1) +
            0.3 * dominance_issue
        )

        # Detect collapse: low effective rank OR high condition number
        is_stagnated = False

        # Criterion 1: Effective rank below threshold
        if effective_rank < 0.15 * dim:
            is_stagnated = True

        # Criterion 2: Severe condition number (ill-conditioned)
        if condition_number > 1e6:
            is_stagnated = True

        # Criterion 3: Single eigenvalue dominates
        if dominance_ratio > 0.95:
            is_stagnated = True

        # Criterion 4: Combined score threshold
        if stagnation_score > 0.75:
            is_stagnated = True

        # Criterion 5: Rapid rank collapse
        if rank_change > 0.5 * dim and effective_rank < 0.25 * dim:
            is_stagnated = True

        # Update history
        self._spectral_history.append({
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'dominance_ratio': dominance_ratio,
            'stagnation_score': stagnation_score
        })

        if len(self._spectral_history) > 20:
            self._spectral_history.pop(0)

        # Track previous eigenvalues
        self._prev_eigenvalues = eigenvalues.copy()

        return is_stagnated
```

# --- From variant_04_catD_t06_idea_0.py (1 wins) ---
```python
def _check_stagnation(self, best_fitness):
        """Detect stagnation using fitness-landscape / rank-based signals.

        Signals used:
        - Spearman rank correlation between consecutive generations
        - Fitness percentile spread (10th-90th percentile)
        - Consecutive generation improvement tracking
        """
        # Initialize tracking attributes
        if not hasattr(self, '_stagnation_counter'):
            self._stagnation_counter = 0
            self._fitness_history = []
            self._prev_pop_fitness = None

        # Track best fitness history (F: temporal/dynamical)
        self._fitness_history.append(float(best_fitness))
        max_history = 20
        if len(self._fitness_history) > max_history:
            self._fitness_history.pop(0)

        # D: Rank-based stagnation detection using Spearman correlation
        stagnation_detected = False

        # Check if population fitness is available for rank analysis
        if hasattr(self, '_current_population_fitness') and self._current_population_fitness is not None:
            pop_fitness = np.asarray(self._current_population_fitness)

            # Filter out NaN values
            valid_mask = ~np.isnan(pop_fitness)
            if valid_mask.sum() >= 5:
                pop_fitness = pop_fitness[valid_mask]

                if self._prev_pop_fitness is not None and len(self._prev_pop_fitness) == len(pop_fitness):
                    prev_valid = ~np.isnan(self._prev_pop_fitness)
                    if prev_valid.sum() == len(pop_fitness):
                        # D: Spearman rank correlation (rank-based)
                        try:
                            from scipy.stats import spearmanr
                            corr, _ = spearmanr(self._prev_pop_fitness, pop_fitness)
                            # High correlation = stable ranks = stagnation
                            if corr is not None and corr > 0.9:
                                stagnation_detected = True
                        except (ValueError, ImportError):
                            # Manual Spearman fallback
                            pass

                self._prev_pop_fitness = pop_fitness.copy()

                # D: Fitness percentile spread (fitness-landscape based)
                # Narrow spread indicates population has converged
                p10, p90 = np.percentile(pop_fitness, [10, 90])
                spread = p90 - p10
                mean_fitness = np.mean(pop_fitness)
                if mean_fitness != 0:
                    rel_spread = spread / (abs(mean_fitness) + 1e-10)
                else:
                    rel_spread = spread

                if rel_spread < 1e-8:
                    stagnation_detected = True
        else:
            # Fallback: best fitness trajectory analysis
            if len(self._fitness_history) >= 3:
                recent = self._fitness_history[-3:]
                improvement = max(recent) - min(recent)
                if improvement < 1e-10:
                    stagnation_detected = True

        # Update stagnation counter (F: temporal tracking over generations)
        if stagnation_detected:
            self._stagnation_counter += 1
        else:
            # Reset on detected progress
            if len(self._fitness_history) >= 2:
                if self._fitness_history[-1] < self._fitness_history[-2] - 1e-12:
                    self._stagnation_counter = 0

        # Threshold for stagnation detection
        stagnation_threshold = 15
        return self._stagnation_counter >= stagnation_threshold
```

# --- From variant_05_catE_t23_idea_0.py (4 wins) ---
```python
def _check_stagnation(self, best_fitness):
        """Detect stagnation via graph-theoretic properties: connected components and MST weight."""
        # Get current population from the optimizer's state
        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        NP, dim = pop.shape

        if NP < 4:
            return False

        # Initialize history for temporal tracking (F: temporal/dynamical component)
        if not hasattr(self, '_topo_stagnation_history'):
            self._topo_stagnation_history = []

        # Build k-NN graph (k adapts to population size)
        k = max(2, min(5, NP // 8))

        # Compute pairwise Euclidean distances (E: topology via graph structure)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # Find k nearest neighbors for each point
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # === Compute connected components using Union-Find ===
        parent = np.arange(NP)
        def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # Path compression
                    x = parent[x]
                return x
        def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
        for i in range(NP):
            for j in nearest_indices[i]:
                union(i, j)

        # Count unique components
        components = np.unique([find(i) for i in range(NP)])
        n_components = len(components)

        # === Compute MST total weight using Prim's algorithm ===
        in_mst = np.zeros(NP, dtype=bool)
        in_mst[0] = True
        mst_edges = []
        mst_total_weight = 0.0

        for _ in range(NP - 1):
            min_weight = np.inf
            min_edge = None
            for i in range(NP):
                if not in_mst[i]:
                    continue
                # Find minimum edge from MST to unvisited vertex
                for j in nearest_indices[i]:
                    if not in_mst[j] and sq_dists[i, j] < min_weight:
                        min_weight = sq_dists[i, j]
                        min_edge = (i, j)
            if min_edge is not None:
                mst_edges.append(min_edge)
                mst_total_weight += np.sqrt(min_weight)
                in_mst[min_edge[1]] = True
            else:
                break

        # Normalize MST weight by population size for scale-invariance
        avg_mst_edge_weight = mst_total_weight / max(len(mst_edges), 1)

        # === Compute average edge length in k-NN graph ===
        edge_lengths = []
        for i in range(NP):
            for j in nearest_indices[i]:
                if i < j:  # Count each edge once
                    edge_lengths.append(np.sqrt(sq_dists[i, j]))

        avg_knn_edge = np.mean(edge_lengths) if edge_lengths else 0.0

        # === Compute component size statistics ===
        component_sizes = []
        for c in components:
            size = sum(1 for i in range(NP) if find(i) == c)
            component_sizes.append(size)

        largest_component_ratio = max(component_sizes) / NP if component_sizes else 1.0

        # Store current metrics for temporal analysis
        current_metrics = {
            'n_components': n_components,
            'avg_mst_edge': avg_mst_edge_weight,
            'largest_component_ratio': largest_component_ratio,
            'avg_knn_edge': avg_knn_edge
        }
        self._topo_stagnation_history.append(current_metrics)

        # Keep history bounded
        if len(self._topo_stagnation_history) > 12:
            self._topo_stagnation_history.pop(0)

        # === Stagnation detection via topology metrics ===
        # Need sufficient history for temporal analysis
        if len(self._topo_stagnation_history) < 4:
            return False

        # Metric 1: Component fragmentation (E: connectivity-based)
        # Too many small components = population fragmented into isolated niches
        fragmentation_threshold = NP / 3
        is_fragmented = n_components > fragmentation_threshold

        # Metric 2: MST weight convergence (E: path-length based)
        # Consistently decreasing MST weight = population converging spatially
        recent_mst = [m['avg_mst_edge'] for m in self._topo_stagnation_history[-4:]]
        mst_decreasing = all(recent_mst[i] >= recent_mst[i+1] * 0.98 for i in range(len(recent_mst)-1))
        mst_shrinking_rate = (recent_mst[0] - recent_mst[-1]) / max(recent_mst[0], 1e-10)

        # Metric 3: Largest component shrinkage
        # If the biggest connected component is shrinking, population is fragmenting
        recent_lcr = [m['largest_component_ratio'] for m in self._topo_stagnation_history[-4:]]
        lcr_decreasing = recent_lcr[-1] < recent_lcr[0] * 0.85

        # Metric 4: k-NN edge length shrinkage (E: neighborhood-based)
        recent_knn = [m['avg_knn_edge'] for m in self._topo_stagnation_history[-4:]]
        knn_decreasing = all(recent_knn[i] >= recent_knn[i+1] * 0.97 for i in range(len(recent_knn)-1))

        # === Combined stagnation decision ===
        stagnation_score = 0

        if is_fragmented:
            stagnation_score += 3
        elif n_components > 2:
            stagnation_score += 1

        if mst_decreasing and mst_shrinking_rate > 0.05:
            stagnation_score += 2

        if lcr_decreasing:
            stagnation_score += 2

        if knn_decreasing:
            stagnation_score += 1

        # Stagnation triggers if score >= 4 (multiple signals align)
        is_stagnated = stagnation_score >= 4

        if is_stagnated:
            self.stagnation_counter += 1
        else:
            # Decay counter when not stagnated (reset slowly)
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        # Also trigger on extreme single signals
        extreme_mst_shrink = mst_shrinking_rate > 0.15
        extreme_fragmentation = n_components > NP / 2

        return is_stagnated or extreme_mst_shrink or extreme_fragmentation
```

# --- From variant_08_catH_t17_idea_0.py (3 wins) ---
```python
def _check_stagnation(self, best_fitness):
        """Hybrid stagnation: temporal fitness + spatial dispersion, variance-weighted."""
        # --- Mechanism 1: Temporal fitness improvement rate ---
        if not hasattr(self, '_stagn_history_fitness'):
            self._stagn_history_fitness = []
            self._stagn_ema_fitness = float('inf')
            self._stagn_ema_alpha = 0.1

        self._stagn_history_fitness.append(float(best_fitness))
        if len(self._stagn_history_fitness) > 20:
            self._stagn_history_fitness.pop(0)

        # EMA of best fitness for trend detection
        if np.isfinite(self._stagn_ema_fitness) and np.isfinite(best_fitness):
            self._stagn_ema_fitness = (1 - self._stagn_ema_alpha) * self._stagn_ema_fitness + self._stagn_ema_alpha * best_fitness
        else:
            self._stagn_ema_fitness = best_fitness

        # Compute normalized fitness improvement rate (relative to initial gap)
        fitness_improvement = abs(self._stagn_ema_fitness - float(best_fitness)) / (abs(self._stagn_ema_fitness) + 1e-12)

        # --- Mechanism 2: Spatial dispersion (population centroid drift) ---
        if not hasattr(self, '_stagn_history_centroid'):
            self._stagn_history_centroid = []

        if hasattr(self, '_current_population') and self._current_population is not None:
            centroid = self._current_population.mean(axis=0)
            self._stagn_history_centroid.append(centroid.copy())
            if len(self._stagn_history_centroid) > 15:
                self._stagn_history_centroid.pop(0)

            # Compute centroid displacement over recent window
            if len(self._stagn_history_centroid) >= 3:
                recent = np.array(self._stagn_history_centroid[-3:])
                centroid_drift = np.mean(np.linalg.norm(np.diff(recent, axis=0), axis=1))
            elif len(self._stagn_history_centroid) >= 2:
                centroid_drift = np.linalg.norm(self._stagn_history_centroid[-1] - self._stagn_history_centroid[0])
            else:
                centroid_drift = 0.0

            # Normalize by dimension and population spread
            pop_spread = self._compute_diversity(self._current_population) + 1e-12
            spatial_signal = centroid_drift / (pop_spread * np.sqrt(self.dim) + 1e-12)
        else:
            spatial_signal = 1.0

        # --- Mechanism 3: Adaptive weighting via inverse-variance ---
        if not hasattr(self, '_stagn_fitness_variance'):
            self._stagn_fitness_variance = 1.0
            self._stagn_spatial_variance = 1.0
            self._stagn_fitness_signal_history = []
            self._stagn_spatial_signal_history = []

        self._stagn_fitness_signal_history.append(fitness_improvement)
        self._stagn_spatial_signal_history.append(spatial_signal)

        if len(self._stagn_fitness_signal_history) > 10:
            self._stagn_fitness_signal_history.pop(0)
        if len(self._stagn_spatial_signal_history) > 10:
            self._stagn_spatial_signal_history.pop(0)

        # Compute variance of each signal
        if len(self._stagn_fitness_signal_history) >= 3:
            self._stagn_fitness_variance = max(np.var(self._stagn_fitness_signal_history), 1e-8)
        if len(self._stagn_spatial_signal_history) >= 3:
            self._stagn_spatial_variance = max(np.var(self._stagn_spatial_signal_history), 1e-8)

        # Inverse-variance weighting: consistent signal gets higher weight
        eps = 1e-10
        w_fitness = 1.0 / (self._stagn_fitness_variance + eps)
        w_spatial = 1.0 / (self._stagn_spatial_variance + eps)
        total_w = w_fitness + w_spatial + eps
        w_fitness /= total_w
        w_spatial /= total_w

        # --- Hybrid stagnation score ---
        hybrid_score = w_fitness * fitness_improvement + w_spatial * spatial_signal

        # --- Adaptive threshold based on generation count and problem scale ---
        if not hasattr(self, '_generation_count'):
            self._generation_count = 0
        self._generation_count += 1

        # Early generations: more lenient; later generations: stricter
        gen_factor = min(1.0, self._generation_count / 100.0)
        base_threshold = 1e-4 * gen_factor
        threshold = base_threshold + 1e-6 * (self.dim / 100.0)

        # --- Decision with minimum confidence requirement ---
        stagnation_detected = (hybrid_score < threshold) and (len(self._stagn_fitness_signal_history) >= 5)

        return stagnation_detected
```

# --- From variant_09_catA_t17_idea_0.py (6 wins) ---
```python
def _check_stagnation(self, best_fitness):
        """Detect stagnation via axis-aligned bounding box volume analysis."""
        if not hasattr(self, '_bb_history'):
            self._bb_history = []
            self._bb_stagnation_counter = 0

        if not hasattr(self, '_current_population') or self._current_population is None:
            return False

        pop = self._current_population
        if len(pop) < 2:
            return False

        try:
            # Compute axis-aligned bounding box
            min_coords = pop.min(axis=0)
            max_coords = pop.max(axis=0)
            ranges = max_coords - min_coords

            # Total volume proxy (product of ranges)
            volume = np.prod(ranges + 1e-10)

            self._bb_history.append(volume)
            if len(self._bb_history) > 10:
                self._bb_history.pop(0)

            # Check for stagnation: volume consistently shrinking
            if len(self._bb_history) >= 3:
                recent = self._bb_history[-3:]
                vol_decreasing = recent[-1] < recent[0] * 0.9

                if vol_decreasing:
                    self._bb_stagnation_counter += 1
                    return self._bb_stagnation_counter >= 3
                else:
                    self._bb_stagnation_counter = 0

            return False
        except Exception:
            return False
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