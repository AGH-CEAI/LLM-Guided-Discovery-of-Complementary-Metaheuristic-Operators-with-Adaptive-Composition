Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_update_strategy_scores` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t05_id  variant_02_catB_t13_id  variant_03_catC_t12_id  variant_04_catD_t16_id  variant_05_catE_t18_id  variant_06_catF_t05_id  variant_07_catG_t05_id  variant_08_catH_t10_id  variant_09_catA_t10_id  variant_10_catB_t10_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
1      6.852831e-06            4.058278e-05            3.581718e-05            -inf                    3.494340e-05            4.593751e-05            5.804848e-05            4.642784e-06            2.702501e-05            4.629597e-05            -inf                    
2      1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
3      1.258548e+00            1.167105e+00            1.702024e+00            -inf                    8.803288e-01            1.458506e+00            1.822217e+00            1.504567e+00            1.224919e+00            1.100431e+00            -inf                    
4      1.000000e-08            1.000000e-08            1.333333e-08            -inf                    1.000000e-08            1.000000e-08            1.322292e-02            1.000000e-08            1.999998e-08            3.999991e-08            -inf                    
5      2.870026e-06            1.249450e-04            3.228327e-04            -inf                    7.476720e-04            5.159220e-04            7.926490e-05            2.912720e-05            1.202837e-04            4.293650e-04            -inf                    
6      4.040447e+01            3.595634e+01            3.717164e+01            -inf                    3.350840e+01            4.455210e+01            4.006696e+01            3.770741e+01            3.719395e+01            4.543668e+01            -inf                    
7      6.274026e-02            1.028764e-01            1.056974e-01            -inf                    9.004226e-02            2.560089e-02            1.342467e-01            3.544228e-02            1.491992e-01            1.212138e-01            -inf                    
8      3.597107e-02            6.561022e-02            6.337158e-02            -inf                    5.195649e-02            5.635814e-02            4.539756e-02            2.498281e-02            5.623335e-02            6.108689e-02            -inf                    
9      9.112324e+00            9.598492e+00            9.644298e+00            -inf                    9.170389e+00            9.562829e+00            9.288506e+00            9.271379e+00            9.223471e+00            9.520356e+00            -inf                    
10     1.141618e-06            1.106713e-04            9.874438e-05            -inf                    8.200596e-05            1.040073e-04            8.756884e-06            1.312519e-06            1.314375e-04            9.350336e-05            -inf                    
11     1.253494e+01            2.473671e+01            4.512751e+01            -inf                    3.211413e+01            1.886916e+01            5.902638e+01            9.596532e+00            2.915143e+01            6.018683e+01            -inf                    
12     6.200226e-03            1.377188e-01            1.156372e-01            -inf                    1.330408e-01            8.474907e-02            3.045828e-02            7.774594e-03            1.314928e-01            9.229173e-02            -inf                    
13     2.454185e+00            4.168047e+00            6.241313e+00            -inf                    5.153298e+00            3.495294e+00            3.442633e+00            2.639818e+00            3.222819e+00            3.430716e+00            -inf                    
14     4.079528e+00            4.298622e+00            3.515906e+00            -inf                    3.855580e+00            3.743708e+00            4.000454e+00            4.033765e+00            3.924423e+00            4.231365e+00            -inf                    
15     1.684079e+00            2.550713e+00            2.070503e+00            -inf                    2.078262e+00            2.072702e+00            2.389287e+00            1.476049e+00            2.182032e+00            2.300895e+00            -inf                    
16     6.513781e+02            8.555179e+02            9.482868e+02            -inf                    1.032235e+03            8.043943e+02            9.527036e+02            4.964649e+02            9.847100e+02            8.878780e+02            -inf                    
17     7.259175e+02            7.784393e+02            2.017026e+03            -inf                    1.358905e+03            1.406315e+03            3.611666e+03            4.728507e+01            1.179742e+03            4.690620e+02            -inf                    
18     2.468271e+01            2.657828e+01            1.980840e+01            -inf                    2.149181e+01            2.668996e+01            3.024297e+01            2.342412e+01            2.399330e+01            2.395380e+01            -inf                    
19     3.609633e+01            3.555582e+01            5.302452e+01            -inf                    4.513906e+01            3.711597e+01            4.981317e+01            4.038839e+01            4.846138e+01            4.642132e+01            -inf                    
20     1.387444e+01            1.694549e+01            1.545713e+01            -inf                    1.621708e+01            1.669397e+01            1.718175e+01            1.526289e+01            1.668464e+01            1.631316e+01            -inf                    
21     4.579532e+00            4.546444e+00            4.580014e+00            -inf                    4.555883e+00            4.619792e+00            4.632560e+00            4.565539e+00            4.487333e+00            4.620072e+00            -inf                    
22     6.142520e+00            4.908418e+00            4.983537e+00            -inf                    5.285874e+00            5.447682e+00            4.778772e+00            4.286764e+00            5.406780e+00            5.181979e+00            -inf                    
23     3.591965e+01            4.023868e+01            3.250992e+01            -inf                    3.282678e+01            3.359843e+01            3.703877e+01            3.471449e+01            3.352034e+01            3.995156e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_07_catG_t05_idea_0.py  (error=4.642784e-06)
Task  2: SKIPPED (trivial)
Task  3: variant_04_catD_t16_idea_0.py  (error=8.803288e-01)
Task  4: SKIPPED (trivial)
Task  5: original.py  (error=2.870026e-06)
Task  6: variant_04_catD_t16_idea_0.py  (error=3.350840e+01)
Task  7: variant_05_catE_t18_idea_0.py  (error=2.560089e-02)
Task  8: variant_07_catG_t05_idea_0.py  (error=2.498281e-02)
Task  9: original.py  (error=9.112324e+00)
Task 10: original.py  (error=1.141618e-06)
Task 11: variant_07_catG_t05_idea_0.py  (error=9.596532e+00)
Task 12: original.py  (error=6.200226e-03)
Task 13: original.py  (error=2.454185e+00)
Task 14: variant_02_catB_t13_idea_0.py  (error=3.515906e+00)
Task 15: variant_07_catG_t05_idea_0.py  (error=1.476049e+00)
Task 16: variant_07_catG_t05_idea_0.py  (error=4.964649e+02)
Task 17: variant_07_catG_t05_idea_0.py  (error=4.728507e+01)
Task 18: variant_02_catB_t13_idea_0.py  (error=1.980840e+01)
Task 19: variant_01_catA_t05_idea_0.py  (error=3.555582e+01)
Task 20: original.py  (error=1.387444e+01)
Task 21: variant_08_catH_t10_idea_0.py  (error=4.487333e+00)
Task 22: variant_07_catG_t05_idea_0.py  (error=4.286764e+00)
Task 23: variant_02_catB_t13_idea_0.py  (error=3.250992e+01)

WIN COUNTS:
  variant_07_catG_t05_idea_0.py: 7 wins
  original.py: 6 wins
  variant_02_catB_t13_idea_0.py: 3 wins
  variant_04_catD_t16_idea_0.py: 2 wins
  variant_05_catE_t18_idea_0.py: 1 wins
  variant_01_catA_t05_idea_0.py: 1 wins
  variant_08_catH_t10_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     1   variant_07_catG_t05_idea_0.py   4.6428e-06       original.py                      6.8528e-06         1.5×    8.2×
     3   variant_04_catD_t16_idea_0.py   8.8033e-01       variant_09_catA_t10_idea_0.py    1.1004e+00         1.3×    1.5×
     5   original.py                     2.8700e-06       variant_07_catG_t05_idea_0.py    2.9127e-05        10.1×   78.0×
     6   variant_04_catD_t16_idea_0.py   3.3508e+01       variant_01_catA_t05_idea_0.py    3.5956e+01         1.1×    1.2×
     7   variant_05_catE_t18_idea_0.py   2.5601e-02       variant_07_catG_t05_idea_0.py    3.5442e-02         1.4×    4.1×
     8   variant_07_catG_t05_idea_0.py   2.4983e-02       original.py                      3.5971e-02         1.4×    2.3×
     9   original.py                     9.1123e+00       variant_04_catD_t16_idea_0.py    9.1704e+00         1.0×    1.0×
    10   original.py                     1.1416e-06       variant_07_catG_t05_idea_0.py    1.3125e-06         1.1×   84.2×
    11   variant_07_catG_t05_idea_0.py   9.5965e+00       original.py                      1.2535e+01         1.3×    3.2×
    12   original.py                     6.2002e-03       variant_07_catG_t05_idea_0.py    7.7746e-03         1.3×   16.8×
    13   original.py                     2.4542e+00       variant_07_catG_t05_idea_0.py    2.6398e+00         1.1×    1.4×
    14   variant_02_catB_t13_idea_0.py   3.5159e+00       variant_05_catE_t18_idea_0.py    3.7437e+00         1.1×    1.1×
    15   variant_07_catG_t05_idea_0.py   1.4760e+00       original.py                      1.6841e+00         1.1×    1.4×
    16   variant_07_catG_t05_idea_0.py   4.9646e+02       original.py                      6.5138e+02         1.3×    1.8×
    17   variant_07_catG_t05_idea_0.py   4.7285e+01       variant_09_catA_t10_idea_0.py    4.6906e+02         9.9×   26.8×
    18   variant_02_catB_t13_idea_0.py   1.9808e+01       variant_04_catD_t16_idea_0.py    2.1492e+01         1.1×    1.2×
    19   variant_01_catA_t05_idea_0.py   3.5556e+01       original.py                      3.6096e+01         1.0×    1.3×
    20   original.py                     1.3874e+01       variant_07_catG_t05_idea_0.py    1.5263e+01         1.1×    1.2×
    21   variant_08_catH_t10_idea_0.py   4.4873e+00       variant_01_catA_t05_idea_0.py    4.5464e+00         1.0×    1.0×
    22   variant_07_catG_t05_idea_0.py   4.2868e+00       variant_06_catF_t05_idea_0.py    4.7788e+00         1.1×    1.2×
    23   variant_02_catB_t13_idea_0.py   3.2510e+01       variant_04_catD_t16_idea_0.py    3.2827e+01         1.0×    1.1×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 21
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_t05_idea_0.py, variant_02_catB_t13_idea_0.py, variant_04_catD_t16_idea_0.py, variant_05_catE_t18_idea_0.py, variant_07_catG_t05_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 2  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 10 — original.py (1.14e-06) vs variant_07_catG_t05_idea_0.py (1.31e-06) = 1.1× gap to runner-up, 84.2× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.23e-06, which is ~1.1× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_t05_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Geometry-based strategy scoring using spatial distribution of success."""
        if not hasattr(self, '_prev_population'):
            self._prev_population = None
            self._strategy_geo_scores = np.ones(len(self.strategy_names))

        # Get population from the optimizer state
        if not hasattr(self, '_current_population') or self._current_population is None:
            return

        pop = self._current_population
        NP, dim = pop.shape

        # Compute pairwise Euclidean distances (vectorized)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        distances = np.sqrt(sq_dists)

        # Global population metrics for normalization
        global_spread = np.mean(distances[np.isfinite(distances)])
        centroid = pop.mean(axis=0)

        # Per-strategy geometric analysis
        new_scores = np.zeros(len(self.strategy_names))

        for s in range(len(self.strategy_names)):
            # Mask for individuals where this strategy was used
            strategy_mask = (strategy_used == s)
            n_used = np.sum(strategy_mask)

            if n_used < 1:
                new_scores[s] = 0.0
                continue

            # Separate successful and unsuccessful individuals for this strategy
            success_mask = strategy_mask & improved
            fail_mask = strategy_mask & ~improved

            n_success = np.sum(success_mask)
            n_fail = np.sum(fail_mask)

            if n_success == 0:
                # Strategy had no successes - check geometric diversity it created
                success_rate = 0.0
                # Measure how spread out the failed attempts were (exploration value)
                if n_used > 1:
                    failed_points = pop[fail_mask]
                    spread = np.std(failed_points, axis=0).mean()
                    exploration_bonus = spread / (global_spread + 1e-10)
                else:
                    exploration_bonus = 0.5
                geometric_score = exploration_bonus
            else:
                success_rate = n_success / n_used

                # Geometric spread of successful mutations
                success_points = pop[success_mask]
                success_centroid = success_points.mean(axis=0)
                success_spread = np.std(success_points, axis=0).mean()

                # Normalized spread: higher is better (found diverse good regions)
                normalized_spread = success_spread / (global_spread + 1e-10)

                # Centroid distance from overall population centroid
                # Successful region away from average = found new territory
                centroid_offset = np.linalg.norm(success_centroid - centroid)
                centroid_bonus = centroid_offset / (global_spread + 1e-10)

                # Local neighborhood quality: avg distance to k nearest neighbors
                k = min(3, n_success)
                if k >= 1 and n_success > 1:
                    s_indices = np.where(success_mask)[0]
                    local_densities = []
                    for idx in s_indices:
                        nn_dists = np.sort(distances[idx, s_indices])
                        if len(nn_dists) > 1:
                            local_densities.append(np.mean(nn_dists[1:k+1]))
                    neighborhood_coherence = 1.0 / (np.mean(local_densities) + 1e-10)
                else:
                    neighborhood_coherence = 1.0

                # Compare success region vs failure region geometrically
                if n_fail > 0:
                    fail_points = pop[fail_mask]
                    fail_centroid = fail_points.mean(axis=0)
                    fail_spread = np.std(fail_points, axis=0).mean()

                    # Distance between success and failure centroids
                    region_separation = np.linalg.norm(success_centroid - fail_centroid)

                    # Ratio of spreads: success more spread = better exploration
                    spread_ratio = (success_spread + 1e-10) / (fail_spread + 1e-10)

                    # Combined geometric quality
                    geometric_score = (normalized_spread * 0.3 + 
                                      centroid_bonus * 0.2 + 
                                      np.clip(region_separation / (global_spread + 1e-10), 0, 1) * 0.3 +
                                      np.clip(spread_ratio, 0, 2) * 0.1 +
                                      np.clip(neighborhood_coherence * 0.01, 0, 1) * 0.1)
                else:
                    # All attempts succeeded - great strategy for this region
                    geometric_score = (normalized_spread * 0.4 + 
                                      centroid_bonus * 0.3 +
                                      np.clip(neighborhood_coherence * 0.01, 0, 1) * 0.3)

            # Combine success rate with geometric quality
            # Geometric quality is weighted by how confident we are (more samples)
            sample_confidence = np.sqrt(n_used / NP)
            new_scores[s] = success_rate * (0.5 + 0.5 * geometric_score * sample_confidence)

        # Apply momentum-based update
        momentum = 0.7
        self._strategy_geo_scores = (momentum * self._strategy_geo_scores + 
                                      (1 - momentum) * new_scores)

        # Ensure positive scores and normalize
        self._strategy_geo_scores = np.clip(self._strategy_geo_scores, 0.01, None)
        self._strategy_geo_scores /= self._strategy_geo_scores.sum()

        # Update main strategy scores
        self.strategy_scores = self._strategy_geo_scores.copy()

        # Store population for next iteration comparison
        self._prev_population = pop.copy()
```

# --- From variant_02_catB_t13_idea_0.py (3 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Spectral credit assignment using population covariance structure.

        Strategies that generate improvements along low-variance (exploratory)
        directions are weighted higher when the population is anisotropic.
        Uses eigenvalue analysis to modulate credit assignment.
        """
        if not hasattr(self, '_strategy_score_history'):
            self._strategy_score_history = []
        if not hasattr(self, '_pop_cov_history'):
            self._pop_cov_history = []

        # Get current population from stored reference
        pop = getattr(self, '_current_population', None)
        if pop is None or len(pop) < 5:
            # Fallback: simple success-rate based scoring
            for s in range(len(self.strategy_scores)):
                mask = strategy_used == s
                if mask.any():
                    success_rate = improved[mask].mean()
                    self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * success_rate
            return

        NP, dim = pop.shape
        n_strategies = len(self.strategy_scores)

        # Step 1: Compute population covariance and eigenvalue decomposition
        centroid = pop.mean(axis=0)
        centered = pop - centroid

        # Regularized covariance for numerical stability
        cov = (centered.T @ centered) / max(NP - 1, 1)
        # Add small regularization to ensure positive definiteness
        cov += np.eye(dim) * 1e-8

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            eigenvalues = eigenvalues[::-1]  # Descending order
            eigenvectors = eigenvectors[:, ::-1]
        except np.linalg.LinAlgError:
            # Fallback on decomposition failure
            for s in range(n_strategies):
                mask = strategy_used == s
                if mask.any():
                    self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * improved[mask].mean()
            return

        # Step 2: Compute spectral metrics
        total_variance = eigenvalues.sum()
        explained_ratio = eigenvalues / total_variance

        # Effective dimensionality: 1 / sum(p_i^2) where p_i = eigenvalue_i / sum(eigenvalues)
        p = explained_ratio + 1e-12
        eff_dim = 1.0 / np.sum(p ** 2)
        eff_dim = np.clip(eff_dim, 1.0, dim)

        # Condition number for anisotropy detection
        cond_num = eigenvalues[0] / eigenvalues[-1]
        cond_log = np.log1p(np.clip(cond_num, 1.0, 1e6))
        anisotropy_factor = np.clip(cond_log / 10.0, 0.1, 2.0)

        # Step 3: Compute improvement vectors and their spectral alignment
        # For each individual that improved, compute the improvement direction
        improved_indices = np.where(improved)[0]

        if len(improved_indices) == 0:
            # No improvements: apply small decay
            self.strategy_scores *= 0.95
            return

        # Compute spectral weights based on effective dimensionality
        # When eff_dim is low (population in low-dimensional subspace), 
        # reward strategies that explore orthogonal directions
        spectral_weights = np.zeros(dim)
        if eff_dim < dim * 0.3:
            # Anisotropic population: reward low-eigenvalue directions
            spectral_weights = 1.0 / (eigenvalues + 1e-6)
            spectral_weights /= spectral_weights.sum()
        elif eff_dim > dim * 0.7:
            # Isotropic population: uniform weights
            spectral_weights = np.ones(dim) / dim
        else:
            # Moderate: smooth transition
            alpha = (eff_dim - dim * 0.3) / (dim * 0.4)
            uniform_weights = np.ones(dim) / dim
            low_var_weights = 1.0 / (eigenvalues + 1e-6)
            low_var_weights /= low_var_weights.sum()
            spectral_weights = alpha * uniform_weights + (1 - alpha) * low_var_weights

        # Step 4: Assign credit to strategies based on spectral analysis
        strategy_credits = np.zeros(n_strategies)

        for s in range(n_strategies):
            mask = (strategy_used == s) & improved
            if not mask.any():
                continue

            # Get improvement vectors for this strategy
            n_improved = mask.sum()

            # Compute average improvement vector (in original space)
            improved_pop = pop[mask]

            # Project onto principal components
            projected = (improved_pop - centroid) @ eigenvectors  # (n_improved, dim)

            # Compute weighted alignment score
            # Higher weight for directions with low variance (exploration contribution)
            projection_variance = np.var(projected, axis=0) + 1e-10
            exploration_score = np.sum(spectral_weights * projection_variance)

            # Also compute raw success rate
            success_rate = improved[mask].mean()

            # Combined credit: success rate + exploration bonus scaled by anisotropy
            exploration_bonus = exploration_score * anisotropy_factor
            strategy_credits[s] = success_rate + 0.3 * exploration_bonus

        # Step 5: Update strategy scores with momentum
        momentum = 0.8
        self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * strategy_credits

        # Normalize to prevent score explosion
        if self.strategy_scores.max() > 10.0:
            self.strategy_scores /= self.strategy_scores.max() / 10.0

        # Ensure minimum scores for unexplored strategies
        min_score = 0.1
        self.strategy_scores = np.maximum(self.strategy_scores, min_score)

        # Store history for adaptive analysis
        self._strategy_score_history.append(self.strategy_scores.copy())
        self._pop_cov_history.append({
            'eff_dim': eff_dim,
            'cond_num': cond_num,
            'eigenvalues': eigenvalues.copy()
        })

        # Keep history bounded
        if len(self._strategy_score_history) > 50:
            self._strategy_score_history.pop(0)
        if len(self._pop_cov_history) > 50:
            self._pop_cov_history.pop(0)
```

# --- From variant_04_catD_t16_idea_0.py (2 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Fitness-landscape / rank-based strategy scoring using Kendall tau correlation and EWM."""
        n_strategies = len(self.strategy_scores)

        # Initialize tracking state
        if not hasattr(self, '_prev_fitness_ranks'):
            self._prev_fitness_ranks = np.full(self.NP, 0.5)

        if not hasattr(self, '_success_ewm'):
            self._success_ewm = np.zeros(n_strategies)

        if not hasattr(self, '_rank_gain_ewm'):
            self._rank_gain_ewm = np.zeros(n_strategies)

        if not hasattr(self, '_score_history'):
            self._score_history = []

        # Compute current fitness ranks (normalized to [0, 1])
        if hasattr(self, '_current_fitness'):
            current_ranks = self._compute_fitness_ranking(self._current_fitness)
        else:
            current_ranks = self._prev_fitness_ranks.copy()

        # Compute per-strategy success rate and average rank improvement
        success_rate = np.zeros(n_strategies)
        avg_rank_gain = np.zeros(n_strategies)

        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() == 0:
                continue

            participants = mask.sum()
            successes = improved[mask].sum()
            success_rate[s] = successes / participants

            # Rank gain: how much did participants improve (higher = better)
            prev_ranks = self._prev_fitness_ranks[mask]
            curr_ranks = current_ranks[mask]
            rank_gains = prev_ranks - curr_ranks
            avg_rank_gain[s] = np.mean(rank_gains)

        # Kendall tau-b correlation between strategy usage and rank improvement
        tau_scores = np.zeros(n_strategies)
        rank_improvement = self._prev_fitness_ranks - current_ranks

        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() >= 2:
                # Binary indicator: 1 if strategy s was used, 0 otherwise
                usage_indicator = mask.astype(float)
                # Correlation between usage and rank improvement
                tau = np.corrcoef(usage_indicator, rank_improvement)[0, 1]
                tau_scores[s] = np.clip(np.nan_to_num(tau), -1, 1)

        # Update EWM estimates with momentum
        self._success_ewm = 0.7 * self._success_ewm + 0.3 * success_rate
        self._rank_gain_ewm = 0.7 * self._rank_gain_ewm + 0.3 * avg_rank_gain

        # Composite score: success EWM + rank gain EWM + Kendall tau
        composite = (
            self._success_ewm * 2.0 +
            np.clip(self._rank_gain_ewm, 0, 1) * 1.5 +
            np.clip(tau_scores, 0, 1) * 2.0
        )

        # Blend with previous scores for stability
        self.strategy_scores = 0.7 * self.strategy_scores + 0.3 * composite
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)

        # Trend detection: boost strategies with improving scores
        self._score_history.append(self.strategy_scores.copy())
        if len(self._score_history) > 10:
            self._score_history.pop(0)

        if len(self._score_history) >= 3:
            recent = self._score_history[-1]
            older = self._score_history[-3]
            score_trend = recent - older
            self.strategy_scores += 0.05 * score_trend
            self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)

        # Store current ranks for next iteration
        self._prev_fitness_ranks = current_ranks
```

# --- From variant_05_catE_t18_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Topology-aware strategy credit using k-NN graph centrality weights."""
        NP = len(strategy_used)

        if NP < 5 or not hasattr(self, '_current_population') or self._current_population is None:
            # Fallback: simple success-rate update
            for s in range(len(self.strategy_scores)):
                mask = strategy_used == s
                if np.any(mask):
                    success_rate = np.mean(improved[mask])
                    self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * success_rate
            return

        pop = self._current_population

        # Build k-NN graph
        k = max(2, min(5, NP // 8))

        # Compute pairwise squared distances
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # Find k nearest neighbors for each individual
        nearest = np.argsort(sq_dists, axis=1)[:, :k]

        # Compute graph centrality: count how often each individual appears in others' neighborhoods
        centrality = np.zeros(NP)
        for i in range(NP):
            centrality[nearest[i]] += 1

        # Normalize centrality to [0, 1]
        if centrality.max() > 0:
            centrality /= centrality.max()
        else:
            centrality = np.ones(NP) / NP

        # Also compute local clustering coefficient for each individual
        clustering = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest[i])
            if len(neighbors) >= 2:
                edges = 0
                possible = 0
                for ni in neighbors:
                    for nj in neighbors:
                        if ni < nj:
                            possible += 1
                            if nj in set(nearest[ni]):
                                edges += 1
                clustering[i] = edges / max(possible, 1)

        # Combine centrality and clustering into topology influence score
        # High centrality + low clustering = valuable bridge individuals
        topology_weight = 0.7 * centrality + 0.3 * (1.0 - clustering)
        topology_weight /= (topology_weight.max() + 1e-12)

        # Track smoothed topology history for adaptive weighting
        if not hasattr(self, '_topology_weight_history'):
            self._topology_weight_history = []
        self._topology_weight_history.append(topology_weight.copy())
        if len(self._topology_weight_history) > 5:
            self._topology_weight_history.pop(0)

        # Compute population fragmentation index from graph
        avg_centrality = np.mean(centrality)
        fragmentation = 1.0 - avg_centrality  # High when population is fragmented

        # Adaptive learning rate based on population state
        if fragmentation > 0.7:
            # Fragmented population: explore more, use higher learning rate
            learn_rate = 0.15
        elif fragmentation < 0.3:
            # Well-connected population: exploit, use lower learning rate
            learn_rate = 0.05
        else:
            learn_rate = 0.10

        # Update strategy scores using topology-weighted credit
        for s in range(len(self.strategy_scores)):
            mask = strategy_used == s
            if np.any(mask):
                # Weight by topology influence of improving individuals
                topology_weights = topology_weight[mask & improved]
                if len(topology_weights) > 0:
                    weighted_success = np.mean(topology_weights)
                else:
                    weighted_success = 0.0

                # Also compute raw success rate for fallback
                raw_success = np.mean(improved[mask])

                # Blend weighted and raw success
                combined = 0.7 * weighted_success + 0.3 * raw_success

                # Exponential moving average update
                self.strategy_scores[s] = 0.85 * self.strategy_scores[s] + learn_rate * combined

        # Apply softmax-like normalization to keep scores in reasonable range
        self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
```

# --- From variant_07_catG_t05_idea_0.py (7 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """
        Bootstrap-based strategy credit assignment with Monte Carlo integration.

        Uses bootstrap resampling to estimate confidence in strategy effectiveness,
        then Monte Carlo integration over random policy weightings to compute
        expected improvement under uncertainty.
        """
        n_strategies = len(self.strategy_scores)

        # Compute empirical success rate per strategy
        emp_success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                emp_success_rate[s] = improved[mask].mean()

        # Bootstrap resampling: estimate variance in success rates via resampling
        n_bootstrap = 50
        bootstrap_success_rates = np.zeros((n_bootstrap, n_strategies))

        for b in range(n_bootstrap):
            # Sample indices with replacement (Monte Carlo sampling of population)
            indices = np.random.randint(0, len(improved), len(improved))
            resampled_improved = improved[indices]
            resampled_strategy = strategy_used[indices]

            for s in range(n_strategies):
                mask = resampled_strategy == s
                if mask.sum() > 0:
                    bootstrap_success_rates[b, s] = resampled_improved[mask].mean()

        # Compute bootstrap statistics
        bootstrap_mean = bootstrap_success_rates.mean(axis=0)
        bootstrap_std = bootstrap_success_rates.std(axis=0)

        # Confidence weighting: inverse of coefficient of variation
        # Low variance = high confidence = stronger update signal
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = bootstrap_std / (bootstrap_mean + 1e-10)
            confidence = 1.0 / (cv + 1.0)
            confidence = np.where(np.isfinite(confidence), confidence, 0.5)

        # Monte Carlo integration over random policy weightings
        # Sample random convex combinations to estimate expected improvement
        n_mc = 20
        mc_expected_improvement = np.zeros(n_strategies)

        for mc in range(n_mc):
            # Sample random Dirichlet weights (policy distribution over simplex)
            raw_weights = np.random.rand(n_strategies)
            policy_weights = raw_weights / raw_weights.sum()

            # Expected improvement under this random policy
            for s in range(n_strategies):
                mc_expected_improvement[s] += bootstrap_mean[s] * policy_weights[s]

        mc_expected_improvement /= n_mc

        # Combine estimates: bootstrap mean weighted by confidence + MC integration
        combined_estimate = 0.6 * bootstrap_mean * confidence + 0.4 * mc_expected_improvement

        # Normalize to get update direction
        combined_estimate = combined_estimate / (combined_estimate.sum() + 1e-10)

        # Compute delta from uniform baseline (each strategy equally likely)
        delta = combined_estimate - 1.0 / n_strategies

        # Apply update with momentum
        lr = 0.1
        momentum = 0.3

        if not hasattr(self, '_score_momentum'):
            self._score_momentum = np.zeros(n_strategies)

        self._score_momentum = momentum * self._score_momentum + (1 - momentum) * delta
        self.strategy_scores += lr * self._score_momentum
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
```

# --- From variant_08_catH_t10_idea_0.py (1 wins) ---
```python
def _update_strategy_scores(self, strategy_used, improved):
        """Hybrid credit: EMA success rate + rank improvement, weighted by sample confidence."""
        n_strategies = len(self.strategy_scores)
        NP = len(strategy_used)

        # Initialize tracking attributes if needed
        if not hasattr(self, '_strategy_ema_success'):
            self._strategy_ema_success = np.ones(n_strategies) * 0.5
            self._strategy_ema_rank_gain = np.zeros(n_strategies)
            self._strategy_samples = np.zeros(n_strategies)
            self._strategy_rank_history = [[] for _ in range(n_strategies)]

        # Compute per-individual rank improvement (how much did trial improve relative rank)
        if not hasattr(self, '_prev_fitness_ranks'):
            self._prev_fitness_ranks = np.ones(NP) * 0.5

        current_ranks = np.ones(NP) * 0.5
        for i in range(NP):
            if improved[i] and self._prev_fitness_ranks[i] < 0.99:
                rank_gain = self._prev_fitness_ranks[i] - 0.5  # Improvement from median
            else:
                rank_gain = 0.0
            current_ranks[i] = rank_gain

        # Update per-strategy statistics
        for s in range(n_strategies):
            mask = strategy_used == s
            count = np.sum(mask)
            if count > 0:
                self._strategy_samples[s] += count
                success_rate = np.mean(improved[mask])
                mean_rank_gain = np.mean(current_ranks[mask])

                # EMA update with momentum (higher momentum = more stable)
                ema_alpha = 0.3  # Base learning rate
                self._strategy_ema_success[s] = (1 - ema_alpha) * self._strategy_ema_success[s] + ema_alpha * success_rate
                self._strategy_ema_rank_gain[s] = (1 - ema_alpha) * self._strategy_ema_rank_gain[s] + ema_alpha * mean_rank_gain

                # Track rank history for variance estimation
                self._strategy_rank_history[s].extend(current_ranks[mask].tolist())
                # Keep bounded history
                if len(self._strategy_rank_history[s]) > 50:
                    self._strategy_rank_history[s] = self._strategy_rank_history[s][-50:]

        # Compute confidence weights based on sample count and signal consistency
        min_samples = 5
        confidence = np.clip(self._strategy_samples / (self._strategy_samples + min_samples), 0.0, 0.95)

        # Also weight by inverse variance of rank gains (stable signals get higher confidence)
        for s in range(n_strategies):
            history = self._strategy_rank_history[s]
            if len(history) >= 5:
                var = np.var(history) + 1e-8
                # Lower variance = higher confidence (capped at 1.5x boost)
                variance_factor = np.clip(1.0 / (1.0 + np.sqrt(var)), 0.5, 1.5)
                confidence[s] *= variance_factor

        # Normalize confidence
        confidence_sum = confidence.sum() + 1e-10
        confidence /= confidence_sum

        # Hybrid score: confidence-weighted combination of success rate and rank improvement
        # Success rate component (normalized to [0,1])
        success_norm = self._strategy_ema_success.copy()

        # Rank gain component (shifted to be positive, normalized)
        rank_gain_norm = np.zeros(n_strategies)
        rank_offset = abs(self._strategy_ema_rank_gain.min()) + 0.1
        for s in range(n_strategies):
            rank_gain_norm[s] = self._strategy_ema_rank_gain[s] + rank_offset
        rank_gain_norm /= rank_gain_norm.max() + 1e-10

        # Combine with confidence weighting
        # When confidence is high, trust the signal more; when low, blend toward uniform
        hybrid_score = np.zeros(n_strategies)
        uniform_prior = 1.0 / n_strategies

        for s in range(n_strategies):
            signal_quality = 0.6 * success_norm[s] + 0.4 * rank_gain_norm[s]
            hybrid_score[s] = confidence[s] * signal_quality + (1 - confidence[s]) * uniform_prior

        # Apply softmax for final selection weights
        hybrid_score -= hybrid_score.max()
        exp_scores = np.exp(hybrid_score * 2.0)  # Temperature = 0.5
        self.strategy_scores = np.log(exp_scores / exp_scores.sum() + 1e-10) + 1.0

        # Store current ranks for next iteration
        self._prev_fitness_ranks = current_ranks.copy()
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