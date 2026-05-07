Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_adapt_parameters` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.118196e-07            7.628715e-06            -inf                    -inf                    5.402258e-06            3.960173e-05            2.470362e-06            -inf                    5.635811e-06            -inf                    
1      1.923990e+00            3.084383e+00            -inf                    -inf                    3.071493e+00            5.153203e+00            2.846461e+00            -inf                    3.302157e+00            -inf                    
2      5.657046e-05            1.249235e-02            -inf                    -inf                    1.628440e-01            4.983648e-03            9.876828e-02            -inf                    3.292221e-03            -inf                    
3      2.592829e+00            1.563292e+01            -inf                    -inf                    1.340172e+01            2.146254e+01            1.272723e+01            -inf                    1.338801e+01            -inf                    
4      8.468928e-01            1.290528e+00            -inf                    -inf                    1.089013e+00            1.150488e+00            1.114175e+00            -inf                    1.185333e+00            -inf                    
5      1.346859e+02            3.079394e+03            -inf                    -inf                    2.825697e+03            1.410643e+04            2.584303e+03            -inf                    2.215994e+03            -inf                    
6      2.654227e+02            6.254004e+02            -inf                    -inf                    6.295850e+02            7.427201e+02            6.701253e+02            -inf                    6.562260e+02            -inf                    
7      6.396815e+00            1.242247e+01            -inf                    -inf                    1.126778e+01            1.811843e+01            1.042054e+01            -inf                    1.149409e+01            -inf                    
8      1.599741e+00            3.541328e+00            -inf                    -inf                    3.679411e+00            4.111575e+00            3.169728e+00            -inf                    3.307877e+00            -inf                    
9      4.692224e+01            6.297823e+01            -inf                    -inf                    5.365337e+01            6.545098e+01            5.952325e+01            -inf                    6.392269e+01            -inf                    
10     1.081160e+00            6.152591e+00            -inf                    -inf                    4.282254e+00            2.068373e-01            3.402376e+01            -inf                    5.522134e+00            -inf                    
11     4.068769e+02            3.851425e+02            -inf                    -inf                    3.681685e+02            8.158755e+02            6.976095e+02            -inf                    3.855575e+02            -inf                    
12     2.837961e+00            6.556655e+00            -inf                    -inf                    7.178531e+00            1.225136e+00            5.058931e+01            -inf                    7.383100e+00            -inf                    
13     3.128272e+01            3.263299e+01            -inf                    -inf                    3.199608e+01            4.023288e+01            3.944534e+01            -inf                    3.220242e+01            -inf                    
14     7.427308e+00            4.244116e+00            -inf                    -inf                    3.738696e+00            9.390656e+00            4.277287e+00            -inf                    3.699314e+00            -inf                    
15     3.591257e+00            3.532027e+00            -inf                    -inf                    3.523211e+00            3.948868e+00            3.822625e+00            -inf                    3.513604e+00            -inf                    
16     5.372293e+03            5.428855e+03            -inf                    -inf                    5.701450e+03            4.549530e+03            5.701125e+03            -inf                    6.740597e+03            -inf                    
17     7.239167e+04            7.457004e+04            -inf                    -inf                    7.563400e+04            2.061464e+05            1.610294e+05            -inf                    8.064829e+04            -inf                    
18     5.912787e+01            5.799658e+01            -inf                    -inf                    5.472906e+01            8.113105e+01            7.629051e+01            -inf                    5.607346e+01            -inf                    
19     1.185155e+02            8.290554e+01            -inf                    -inf                    1.458282e+02            2.072199e+02            1.434593e+02            -inf                    9.280179e+01            -inf                    
20     2.008751e+01            2.064489e+01            -inf                    -inf                    2.015844e+01            2.012003e+01            2.612240e+01            -inf                    2.031622e+01            -inf                    
21     5.688817e+00            5.647390e+00            -inf                    -inf                    5.678037e+00            5.917791e+00            5.720989e+00            -inf                    5.696816e+00            -inf                    
22     1.318928e+01            1.277766e+01            -inf                    -inf                    1.257918e+01            1.778442e+01            1.584128e+01            -inf                    1.303120e+01            -inf                    
23     5.451930e+01            4.531310e+01            -inf                    -inf                    4.431743e+01            5.277494e+01            4.802243e+01            -inf                    4.349616e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.118196e-07)
Task  1: original.py  (error=1.923990e+00)
Task  2: original.py  (error=5.657046e-05)
Task  3: original.py  (error=2.592829e+00)
Task  4: original.py  (error=8.468928e-01)
Task  5: original.py  (error=1.346859e+02)
Task  6: original.py  (error=2.654227e+02)
Task  7: original.py  (error=6.396815e+00)
Task  8: original.py  (error=1.599741e+00)
Task  9: original.py  (error=4.692224e+01)
Task 10: variant_06_catF_idea_0.py  (error=2.068373e-01)
Task 11: variant_04_catD_idea_0.py  (error=3.681685e+02)
Task 12: variant_06_catF_idea_0.py  (error=1.225136e+00)
Task 13: original.py  (error=3.128272e+01)
Task 14: variant_09_catA_idea_0.py  (error=3.699314e+00)
Task 15: variant_09_catA_idea_0.py  (error=3.513604e+00)
Task 16: variant_06_catF_idea_0.py  (error=4.549530e+03)
Task 17: original.py  (error=7.239167e+04)
Task 18: variant_04_catD_idea_0.py  (error=5.472906e+01)
Task 19: variant_01_catA_idea_0.py  (error=8.290554e+01)
Task 20: original.py  (error=2.008751e+01)
Task 21: variant_01_catA_idea_0.py  (error=5.647390e+00)
Task 22: variant_04_catD_idea_0.py  (error=1.257918e+01)
Task 23: variant_09_catA_idea_0.py  (error=4.349616e+01)

WIN COUNTS:
  original.py: 13 wins
  variant_06_catF_idea_0.py: 3 wins
  variant_04_catD_idea_0.py: 3 wins
  variant_09_catA_idea_0.py: 3 wins
  variant_01_catA_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (2 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using geometric/spatial population characteristics."""
        population = self.population if hasattr(self, 'population') else None

        if not hasattr(self, 'geo_centroid_history'):
            self.geo_centroid_history = []
            self.geo_spread_history = []

        if population is not None and len(population) > 0:
            # Compute population centroid
            centroid = np.mean(population, axis=0)

            # Track centroid drift (temporal, but only as geometric signal)
            self.geo_centroid_history.append(centroid.copy())
            if len(self.geo_centroid_history) > 10:
                self.geo_centroid_history.pop(0)

            centroid_drift = 0.0
            if len(self.geo_centroid_history) >= 2:
                centroid_drift = np.linalg.norm(self.geo_centroid_history[-1] - self.geo_centroid_history[0])

            # Geometric spread: average distance to centroid
            dist_to_centroid = np.linalg.norm(population - centroid, axis=1)
            mean_dist = np.mean(dist_to_centroid)
            std_dist = np.std(dist_to_centroid)

            # Normalize spread relative to search space
            search_range = self.upper - self.lower
            normalized_spread = mean_dist / (search_range + 1e-10)

            # Axis-aligned bounding box volume (log scale)
            pop_min = np.min(population, axis=0)
            pop_max = np.max(population, axis=0)
            bbox_sizes = pop_max - pop_min
            bbox_volume = np.prod(bbox_sizes + 1e-10)
            log_bbox_volume = np.log(bbox_volume + 1e-30)
            max_log_volume = np.log((search_range + 1e-10) ** self.dim)
            normalized_bbox = np.clip(log_bbox_volume / (max_log_volume + 1e-10), 0.0, 1.0)

            # k-NN connectivity: average distance to 5 nearest neighbors
            k = min(5, len(population) - 1)
            knn_distances = []
            for i in range(len(population)):
                distances = np.linalg.norm(population - population[i], axis=1)
                distances[i] = np.inf
                knn_distances.append(np.mean(np.partition(distances, k)[:k]))
            mean_knn_dist = np.mean(knn_distances)
            normalized_knn = np.clip(mean_knn_dist / (search_range * 0.5 + 1e-10), 0.0, 1.0)

            # Store spread for history
            self.geo_spread_history.append(normalized_spread)
            if len(self.geo_spread_history) > 15:
                self.geo_spread_history.pop(0)

            # Compute spread trend (increasing = exploring, decreasing = converging)
            spread_trend = 0.0
            if len(self.geo_spread_history) >= 3:
                recent = np.mean(self.geo_spread_history[-2:])
                older = np.mean(self.geo_spread_history[:2])
                spread_trend = (recent - older) / (older + 1e-10)

            # Combine geometric signals into adaptation factors
            # Low spread + low knn = clustered population → need more exploration
            # High spread + high knn = well-distributed → can exploit more

            exploration_pressure = 1.0 - normalized_spread  # Higher when clustered
            exploration_pressure += (1.0 - normalized_knn) * 0.5
            exploration_pressure = np.clip(exploration_pressure, 0.0, 2.0)

            # Adjustments based on geometry
            if exploration_pressure > 1.2:
                # Population is clustered - increase exploration
                F_adjustment = 1.0 + 0.15 * exploration_pressure
                Cr_adjustment = 0.90
            elif exploration_pressure < 0.5:
                # Population is well spread - increase exploitation
                F_adjustment = 0.85
                Cr_adjustment = 1.0 + 0.1 * (0.5 - exploration_pressure)
            else:
                # Balanced regime - use spread trend
                F_adjustment = 1.0 + 0.05 * spread_trend
                Cr_adjustment = 1.0 - 0.03 * spread_trend

            # Factor in centroid drift (high drift = population moving = exploration needed)
            if centroid_drift > 0.1 * search_range:
                F_adjustment *= 1.1
                Cr_adjustment *= 0.95

            # Apply adjustments
            self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
        else:
            # Fallback: gentle reversion if no population available
            self.F = np.clip(self.F * 0.98 + self.F_base * 0.02, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * 0.98 + self.Cr_base * 0.02, 0.1, 0.9)
```

# --- From variant_04_catD_idea_0.py (3 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using Spearman rank correlation and fitness percentile landscape analysis."""
        from scipy.stats import spearmanr

        # Get current fitness from stored state
        if not hasattr(self, '_cached_fitness'):
            return
        current_fitness = self._cached_fitness
        pop_size = len(current_fitness)

        # Initialize rank tracking on first call
        if not hasattr(self, 'fitness_rank_history'):
            self.fitness_rank_history = []
            self.improvement_percentile_history = []
            self.fitness_spread_history = []

        # Compute current fitness ranks (pure fitness signal)
        current_ranks = np.argsort(np.argsort(current_fitness)) / max(pop_size - 1, 1)

        # Compute Spearman rank correlation between consecutive generations
        if len(self.fitness_rank_history) >= 1:
            prev_ranks = self.fitness_rank_history[-1]
            rank_corr, _ = spearmanr(prev_ranks, current_ranks)
            if np.isnan(rank_corr):
                rank_corr = 0.5
        else:
            rank_corr = 0.5

        # Compute improvement magnitude percentiles (fitness signal: how good are the wins?)
        if np.sum(improved_mask) > 0 and hasattr(self, '_cached_trial_fitness'):
            trial_fitness = self._cached_trial_fitness
            fitness_diff = current_fitness - trial_fitness
            improvements = fitness_diff[improved_mask]
            if len(improvements) > 0:
                improv_percentile = np.percentile(improvements, 75)
            else:
                improv_percentile = 0.0
        else:
            improv_percentile = 0.0

        # Compute fitness percentile spread (landscape clustering signal)
        fit_25 = np.percentile(current_fitness, 25)
        fit_75 = np.percentile(current_fitness, 75)
        fit_mean = np.mean(current_fitness)
        fit_spread = (fit_75 - fit_25) / (np.abs(fit_mean) + 1e-10)

        # Regime detection via rank-based signals
        high_rank_stability = rank_corr > 0.75
        low_rank_stability = rank_corr < 0.4
        tight_clustering = fit_spread < 0.15
        small_improvements = improv_percentile < 0.25 * (fit_75 - fit_25 + 1e-10)

        # Fitness-landscape / rank-based adaptation logic
        if high_rank_stability and tight_clustering:
            # CONVERGENCE TRAP: ranks stable + population clustered = stuck in local optimum
            # Need aggressive exploration with high F
            F_adjustment = 1.35
            Cr_adjustment = 0.75
        elif low_rank_stability and improv_percentile > 0:
            # ACTIVE EXPLORATION: ranks shuffling + good improvements = healthy search
            # Can afford more exploitation with lower F
            F_adjustment = 0.85
            Cr_adjustment = 1.1
        elif small_improvements and tight_clustering:
            # DEEP TRAP: tiny wins in tight cluster = deceptive local optimum
            # Need maximum perturbation
            F_adjustment = 1.4
            Cr_adjustment = 0.7
        elif high_rank_stability and not tight_clustering:
            # RANK STABLE BUT SPREAD: converging to good region
            # Moderate exploitation
            F_adjustment = 0.9
            Cr_adjustment = 1.05
        else:
            # NEUTRAL: rank correlation moderate = normal search dynamics
            # Gentle correction based on rank correlation magnitude
            F_adjustment = 0.95 + 0.1 * (1.0 - rank_corr)
            Cr_adjustment = 0.95 + 0.15 * rank_corr

        # Apply adjustments with bounds
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)

        # Store current state for next generation's rank correlation
        self.fitness_rank_history.append(current_ranks.copy())
        if len(self.fitness_rank_history) > 8:
            self.fitness_rank_history.pop(0)

        self.improvement_percentile_history.append(improv_percentile)
        if len(self.improvement_percentile_history) > 8:
            self.improvement_percentile_history.pop(0)

        self.fitness_spread_history.append(fit_spread)
        if len(self.fitness_spread_history) > 8:
            self.fitness_spread_history.pop(0)
```

# --- From variant_06_catF_idea_0.py (3 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using autocorrelation-based regime detection and parameter momentum."""
        success_rate = np.mean(improved_mask)

        # Initialize temporal state on first call
        if not hasattr(self, 'success_history'):
            self.success_history = []
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.F_momentum = 0.0
            self.Cr_momentum = 0.0
            self.stagnation_counter = 0
            self.prev_best_fitness = np.inf
            self.best_fitness_streak = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        # Update EMAs with adaptive smoothing
        alpha_short = 0.4
        alpha_long = 0.15
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        # Track rolling success history for autocorrelation
        self.success_history.append(success_rate)
        if len(self.success_history) > 12:
            self.success_history.pop(0)

        # Compute autocorrelation at lag-1: detects oscillatory vs steady behavior
        if len(self.success_history) >= 4:
            sh = np.array(self.success_history)
            n = len(sh)
            mean_s = np.mean(sh)
            var_s = np.var(sh) + 1e-10
            autocorr_lag1 = np.mean((sh[:-1] - mean_s) * (sh[1:] - mean_s)) / var_s
        else:
            autocorr_lag1 = 0.5  # Default to no oscillation

        # Compute success rate derivative (acceleration/deceleration)
        success_derivative = success_rate - self.success_ewma_long

        # Track parameter history for momentum and drift
        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)

        # Compute parameter momentum (rate of change)
        if len(self.F_history) >= 3:
            F_velocity = (self.F_history[-1] - self.F_history[-3]) / 2.0
            Cr_velocity = (self.Cr_history[-1] - self.Cr_history[-3]) / 2.0
            self.F_momentum = 0.7 * self.F_momentum + 0.3 * F_velocity
            self.Cr_momentum = 0.7 * self.Cr_momentum + 0.3 * Cr_velocity

        # Detect stagnation via best fitness not improving
        current_best = np.min(self.population.fitness) if hasattr(self, 'population') else np.inf
        if current_best < self.prev_best_fitness - 1e-10:
            self.best_fitness_streak += 1
        else:
            self.best_fitness_streak = 0
        self.prev_best_fitness = current_best

        # Regime detection via autocorrelation and derivative
        oscillating = autocorr_lag1 < -0.15  # Alternating success/failure
        steady_converging = autocorr_lag1 > 0.5 and success_rate > 0.2
        accelerating = success_derivative > 0.05
        decelerating = success_derivative < -0.05
        stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
        improving_streak = self.best_fitness_streak > 10

        # Compute momentum factor to damp runaway parameters
        F_hist_std = np.std(self.F_history) + 1e-10
        momentum_damp = np.clip(1.0 - 0.3 * abs(self.F_momentum) / F_hist_std, 0.6, 1.0)

        if stagnant or improving_streak and success_rate < 0.15:
            # Deep stagnation: reset with perturbation, damp momentum
            F_adjustment = 0.7 * momentum_damp
            Cr_adjustment = 1.2
            self.F_momentum *= 0.5
            self.Cr_momentum *= 0.5
            self.stagnation_counter += 2
        elif oscillating:
            # Oscillatory regime: counteract momentum, increase stability
            F_adjustment = 0.85 * momentum_damp
            Cr_adjustment = 0.9 - 0.2 * autocorr_lag1
            self.stagnation_counter += 1
        elif accelerating and steady_converging:
            # Exploitation: success accelerating steadily, increase convergence pressure
            F_adjustment = 1.08 * momentum_damp
            Cr_adjustment = 1.05
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        elif decelerating and not oscillating:
            # Deceleration without oscillation: moderate diversity injection
            F_adjustment = 0.92
            Cr_adjustment = 0.95
            self.stagnation_counter += 1
        else:
            # Neutral regime: gentle correction toward baseline
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.1 * delta
            Cr_adjustment = 1.0 - 0.05 * delta

        # Apply adjustments with momentum correction
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```

# --- From variant_09_catA_idea_0.py (3 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using geometric properties of population layout."""
        # Access stored population from optimization loop
        if not hasattr(self, 'stored_population') or self.stored_population is None:
            self.F = np.clip(self.F * 1.0, 0.1, 1.5)
            self.Cr = np.clip(self.Cr * 1.0, 0.1, 0.9)
            return

        pop = self.stored_population
        if len(pop) < 2:
            return

        # --- Geometric Feature Computation ---
        # 1. Axis-aligned spread per dimension (geometric range)
        dim_min = np.min(pop, axis=0)
        dim_max = np.max(pop, axis=0)
        dim_spread = dim_max - dim_min
        total_spread = np.sum(dim_spread)
        range_normalized_spread = total_spread / (self.dim * (self.upper - self.lower) + 1e-10)

        # 2. Centroid position relative to bounds (geometric centrality)
        centroid = np.mean(pop, axis=0)
        centroid_normalized = (centroid - self.lower) / (self.upper - self.lower + 1e-10)
        corner_proximity = np.min(centroid_normalized)  # Low = near corner/boundary
        edge_bias = np.max(np.abs(centroid_normalized - 0.5))  # High = off-center

        # 3. Mean distance to centroid (geometric compactness)
        distances_to_centroid = np.linalg.norm(pop - centroid, axis=1)
        mean_dist = np.mean(distances_to_centroid)
        max_possible_dist = np.sqrt(self.dim) * (self.upper - self.lower) / 2
        compactness = mean_dist / (max_possible_dist + 1e-10)

        # 4. Pairwise distance statistics (clustering detection)
        centroid_dist_matrix = distances_to_centroid[:, np.newaxis] + distances_to_centroid[np.newaxis, :]
        pairwise_dist = np.linalg.norm(pop[:, np.newaxis, :] - pop[np.newaxis, :, :], axis=2)
        np.fill_diagonal(pairwise_dist, np.inf)
        mean_pairwise = np.mean(np.min(pairwise_dist, axis=1))
        max_pairwise = np.max(np.min(pairwise_dist, axis=1))
        clustering_ratio = mean_pairwise / (max_pairwise + 1e-10)

        # --- Geometric Regime Classification ---
        # Classify population state purely from spatial geometry
        is_tight = compactness < 0.15
        is_dispersed = compactness > 0.4
        is_corner_trapped = corner_proximity < 0.1 or edge_bias > 0.4
        is_clustered = clustering_ratio < 0.3

        # --- Parameter Adjustment via Geometry ---
        if is_tight and is_corner_trapped:
            # Population is compact AND near boundary corner → trapped, need aggressive exploration
            F_adjustment = 1.30
            Cr_adjustment = 0.80
        elif is_tight and not is_corner_trapped:
            # Compact but well-centered → exploit local region
            F_adjustment = 0.85
            Cr_adjustment = 1.10
        elif is_dispersed:
            # Wide spread → reduce F, allow convergence
            F_adjustment = 0.80
            Cr_adjustment = 1.05
        elif is_clustered and not is_tight:
            # Multiple subclusters → moderate F, higher Cr to recombine
            F_adjustment = 1.10
            Cr_adjustment = 0.90
        else:
            # Neutral geometric state → proportional adjustment based on spread
            F_adjustment = 1.0 + 0.20 * (range_normalized_spread - 0.3)
            Cr_adjustment = 1.0 - 0.15 * (range_normalized_spread - 0.3)

        # Apply geometrically-motivated adjustments
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
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
        """Greedy selection with diversity maintenance."""
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
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