Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_adapt_parameters` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_t16_id  variant_02_catB_t15_id  variant_03_catC_t12_id  variant_04_catD_t09_id  variant_05_catE_t10_id  variant_06_catF_t04_id  variant_07_catG_t05_id  variant_10_catB_t05_id  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.014452e-06            4.459151e-08            -inf                    -inf                    -inf                    4.259914e-08            1.327094e-01            8.283226e-01            -inf                    
1      5.455622e+00            3.083820e-05            -inf                    -inf                    -inf                    1.786280e-05            1.201165e+01            3.534255e-01            -inf                    
2      9.038567e-05            1.000000e-08            -inf                    -inf                    -inf                    1.000000e-08            1.435745e+01            1.491091e+01            -inf                    
3      2.683252e+00            1.401853e+00            -inf                    -inf                    -inf                    1.378117e+00            5.015354e+01            7.060034e+00            -inf                    
4      4.173622e-01            1.644354e-02            -inf                    -inf                    -inf                    1.999998e-08            1.761369e+00            1.217234e+00            -inf                    
5      4.852106e+01            3.306318e-05            -inf                    -inf                    -inf                    4.452708e-05            8.476447e+07            8.979913e+03            -inf                    
6      4.739891e+01            3.903814e+01            -inf                    -inf                    -inf                    4.173817e+01            3.934352e+02            4.876458e+01            -inf                    
7      2.329600e+00            7.195005e-02            -inf                    -inf                    -inf                    5.894093e-02            1.086336e+02            1.031538e+01            -inf                    
8      2.487993e+00            5.399196e-02            -inf                    -inf                    -inf                    3.545024e-02            2.030306e+01            2.781174e+00            -inf                    
9      3.784599e+01            9.520159e+00            -inf                    -inf                    -inf                    9.404178e+00            7.160975e+01            9.887447e+00            -inf                    
10     1.127008e-07            2.085314e-06            -inf                    -inf                    -inf                    1.870306e-05            1.403287e+02            7.973609e+01            -inf                    
11     3.576802e+02            6.915394e+01            -inf                    -inf                    -inf                    2.729658e+01            1.855122e+03            3.256135e+02            -inf                    
12     2.629879e-01            1.173325e+01            -inf                    -inf                    -inf                    1.188863e-01            1.783952e+02            2.155503e+01            -inf                    
13     1.312397e+01            2.753343e+00            -inf                    -inf                    -inf                    2.449321e+00            5.104847e+01            2.043809e+01            -inf                    
14     4.878480e+00            8.446624e+00            -inf                    -inf                    -inf                    8.226955e+00            9.618048e+00            6.406443e+00            -inf                    
15     3.745465e+00            3.076808e+00            -inf                    -inf                    -inf                    3.055214e+00            4.462100e+00            3.292007e+00            -inf                    
16     2.220722e+03            8.224661e+02            -inf                    -inf                    -inf                    1.018866e+03            5.280723e+03            9.044207e+02            -inf                    
17     3.095278e+04            1.046539e+03            -inf                    -inf                    -inf                    4.467852e+03            7.383986e+04            6.142309e+03            -inf                    
18     4.269711e+01            2.496681e+01            -inf                    -inf                    -inf                    3.212187e+01            6.398556e+01            4.889178e+01            -inf                    
19     1.059782e+02            1.623261e+02            -inf                    -inf                    -inf                    1.356591e+02            3.568707e+02            1.386011e+02            -inf                    
20     2.044660e+01            2.149571e+01            -inf                    -inf                    -inf                    2.391089e+01            3.799587e+01            3.250274e+01            -inf                    
21     4.859584e+00            5.179678e+00            -inf                    -inf                    -inf                    5.288898e+00            5.659800e+00            5.372394e+00            -inf                    
22     1.448734e+01            1.343048e+01            -inf                    -inf                    -inf                    1.348507e+01            2.202010e+01            1.471889e+01            -inf                    
23     4.193511e+01            4.127380e+01            -inf                    -inf                    -inf                    3.652570e+01            8.450506e+01            3.405153e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_05_catE_t10_idea_0.py  (error=4.259914e-08)
Task  1: variant_05_catE_t10_idea_0.py  (error=1.786280e-05)
Task  2: variant_01_catA_t16_idea_0.py  (error=1.000000e-08)
Task  3: variant_05_catE_t10_idea_0.py  (error=1.378117e+00)
Task  4: variant_05_catE_t10_idea_0.py  (error=1.999998e-08)
Task  5: variant_01_catA_t16_idea_0.py  (error=3.306318e-05)
Task  6: variant_01_catA_t16_idea_0.py  (error=3.903814e+01)
Task  7: variant_05_catE_t10_idea_0.py  (error=5.894093e-02)
Task  8: variant_05_catE_t10_idea_0.py  (error=3.545024e-02)
Task  9: variant_05_catE_t10_idea_0.py  (error=9.404178e+00)
Task 10: original.py  (error=1.127008e-07)
Task 11: variant_05_catE_t10_idea_0.py  (error=2.729658e+01)
Task 12: variant_05_catE_t10_idea_0.py  (error=1.188863e-01)
Task 13: variant_05_catE_t10_idea_0.py  (error=2.449321e+00)
Task 14: original.py  (error=4.878480e+00)
Task 15: variant_05_catE_t10_idea_0.py  (error=3.055214e+00)
Task 16: variant_01_catA_t16_idea_0.py  (error=8.224661e+02)
Task 17: variant_01_catA_t16_idea_0.py  (error=1.046539e+03)
Task 18: variant_01_catA_t16_idea_0.py  (error=2.496681e+01)
Task 19: original.py  (error=1.059782e+02)
Task 20: original.py  (error=2.044660e+01)
Task 21: original.py  (error=4.859584e+00)
Task 22: variant_01_catA_t16_idea_0.py  (error=1.343048e+01)
Task 23: variant_07_catG_t05_idea_0.py  (error=3.405153e+01)

WIN COUNTS:
  variant_05_catE_t10_idea_0.py: 11 wins
  variant_01_catA_t16_idea_0.py: 7 wins
  original.py: 5 wins
  variant_07_catG_t05_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_05_catE_t10_idea_0.py   4.2599e-08       variant_01_catA_t16_idea_0.py    4.4592e-08         1.0×    >1e6×
     1   variant_05_catE_t10_idea_0.py   1.7863e-05       variant_01_catA_t16_idea_0.py    3.0838e-05         1.7×  162602×
     2   variant_01_catA_t16_idea_0.py   1.0000e-08       variant_05_catE_t10_idea_0.py    1.0000e-08         1.0×    >1e6×
     3   variant_05_catE_t10_idea_0.py   1.3781e+00       variant_01_catA_t16_idea_0.py    1.4019e+00         1.0×    3.5×
     4   variant_05_catE_t10_idea_0.py   2.0000e-08       variant_01_catA_t16_idea_0.py    1.6444e-02       822177×    >1e6×
     5   variant_01_catA_t16_idea_0.py   3.3063e-05       variant_05_catE_t10_idea_0.py    4.4527e-05         1.3×    >1e6×
     6   variant_01_catA_t16_idea_0.py   3.9038e+01       variant_05_catE_t10_idea_0.py    4.1738e+01         1.1×    1.2×
     7   variant_05_catE_t10_idea_0.py   5.8941e-02       variant_01_catA_t16_idea_0.py    7.1950e-02         1.2×  107.3×
     8   variant_05_catE_t10_idea_0.py   3.5450e-02       variant_01_catA_t16_idea_0.py    5.3992e-02         1.5×   74.3×
     9   variant_05_catE_t10_idea_0.py   9.4042e+00       variant_01_catA_t16_idea_0.py    9.5202e+00         1.0×    2.5×
    10   original.py                     1.1270e-07       variant_01_catA_t16_idea_0.py    2.0853e-06        18.5×    >1e6×
    11   variant_05_catE_t10_idea_0.py   2.7297e+01       variant_01_catA_t16_idea_0.py    6.9154e+01         2.5×   12.5×
    12   variant_05_catE_t10_idea_0.py   1.1889e-01       original.py                      2.6299e-01         2.2×  140.0×
    13   variant_05_catE_t10_idea_0.py   2.4493e+00       variant_01_catA_t16_idea_0.py    2.7533e+00         1.1×    6.9×
    14   original.py                     4.8785e+00       variant_07_catG_t05_idea_0.py    6.4064e+00         1.3×    1.7×
    15   variant_05_catE_t10_idea_0.py   3.0552e+00       variant_01_catA_t16_idea_0.py    3.0768e+00         1.0×    1.2×
    16   variant_01_catA_t16_idea_0.py   8.2247e+02       variant_07_catG_t05_idea_0.py    9.0442e+02         1.1×    2.0×
    17   variant_01_catA_t16_idea_0.py   1.0465e+03       variant_05_catE_t10_idea_0.py    4.4679e+03         4.3×   17.7×
    18   variant_01_catA_t16_idea_0.py   2.4967e+01       variant_05_catE_t10_idea_0.py    3.2122e+01         1.3×    1.8×
    19   original.py                     1.0598e+02       variant_05_catE_t10_idea_0.py    1.3566e+02         1.3×    1.4×
    20   original.py                     2.0447e+01       variant_01_catA_t16_idea_0.py    2.1496e+01         1.1×    1.4×
    21   original.py                     4.8596e+00       variant_01_catA_t16_idea_0.py    5.1797e+00         1.1×    1.1×
    22   variant_01_catA_t16_idea_0.py   1.3430e+01       variant_05_catE_t10_idea_0.py    1.3485e+01         1.0×    1.1×
    23   variant_07_catG_t05_idea_0.py   3.4052e+01       variant_05_catE_t10_idea_0.py    3.6526e+01         1.1×    1.2×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 4 (original.py, variant_01_catA_t16_idea_0.py, variant_05_catE_t10_idea_0.py, variant_07_catG_t05_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 9  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 6  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 2 — variant_01_catA_t16_idea_0.py (1.00e-08) vs variant_05_catE_t10_idea_0.py (1.00e-08) = 1.0× gap to runner-up, 717877193.8× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.00e-08, which is ~1.0× WORSE than the winner alone.

MANDATORY MECHANISM: per-task fingerprint-and-commit (the PROBE-AND-COMMIT pattern below — a specialisation of mechanism #4). Mechanism #2 (ensemble/blending) is EXPLICITLY FORBIDDEN: the per-task gap table shows multiple tasks where the winner beats the runner-up by >=100×, and a linear blend of the two cannot recover the winner's value. You may also use mechanism #3 (rule-based dispatch) if you can derive a CHEAP, OBSERVABLE rule that maps a landscape fingerprint to a winner.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_t16_idea_0.py (7 wins) ---
```python
def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on geometric spatial spread of population."""
        if not hasattr(self, '_pop_history'):
            self._pop_history = []

        # Get current population from the optimizer context
        # We access it via self's stored reference or reconstruct from available state
        # Since we don't have direct population access, we track spread via stored metrics

        # Compute axis-aligned spread (range per dimension)
        # This requires population - we store it in a class attribute set by __call__
        if hasattr(self, '_current_population') and self._current_population is not None:
            pop = self._current_population
            pop_min = pop.min(axis=0)
            pop_max = pop.max(axis=0)
            axis_spread = pop_max - pop_min

            # Normalize spread relative to search space (assuming [-100, 100])
            normalized_spread = axis_spread / 200.0
            mean_spread = normalized_spread.mean()
            max_spread = normalized_spread.max()

            # Compute pairwise distances from centroid
            centroid = pop.mean(axis=0)
            centroid_distances = np.linalg.norm(pop - centroid, axis=1)
            mean_centroid_dist = centroid_distances.mean()

            # Normalize by expected random spread (roughly sqrt(dim) * range/sqrt(NP))
            expected_dist = 200.0 * np.sqrt(self.dim / self.NP)
            normalized_centroid_dist = mean_centroid_dist / expected_dist

            # Geometric composite: combine spread metrics
            # Low spread -> population clustered -> need larger F to explore
            # High spread -> population dispersed -> reduce F, increase CR for exploitation
            spread_factor = np.clip(normalized_centroid_dist, 0.1, 2.0)

            # F adaptation: inversely proportional to spread
            # When spread is low (clustered), increase F significantly
            target_F = 0.6 / (spread_factor + 0.3)
            target_F = np.clip(target_F, 0.3, 1.5)

            # CR adaptation: proportional to spread (more exploration when dispersed)
            target_CR = 0.5 + 0.4 * np.clip(spread_factor - 0.5, 0, 1)
            target_CR = np.clip(target_CR, 0.3, 0.95)

            # Smooth transition with momentum
            if len(self.F_history) > 0:
                alpha = 0.3
                self.F = alpha * target_F + (1 - alpha) * self.F_history[-1]
                self.CR = alpha * target_CR + (1 - alpha) * self.CR_history[-1]
            else:
                self.F = target_F
                self.CR = target_CR

            # Clamp final values
            self.F = float(np.clip(self.F, 0.1, 2.0))
            self.CR = float(np.clip(self.CR, 0.1, 0.99))

            self.F_history.append(self.F)
            self.CR_history.append(self.CR)

            # Keep history bounded
            if len(self.F_history) > 50:
                self.F_history = self.F_history[-50:]
                self.CR_history = self.CR_history[-50:]
```

# --- From variant_05_catE_t10_idea_0.py (11 wins) ---
```python
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
```

# --- From variant_07_catG_t05_idea_0.py (1 wins) ---
```python
def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR using bootstrap confidence intervals of improvement rates.
        Category G: Stochastic sampling-based approach using bootstrap estimation."""

        # Maintain sliding window of improvement rates
        if not hasattr(self, '_improvement_history'):
            self._improvement_history = []

        self._improvement_history.append(improvement_rate)
        if len(self._improvement_history) > 50:
            self._improvement_history.pop(0)

        n_history = len(self._improvement_history)
        if n_history < 5:
            # Not enough data: use random walk with small variance
            self.F = np.clip(self.F + np.random.randn() * 0.05, 0.3, 1.5)
            self.CR = np.clip(self.CR + np.random.randn() * 0.03, 0.3, 0.99)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return

        # Bootstrap resampling: fundamentally stochastic computation
        history_array = np.array(self._improvement_history)
        n_bootstrap = 100  # Monte Carlo samples

        # Bootstrap means: sample with replacement, compute mean for each
        bootstrap_indices = np.random.randint(0, n_history, size=(n_bootstrap, n_history))
        bootstrap_means = history_array[bootstrap_indices].mean(axis=1)

        # Construct bootstrap confidence interval
        ci_lower = np.percentile(bootstrap_means, 10)
        ci_upper = np.percentile(bootstrap_means, 90)
        ci_width = max(ci_upper - ci_lower, 0.01)

        # Sample F and CR stochastically within confidence interval
        # Higher improvement rate -> scale up F (more exploration) and CR (more recombination)
        baseline_F = 0.6
        baseline_CR = 0.85

        # Map CI position to parameter adjustment
        norm_rate = np.clip((ci_lower + ci_width * np.random.rand()) / max(ci_upper, 0.1), 0, 1)

        # F: lower improvement -> smaller F (exploit), higher improvement -> larger F (explore)
        target_F = baseline_F * (0.5 + 1.5 * norm_rate)
        # CR: inverse relationship with improvement rate
        target_CR = baseline_CR * (1.3 - 0.5 * norm_rate)

        # Stochastic smoothing: blend with momentum
        alpha = 0.3
        self.F = np.clip(alpha * target_F + (1 - alpha) * self.F + np.random.randn() * 0.05, 0.3, 1.5)
        self.CR = np.clip(alpha * target_CR + (1 - alpha) * self.CR + np.random.randn() * 0.03, 0.3, 0.99)

        # Record history
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)

        # Keep history bounded
        if len(self.F_history) > 100:
            self.F_history = self.F_history[-100:]
            self.CR_history = self.CR_history[-100:]
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
        """Adapt F and CR based on improvement rate."""
        # Momentum-based F adaptation
        if improvement_rate > 0.25:
            target_F = min(self.F * 1.15, 1.5)
        elif improvement_rate < 0.1:
            target_F = max(self.F * 0.85, 0.2)
        else:
            target_F = self.F
        self.F = 0.7 * self.F + 0.3 * target_F
        
        # CR adaptation toward success
        if improvement_rate > 0.2:
            target_CR = min(self.CR * 1.05, 0.98)
        else:
            target_CR = max(self.CR * 0.95, 0.4)
        self.CR = 0.6 * self.CR + 0.4 * target_CR
        
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