Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_mutate_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.148441e-07            6.682951e+00            -inf                    nan                     1.295053e-04            7.081466e-03            5.416009e-01            -inf                    1.479836e+00            6.298970e-02            
1      3.760932e+00            1.634347e+01            -inf                    nan                     4.027570e+00            3.784415e+00            1.143248e+01            -inf                    2.043940e+01            3.112068e+00            
2      7.425561e-05            3.767727e+01            -inf                    nan                     9.793779e-01            3.383508e-01            7.542253e+00            -inf                    1.127886e+00            5.829855e-01            
3      2.769512e+00            7.335725e+01            -inf                    nan                     2.247354e+01            2.614893e+00            5.391735e+01            -inf                    4.681580e+01            1.282978e+01            
4      9.949669e-01            2.738580e+00            -inf                    nan                     1.233049e+00            1.192279e+00            1.885398e+00            -inf                    4.197644e+00            1.234991e+00            
5      2.702739e+01            1.016139e+06            -inf                    nan                     4.712971e+03            1.603931e+01            4.179313e+05            -inf                    2.817262e+05            1.740717e+03            
6      1.258695e+02            6.809487e+02            -inf                    nan                     1.095469e+03            6.521732e+02            7.665603e+02            -inf                    5.638544e+02            6.773326e+02            
7      5.661689e+00            5.372500e+01            -inf                    nan                     1.934269e+01            3.273042e+00            5.551498e+01            -inf                    1.976840e+02            8.034046e+00            
8      2.364168e+00            1.398307e+01            -inf                    nan                     5.269928e+00            2.225932e+00            1.079033e+01            -inf                    1.620268e+01            3.391320e+00            
9      4.033458e+01            6.985999e+01            -inf                    nan                     1.721309e+02            4.211920e+01            1.065014e+02            -inf                    1.149547e+02            8.930483e+01            
10     1.263143e+00            4.000798e+02            -inf                    nan                     2.267383e+00            1.309457e+00            7.074774e+01            -inf                    1.822544e+01            4.756702e+00            
11     3.546620e+02            1.813688e+03            -inf                    nan                     6.850322e+02            5.320561e+02            1.039047e+03            -inf                    2.815399e+02            7.530685e+02            
12     2.525282e+00            2.581156e+02            -inf                    nan                     2.913464e+00            1.090951e+00            6.929907e+01            -inf                    3.519995e+01            3.657443e+00            
13     3.141164e+01            6.353079e+01            -inf                    nan                     2.791455e+01            3.476900e+01            4.520256e+01            -inf                    2.744864e+01            3.396043e+01            
14     7.261424e+00            2.140415e+01            -inf                    nan                     1.621489e+01            7.202844e+00            1.541621e+01            -inf                    5.634804e+00            1.442649e+01            
15     3.614938e+00            4.325674e+00            -inf                    nan                     4.086919e+00            3.828655e+00            4.131412e+00            -inf                    3.407019e+00            3.941457e+00            
16     4.857326e+03            4.840404e+04            -inf                    nan                     1.059327e+04            4.673378e+03            1.141036e+04            -inf                    4.205635e+03            5.812771e+03            
17     6.409501e+04            3.986530e+05            -inf                    nan                     7.313672e+04            1.508082e+05            2.903961e+05            -inf                    8.919415e+04            1.609063e+05            
18     5.790753e+01            1.066147e+02            -inf                    nan                     6.028920e+01            7.355798e+01            9.170275e+01            -inf                    6.051875e+01            7.441788e+01            
19     1.462088e+02            5.893938e+02            -inf                    nan                     3.945861e+02            1.714560e+02            3.582674e+02            -inf                    1.090491e+02            3.044899e+02            
20     2.043047e+01            7.021578e+01            -inf                    nan                     3.305804e+01            2.314093e+01            3.687427e+01            -inf                    4.500954e+01            2.660412e+01            
21     5.707348e+00            6.118364e+00            -inf                    nan                     5.749221e+00            5.795147e+00            6.029872e+00            -inf                    5.240670e+00            5.821977e+00            
22     1.334288e+01            2.347118e+01            -inf                    nan                     1.985980e+01            1.653051e+01            2.017367e+01            -inf                    1.107293e+01            1.712349e+01            
23     4.747748e+01            1.639635e+02            -inf                    nan                     1.039408e+02            6.931755e+01            8.952827e+01            -inf                    6.622843e+01            9.229098e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.148441e-07)
Task  1: variant_10_catB_idea_0.py  (error=3.112068e+00)
Task  2: original.py  (error=7.425561e-05)
Task  3: variant_05_catE_idea_0.py  (error=2.614893e+00)
Task  4: original.py  (error=9.949669e-01)
Task  5: variant_05_catE_idea_0.py  (error=1.603931e+01)
Task  6: original.py  (error=1.258695e+02)
Task  7: variant_05_catE_idea_0.py  (error=3.273042e+00)
Task  8: variant_05_catE_idea_0.py  (error=2.225932e+00)
Task  9: original.py  (error=4.033458e+01)
Task 10: original.py  (error=1.263143e+00)
Task 11: variant_08_catH_idea_0.py  (error=2.815399e+02)
Task 12: variant_05_catE_idea_0.py  (error=1.090951e+00)
Task 13: variant_08_catH_idea_0.py  (error=2.744864e+01)
Task 14: variant_08_catH_idea_0.py  (error=5.634804e+00)
Task 15: variant_08_catH_idea_0.py  (error=3.407019e+00)
Task 16: variant_08_catH_idea_0.py  (error=4.205635e+03)
Task 17: original.py  (error=6.409501e+04)
Task 18: original.py  (error=5.790753e+01)
Task 19: variant_08_catH_idea_0.py  (error=1.090491e+02)
Task 20: original.py  (error=2.043047e+01)
Task 21: variant_08_catH_idea_0.py  (error=5.240670e+00)
Task 22: variant_08_catH_idea_0.py  (error=1.107293e+01)
Task 23: original.py  (error=4.747748e+01)

WIN COUNTS:
  original.py: 10 wins
  variant_08_catH_idea_0.py: 8 wins
  variant_05_catE_idea_0.py: 5 wins
  variant_10_catB_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_05_catE_idea_0.py (5 wins) ---
```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))

        # Build k-NN graph based on Euclidean distance
        k = max(3, min(10, np_pop // 4))

        # Compute pairwise distances
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)

        # Find k-nearest neighbors
        knn_indices = np.argsort(dists_sq, axis=1)[:, :k]

        # Graph-based property 1: local connectivity (inverse mean distance to k-NN)
        local_density = np.zeros(np_pop)
        for i in range(np_pop):
            local_density[i] = np.mean(dists_sq[i, knn_indices[i]])
        local_density = np.clip(local_density, 1e-10, None)
        connectivity = 1.0 / local_density

        # Normalize connectivity to [0, 1]
        conn_min, conn_max = np.min(connectivity), np.max(connectivity)
        if conn_max > conn_min:
            conn_norm = (connectivity - conn_min) / (conn_max - conn_min)
        else:
            conn_norm = np.ones(np_pop) * 0.5

        # Graph-based property 2: find individuals in sparse regions (low connectivity)
        # These are under-explored — bias mutation toward global best
        sparse_mask = conn_norm < 0.3

        # Find global best and second-best for cross-cluster guidance
        best_idx = np.argmin(fitness)
        sorted_idx = np.argsort(fitness)
        second_best_idx = sorted_idx[1] if np_pop > 1 else best_idx

        # Build graph-based parent indices: select from different local neighborhoods
        # For each individual, find one k-NN parent and one "far" parent from different region
        graph_parent1 = np.array([knn_indices[i, np.random.randint(k)] for i in range(np_pop)])

        # Second parent: pick from k-NN of a distant individual (promotes cross-cluster mating)
        far_indices = np.argsort(dists_sq, axis=1)[:, -1]  # farthest neighbor
        graph_parent2 = far_indices[graph_parent1]  # far from first graph parent

        # Adaptive F
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)

        # Graph-guided mutation for all individuals
        p1 = population[graph_parent1]
        p2 = population[graph_parent2]

        # Base graph-guided mutation: rand/1 using graph-derived parents
        graph_mutation = p1 + current_F * (p2 - population)

        # Sparse-region correction: additionally bias toward best for under-explored areas
        best_vector = population[best_idx]
        bias_strength = np.maximum(0, 0.3 - conn_norm) * 2.0  # 0 to 0.3 based on sparsity

        for i in range(np_pop):
            if sparse_mask[i]:
                # Under-explored: mix graph mutation with best-guided mutation
                trials[i] = (1 - bias_strength[i]) * graph_mutation[i] + \
                            bias_strength[i] * (population[i] + current_F * (best_vector - population[i]))
            else:
                trials[i] = graph_mutation[i]

        # Final clipping to bounds
        trials = np.clip(trials, self.lower, self.upper)

        return trials, current_F
```

# --- From variant_08_catH_idea_0.py (8 wins) ---
```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
            """
            Hybrid mutation combining:
            - k-NN spatial mechanism: centroid from k nearest better fitness members (Category A)
            - Fitness improvement mechanism: success-rate based perturbation (Category D)
            Switching is data-driven based on population diversity state.
            """
            np_pop, dim = population.shape
            trials = np.zeros((np_pop, dim))

            # Initialize hybrid state tracking
            if not hasattr(self, 'hybrid_success_ewma'):
                self.hybrid_success_ewma = 0.5
                self.hybrid_diversity_ewma = 1.0

            # --- Mechanism 1: k-NN Spatial (Category A) ---
            # For each individual, find k nearest neighbors with better fitness
            k = min(5, np_pop - 1)
            sorted_indices = np.argsort(fitness)  # 0 = best
            knn_centroids = np.zeros((np_pop, dim))

            for i in range(np_pop):
                # Find k individuals with better fitness (earlier in sorted list)
                better_mask = np.isin(sorted_indices, np.arange(0, k + 1))
                better_indices = sorted_indices[:min(k + 1, np_pop)]
                better_indices = better_indices[better_indices != i]

                if len(better_indices) > 0:
                    knn_neighbors = population[better_indices]
                    knn_centroids[i] = np.mean(knn_neighbors, axis=0)
                else:
                    knn_centroids[i] = np.mean(population, axis=0)

            # --- Mechanism 2: Fitness Improvement Rate (Category D) ---
            # Track improvement rate for switching decision
            current_best = np.min(fitness)
            if hasattr(self, 'prev_best_fitness'):
                improvement_occurred = current_best < self.prev_best_fitness
                improvement_signal = 1.0 if improvement_occurred else 0.0
                self.hybrid_success_ewma = 0.7 * self.hybrid_success_ewma + 0.3 * improvement_signal
            self.prev_best_fitness = current_best

            # --- Compute Population Diversity for Switching ---
            centroid = np.mean(population, axis=0)
            avg_dist = np.mean(np.linalg.norm(population - centroid, axis=1))
            self.hybrid_diversity_ewma = 0.9 * self.hybrid_diversity_ewma + 0.1 * avg_dist

            # Diversity-based switching threshold (principled: relative to search space)
            search_space_scale = self.upper - self.lower
            diversity_ratio = avg_dist / (search_space_scale + 1e-10)
            diversity_threshold = 0.05  # 5% of search space as low-diversity threshold

            # --- Principled Weighting/Switching Logic ---
            # If diversity is low OR improvement rate is low: favor spatial mechanism
            # If diversity is adequate AND improvement is occurring: balance both
            low_diversity = diversity_ratio < diversity_threshold
            low_improvement = self.hybrid_success_ewma < 0.3

            if low_diversity:
                # Exploration mode: prioritize spatial k-NN mechanism
                spatial_weight = 0.8
            elif low_improvement:
                # Stagnation mode: increase spatial weight for escape
                spatial_weight = 0.65
            else:
                # Normal mode: balanced hybrid
                spatial_weight = 0.5

            # --- Generate Mutation Vectors ---
            # Get ring-based parents
            r1 = ring_prev
            r2 = ring_next
            rand1_vectors = population[r1]
            rand2_vectors = population[r2]

            # Adaptive F based on diversity
            diversity_factor = np.clip(avg_dist / 50.0, 0.5, 2.0)
            current_F = self.F * diversity_factor * (1.0 + 0.1 * np.random.randn())
            current_F = np.clip(current_F, 0.1, 2.0)

            # Mechanism 1: k-NN spatial mutation
            spatial_mutation = knn_centroids + current_F * (rand1_vectors - rand2_vectors)

            # Mechanism 2: Fitness-weighted centroid perturbation
            weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
            weights = weights / np.sum(weights)
            fitness_centroid = np.sum(population * weights[:, np.newaxis], axis=0)

            # Direction toward fitness centroid
            to_centroid = fitness_centroid - population
            fitness_mutation = population + current_F * to_centroid + 0.5 * (rand1_vectors - rand2_vectors)

            # --- Hybrid Combination ---
            trials = spatial_weight * spatial_mutation + (1 - spatial_weight) * fitness_mutation

            return trials, current_F
```

# --- From variant_10_catB_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using eigendecomposition of population covariance.
        Modulate mutation strength along each eigenvector based on eigenvalue magnitude.
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))

        # Compute population covariance matrix
        centroid = np.mean(population, axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / max(np_pop - 1, 1)

        # Regularize covariance for numerical stability
        cov_reg = cov + 1e-6 * np.eye(dim)

        # Eigendecomposition: cov = V @ diag(eigvals) @ V.T
        try:
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            # Sort eigenvalues ascending (smallest = most neglected direction)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        except np.linalg.LinAlgError:
            # Fallback to standard rand/1 if eigendecomposition fails
            r1 = ring_prev
            r2 = ring_next
            current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
            trials = population[r1] + current_F * (population[r2] - population)
            return trials, current_F

        # Compute condition number for anisotropy detection
        eigvals_safe = np.clip(eigvals, 1e-12, None)
        cond_num = np.max(eigvals_safe) / np.min(eigvals_safe)

        # Compute spectral weights: inverse of normalized eigenvalue
        # Small eigenvalues -> large weight (explore neglected subspace)
        eigvals_norm = eigvals_safe / (np.sum(eigvals_safe) + 1e-12)
        spectral_weights = 1.0 / (eigvals_norm + 0.01)
        spectral_weights = spectral_weights / np.sum(spectral_weights)  # Normalize

        # Anisotropy factor: when highly anisotropic, emphasize small-eigenvalue directions
        anisotropy_factor = np.log1p(cond_num) / np.log1p(dim * dim)
        anisotropy_factor = np.clip(anisotropy_factor, 0.0, 1.0)

        # Base mutation: ring-based rand/1
        r1 = ring_prev
        r2 = ring_next
        current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
        base_mutation = population[r1] + current_F * (population[r2] - population)

        # Project base mutation to eigenbasis
        base_mutation_centered = base_mutation - centroid
        proj_coeffs = base_mutation_centered @ eigvecs  # (np_pop, dim) in eigenbasis

        # Modulate mutation coefficients by spectral weights
        # High anisotropy -> stronger modulation toward small-eigenvalue directions
        modulation_strength = 0.3 + 0.4 * anisotropy_factor
        modulation = 1.0 + modulation_strength * (spectral_weights - np.mean(spectral_weights)) / (np.std(spectral_weights) + 1e-8)
        modulation = np.clip(modulation, 0.3, 2.5)

        # Apply modulation in eigenbasis
        modulated_coeffs = proj_coeffs * modulation

        # Transform back to original space
        modulated_mutation = modulated_coeffs @ eigvecs.T + centroid

        # Blend: modulated vs base based on anisotropy
        blend_weight = 0.3 + 0.4 * anisotropy_factor
        blend_weight = np.clip(blend_weight, 0.2, 0.7)
        trials = (1 - blend_weight) * base_mutation + blend_weight * modulated_mutation

        # Fitness-weighted directional component for additional exploitation
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid_weighted = np.sum(population * weights[:, np.newaxis], axis=0)

        directional_component = centroid_weighted - centroid
        trials = trials + 0.15 * directional_component

        return trials, current_F
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