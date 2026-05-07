Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_mutate_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.118196e-07            1.255132e+00            1.074463e+02            3.494991e+01            8.826738e+01            -inf                    4.635525e+00            nan                     2.136877e-06            
1      1.923990e+00            9.755409e+00            2.289316e+02            1.057776e+02            1.896382e+02            -inf                    1.106164e+01            nan                     4.272179e+00            
2      5.657046e-05            5.091886e+00            7.760391e+02            6.698620e+02            6.741288e+02            -inf                    6.483936e+01            nan                     1.161960e-04            
3      2.592829e+00            4.008652e+01            3.349533e+02            3.245204e+02            3.009288e+02            -inf                    4.924310e+01            nan                     1.841215e+01            
4      8.468928e-01            1.861696e+00            6.161163e+00            4.961430e+00            5.818221e+00            -inf                    2.662195e+00            nan                     1.330941e+00            
5      1.346859e+02            9.662691e+04            4.191339e+09            9.691343e+08            2.212468e+09            -inf                    8.996026e+05            nan                     3.112419e+03            
6      2.654227e+02            9.182971e+02            9.834476e+03            1.266667e+04            6.923254e+03            -inf                    6.595117e+02            nan                     4.294998e+02            
7      6.396815e+00            2.518175e+01            4.113205e+02            5.203181e+02            3.785971e+02            -inf                    4.109231e+01            nan                     1.516552e+01            
8      1.599741e+00            8.677900e+00            5.464339e+01            6.055877e+01            5.187244e+01            -inf                    1.312211e+01            nan                     3.377309e+00            
9      4.692224e+01            1.505598e+02            3.409533e+02            4.649791e+02            2.829359e+02            -inf                    7.619079e+01            nan                     7.412623e+01            
10     1.081160e+00            8.675742e+00            3.228733e+02            4.310097e+02            2.697619e+02            -inf                    1.800835e+01            nan                     4.714497e-01            
11     4.068769e+02            6.015040e+02            1.573835e+03            1.949325e+03            1.428524e+03            -inf                    3.452842e+02            nan                     9.817003e+02            
12     2.837961e+00            1.355453e+01            2.177755e+02            2.956029e+02            1.886493e+02            -inf                    1.976889e+01            nan                     8.671437e-01            
13     3.128272e+01            3.248895e+01            5.885565e+01            6.240676e+01            5.589925e+01            -inf                    2.386031e+01            nan                     3.394585e+01            
14     7.427308e+00            1.231472e+01            1.938924e+01            2.067601e+01            1.832699e+01            -inf                    1.089328e+01            nan                     1.426960e+01            
15     3.591257e+00            3.375417e+00            4.265505e+00            4.439886e+00            4.211957e+00            -inf                    3.425541e+00            nan                     4.182131e+00            
16     5.372293e+03            7.218773e+03            2.826653e+04            4.016791e+04            2.409382e+04            -inf                    3.036580e+03            nan                     5.879108e+03            
17     7.239167e+04            1.042739e+05            3.902179e+05            2.590525e+05            3.577583e+05            -inf                    4.756919e+04            nan                     1.506220e+05            
18     5.912787e+01            6.212168e+01            1.039610e+02            9.014445e+01            9.960855e+01            -inf                    4.733852e+01            nan                     7.491922e+01            
19     1.185155e+02            2.289982e+02            4.927039e+02            5.939941e+02            4.493582e+02            -inf                    1.984164e+02            nan                     3.623287e+02            
20     2.008751e+01            3.143393e+01            6.461851e+01            7.331536e+01            5.851894e+01            -inf                    2.309575e+01            nan                     2.205306e+01            
21     5.688817e+00            5.621316e+00            6.112758e+00            6.020295e+00            6.070954e+00            -inf                    5.531673e+00            nan                     5.807553e+00            
22     1.318928e+01            1.506816e+01            2.269239e+01            2.355062e+01            2.186661e+01            -inf                    1.410852e+01            nan                     1.948237e+01            
23     5.451930e+01            7.529116e+01            1.344578e+02            1.536709e+02            1.200930e+02            -inf                    7.167682e+01            nan                     1.019921e+02            

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
Task 10: variant_10_catB_idea_0.py  (error=4.714497e-01)
Task 11: variant_08_catH_idea_0.py  (error=3.452842e+02)
Task 12: variant_10_catB_idea_0.py  (error=8.671437e-01)
Task 13: variant_08_catH_idea_0.py  (error=2.386031e+01)
Task 14: original.py  (error=7.427308e+00)
Task 15: variant_01_catA_idea_0.py  (error=3.375417e+00)
Task 16: variant_08_catH_idea_0.py  (error=3.036580e+03)
Task 17: variant_08_catH_idea_0.py  (error=4.756919e+04)
Task 18: variant_08_catH_idea_0.py  (error=4.733852e+01)
Task 19: original.py  (error=1.185155e+02)
Task 20: original.py  (error=2.008751e+01)
Task 21: variant_08_catH_idea_0.py  (error=5.531673e+00)
Task 22: original.py  (error=1.318928e+01)
Task 23: original.py  (error=5.451930e+01)

WIN COUNTS:
  original.py: 15 wins
  variant_08_catH_idea_0.py: 6 wins
  variant_10_catB_idea_0.py: 2 wins
  variant_01_catA_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
            """
            Generate mutation vectors using geometric structure:
            - k-NN local direction vectors for each individual
            - Convex hull boundary detection for adaptive scaling
            - Axis-aligned spread normalization
            """
            np_pop, dim = population.shape

            # Compute pairwise Euclidean distance matrix
            expanded = population[:, np.newaxis, :]
            diff_matrix = expanded - population[np.newaxis, :, :]
            pairwise_dist = np.linalg.norm(diff_matrix, axis=2)

            # Replace zero diagonal with inf to exclude self
            np.fill_diagonal(pairwise_dist, np.inf)

            # k-NN structure: find k nearest neighbors
            k = min(5, np_pop - 1)
            knn_indices = np.argsort(pairwise_dist, axis=1)[:, :k]

            # Compute local direction vectors from k-NN
            # For each individual, direction = weighted mean of neighbor offsets
            neighbor_weights = 1.0 / (pairwise_dist[np.arange(np_pop)[:, None], knn_indices] + 1e-10)
            neighbor_weights = neighbor_weights / (neighbor_weights.sum(axis=1, keepdims=True) + 1e-10)

            local_directions = np.zeros((np_pop, dim))
            for i in range(np_pop):
                neighbors = population[knn_indices[i]]
                offsets = neighbors - population[i]
                local_directions[i] = np.sum(offsets * neighbor_weights[i, :, np.newaxis], axis=0)

            # Compute population centroid and centroid distances (geometric only)
            centroid = np.mean(population, axis=0)
            centroid_distances = np.linalg.norm(population - centroid, axis=1)
            max_centroid_dist = np.max(centroid_distances) + 1e-10

            # Normalize centroid distance to [0.1, 1.0] range
            norm_centroid_dist = 0.1 + 0.9 * (centroid_distances / max_centroid_dist)

            # Identify convex hull boundary points via Graham scan approach
            # Use axis-aligned bounding box corners as reference
            mins = np.min(population, axis=0)
            maxs = np.max(population, axis=0)

            # Compute boundary score: higher = closer to boundary
            boundary_score = np.zeros(np_pop)
            for d in range(dim):
                boundary_score += (population[:, d] - mins[d]) / (maxs[d] - mins[d] + 1e-10)
                boundary_score += (maxs[d] - population[:, d]) / (maxs[d] - mins[d] + 1e-10)
            boundary_score = boundary_score / (2 * dim)

            # Convex hull approximation: extreme points have boundary_score near 0 or 2
            is_hull_point = np.any([
                np.allclose(population, mins, atol=1e-5),
                np.allclose(population, maxs, atol=1e-5)
            ], axis=0)

            # Ring-based parents for baseline differential component
            r1 = ring_prev
            r2 = ring_next
            rand1_vectors = population[r1]
            rand2_vectors = population[r2]

            # Adaptive F with geometric scaling
            current_F = self.F * (1.0 + 0.1 * np.random.randn())
            current_F = np.clip(current_F, 0.1, 2.0)

            # Base differential mutation
            base_mutation = rand1_vectors + current_F * (rand2_vectors - population)

            # Geometric mutation strength: inverse scaling from centroid distance
            # Points near centroid (low norm_dist) get stronger exploration
            exploration_factor = 1.0 / (norm_centroid_dist + 0.2)

            # Hull points get additional outward bias
            hull_boost = np.where(is_hull_point, 1.3, 1.0)
            total_scale = exploration_factor * hull_boost
            total_scale = np.clip(total_scale, 0.5, 2.5)

            # Geometric component: centroid + local k-NN direction
            geometric_component = centroid + total_scale[:, np.newaxis] * local_directions

            # Blend geometric and base mutations
            # Boundary points favor geometric (exploration), interior favor base (exploitation)
            blend_weight = 0.15 + 0.35 * boundary_score
            blend_weight = np.clip(blend_weight, 0.1, 0.5)

            trials = (1 - blend_weight[:, np.newaxis]) * base_mutation + blend_weight[:, np.newaxis] * geometric_component

            return trials, current_F
```

# --- From variant_08_catH_idea_0.py (6 wins) ---
```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
            """
            Hybrid mutation: switch between exploration (DE/rand/1) and 
            exploitation (best+centroid) based on diversity and success rate.
            """
            np_pop = len(population)
            trials = np.zeros((np_pop, self.dim))

            # Initialize regime tracking
            if not hasattr(self, 'exploration_ema'):
                self.exploration_ema = 0.8  # Start exploratory
                self.recent_success_rate = 0.5

            # Compute diversity ratio (current spread / initial spread)
            current_spread = np.std(population, axis=0)
            initial_spread = (self.upper - self.lower) / 3.0  # ~3-sigma coverage
            diversity_ratio = np.mean(current_spread / (initial_spread + 1e-10))
            diversity_ratio = np.clip(diversity_ratio, 0.01, 1.0)

            # Update success rate EMA
            if hasattr(self, 'last_improved_count'):
                current_success = self.last_improved_count / np_pop
                self.recent_success_rate = 0.3 * current_success + 0.7 * self.recent_success_rate

            # Regime detection: exploration vs exploitation weighting
            # High diversity + low success -> favor exploitation (intensify)
            # Low diversity + high success -> favor exploration (diversify)
            # High diversity + high success -> favor exploitation (ride the wave)
            # Low diversity + low success -> favor exploration (escape)

            diversity_score = diversity_ratio  # 0 = clustered, 1 = spread
            success_score = self.recent_success_rate  # 0 = failing, 1 = succeeding

            # Exploration weight: high when diversity is low OR when failing badly
            exploration_weight = 0.5 + 0.4 * (1.0 - diversity_score) + 0.2 * (0.5 - success_score)
            exploration_weight = np.clip(exploration_weight, 0.1, 0.9)

            # Smooth the exploration weight (regime shouldn't flip wildly)
            self.exploration_ema = 0.7 * self.exploration_ema + 0.3 * exploration_weight
            final_exploration_weight = self.exploration_ema

            # Strategy 1: DE/rand/1 (pure exploration)
            r1 = ring_prev
            r2 = ring_next
            rand1_vectors = population[r1]
            rand2_vectors = population[r2]

            F_explore = np.clip(self.F * (1.0 + 0.15 * np.random.randn()), 0.3, 2.0)
            explore_mutation = rand1_vectors + F_explore * (rand2_vectors - population)

            # Strategy 2: best + centroid direction (exploitation)
            best_idx = np.argmin(fitness)
            best_vector = population[best_idx]

            centroid = self._compute_population_centroid(population, fitness)
            centroid_direction = centroid - best_vector  # Direction from best toward centroid

            # Scale by distance to prevent overshooting
            centroid_dist = np.linalg.norm(centroid_direction) + 1e-10
            F_exploit = np.clip(0.5 * self.F, 0.1, 1.0)
            exploit_mutation = best_vector + F_exploit * centroid_direction

            # Broadcast exploitation vector to all individuals
            exploit_mutation = np.tile(exploit_mutation, (np_pop, 1))

            # Blend based on regime
            trials = final_exploration_weight * explore_mutation + (1.0 - final_exploration_weight) * exploit_mutation

            # Add small perturbation to prevent stagnation
            perturbation = 0.01 * (np.random.rand(np_pop, self.dim) - 0.5) * (self.upper - self.lower)
            trials = trials + perturbation

            return trials, F_explore
```

# --- From variant_10_catB_idea_0.py (2 wins) ---
```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using eigendecomposition-driven covariance adaptation.
        - Track covariance of successful mutations (not just population)
        - Eigendecompose to get principal axes of successful search directions
        - Scale differential vectors inversely proportional to eigenvalue magnitude
        - Blend with centroid for stability
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))

        # Compute population centroid for fallback
        centroid = self._compute_population_centroid(population, fitness)

        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next

        rand1_vectors = population[r1]
        rand2_vectors = population[r2]

        # Adaptive F
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)

        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)

        # --- NEW: Eigendecomposition-driven scaling ---
        # Build covariance of successful mutations from direction memory
        if len(self.successful_directions) >= dim + 1:
            # Stack successful mutation vectors
            D = np.array(self.successful_directions)
            # Center the data
            D_centered = D - np.mean(D, axis=0)
            # Covariance of successful mutations (not population covariance)
            cov_mutations = np.cov(D_centered, rowvar=False)

            # Regularize for numerical stability
            cov_mutations += 0.01 * np.eye(dim)

            # Eigendecomposition
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_mutations)
                eigenvalues = np.maximum(eigenvalues, 1e-10)

                # Compute inverse sqrt eigenvalues for scaling
                inv_sqrt_eig = 1.0 / np.sqrt(eigenvalues)
                # Clip to prevent extreme scaling
                inv_sqrt_eig = np.clip(inv_sqrt_eig, 0.1, 10.0)
                # Normalize to prevent overall scale change
                inv_sqrt_eig = inv_sqrt_eig / np.mean(inv_sqrt_eig)

                # Build scaling matrix and apply to differential vectors
                scaling_matrix = eigenvectors @ np.diag(inv_sqrt_eig) @ eigenvectors.T

                # Transform the differential component
                diff_vectors = rand2_vectors - population
                scaled_diff = diff_vectors @ scaling_matrix

                # Eigenscaled mutation component
                eigen_scaled = rand1_vectors + current_F * scaled_diff

                # Blend eigenscaled with base mutation
                blend_weight = 0.3 + 0.2 * (1.0 - self.Cr)
                trials = (1 - blend_weight) * base_mutation + blend_weight * eigen_scaled
            except np.linalg.LinAlgError:
                # Fallback on decomposition failure
                trials = base_mutation
        else:
            # Insufficient data for eigendecomposition, use base mutation
            trials = base_mutation

        # Clamp F for numerical safety
        current_F = np.clip(current_F, 0.1, 2.0)

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