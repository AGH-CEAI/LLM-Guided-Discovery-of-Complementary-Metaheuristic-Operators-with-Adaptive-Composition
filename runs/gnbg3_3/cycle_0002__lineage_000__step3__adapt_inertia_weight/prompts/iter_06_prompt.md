This is iteration 6 of 10. Propose ONE replacement implementation for `_adapt_inertia_weight`.

═══════════════════════════════════════════════════════════════════════
ASSIGNED STRATEGY CATEGORY FOR THIS ITERATION:
  Category F — Temporal / dynamical
  Track quantities ACROSS GENERATIONS: rates of change, autocorrelation, exponential moving averages of state, stagnation detection over a time window, drift of the mean. Single-generation snapshots are insufficient — you must use temporal information.

  (This category has been used 0 time(s) before. If > 0, your
  variant MUST take a different angle within the category — same family,
  different mechanism.)
═══════════════════════════════════════════════════════════════════════

For reference, the full category menu is:
  A) Geometry / spatial: Operate on the literal geometric layout of points: pairwise distances, convex hull, centroid distances, k-NN structure, axis-aligned spread, projection onto principal axes. Avoid information-theoretic or fitness-based reasoning.
  B) Spectral / linear-algebraic: Use eigenvalues, singular values, condition numbers, rank, covariance shape, or matrix decompositions. Reason in terms of subspace alignment, anisotropy, or effective dimensionality.
  C) Information-theoretic / distributional: Use entropy, KL divergence, mutual information, distribution fitting (Gaussian, GMM, kernel density), or log-likelihood-based reasoning. Treat the population as a probability distribution to be characterised.
  D) Fitness-landscape / rank-based: Reason about fitness values directly: rank correlations, Spearman/Kendall coefficients, fitness percentiles, fitness-distance correlation, success-history. Do NOT use raw distances or covariance — only fitness signals.
  E) Topology / graph-based: Build a graph or topology over the population (k-NN graph, minimum spanning tree, neighborhood lattice, ring/star topology) and derive properties from connectivity, clustering, or path lengths.
  F) Temporal / dynamical: Track quantities ACROSS GENERATIONS: rates of change, autocorrelation, exponential moving averages of state, stagnation detection over a time window, drift of the mean. Single-generation snapshots are insufficient — you must use temporal information.
  G) Stochastic / sampling-based: Use Monte-Carlo sampling, bootstrap estimates, Latin hypercube / Sobol probes, or randomised projections to estimate the quantity of interest. The variant must be fundamentally stochastic in its computation.
  H) Hybrid / multi-strategy: Combine TWO clearly-distinct mechanisms with explicit weighting or switching logic. The weighting/switching itself must be principled (e.g. data-driven, schedule-based) rather than an arbitrary linear combination of magic constants.

You must NOT cross categories. If your variant computes something that
properly belongs to a different category (e.g. you were assigned 'Geometry'
but you end up doing eigendecomposition, which is 'Spectral'), STOP and
re-design within the assigned category.

Universal requirements:
- Keep the EXACT function signature: `def _adapt_inertia_weight(self):`
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects (or blends) operators FOR EACH TASK at runtime. This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that reaches error <= 1e-08 on EACH task.
- Tasks marked UNSOLVED below are critical gaps. The bigger the remaining
  error, the higher the priority — your variant should specifically target
  the WORST unsolved tasks at the top of the priority list, AS APPROACHED
  THROUGH THE LENS OF YOUR ASSIGNED CATEGORY.

TASK COVERAGE SUMMARY: 0 SOLVED (<= 1e-08), 24 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 17: *** UNSOLVED *** — best=2.608e+04 by original.py               (target=1e-08, ~+12.4 decades above target)
  Task 16: *** UNSOLVED *** — best=3.423e+03 by original.py               (target=1e-08, ~+11.5 decades above target)
  Task  6: *** UNSOLVED *** — best=9.004e+02 by variant_01_catA_idea_0.py (target=1e-08, ~+11.0 decades above target)
  Task 19: *** UNSOLVED *** — best=1.903e+02 by original.py               (target=1e-08, ~+10.3 decades above target)
  Task 11: *** UNSOLVED *** — best=1.585e+02 by original.py               (target=1e-08, ~+10.2 decades above target)
  Task  9: *** UNSOLVED *** — best=1.260e+02 by original.py               (target=1e-08, ~+10.1 decades above target)
  Task  2: *** UNSOLVED *** — best=1.170e+02 by original.py               (target=1e-08, ~+10.1 decades above target)
  Task 12: *** UNSOLVED *** — best=6.199e+01 by variant_04_catD_idea_0.py (target=1e-08, ~+9.8 decades above target)
  Task 10: *** UNSOLVED *** — best=5.639e+01 by original.py               (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=5.459e+01 by original.py               (target=1e-08, ~+9.7 decades above target)
  Task 18: *** UNSOLVED *** — best=4.736e+01 by variant_02_catB_idea_0.py (target=1e-08, ~+9.7 decades above target)
  Task  5: *** UNSOLVED *** — best=3.531e+01 by variant_01_catA_idea_0.py (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=3.241e+01 by original.py               (target=1e-08, ~+9.5 decades above target)
  Task 13: *** UNSOLVED *** — best=2.336e+01 by original.py               (target=1e-08, ~+9.4 decades above target)
  Task 22: *** UNSOLVED *** — best=1.436e+01 by variant_04_catD_idea_0.py (target=1e-08, ~+9.2 decades above target)
  Task  0: *** UNSOLVED *** — best=1.139e+01 by original.py               (target=1e-08, ~+9.1 decades above target)
  Task 14: *** UNSOLVED *** — best=1.061e+01 by original.py               (target=1e-08, ~+9.0 decades above target)
  Task  7: *** UNSOLVED *** — best=7.620e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.9 decades above target)
  Task 21: *** UNSOLVED *** — best=5.648e+00 by variant_04_catD_idea_0.py (target=1e-08, ~+8.8 decades above target)
  Task 15: *** UNSOLVED *** — best=3.466e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.5 decades above target)
  Task  8: *** UNSOLVED *** — best=3.411e+00 by variant_04_catD_idea_0.py (target=1e-08, ~+8.5 decades above target)
  Task  3: *** UNSOLVED *** — best=2.293e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.4 decades above target)
  Task  4: *** UNSOLVED *** — best=1.787e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task  1: *** UNSOLVED *** — best=1.089e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.0 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_catA_idea  variant_02_catB_idea  variant_03_catC_idea  variant_04_catD_idea  variant_05_catE_idea  
-----------------------------------------------------------------------------------------------------------------------------------------
0    1.1387e+01            1.2388e+01            1.3934e+01            3.2509e+01            1.3702e+01            4.4927e+01            
1    4.2639e+00            1.0893e+00            1.2650e+00            7.2199e+01            1.7057e+00            9.4735e+01            
2    1.1701e+02            1.3348e+02            1.4858e+02            3.1586e+02            1.2307e+02            3.8698e+02            
3    7.9957e+00            2.2933e+00            2.7068e+00            1.7244e+02            3.2705e+00            1.9358e+02            
4    1.7874e+00            2.2079e+00            2.6779e+00            4.3973e+00            2.8655e+00            4.7631e+00            
5    2.4412e+03            3.5311e+01            1.4616e+06            1.8033e+07            6.8001e+02            3.3106e+08            
6    9.1266e+02            9.0038e+02            1.3177e+03            3.3016e+03            1.3439e+03            4.2089e+03            
7    9.3029e+00            7.6199e+00            7.9116e+00            1.8853e+02            8.0237e+00            2.2889e+02            
8    5.9575e+00            6.2253e+00            1.1322e+01            3.2412e+01            3.4110e+00            3.6794e+01            
9    1.2601e+02            1.3245e+02            1.6783e+02            2.4160e+02            1.3079e+02            2.7591e+02            
10   5.6393e+01            6.4793e+01            9.2473e+01            1.3841e+02            6.9683e+01            1.9229e+02            
11   1.5845e+02            2.9216e+02            2.1208e+02            5.8011e+02            2.0905e+02            7.9485e+02            
12   7.0639e+01            6.7906e+01            6.6116e+01            1.1633e+02            6.1985e+01            1.4728e+02            
13   2.3355e+01            2.5194e+01            2.5631e+01            3.1941e+01            2.6476e+01            3.9302e+01            
14   1.0608e+01            1.0779e+01            1.1827e+01            1.4940e+01            1.1543e+01            1.5578e+01            
15   4.0276e+00            3.4656e+00            3.6680e+00            3.6636e+00            3.7240e+00            3.7835e+00            
16   3.4231e+03            3.8748e+03            7.2469e+03            8.8913e+03            4.5705e+03            1.2029e+04            
17   2.6075e+04            9.2962e+04            6.8715e+04            7.2847e+04            9.4515e+04            1.0350e+05            
18   4.7733e+01            4.8424e+01            4.7360e+01            6.1182e+01            4.9399e+01            6.7095e+01            
19   1.9027e+02            2.1071e+02            2.3740e+02            3.0535e+02            2.1189e+02            3.1146e+02            
20   3.2414e+01            3.3957e+01            3.7017e+01            4.3952e+01            3.4051e+01            4.7422e+01            
21   5.8698e+00            5.6797e+00            5.6965e+00            5.7176e+00            5.6478e+00            5.7489e+00            
22   1.5008e+01            1.5022e+01            1.4708e+01            1.6999e+01            1.4359e+01            1.7558e+01            
23   5.4592e+01            6.2763e+01            6.8276e+01            9.3772e+01            6.9074e+01            9.9341e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 2.608e+04  (target = 1e-08)
  Task 16: best error so far = 3.423e+03  (target = 1e-08)
  Task  6: best error so far = 9.004e+02  (target = 1e-08)
  Task 19: best error so far = 1.903e+02  (target = 1e-08)
  Task 11: best error so far = 1.585e+02  (target = 1e-08)
  Task  9: best error so far = 1.260e+02  (target = 1e-08)
  Task  2: best error so far = 1.170e+02  (target = 1e-08)
  Task 12: best error so far = 6.199e+01  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

CATEGORY ROTATION SO FAR (per-category proposal count):
  A×1  B×1  C×1  D×1  E×1  F×0  G×0  H×0

Your new proposal MUST be designed to crush the error on the WORST unsolved
tasks above, USING THE MECHANISMS OF YOUR ASSIGNED CATEGORY (see below).
It is acceptable — even expected — for the new variant to be worse than
existing variants on already-SOLVED tasks; the adaptive selector will
handle that.

Reason explicitly about the priority targets:
1. What property of those WORST unsolved tasks (multimodality, ill-
   conditioning, separability, ruggedness, deceptive local optima, narrow
   basins, non-separable rotation, etc.) is preventing existing operators
   from reaching 1e-08? Use the per-task error magnitudes as
   evidence — errors stuck at ~1e+1 vs ~1e-3 vs ~1e-6 imply different
   failure modes.
2. How does YOUR ASSIGNED CATEGORY ('Temporal / dynamical') give you a fundamentally
   different angle of attack on those obstacles than the prior variants
   in different categories?
3. What concrete computation, distinctive to 'Temporal / dynamical', will you use?

Current implementation:
```python
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
```

Full algorithm for context:
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

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description (must mention category F: Temporal / dynamical).
```python
def _adapt_inertia_weight(self, ...):
    ...
```