This is iteration 8 of 10. Propose ONE replacement implementation for `_select_survivors_batch`.

═══════════════════════════════════════════════════════════════════════
ASSIGNED STRATEGY CATEGORY FOR THIS ITERATION:
  Category H — Hybrid / multi-strategy
  Combine TWO clearly-distinct mechanisms with explicit weighting or switching logic. The weighting/switching itself must be principled (e.g. data-driven, schedule-based) rather than an arbitrary linear combination of magic constants.

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
- Keep the EXACT function signature: `def _select_survivors_batch(self, population, fitness, trials, trial_fitness):`
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
  Task 17: *** UNSOLVED *** — best=7.239e+04 by original.py               (target=1e-08, ~+12.9 decades above target)
  Task 16: *** UNSOLVED *** — best=4.217e+03 by variant_03_catC_idea_0.py (target=1e-08, ~+11.6 decades above target)
  Task 11: *** UNSOLVED *** — best=3.201e+02 by variant_06_catF_idea_0.py (target=1e-08, ~+10.5 decades above target)
  Task  6: *** UNSOLVED *** — best=1.840e+02 by variant_05_catE_idea_0.py (target=1e-08, ~+10.3 decades above target)
  Task 19: *** UNSOLVED *** — best=1.077e+02 by variant_04_catD_idea_0.py (target=1e-08, ~+10.0 decades above target)
  Task 18: *** UNSOLVED *** — best=5.733e+01 by variant_05_catE_idea_0.py (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=4.169e+01 by variant_06_catF_idea_0.py (target=1e-08, ~+9.6 decades above target)
  Task  9: *** UNSOLVED *** — best=3.847e+01 by variant_01_catA_idea_0.py (target=1e-08, ~+9.6 decades above target)
  Task 13: *** UNSOLVED *** — best=2.875e+01 by variant_03_catC_idea_0.py (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=1.988e+01 by variant_06_catF_idea_0.py (target=1e-08, ~+9.3 decades above target)
  Task 22: *** UNSOLVED *** — best=1.272e+01 by variant_06_catF_idea_0.py (target=1e-08, ~+9.1 decades above target)
  Task  5: *** UNSOLVED *** — best=9.047e+00 by variant_05_catE_idea_0.py (target=1e-08, ~+9.0 decades above target)
  Task 21: *** UNSOLVED *** — best=5.629e+00 by variant_03_catC_idea_0.py (target=1e-08, ~+8.8 decades above target)
  Task  7: *** UNSOLVED *** — best=5.011e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.7 decades above target)
  Task 14: *** UNSOLVED *** — best=4.027e+00 by variant_04_catD_idea_0.py (target=1e-08, ~+8.6 decades above target)
  Task 15: *** UNSOLVED *** — best=3.486e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.5 decades above target)
  Task 12: *** UNSOLVED *** — best=2.585e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.4 decades above target)
  Task  3: *** UNSOLVED *** — best=2.270e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.4 decades above target)
  Task  1: *** UNSOLVED *** — best=1.737e+00 by variant_04_catD_idea_0.py (target=1e-08, ~+8.2 decades above target)
  Task  8: *** UNSOLVED *** — best=1.600e+00 by original.py               (target=1e-08, ~+8.2 decades above target)
  Task 10: *** UNSOLVED *** — best=1.010e+00 by variant_05_catE_idea_0.py (target=1e-08, ~+8.0 decades above target)
  Task  4: *** UNSOLVED *** — best=8.469e-01 by original.py               (target=1e-08, ~+7.9 decades above target)
  Task  2: *** UNSOLVED *** — best=5.433e-05 by variant_01_catA_idea_0.py (target=1e-08, ~+3.7 decades above target)
  Task  0: *** UNSOLVED *** — best=1.023e-07 by variant_01_catA_idea_0.py (target=1e-08, ~+1.0 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_catA_idea  variant_02_catB_idea  variant_03_catC_idea  variant_04_catD_idea  variant_05_catE_idea  variant_06_catF_idea  variant_07_catG_idea  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    1.1182e-07            1.0225e-07            -inf                  7.3761e-07            2.2259e-05            1.0798e-07            1.0638e-07            nan                   
1    1.9240e+00            2.5377e+00            -inf                  3.1279e+00            1.7372e+00            1.8725e+00            3.3600e+00            nan                   
2    5.6570e-05            5.4328e-05            -inf                  5.1397e-02            2.2335e-04            6.4030e-05            8.9569e-05            nan                   
3    2.5928e+00            2.2699e+00            -inf                  8.8582e+00            3.6239e+00            2.6396e+00            2.6080e+00            nan                   
4    8.4689e-01            1.0448e+00            -inf                  1.3684e+00            1.2147e+00            9.4674e-01            1.1597e+00            nan                   
5    1.3469e+02            2.6900e+01            -inf                  1.3025e+03            2.3252e+02            9.0473e+00            4.8729e+01            nan                   
6    2.6542e+02            3.0171e+02            -inf                  6.2248e+02            5.5260e+02            1.8400e+02            2.7760e+02            nan                   
7    6.3968e+00            5.0114e+00            -inf                  1.0010e+01            7.0166e+00            5.8139e+00            6.0744e+00            nan                   
8    1.5997e+00            1.7343e+00            -inf                  2.8814e+00            1.8270e+00            2.0440e+00            1.6142e+00            nan                   
9    4.6922e+01            3.8473e+01            -inf                  5.4757e+01            4.4319e+01            4.2400e+01            4.2143e+01            nan                   
10   1.0812e+00            1.0129e+00            -inf                  1.1978e+00            1.6322e+00            1.0101e+00            1.5172e+00            nan                   
11   4.0688e+02            3.8704e+02            -inf                  3.5926e+02            4.3677e+02            3.6485e+02            3.2010e+02            nan                   
12   2.8380e+00            2.5847e+00            -inf                  3.0835e+00            3.6004e+00            2.8946e+00            2.6629e+00            nan                   
13   3.1283e+01            2.9062e+01            -inf                  2.8745e+01            3.4072e+01            3.1115e+01            3.0913e+01            nan                   
14   7.4273e+00            7.3019e+00            -inf                  8.1263e+00            4.0266e+00            7.2239e+00            4.4691e+00            nan                   
15   3.5913e+00            3.4861e+00            -inf                  3.6037e+00            3.6629e+00            3.6221e+00            3.6420e+00            nan                   
16   5.3723e+03            4.7720e+03            -inf                  4.2171e+03            6.2554e+03            4.5991e+03            5.1439e+03            nan                   
17   7.2392e+04            8.7265e+04            -inf                  8.5945e+04            1.2454e+05            7.8604e+04            1.0789e+05            nan                   
18   5.9128e+01            5.8948e+01            -inf                  5.9385e+01            6.5973e+01            5.7331e+01            6.2001e+01            nan                   
19   1.1852e+02            1.3262e+02            -inf                  1.3385e+02            1.0767e+02            1.5599e+02            1.2819e+02            nan                   
20   2.0088e+01            1.9975e+01            -inf                  2.0118e+01            2.2374e+01            1.9963e+01            1.9877e+01            nan                   
21   5.6888e+00            5.6882e+00            -inf                  5.6291e+00            5.6322e+00            5.6518e+00            5.8587e+00            nan                   
22   1.3189e+01            1.3364e+01            -inf                  1.3705e+01            1.3885e+01            1.3296e+01            1.2725e+01            nan                   
23   5.4519e+01            5.0931e+01            -inf                  6.3007e+01            4.7047e+01            5.2156e+01            4.1693e+01            nan                   

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 7.239e+04  (target = 1e-08)
  Task 16: best error so far = 4.217e+03  (target = 1e-08)
  Task 11: best error so far = 3.201e+02  (target = 1e-08)
  Task  6: best error so far = 1.840e+02  (target = 1e-08)
  Task 19: best error so far = 1.077e+02  (target = 1e-08)
  Task 18: best error so far = 5.733e+01  (target = 1e-08)
  Task 23: best error so far = 4.169e+01  (target = 1e-08)
  Task  9: best error so far = 3.847e+01  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

CATEGORY ROTATION SO FAR (per-category proposal count):
  A×1  B×1  C×1  D×1  E×1  F×1  G×1  H×0

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
2. How does YOUR ASSIGNED CATEGORY ('Hybrid / multi-strategy') give you a fundamentally
   different angle of attack on those obstacles than the prior variants
   in different categories?
3. What concrete computation, distinctive to 'Hybrid / multi-strategy', will you use?

Current implementation:
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with diversity maintenance."""
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
```

Full algorithm for context:
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

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description (must mention category H: Hybrid / multi-strategy).
```python
def _select_survivors_batch(self, ...):
    ...
```