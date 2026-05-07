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
  Task 17: *** UNSOLVED *** — best=7.983e+04 by original.py               (target=1e-08, ~+12.9 decades above target)
  Task 16: *** UNSOLVED *** — best=4.770e+03 by variant_05_catE_idea_0.py (target=1e-08, ~+11.7 decades above target)
  Task 11: *** UNSOLVED *** — best=3.656e+02 by variant_05_catE_idea_0.py (target=1e-08, ~+10.6 decades above target)
  Task 19: *** UNSOLVED *** — best=1.352e+02 by variant_06_catF_idea_0.py (target=1e-08, ~+10.1 decades above target)
  Task  6: *** UNSOLVED *** — best=1.347e+02 by variant_02_catB_idea_0.py (target=1e-08, ~+10.1 decades above target)
  Task 18: *** UNSOLVED *** — best=5.798e+01 by original.py               (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=4.517e+01 by variant_07_catG_idea_0.py (target=1e-08, ~+9.7 decades above target)
  Task  9: *** UNSOLVED *** — best=3.643e+01 by variant_02_catB_idea_0.py (target=1e-08, ~+9.6 decades above target)
  Task 13: *** UNSOLVED *** — best=2.996e+01 by original.py               (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=1.991e+01 by variant_06_catF_idea_0.py (target=1e-08, ~+9.3 decades above target)
  Task 22: *** UNSOLVED *** — best=1.325e+01 by original.py               (target=1e-08, ~+9.1 decades above target)
  Task  5: *** UNSOLVED *** — best=1.233e+01 by variant_06_catF_idea_0.py (target=1e-08, ~+9.1 decades above target)
  Task 21: *** UNSOLVED *** — best=5.665e+00 by variant_05_catE_idea_0.py (target=1e-08, ~+8.8 decades above target)
  Task 14: *** UNSOLVED *** — best=4.110e+00 by variant_07_catG_idea_0.py (target=1e-08, ~+8.6 decades above target)
  Task  7: *** UNSOLVED *** — best=3.700e+00 by variant_05_catE_idea_0.py (target=1e-08, ~+8.6 decades above target)
  Task 15: *** UNSOLVED *** — best=3.535e+00 by variant_05_catE_idea_0.py (target=1e-08, ~+8.5 decades above target)
  Task 12: *** UNSOLVED *** — best=2.404e+00 by variant_05_catE_idea_0.py (target=1e-08, ~+8.4 decades above target)
  Task  1: *** UNSOLVED *** — best=2.012e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task  8: *** UNSOLVED *** — best=1.711e+00 by variant_07_catG_idea_0.py (target=1e-08, ~+8.2 decades above target)
  Task  3: *** UNSOLVED *** — best=1.625e+00 by variant_07_catG_idea_0.py (target=1e-08, ~+8.2 decades above target)
  Task 10: *** UNSOLVED *** — best=1.075e+00 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task  4: *** UNSOLVED *** — best=8.754e-01 by variant_02_catB_idea_0.py (target=1e-08, ~+7.9 decades above target)
  Task  2: *** UNSOLVED *** — best=3.759e-05 by original.py               (target=1e-08, ~+3.6 decades above target)
  Task  0: *** UNSOLVED *** — best=1.138e-07 by variant_06_catF_idea_0.py (target=1e-08, ~+1.1 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_catA_idea  variant_02_catB_idea  variant_03_catC_idea  variant_04_catD_idea  variant_05_catE_idea  variant_06_catF_idea  variant_07_catG_idea  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    1.1532e-07            6.3042e+01            1.2089e-07            -inf                  nan                   1.1811e-07            1.1381e-07            5.5911e-06            
1    2.0123e+00            1.9186e+02            2.0250e+00            -inf                  nan                   3.4004e+00            2.7920e+00            2.0290e+00            
2    3.7587e-05            5.0178e+02            8.3921e-05            -inf                  nan                   7.1429e-05            6.3439e-05            9.8281e-04            
3    2.3821e+00            2.7791e+02            1.8108e+00            -inf                  nan                   2.5948e+00            2.0379e+00            1.6251e+00            
4    9.2371e-01            5.4628e+00            8.7545e-01            -inf                  nan                   9.5824e-01            9.2228e-01            1.2300e+00            
5    4.9558e+01            9.8337e+08            4.0129e+01            -inf                  nan                   3.2807e+01            1.2332e+01            2.1870e+01            
6    2.1397e+02            4.4439e+03            1.3467e+02            -inf                  nan                   1.9546e+02            2.0054e+02            2.7585e+02            
7    4.4730e+00            3.2463e+02            5.5991e+00            -inf                  nan                   3.6996e+00            5.0195e+00            4.1374e+00            
8    1.7997e+00            4.7318e+01            2.2861e+00            -inf                  nan                   1.8097e+00            1.9124e+00            1.7107e+00            
9    3.9904e+01            2.0463e+02            3.6428e+01            -inf                  nan                   4.1954e+01            4.0888e+01            4.1968e+01            
10   1.0752e+00            2.7305e+02            1.1138e+00            -inf                  nan                   1.0831e+00            1.1887e+00            2.1673e+01            
11   4.0109e+02            1.2096e+03            4.2219e+02            -inf                  nan                   3.6559e+02            4.0675e+02            4.4028e+02            
12   2.6534e+00            2.0225e+02            2.8195e+00            -inf                  nan                   2.4035e+00            2.7971e+00            3.7481e+01            
13   2.9959e+01            5.1116e+01            3.1138e+01            -inf                  nan                   3.1259e+01            3.2322e+01            3.3999e+01            
14   7.4573e+00            1.7165e+01            7.2438e+00            -inf                  nan                   7.2695e+00            7.5286e+00            4.1101e+00            
15   3.5752e+00            4.2249e+00            3.5982e+00            -inf                  nan                   3.5350e+00            3.5749e+00            3.7005e+00            
16   5.0253e+03            1.5220e+04            5.5546e+03            -inf                  nan                   4.7704e+03            5.4209e+03            9.1204e+03            
17   7.9831e+04            3.4776e+05            8.4941e+04            -inf                  nan                   8.0018e+04            8.1585e+04            1.0250e+05            
18   5.7977e+01            9.7929e+01            5.8959e+01            -inf                  nan                   5.8599e+01            5.8386e+01            6.0226e+01            
19   1.4758e+02            3.8331e+02            1.5291e+02            -inf                  nan                   1.4674e+02            1.3518e+02            1.5804e+02            
20   2.0669e+01            5.7156e+01            2.0397e+01            -inf                  nan                   2.0568e+01            1.9910e+01            2.4009e+01            
21   5.7481e+00            6.0187e+00            5.7064e+00            -inf                  nan                   5.6646e+00            5.7520e+00            5.8589e+00            
22   1.3255e+01            2.1651e+01            1.3484e+01            -inf                  nan                   1.3595e+01            1.3774e+01            1.3762e+01            
23   5.1727e+01            9.4750e+01            5.2768e+01            -inf                  nan                   5.3258e+01            5.4701e+01            4.5169e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 7.983e+04  (target = 1e-08)
  Task 16: best error so far = 4.770e+03  (target = 1e-08)
  Task 11: best error so far = 3.656e+02  (target = 1e-08)
  Task 19: best error so far = 1.352e+02  (target = 1e-08)
  Task  6: best error so far = 1.347e+02  (target = 1e-08)
  Task 18: best error so far = 5.798e+01  (target = 1e-08)
  Task 23: best error so far = 4.517e+01  (target = 1e-08)
  Task  9: best error so far = 3.643e+01  (target = 1e-08)
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
            """Greedy selection with spectral diversity modulation."""
            improved_mask = trial_fitness < fitness
            improvement = fitness - trial_fitness

            # Compute population covariance eigenvalue spread
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
            # High spread → anisotropic population → reduce selection pressure
            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0

            # Modulation: in ill-conditioned populations, accept smaller improvements
            # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
            diversity_bonus = np.log1p(condition)

            # Scaled improvement relative to population fitness range
            fit_range = np.ptp(fitness)
            fit_range = max(fit_range, 1e-10)
            scaled_improvement = improvement / fit_range

            # Accept if improved OR if scaled improvement exceeds diversity threshold
            accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
            accept_mask = improved_mask | (scaled_improvement > accept_threshold)

            new_population = population.copy()
            new_fitness = fitness.copy()

            new_population[accept_mask] = trials[accept_mask]
            new_fitness[accept_mask] = trial_fitness[accept_mask]

            return new_population, new_fitness, accept_mask
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
            """Greedy selection with spectral diversity modulation."""
            improved_mask = trial_fitness < fitness
            improvement = fitness - trial_fitness

            # Compute population covariance eigenvalue spread
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
            # High spread → anisotropic population → reduce selection pressure
            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0

            # Modulation: in ill-conditioned populations, accept smaller improvements
            # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
            diversity_bonus = np.log1p(condition)

            # Scaled improvement relative to population fitness range
            fit_range = np.ptp(fitness)
            fit_range = max(fit_range, 1e-10)
            scaled_improvement = improvement / fit_range

            # Accept if improved OR if scaled improvement exceeds diversity threshold
            accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
            accept_mask = improved_mask | (scaled_improvement > accept_threshold)

            new_population = population.copy()
            new_fitness = fitness.copy()

            new_population[accept_mask] = trials[accept_mask]
            new_fitness[accept_mask] = trial_fitness[accept_mask]

            return new_population, new_fitness, accept_mask
    
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