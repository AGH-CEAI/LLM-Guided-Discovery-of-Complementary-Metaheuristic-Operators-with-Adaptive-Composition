This is iteration 4 of 10. Propose ONE replacement implementation for `_velocity_update_batch`.

═══════════════════════════════════════════════════════════════════════
ASSIGNED STRATEGY CATEGORY FOR THIS ITERATION:
  Category D — Fitness-landscape / rank-based
  Reason about fitness values directly: rank correlations, Spearman/Kendall coefficients, fitness percentiles, fitness-distance correlation, success-history. Do NOT use raw distances or covariance — only fitness signals.

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
- Keep the EXACT function signature: `def _velocity_update_batch(self):`
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
  Task 17: *** UNSOLVED *** — best=6.764e+04 by original.py               (target=1e-08, ~+12.8 decades above target)
  Task 16: *** UNSOLVED *** — best=5.401e+03 by original.py               (target=1e-08, ~+11.7 decades above target)
  Task  6: *** UNSOLVED *** — best=7.869e+02 by variant_01_catA_idea_0.py (target=1e-08, ~+10.9 decades above target)
  Task 11: *** UNSOLVED *** — best=2.596e+02 by original.py               (target=1e-08, ~+10.4 decades above target)
  Task 19: *** UNSOLVED *** — best=2.309e+02 by original.py               (target=1e-08, ~+10.4 decades above target)
  Task 12: *** UNSOLVED *** — best=1.096e+02 by original.py               (target=1e-08, ~+10.0 decades above target)
  Task  9: *** UNSOLVED *** — best=9.875e+01 by variant_03_catC_idea_0.py (target=1e-08, ~+10.0 decades above target)
  Task  5: *** UNSOLVED *** — best=9.845e+01 by original.py               (target=1e-08, ~+10.0 decades above target)
  Task 10: *** UNSOLVED *** — best=7.890e+01 by original.py               (target=1e-08, ~+9.9 decades above target)
  Task 23: *** UNSOLVED *** — best=7.888e+01 by original.py               (target=1e-08, ~+9.9 decades above target)
  Task 18: *** UNSOLVED *** — best=5.352e+01 by original.py               (target=1e-08, ~+9.7 decades above target)
  Task  2: *** UNSOLVED *** — best=4.862e+01 by variant_01_catA_idea_0.py (target=1e-08, ~+9.7 decades above target)
  Task 20: *** UNSOLVED *** — best=3.724e+01 by original.py               (target=1e-08, ~+9.6 decades above target)
  Task 13: *** UNSOLVED *** — best=2.382e+01 by original.py               (target=1e-08, ~+9.4 decades above target)
  Task 22: *** UNSOLVED *** — best=1.537e+01 by original.py               (target=1e-08, ~+9.2 decades above target)
  Task 14: *** UNSOLVED *** — best=1.198e+01 by variant_03_catC_idea_0.py (target=1e-08, ~+9.1 decades above target)
  Task  8: *** UNSOLVED *** — best=1.150e+01 by variant_01_catA_idea_0.py (target=1e-08, ~+9.1 decades above target)
  Task  7: *** UNSOLVED *** — best=8.312e+00 by original.py               (target=1e-08, ~+8.9 decades above target)
  Task  0: *** UNSOLVED *** — best=7.000e+00 by variant_03_catC_idea_0.py (target=1e-08, ~+8.8 decades above target)
  Task  3: *** UNSOLVED *** — best=6.844e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 21: *** UNSOLVED *** — best=5.668e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 15: *** UNSOLVED *** — best=3.861e+00 by original.py               (target=1e-08, ~+8.6 decades above target)
  Task  1: *** UNSOLVED *** — best=2.918e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task  4: *** UNSOLVED *** — best=2.547e+00 by variant_01_catA_idea_0.py (target=1e-08, ~+8.4 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_catA_idea  variant_02_catB_idea  variant_03_catC_idea  
---------------------------------------------------------------------------------------------
0    9.6395e+00            7.6744e+00            nan                   6.9998e+00            
1    2.9175e+00            8.7971e+00            nan                   1.9226e+01            
2    1.4261e+02            4.8622e+01            nan                   9.2774e+01            
3    6.8438e+00            4.2338e+01            nan                   6.8241e+01            
4    2.8046e+00            2.5470e+00            nan                   2.9398e+00            
5    9.8449e+01            3.3650e+05            nan                   2.1439e+06            
6    1.3178e+03            7.8691e+02            nan                   9.5125e+02            
7    8.3120e+00            3.3207e+01            nan                   5.2562e+01            
8    1.1768e+01            1.1498e+01            nan                   1.6284e+01            
9    1.5675e+02            1.0335e+02            nan                   9.8747e+01            
10   7.8895e+01            3.2216e+02            1.7565e+02            1.5876e+02            
11   2.5965e+02            1.3339e+03            7.5887e+02            1.0559e+03            
12   1.0957e+02            2.4766e+02            1.7212e+02            1.7091e+02            
13   2.3822e+01            5.1097e+01            3.3342e+01            3.0521e+01            
14   1.2736e+01            1.3549e+01            1.3874e+01            1.1979e+01            
15   3.8607e+00            4.1727e+00            4.0798e+00            4.0853e+00            
16   5.4011e+03            1.1651e+04            nan                   5.7261e+03            
17   6.7644e+04            3.4769e+05            1.9552e+05            2.0800e+05            
18   5.3519e+01            9.8527e+01            7.9901e+01            8.2094e+01            
19   2.3093e+02            3.1461e+02            2.7769e+02            2.4688e+02            
20   3.7245e+01            6.0878e+01            nan                   4.1354e+01            
21   5.6677e+00            6.0493e+00            5.7479e+00            5.7478e+00            
22   1.5368e+01            2.0365e+01            1.7533e+01            1.8358e+01            
23   7.8882e+01            9.9473e+01            9.3779e+01            8.3359e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 6.764e+04  (target = 1e-08)
  Task 16: best error so far = 5.401e+03  (target = 1e-08)
  Task  6: best error so far = 7.869e+02  (target = 1e-08)
  Task 11: best error so far = 2.596e+02  (target = 1e-08)
  Task 19: best error so far = 2.309e+02  (target = 1e-08)
  Task 12: best error so far = 1.096e+02  (target = 1e-08)
  Task  9: best error so far = 9.875e+01  (target = 1e-08)
  Task  5: best error so far = 9.845e+01  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0

CATEGORY ROTATION SO FAR (per-category proposal count):
  A×1  B×1  C×1  D×0  E×0  F×0  G×0  H×0

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
2. How does YOUR ASSIGNED CATEGORY ('Fitness-landscape / rank-based') give you a fundamentally
   different angle of attack on those obstacles than the prior variants
   in different categories?
3. What concrete computation, distinctive to 'Fitness-landscape / rank-based', will you use?

Current implementation:
```python
def _velocity_update_batch(self):
        """Update velocities with adaptive components and DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)
        
        # DE/rand/1 mutation injection for enhanced exploration
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)  # Decreases over time
        mutation_active = mutation_mask < mutation_threshold
        
        # Generate mutation vectors from random distinct indices
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        # Apply mutation where active
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

Full algorithm for context:
```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Adaptive Topology Particle Swarm Optimizer with DE Mutation (ATPSO-DM).
    
    Combines:
    - Ring topology PSO with adaptive neighborhood size
    - Adaptive inertia weight based on swarm diversity
    - DE/rand/1 mutation injection for exploration
    - Adaptive cognitive/social coefficients
    - Diversity-guided restart mechanism
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
    
    def _velocity_update_batch(self):
        """Update velocities with adaptive components and DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)
        
        # DE/rand/1 mutation injection for enhanced exploration
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)  # Decreases over time
        mutation_active = mutation_mask < mutation_threshold
        
        # Generate mutation vectors from random distinct indices
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        # Apply mutation where active
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
        # Compute population covariance for spectral analysis
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)

        # Eigendecomposition for spectral scaling
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute condition-number-based scaling (stuck populations have high condition number)
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)

            # Project velocity onto principal axes and scale
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale

            # Transform back to original space
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            # Fallback to standard update on decomposition failure
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            # Reinitialize worst 30% of particles
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            # Generate new random particles
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            # Reset fitness for replaced particles
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            # Perturb global best slightly
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
        Run the optimizer.
        
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
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        # Handle truncated initial evaluation
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        # Check initial stopping condition
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Main optimization loop
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            # Check for budget exhaustion
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Update bests
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            
            # Restart check
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description (must mention category D: Fitness-landscape / rank-based).
```python
def _velocity_update_batch(self, ...):
    ...
```