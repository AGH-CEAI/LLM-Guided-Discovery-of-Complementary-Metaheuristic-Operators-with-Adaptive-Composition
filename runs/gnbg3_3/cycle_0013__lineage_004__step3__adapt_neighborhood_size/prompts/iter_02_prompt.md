This is iteration 2 of 10. Propose ONE replacement implementation for `_adapt_neighborhood_size`.

═══════════════════════════════════════════════════════════════════════
ASSIGNED STRATEGY CATEGORY FOR THIS ITERATION:
  Category B — Spectral / linear-algebraic
  Use eigenvalues, singular values, condition numbers, rank, covariance shape, or matrix decompositions. Reason in terms of subspace alignment, anisotropy, or effective dimensionality.

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
- Keep the EXACT function signature: `def _adapt_neighborhood_size(self):`
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
  Task 17: *** UNSOLVED *** — best=5.021e+04 by original.py               (target=1e-08, ~+12.7 decades above target)
  Task 16: *** UNSOLVED *** — best=2.123e+03 by original.py               (target=1e-08, ~+11.3 decades above target)
  Task  6: *** UNSOLVED *** — best=6.862e+02 by original.py               (target=1e-08, ~+10.8 decades above target)
  Task 11: *** UNSOLVED *** — best=1.966e+02 by original.py               (target=1e-08, ~+10.3 decades above target)
  Task 19: *** UNSOLVED *** — best=1.402e+02 by original.py               (target=1e-08, ~+10.1 decades above target)
  Task  9: *** UNSOLVED *** — best=1.049e+02 by original.py               (target=1e-08, ~+10.0 decades above target)
  Task 12: *** UNSOLVED *** — best=7.251e+01 by original.py               (target=1e-08, ~+9.9 decades above target)
  Task 23: *** UNSOLVED *** — best=6.506e+01 by original.py               (target=1e-08, ~+9.8 decades above target)
  Task 18: *** UNSOLVED *** — best=5.120e+01 by original.py               (target=1e-08, ~+9.7 decades above target)
  Task  5: *** UNSOLVED *** — best=2.634e+01 by original.py               (target=1e-08, ~+9.4 decades above target)
  Task 20: *** UNSOLVED *** — best=2.550e+01 by original.py               (target=1e-08, ~+9.4 decades above target)
  Task 13: *** UNSOLVED *** — best=1.685e+01 by original.py               (target=1e-08, ~+9.2 decades above target)
  Task 10: *** UNSOLVED *** — best=1.416e+01 by original.py               (target=1e-08, ~+9.2 decades above target)
  Task 22: *** UNSOLVED *** — best=1.376e+01 by original.py               (target=1e-08, ~+9.1 decades above target)
  Task  7: *** UNSOLVED *** — best=1.071e+01 by original.py               (target=1e-08, ~+9.0 decades above target)
  Task 14: *** UNSOLVED *** — best=8.846e+00 by original.py               (target=1e-08, ~+8.9 decades above target)
  Task  3: *** UNSOLVED *** — best=6.780e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 21: *** UNSOLVED *** — best=5.474e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task 15: *** UNSOLVED *** — best=3.680e+00 by original.py               (target=1e-08, ~+8.6 decades above target)
  Task  8: *** UNSOLVED *** — best=2.950e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task  1: *** UNSOLVED *** — best=2.701e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task  4: *** UNSOLVED *** — best=1.189e+00 by original.py               (target=1e-08, ~+8.1 decades above target)
  Task  2: *** UNSOLVED *** — best=4.252e-01 by original.py               (target=1e-08, ~+7.6 decades above target)
  Task  0: *** UNSOLVED *** — best=1.632e-01 by original.py               (target=1e-08, ~+7.2 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_catA_idea  
-------------------------------------------------
0    1.6316e-01            -inf                  
1    2.7012e+00            -inf                  
2    4.2523e-01            -inf                  
3    6.7802e+00            -inf                  
4    1.1887e+00            -inf                  
5    2.6338e+01            -inf                  
6    6.8615e+02            -inf                  
7    1.0714e+01            -inf                  
8    2.9501e+00            -inf                  
9    1.0488e+02            -inf                  
10   1.4162e+01            -inf                  
11   1.9660e+02            -inf                  
12   7.2507e+01            -inf                  
13   1.6845e+01            -inf                  
14   8.8464e+00            -inf                  
15   3.6800e+00            -inf                  
16   2.1229e+03            -inf                  
17   5.0208e+04            -inf                  
18   5.1198e+01            -inf                  
19   1.4016e+02            -inf                  
20   2.5500e+01            -inf                  
21   5.4742e+00            -inf                  
22   1.3764e+01            -inf                  
23   6.5064e+01            -inf                  

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 5.021e+04  (target = 1e-08)
  Task 16: best error so far = 2.123e+03  (target = 1e-08)
  Task  6: best error so far = 6.862e+02  (target = 1e-08)
  Task 11: best error so far = 1.966e+02  (target = 1e-08)
  Task 19: best error so far = 1.402e+02  (target = 1e-08)
  Task  9: best error so far = 1.049e+02  (target = 1e-08)
  Task 12: best error so far = 7.251e+01  (target = 1e-08)
  Task 23: best error so far = 6.506e+01  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0

CATEGORY ROTATION SO FAR (per-category proposal count):
  A×1  B×0  C×0  D×0  E×0  F×0  G×0  H×0

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
2. How does YOUR ASSIGNED CATEGORY ('Spectral / linear-algebraic') give you a fundamentally
   different angle of attack on those obstacles than the prior variants
   in different categories?
3. What concrete computation, distinctive to 'Spectral / linear-algebraic', will you use?

Current implementation:
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
```

Full algorithm for context:
```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE:
      - Task 5: original wins with 768× gap to runner-up, 544618× to median.
        A 50/50 blend of 63.5 and 48742 yields ~24403, which is ~384× WORSE
        than the winner alone. Linear blending cannot preserve this advantage.
      - Task 1: original wins with 11.5× gap.
      - Task 3: original wins with 6.3× gap.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin burst with each of the 5 winning
        operators, tracking rank-based improvement scores on the ACTUAL
        landscape. Each operator gets 3 generations in a sub-population.
      - PHASE B (commit): run ONLY the winning operator for the remaining
        budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if committed operator stagnates (>20 gens, p=0.05),
        allow a small probability of switching to the runner-up.
    
    Win-count distribution (8/6/4/3/3 across 5 winners) confirms that
    different operators win on different tasks. The probe extracts a
    cheap FINGERPRINT from the landscape (convergence slope, rank-
    correlation, effective dimensionality) to route to the right operator.
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
        self.current_fitness = None
        
        # --- PROBE-AND-COMMIT STATE ---
        # Map operator ID -> list of rank-based scores
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        
        # THE 5 ACTUAL WINNING POSITION UPDATE STRATEGIES
        self._operators = [
            'original',      # variant_00: eigenvalue-anisotropic velocity modulation
            'svd_whitening', # variant_02: SVD-based population whitening (3 wins)
            'fitness_rank',  # variant_04: fitness-landscape rank signals only (8 wins)
            'temporal_drift',# variant_06: temporal-drift-modulated velocity scaling (3 wins)
            'centroid_knn',  # variant_09: centroid gravity + k-NN density (6 wins)
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator, min 15 total
        self._probe_gens_per_op = 3
        self._probe_total_budget = max(len(self._operators) * self._probe_gens_per_op, 15)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        # Running history for rank-based scoring
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Fingerprint signals (computed during probe)
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
        
        # Commit phase stagnation tracking
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
    
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
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
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
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
    
    def _compute_fingerprint(self):
        """Compute cheap landscape fingerprint from current population state.
        
        Returns a dict of signals observable from any black-box func:
          - convergence_slope: slope of log(best_fitness) vs generation
          - rank_corr: Spearman-like correlation between fitness rank and
                       distance to global best (proxy for landscape smoothness)
          - eff_dim: effective dimensionality from early covariance
          - diversity: current population diversity
        """
        signals = {}
        
        # 1. Convergence slope (from probe history)
        if len(self._probe_fitness_history) >= 3:
            gens = np.arange(len(self._probe_fitness_history))
            bests = np.array(self._probe_fitness_history)
            # Use log-slope for positive fitness; fallback to linear for negative/zero
            valid = bests > 0
            if np.sum(valid) >= 2:
                log_bests = np.log1p(bests[valid])
                signals['convergence_slope'] = np.polyfit(gens[valid], log_bests, 1)[0]
            else:
                signals['convergence_slope'] = np.polyfit(gens, bests, 1)[0]
        else:
            signals['convergence_slope'] = 0.0
        
        # 2. Rank correlation (fitness vs distance to global best)
        if self.global_best is not None and self.np > 2:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                signals['rank_corr'] = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                signals['rank_corr'] = 0.0
        else:
            signals['rank_corr'] = 0.0
        
        # 3. Effective dimensionality from covariance
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cumvar = np.cumsum(sorted(eigenvalues, reverse=True)) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        # 4. Diversity
        signals['diversity'] = self._compute_diversity()
        
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        
        return signals
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight using condition number of population covariance."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            # Spectral signal: map condition number to inertia adjustment
            # High cond = anisotropic/ill-conditioned → more exploration (higher inertia)
            # Low cond = well-conditioned → more exploitation (lower inertia)
            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            # Map spectral signal to inertia range [0.4, 0.95]
            target_inertia = 0.4 + 0.55 * spectral_signal

            # Blend with time-based baseline
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (shared across all position update strategies)
    # -------------------------------------------------------------------------
    
    def _velocity_update_base(self):
        """Base velocity update: cognitive + social + DE mutation."""
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
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (the 5 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation."""
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
    
    def _position_update_svd_whitening(self):
        """SVD-based population whitening with anisotropic damping (variant_02).
        
        3 wins on tasks 2, 4, 8. Key insight: counteracts anisotropy by
        dampening high-spread directions and amplifying collapsed directions.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            singular_values = np.clip(singular_values, 1e-10, None)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Inverse-square-law: stronger effect on collapsed directions
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_fitness_rank(self):
        """Graph-Laplacian eigenvalue-based velocity modulation (Category E).

        Key insight: Build k-NN graph over population; use Laplacian eigenvalues
        to detect clustering/convergence. Small spectral gap = clustered population
        → increase exploration. Small component particles → stronger nudges toward
        larger components. This topology-aware approach is orthogonal to fitness-
        landscape (D) and spectral-matrix (B) approaches used previously.
        """
        # Build k-NN graph over population
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Connected component analysis using graph traversal
        visited = np.zeros(self.np, dtype=bool)
        components = []
        for start in range(self.np):
            if visited[start]:
                continue
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for neighbor in knn_indices[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            components.append(component)

        component_sizes = np.array([len(c) for c in components])
        particle_component_size = np.array([component_sizes[np.argmax([i in c for c in components])] for i in range(self.np)])

        # Graph Laplacian spectral analysis: second-smallest eigenvalue of normalized Laplacian
        try:
            # Compute adjacency (k-NN symmetric)
            row_indices = np.repeat(np.arange(self.np), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            adj_size = self.np * self.np
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
            adj = adj + adj.T
            adj.data[:] = 1.0

            # Normalized Laplacian: I - D^{-1/2} * A * D^{-1/2}
            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
            lap = dspy.identity(self.np) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)

            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)

            # Modularity-like signal from top eigenvalues
            modularity_signal = 1.0 - eigenvalues[0] if len(eigenvalues) > 0 else 0.0
        except:
            spectral_gap = 1.0
            modularity_signal = 0.0

        # Fitness rank signal (from Category D)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Graph-based exploration factor
        # Small spectral_gap = clustered → more exploration needed
        clustering_explore = 1.0 + 0.5 * (1.0 - spectral_gap)
        # Small component → particles may be isolated → stronger exploration
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.3, 1.0)
        graph_explore = clustering_explore * (2.0 - size_factor)
        graph_explore = np.clip(graph_explore, 0.5, 2.0)

        # Success history (from Category D)
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # Velocity scaling combines graph signals with fitness rank
        vel_scale = graph_explore * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.5, 2.5)

        # Fitness-based directional perturbation
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        random_perturb = graph_explore[:, np.newaxis] * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_temporal_drift(self):
        """Hybrid: Temporal drift tracking + topology-based fragmentation detection (Category H).

        Combines TWO distinct mechanisms:
          1. Temporal: Centroid drift rate across generations to detect convergence phases
          2. Topology: k-NN connected component analysis to detect population fragmentation

        Switching logic is data-driven:
          - Low drift + fragmented population → strong exploration boost
          - High drift (exploring) → trust spectral signals
          - Fragmented + improving → moderate exploration
          - Well-connected + converging → exploit via spectral damping

        Targets worst tasks (17, 16, 6, 11) where premature fragmentation causes
        the population to get trapped in disconnected local optima.
        """
        try:
            # === MECHANISM 1: TEMPORAL DRIFT TRACKING ===
            centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_prev_centroid'):
                self._prev_centroid = centroid.copy()
                self._centroid_drift_history = []

            drift = np.linalg.norm(centroid - self._prev_centroid)
            self._prev_centroid = centroid.copy()

            if not hasattr(self, '_centroid_drift_history'):
                self._centroid_drift_history = []
            self._centroid_drift_history.append(drift)
            if len(self._centroid_drift_history) > 20:
                self._centroid_drift_history.pop(0)

            # EMA of drift rate
            if not hasattr(self, '_ema_drift'):
                self._ema_drift = 0.0
            self._ema_drift = 0.3 * drift + 0.7 * self._ema_drift

            # Normalize drift by population spread
            population_spread = np.std(self.population) + 1e-10
            normalized_drift = self._ema_drift / population_spread

            # === MECHANISM 2: TOPOLOGY FRAGMENTATION DETECTION ===
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

            # Connected component analysis via BFS
            visited = np.zeros(self.np, dtype=bool)
            components = []
            for start in range(self.np):
                if visited[start]:
                    continue
                component = []
                stack = [start]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    component.append(node)
                    for neighbor in knn_indices[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
                components.append(component)

            n_components = len(components)
            component_sizes = np.array([len(c) for c in components])

            # Fragmentation ratio: how uneven is the distribution?
            max_component_size = np.max(component_sizes)
            fragmentation_ratio = 1.0 - (max_component_size / (self.np + 1e-10))

            # Per-particle component membership
            particle_to_component = np.zeros(self.np, dtype=int)
            for idx, comp in enumerate(components):
                for particle_idx in comp:
                    particle_to_component[particle_idx] = idx

            # === PRINCIPLED SWITCHING LOGIC ===
            # State classification based on temporal and topological signals
            is_converging = normalized_drift < 0.05  # Low drift = converging
            is_fragmented = n_components > 1 and fragmentation_ratio > 0.3

            # === SPECTRAL ANALYSIS (for when not fragmented) ===
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

            # === TEMPORAL IMPROVEMENT SIGNAL ===
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_spectral_ema_improvement'):
                self._spectral_ema_improvement = 0.0
            self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement

            # === DECISION TREE WITH PRINCIPLED SWITCHING ===
            if is_fragmented and is_converging:
                # CRITICAL: Population is fragmented AND stuck → aggressive exploration
                # Weight heavily toward topology-based correction
                base_scale = 1.5
                topology_weight = 0.8
                spectral_weight = 0.2
            elif is_fragmented and not is_converging:
                # Fragmented but still moving → moderate exploration
                base_scale = 1.3
                topology_weight = 0.6
                spectral_weight = 0.4
            elif not is_fragmented and is_converging:
                # Well-connected but converging → use spectral exploitation
                base_scale = 0.9
                topology_weight = 0.2
                spectral_weight = 0.8
            else:
                # Well-connected and exploring → balanced
                base_scale = 1.1
                topology_weight = 0.4
                spectral_weight = 0.6

            # === COMBINE MECHANISMS ===
            # Spectral contribution
            if self._spectral_ema_improvement > 1e-6:
                spectral_base = 1.2
            elif self._spectral_ema_improvement > 1e-10:
                spectral_base = 1.0
            else:
                spectral_base = 0.7

            spectral_scale = spectral_base * (1.0 + 0.4 * spectral_signal)
            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                spectral_scale *= (1.0 + 0.3 * log_damp)

            # Topology contribution: inter-component repulsion
            # Particles in smaller components get pushed toward larger ones
            component_centroids = np.array([np.mean(self.population[c], axis=0) for c in components])
            largest_comp_idx = np.argmax(component_sizes)
            target_centroid = component_centroids[largest_comp_idx]

            # Per-particle correction toward largest component centroid
            topology_correction = np.zeros((self.np, self.dim))
            for i in range(self.np):
                comp_idx = particle_to_component[i]
                comp_size = component_sizes[comp_idx]
                if comp_idx != largest_comp_idx:
                    # Scale correction by inverse component size (smaller = stronger push)
                    correction_strength = 0.5 * (1.0 - comp_size / self.np)
                    direction = target_centroid - self.population[i]
                    direction_norm = np.linalg.norm(direction) + 1e-10
                    topology_correction[i] = correction_strength * (direction / direction_norm)

            # Combine spectral and topology contributions
            combined_scale = spectral_weight * spectral_scale + topology_weight * (1.0 + 0.5 * fragmentation_ratio)

            # Apply combined scale to velocity
            U = np.linalg.eigvalsh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = np.linalg.eigh(cov)[1][:, idx]

            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            per_comp_scale = np.clip(sv_norm, 0.1, 2.0)
            vel_scaled = vel_proj * per_comp_scale

            # Final update: spectral velocity + topology correction
            new_population = self.population + combined_scale * (vel_scaled @ V.T) + topology_correction

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_centroid_knn(self):
        """Centroid gravity + k-NN density modulation (variant_09).
        
        6 wins on tasks 6, 9, 10, 14, 19, 23. Key insight: particles in
        sparse regions get more centroid pull; dense regions get repulsion.
        """
        centroid = np.mean(self.population, axis=0)

        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10

        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)

        correction = centroid_attraction * to_centroid_dir * density_modulation

        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'svd_whitening':
            self._position_update_svd_whitening()
        elif operator == 'fitness_rank':
            self._position_update_fitness_rank()
        elif operator == 'temporal_drift':
            self._position_update_temporal_drift()
        elif operator == 'centroid_knn':
            self._position_update_centroid_knn()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score using rank-based (scale-independent) scoring.
        
        Score = fraction of probe generations with WORSE (higher) fitness.
        This is rank-based, not raw fitness — no scale-dependent constants.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        if len(self._probe_fitness_history) < 2:
            score = 1.0
        else:
            current_best = min(self._probe_fitness_history)
            current_worst = max(self._probe_fitness_history)
            if current_worst > current_best:
                # Higher score = better (lower fitness)
                normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
                score = np.clip(normalized, 0.0, 1.0)
            else:
                score = 1.0
        
        self._operator_scores[operator].append(score)
        self._operator_total_score[operator] += score
        self._operator_sample_count[operator] += 1
    
    def _compute_ucb_score(self, operator):
        """Compute UCB1-style score for operator selection.
        
        Uses rank-based average (already in [0,1]) so exploration bonus
        is on the same scale — no magic constants.
        """
        n = self._operator_sample_count[operator]
        if n == 0:
            return float('inf')
        
        total_n = sum(self._operator_sample_count.values())
        avg_score = self._operator_total_score[operator] / n
        
        # UCB1 exploration bonus — scale is [0,1], so this is well-calibrated
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        
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
    # RESTART & LOCAL BEST
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT position-update selection.
        
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
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset EMA states for temporal_drift
        if hasattr(self, '_ema_centroid'):
            del self._ema_centroid
        if hasattr(self, '_ema_centroid_velocity'):
            del self._ema_centroid_velocity
        if hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        if hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        if hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        
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
            if hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Velocity update (shared base), then position update (PROBED)
                self._velocity_update_base()
                self._dispatch_position_update(current_op)
                
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
                
                # Record probe score (rank-based, scale-independent)
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
                    # Compute fingerprint from probe data
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._velocity_update_base()
                    self._dispatch_position_update(self._runner_up_operator)
                else:
                    self._velocity_update_base()
                    self._dispatch_position_update(self._committed_operator)
                
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
                
                # Track commit-phase improvement for potential review
                if self.global_best_fitness < self._commit_best_fitness:
                    self._commit_best_fitness = self.global_best_fitness
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description (must mention category B: Spectral / linear-algebraic).
```python
def _adapt_neighborhood_size(self, ...):
    ...
```