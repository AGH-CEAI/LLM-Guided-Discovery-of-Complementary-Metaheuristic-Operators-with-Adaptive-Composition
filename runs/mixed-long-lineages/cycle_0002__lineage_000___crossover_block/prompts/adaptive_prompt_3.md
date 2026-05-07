Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_crossover_block` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            
1      1.222014e+00            3.411675e+00            -inf                    4.712748e+00            2.064306e+00            -inf                    2.959998e+00            2.971570e+00            3.081107e+00            -inf                    1.910987e+00            
2      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            
3      1.501009e+00            2.881465e+00            -inf                    3.227897e+01            1.780098e+00            -inf                    1.452526e+00            2.154406e+00            1.292618e+00            -inf                    1.318681e+00            
4      1.000000e-08            5.180537e-02            -inf                    1.447814e-01            3.577387e-02            -inf                    2.971992e-02            2.126460e-02            5.852970e-02            -inf                    5.155277e-02            
5      1.443927e+01            1.787792e+02            -inf                    2.389223e+05            1.072648e+02            -inf                    3.761383e+01            1.380679e+01            1.637909e+01            -inf                    1.442251e+00            
6      3.662872e+01            4.050146e+01            -inf                    4.661467e+01            3.502609e+01            -inf                    3.920095e+01            4.200829e+01            2.863740e+01            -inf                    3.900984e+01            
7      3.852420e+00            5.943993e+00            -inf                    1.213236e+01            4.262568e+00            -inf                    3.576812e+00            4.769500e+00            4.551279e+00            -inf                    2.630420e+00            
8      1.635026e+00            1.681228e+00            -inf                    2.062919e+00            1.690042e+00            -inf                    2.128102e+00            1.926354e+00            1.772283e+00            -inf                    1.876179e+00            
9      2.149761e+01            1.356326e+01            -inf                    3.542912e+01            1.743051e+01            -inf                    1.643066e+01            1.988959e+01            8.224790e+00            -inf                    1.336138e+01            
10     1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.102104e-05            -inf                    2.565397e-07            
11     8.566704e+01            8.284871e+01            -inf                    8.706780e+01            5.143995e+01            -inf                    2.876143e+01            7.235102e+01            1.885961e+02            -inf                    5.920067e+01            
12     1.816445e-08            5.081466e-06            -inf                    7.673078e-06            1.000000e-08            -inf                    1.992350e-08            4.434533e-06            2.809059e-01            -inf                    4.171731e-03            
13     1.277921e+01            2.318130e+01            -inf                    2.202784e+01            1.119432e+01            -inf                    9.223490e+00            1.784760e+01            1.216350e+01            -inf                    1.605938e+01            
14     6.927423e+00            6.855312e+00            -inf                    4.492523e+00            7.576968e+00            -inf                    6.701086e+00            8.658788e+00            8.576519e+00            -inf                    7.627232e+00            
15     2.767830e+00            3.803241e+00            -inf                    2.854781e+00            2.642095e+00            -inf                    2.413169e+00            2.736123e+00            3.358978e+00            -inf                    2.757144e+00            
16     8.035295e+02            1.071725e+03            -inf                    9.065316e+02            6.482112e+02            -inf                    6.992322e+02            6.724095e+02            8.665658e+02            -inf                    7.091101e+02            
17     1.045826e+04            6.520501e+03            -inf                    2.087392e+04            5.919219e+03            -inf                    1.826202e+03            3.955532e+03            6.957754e+03            -inf                    6.881655e+03            
18     2.646899e+01            3.123494e+01            -inf                    3.929132e+01            3.026164e+01            -inf                    2.800493e+01            3.106926e+01            3.191287e+01            -inf                    2.760224e+01            
19     8.724815e+01            8.040406e+01            -inf                    6.960259e+01            1.022119e+02            -inf                    9.025283e+01            1.121598e+02            1.402022e+02            -inf                    1.017425e+02            
20     1.633115e+01            1.922343e+01            -inf                    2.065336e+01            1.755656e+01            -inf                    1.842932e+01            1.849673e+01            1.979348e+01            -inf                    1.965091e+01            
21     4.883991e+00            4.895486e+00            -inf                    4.669386e+00            4.967338e+00            -inf                    4.869167e+00            5.089319e+00            5.212070e+00            -inf                    5.024379e+00            
22     1.001935e+01            1.123693e+01            -inf                    1.091456e+01            1.049047e+01            -inf                    1.008281e+01            1.115928e+01            1.272219e+01            -inf                    1.103027e+01            
23     4.183043e+01            4.142247e+01            -inf                    4.007639e+01            4.165225e+01            -inf                    4.975248e+01            4.887398e+01            5.258898e+01            -inf                    4.506870e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: original.py  (error=1.222014e+00)
Task  2: SKIPPED (trivial)
Task  3: variant_08_idea_0.py  (error=1.292618e+00)
Task  4: original.py  (error=1.000000e-08)
Task  5: variant_10_idea_0.py  (error=1.442251e+00)
Task  6: variant_08_idea_0.py  (error=2.863740e+01)
Task  7: variant_10_idea_0.py  (error=2.630420e+00)
Task  8: original.py  (error=1.635026e+00)
Task  9: variant_08_idea_0.py  (error=8.224790e+00)
Task 10: SKIPPED (trivial)
Task 11: variant_06_idea_0.py  (error=2.876143e+01)
Task 12: variant_04_idea_0.py  (error=1.000000e-08)
Task 13: variant_06_idea_0.py  (error=9.223490e+00)
Task 14: variant_03_idea_0.py  (error=4.492523e+00)
Task 15: variant_06_idea_0.py  (error=2.413169e+00)
Task 16: variant_04_idea_0.py  (error=6.482112e+02)
Task 17: variant_06_idea_0.py  (error=1.826202e+03)
Task 18: original.py  (error=2.646899e+01)
Task 19: variant_03_idea_0.py  (error=6.960259e+01)
Task 20: original.py  (error=1.633115e+01)
Task 21: variant_03_idea_0.py  (error=4.669386e+00)
Task 22: original.py  (error=1.001935e+01)
Task 23: variant_03_idea_0.py  (error=4.007639e+01)

WIN COUNTS:
  original.py: 6 wins
  variant_06_idea_0.py: 4 wins
  variant_03_idea_0.py: 4 wins
  variant_08_idea_0.py: 3 wins
  variant_10_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_03_idea_0.py (4 wins) ---
```python
def _crossover_block(self, population, trial_population):
        """Op 3: Success-rate guided crossover with adaptive mixing."""
        n_trials = self.NP
        dim = self.dim

        # Initialize per-dimension success tracking if needed
        if not hasattr(self, 'dim_cx_success') or self.dim_cx_success is None or len(self.dim_cx_success) != dim:
            self.dim_cx_success = np.zeros(dim)
            self.dim_cx_total = np.zeros(dim) + 1e-10

        # Compute per-dimension success rates for crossover
        success_rates = self.dim_cx_success / (self.dim_cx_total + 1e-10)

        # Global diversity measure
        pop_std = np.std(population[:n_trials], axis=0)
        max_spread = self.upper - self.lower
        diversity = np.mean(pop_std / (max_spread + 1e-10))

        # Adaptive mixing factor: conservative when diverse, aggressive when clustered
        mix_base = np.clip(0.3 + 0.2 * diversity, 0.15, 0.5)

        # Per-individual adaptive mixing based on relative fitness
        fitness = self.population_fitness[:n_trials] if hasattr(self, 'population_fitness') else np.zeros(n_trials)
        best_fit = np.min(fitness) if len(fitness) > 0 and np.any(np.isfinite(fitness)) else 0.0
        worst_fit = np.max(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else 1.0
        fit_range = worst_fit - best_fit + 1e-10

        # Individuals worse than median get higher mixing (more exploration)
        median_fit = np.median(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else best_fit
        mix_factor = np.clip(
            mix_base + 0.2 * np.maximum(0, (median_fit - fitness) / fit_range),
            0.1, 0.6
        )

        # Create blended population (weighted average)
        mix_expanded = mix_factor[:, np.newaxis]
        blended = (
            mix_expanded * trial_population +
            (1.0 - mix_expanded) * population[:n_trials]
        )

        # Per-dimension inheritance probability based on historical success
        inherit_prob = np.clip(success_rates, 0.2, 0.8)

        # Generate crossover mask using success-rate guided probabilities
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < inherit_prob[np.newaxis, :]

        # Ensure at least one dimension from trial is taken
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        # Create offspring from blended population
        offspring = np.where(cross_mask, blended, population[:n_trials])

        # Update success tracking based on trial performance
        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_improved = np.mean(trial_better, axis=0)
        self.dim_cx_total += 1
        alpha = 0.1
        self.dim_cx_success = alpha * dim_improved * n_trials + (1 - alpha) * self.dim_cx_success

        return offspring
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _crossover_block(self, population, trial_population):
        """Op 3: Success-guided adaptive crossover."""
        n_trials = self.NP
        dim = self.dim

        # Initialize per-dimension success tracking
        if not hasattr(self, 'dim_improve_count'):
            self.dim_improve_count = np.zeros(dim)
            self.dim_attempt_count = np.zeros(dim)
            self.prev_trial_pop = np.zeros((n_trials, dim))

        # Track which dimensions led to improvements (comparing consecutive trials)
        if self.prev_trial_pop.shape[0] == n_trials:
            improved_mask = trial_population < self.prev_trial_pop
            self.dim_improve_count += np.sum(improved_mask, axis=0)
            self.dim_attempt_count += n_trials

        self.prev_trial_pop = trial_population.copy()

        # Compute per-dimension success rates (avoid division by zero)
        dim_success_rate = np.zeros(dim)
        valid_dims = self.dim_attempt_count > 10
        dim_success_rate[valid_dims] = (
            self.dim_improve_count[valid_dims] / self.dim_attempt_count[valid_dims]
        )

        # Adaptive CR per individual based on population diversity
        pop_std = np.std(population, axis=0)
        max_spread = self.upper - self.lower
        diversity_factor = np.clip(np.mean(pop_std) / (max_spread + 1e-10), 0.05, 0.95)

        # Base CR from diversity, modulated by dimension success
        base_cr = 0.3 + 0.4 * diversity_factor

        # Per-dimension CR: higher for historically successful dimensions
        dim_weights = np.clip(dim_success_rate, 0.1, 0.9)
        dim_weights_norm = dim_weights / (np.sum(dim_weights) + 1e-10)

        # Generate crossover masks
        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for i in range(n_trials):
            # Sample number of dimensions to cross from distribution
            n_cross = max(1, int(np.clip(
                base_cr * dim * (0.5 + 0.5 * self.rng.random()),
                1, dim
            )))

            # Weighted selection of dimensions (prefer historically successful ones)
            probs = dim_weights_norm.copy()
            selected_dims = set()
            attempts = 0
            while len(selected_dims) < n_cross and attempts < n_cross * 3:
                d = self.rng.choice(dim, p=probs)
                selected_dims.add(d)
                attempts += 1
            selected_dims = list(selected_dims)

            # Also ensure at least one random dimension gets crossed
            if len(selected_dims) == 0:
                selected_dims = [self.rng.integers(0, dim)]

            cross_mask[i, selected_dims] = True

        # Ensure at least one dimension crosses for each individual
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        # Apply crossover with weighted blending for successful dimensions
        blend_weights = 0.7 + 0.3 * dim_weights
        blended_trial = (
            blend_weights[np.newaxis, :] * trial_population +
            (1 - blend_weights[np.newaxis, :]) * population[:n_trials]
        )

        offspring = np.where(cross_mask, blended_trial, population[:n_trials])
        return offspring
```

# --- From variant_06_idea_0.py (4 wins) ---
```python
def _crossover_block(self, population, trial_population):
        """Op 6: Confidence-weighted parent preservation crossover."""
        n_trials = self.NP
        dim = self.dim

        # Initialize per-dimension tracking if needed
        if not hasattr(self, '_dim_success_count') or self._dim_success_count is None or len(self._dim_success_count) != dim:
            self._dim_success_count = np.zeros(dim)
            self._dim_attempt_count = np.zeros(dim)
            self._dim_history_fitness = np.full(dim, np.inf)

        # Track which dimensions improved in trial vectors (global, not per-individual)
        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_signal = np.mean(trial_better, axis=0)

        # Update per-dimension statistics
        self._dim_attempt_count += 1
        update_mask = self._dim_attempt_count > 0
        alpha = 0.1
        self._dim_success_count[update_mask] = (
            alpha * dim_signal[update_mask] * float(n_trials) +
            (1 - alpha) * self._dim_success_count[update_mask]
        )

        # Compute confidence: higher success rate = higher confidence to preserve parent
        success_rate = np.zeros(dim)
        valid = self._dim_attempt_count > 5
        success_rate[valid] = self._dim_success_count[valid] / self._dim_attempt_count[valid]

        # Normalize confidence to probability of preserving parent
        conf_preserve = np.clip(success_rate, 0.05, 0.95)

        # For each trial, decide per-dimension whether to keep parent or take trial
        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for i in range(n_trials):
            # Dimensions with high confidence: preserve parent
            # Dimensions with low confidence: take trial vector
            preserve_prob = conf_preserve + 0.2 * (self.CR[i] - 0.5)
            preserve_prob = np.clip(preserve_prob, 0.0, 1.0)

            rand_vals = self.rng.uniform(size=dim)
            cross_mask[i] = rand_vals >= preserve_prob

        # Ensure at least one dimension takes from trial (diversity guarantee)
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        # Create offspring
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring
```

# --- From variant_08_idea_0.py (3 wins) ---
```python
def _crossover_block(self, population, trial_population):
        """Op 5: variant_11 - Ultra-conservative crossover for sensitive tasks."""
        n_trials = self.NP
        dim = self.dim

        # Ultra-low CR to preserve parent information on sensitive landscapes
        cr_batch = np.clip(self.CR.copy() * 0.15, 0.02, 0.08)

        # Single-dimension crossover mask
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]

        # Ensure at least one dimension from trial (or fall back to parent-preserving)
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                # No crossover: preserve parent entirely
                cross_mask[i, j_rand[i]] = True

        # Blend: small weight to trial vector to minimize disruption
        blend_weight = 0.1
        blended = (
            (1.0 - blend_weight) * population[:n_trials] +
            blend_weight * trial_population
        )

        # Apply crossover: mostly parent, minimally perturbed trial
        offspring = np.where(cross_mask, blended, population[:n_trials])

        # Clip to bounds
        offspring = np.clip(offspring, self.lower, self.upper)
        return offspring
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _crossover_block(self, population, trial_population):
        """Op 3: Coordinate-wise elite-preserving conservative crossover."""
        n_trials = self.NP
        dim = self.dim

        # Initialize per-individual elite memory (best seen values)
        if not hasattr(self, 'cx_elite_memory') or self.cx_elite_memory.shape != (n_trials, dim):
            self.cx_elite_memory = population[:n_trials].copy()

        # Determine global diversity to set exploration level
        pop_centroid = np.mean(population, axis=0)
        avg_distance = np.mean(np.linalg.norm(population - pop_centroid, axis=1))
        max_spread = self.upper - self.lower
        diversity_ratio = avg_distance / (max_spread + 1e-10)

        # Conservative exploration: reduce when diversity is low, increase when high
        # But always stay conservative (max 0.4 even at high diversity)
        cr_base = np.clip(0.15 + 0.25 * diversity_ratio, 0.1, 0.4)
        cr_batch = np.clip(cr_base + self.rng.uniform(-0.05, 0.05, size=n_trials), 0.05, 0.5)

        # Compute trial vs parent fitness gain per individual
        # (We don't have trial fitness yet, so use distance as proxy for risk)
        parent_trial_dist = np.linalg.norm(
            trial_population - population[:n_trials], axis=1, keepdims=True
        )
        dist_threshold = 0.3 * dim * (self.upper - self.lower) / (n_trials ** 0.5 + 1e-10)

        # Distance-based risk: large jumps are risky
        jump_risk = np.clip(parent_trial_dist / (dist_threshold + 1e-10), 0.0, 1.0)

        # Only accept trial values when:
        # 1. We're not making a large jump (low risk)
        # 2. OR the coordinate is currently at a bad value (exploration needed)
        # Use current CR as base acceptance probability, reduced by jump risk
        effective_cr = cr_batch[:, np.newaxis] * (1.0 - 0.5 * jump_risk)

        # Update elite memory: track best values seen
        better_mask = trial_population < self.cx_elite_memory
        self.cx_elite_memory = np.where(better_mask, trial_population, self.cx_elite_memory)

        # Create crossover mask: coordinate-wise decisions
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < effective_cr

        # For coordinates where parent is close to its personal best, be extra conservative
        coord_dist_from_elite = np.abs(population[:n_trials] - self.cx_elite_memory)
        max_coord_dist = (self.upper - self.lower) * 0.1
        stable_mask = coord_dist_from_elite < max_coord_dist
        cross_mask = cross_mask & ~stable_mask

        # Always force at least one crossover per individual
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        # Blend: use a small alpha to prevent drastic changes
        alpha = 0.2  # Very conservative blending
        blended_trial = (
            alpha * trial_population +
            (1.0 - alpha) * population[:n_trials]
        )

        # Final offspring selection
        offspring = np.where(cross_mask, blended_trial, population[:n_trials])

        # Clamp and return
        offspring = np.clip(offspring, self.lower, self.upper)
        return offspring
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
import numpy as np


class ReflectiveDEWithStrategyPortfolio:
    """
    Differential Evolution using reflective boundary handling and an adaptive
    portfolio of mutation strategies. Each individual maintains its own
    strategy label and per-strategy success memories that drive strategy
    selection and parameter adaptation.
    
    This version uses adaptive operator selection (sliding window + Thompson Sampling)
    to choose among multiple crossover strategies during optimization.
    """

    def __init__(self, dim, np_factor=6, seed=None):
        self.dim = dim
        self.NP = np_factor * dim
        self.NP = max(20, min(self.NP, 300))
        self.lower = -100.0
        self.upper = 100.0
        self.rng = np.random.default_rng(seed)

        # Strategy definitions (name, n_differences, uses_best)
        self.strategy_names = [
            "rand_1",
            "best_1",
            "current_1",
            "rand_2",
            "best_2",
        ]
        self.n_strategies = len(self.strategy_names)

        # Per-individual state
        self.F = np.empty(self.NP)
        self.CR = np.empty(self.NP)
        self.strategy_idx = np.zeros(self.NP, dtype=int)

        # Per-strategy success memory for adaptation
        self.strategy_success_fitness = np.zeros(self.n_strategies)
        self.strategy_success_count = np.zeros(self.n_strategies)
        self.strategy_attempt_count = np.zeros(self.n_strategies)

        # Global memory for F/CR adaptation (jADE-style)
        self.archive = []
        self.archive_max_size = self.NP

        # Stagnation detection
        self.generation_without_improvement = 0
        self.last_best_fitness = np.inf

        # Diversity tracking
        self.diversity_history = []

        # --- Adaptive crossover operator selection ---
        # We have 5 crossover operators indexed 0..4:
        # 0: original (binomial)
        # 1: variant_02 (dim-success adaptive CR)
        # 2: variant_03 (diversity-adaptive blending)
        # 3: variant_05 (block crossover) -- dominant winner
        # 4: variant_09 (binomial/exponential hybrid)
        self.n_crossover_ops = 5
        # Thompson Sampling parameters (Beta distribution)
        self.cx_alpha = np.ones(self.n_crossover_ops)  # successes + 1
        self.cx_beta = np.ones(self.n_crossover_ops)   # failures + 1
        # Sliding window for credit assignment
        self.cx_window_size = 50
        self.cx_history = []  # list of (op_idx, reward)
        # Current operator index
        self.current_cx_op = 3  # start with the best overall (variant_05)
        # Per-operator tracking for dim-success (variant_02)
        self.dim_success_rate = None
        self.dim_attempt_count = None
        # Track improvement rate for variant_06-style temp
        self.last_improvement_rate = 0.5

    def _initialize_population(self):
        lo = self.lower
        hi = self.upper
        pop = lo + (hi - lo) * self.rng.uniform(size=(self.NP, self.dim))
        self.F[:] = 0.5
        self.CR[:] = 0.5
        self.strategy_idx[:] = self.rng.integers(0, self.n_strategies, size=self.NP)
        self.strategy_success_fitness[:] = 0.0
        self.strategy_success_count[:] = 1
        self.strategy_attempt_count[:] = 1
        self.archive = []
        # Reset crossover adaptation
        self.cx_alpha = np.ones(self.n_crossover_ops)
        self.cx_beta = np.ones(self.n_crossover_ops)
        self.cx_history = []
        self.dim_success_rate = np.zeros(self.dim)
        self.dim_attempt_count = np.zeros(self.dim)
        return pop

    def _evaluate_batch(self, population, func):
        fitness = func(population)
        if len(fitness) < len(population):
            return fitness[:len(fitness)]
        return fitness

    def _clip_reflective(self, population):
        pop = population.copy()
        mask_lo = pop < self.lower
        mask_hi = pop > self.upper
        pop[mask_lo] = 2.0 * self.lower - pop[mask_lo]
        pop[mask_hi] = 2.0 * self.upper - pop[mask_hi]
        pop = np.clip(pop, self.lower, self.upper)
        return pop

    def _select_three_distinct(self, exclude_idx):
        pool = np.arange(self.NP)
        pool = pool[pool != exclude_idx]
        selected = self.rng.choice(pool, size=3, replace=False)
        return selected[0], selected[1], selected[2]

    def _select_five_distinct(self, exclude_idx):
        pool = np.arange(self.NP)
        pool = pool[pool != exclude_idx]
        selected = self.rng.choice(pool, size=5, replace=False)
        return selected[0], selected[1], selected[2], selected[3], selected[4]

    def _mutate_single_reflective(self, idx, population, fitness, trial_pop, trial_fit):
        x_i = population[idx]
        s_idx = self.strategy_idx[idx]
        strategy = self.strategy_names[s_idx]

        a, b, c = self._select_three_distinct(idx)
        x_a, x_b, x_c = population[a], population[b], population[c]
        f_scale = self.F[idx]

        if strategy == "rand_1":
            mutant = x_a + f_scale * (x_b - x_c)
        elif strategy == "best_1":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            mutant = x_best + f_scale * (x_a - x_b)
        elif strategy == "current_1":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            mutant = x_i + f_scale * (x_best - x_i) + f_scale * (x_a - x_b)
        elif strategy == "rand_2":
            d, e = self._select_three_distinct(idx)[:2]
            x_d, x_e = population[d], population[e]
            mutant = x_a + f_scale * (x_b - x_c) + f_scale * (x_d - x_e)
        elif strategy == "best_2":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            d, e = self._select_three_distinct(idx)[:2]
            x_d, x_e = population[d], population[e]
            mutant = x_best + f_scale * (x_a - x_b) + f_scale * (x_d - x_e)
        else:
            mutant = x_a + f_scale * (x_b - x_c)

        return self._clip_reflective(mutant.reshape(1, -1))[0]

    def _mutate_batch(self, population, fitness):
        n_trials = self.NP
        trial_population = np.empty((n_trials, self.dim))

        for idx in range(n_trials):
            trial_population[idx] = self._mutate_single_reflective(
                idx, population, fitness, trial_population, None
            )
        return trial_population

    # --- Crossover operator implementations ---

    def _crossover_original(self, population, trial_population):
        """Op 0: Standard binomial crossover."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_dim_adaptive(self, population, trial_population):
        """Op 1: variant_02 - Per-dimension success adaptive CR."""
        n_trials = self.NP
        dim = self.dim
        if self.dim_success_rate is None or len(self.dim_success_rate) != dim:
            self.dim_success_rate = np.zeros(dim)
            self.dim_attempt_count = np.zeros(dim)

        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_signal = np.mean(trial_better, axis=0)
        self.dim_attempt_count += 1
        alpha = 0.1
        self.dim_success_rate = alpha * dim_signal + (1 - alpha) * self.dim_success_rate

        dim_probs = np.clip(self.dim_success_rate, 0.01, 0.99)
        dim_probs_norm = dim_probs / (np.sum(dim_probs) + 1e-10)
        cr_base = 0.3 + 0.4 * np.mean(dim_probs_norm)
        cr_batch = np.full(n_trials, np.clip(cr_base, 0.0, 1.0))

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_diversity_blend(self, population, trial_population):
        """Op 2: variant_03 - Diversity-adaptive blending crossover."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        max_spread = self.upper - self.lower
        pop_spread = np.std(population, axis=0)
        normalized_diversity = np.mean(pop_spread / (max_spread + 1e-10))

        alpha_base = np.clip(0.5 - normalized_diversity, 0.1, 0.5)
        alpha_arr = np.clip(
            alpha_base + self.rng.uniform(-0.05, 0.05, size=n_trials),
            0.05, 0.6
        )

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True

        blended = (
            alpha_arr[:, np.newaxis] * trial_population +
            (1.0 - alpha_arr[:, np.newaxis]) * population[:n_trials]
        )

        offspring = np.where(cross_mask, blended, population[:n_trials])
        return offspring

    def _crossover_block(self, population, trial_population):
        """Op 3: variant_05 - Block crossover (dominant winner)."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        block_size = max(1, dim // 8)
        n_blocks = (dim + block_size - 1) // block_size

        cross_blocks = self.rng.uniform(size=(n_trials, n_blocks)) < cr_batch[:, np.newaxis]

        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, dim)
            cross_mask[:, start:end] = cross_blocks[:, b:b+1]

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_hybrid_binexp(self, population, trial_population):
        """Op 4: variant_09 - Binomial/exponential hybrid crossover."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        cross_mask = np.zeros((n_trials, dim), dtype=bool)

        for i in range(n_trials):
            if self.rng.random() < 0.5:
                # Binomial crossover
                mask = self.rng.uniform(size=dim) < cr_batch[i]
                j_rand = self.rng.integers(0, dim)
                mask[j_rand] = True
            else:
                # Exponential crossover
                j_start = self.rng.integers(0, dim)
                mask = np.zeros(dim, dtype=bool)
                L = 0
                while L < dim and self.rng.random() < cr_batch[i]:
                    j = (j_start + L) % dim
                    mask[j] = True
                    L += 1
                if not np.any(mask):
                    mask[j_start] = True
            cross_mask[i] = mask

        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _select_crossover_operator(self):
        """Thompson Sampling to select crossover operator."""
        samples = np.array([
            self.rng.beta(self.cx_alpha[i], self.cx_beta[i])
            for i in range(self.n_crossover_ops)
        ])
        return int(np.argmax(samples))

    def _update_crossover_reward(self, op_idx, n_improved, n_total, fitness_improvement):
        """Update Thompson Sampling parameters based on observed reward."""
        op_idx = int(op_idx)
        # Reward is the fraction of improved individuals + bonus for fitness improvement
        if n_total > 0:
            success_rate = float(n_improved) / float(n_total)
        else:
            success_rate = 0.0
        
        # Normalize fitness improvement to [0, 1] range
        fit_reward = float(np.clip(fitness_improvement, 0.0, 1.0))
        
        # Combined reward
        reward = 0.7 * success_rate + 0.3 * fit_reward
        reward = float(np.clip(reward, 0.0, 1.0))
        
        # Store in sliding window
        self.cx_history.append((op_idx, reward))
        if len(self.cx_history) > self.cx_window_size:
            self.cx_history.pop(0)
        
        # Recompute alpha/beta from sliding window
        self.cx_alpha = np.ones(self.n_crossover_ops)
        self.cx_beta = np.ones(self.n_crossover_ops)
        for (oidx, rew) in self.cx_history:
            oidx = int(oidx)
            self.cx_alpha[oidx] += float(rew)
            self.cx_beta[oidx] += float(1.0 - rew)

    def _crossover_batch(self, population, trial_population):
        """Adaptive crossover: selects among 5 operators using Thompson Sampling."""
        self.current_cx_op = self._select_crossover_operator()
        
        try:
            if self.current_cx_op == 0:
                result = self._crossover_original(population, trial_population)
            elif self.current_cx_op == 1:
                result = self._crossover_dim_adaptive(population, trial_population)
            elif self.current_cx_op == 2:
                result = self._crossover_diversity_blend(population, trial_population)
            elif self.current_cx_op == 3:
                result = self._crossover_block(population, trial_population)
            elif self.current_cx_op == 4:
                result = self._crossover_hybrid_binexp(population, trial_population)
            else:
                result = self._crossover_block(population, trial_population)
        except Exception:
            # Fallback to block crossover (best overall)
            self.current_cx_op = 3
            result = self._crossover_block(population, trial_population)
        
        # Safety: clip and handle NaN
        result = np.nan_to_num(result, nan=0.0, posinf=self.upper, neginf=self.lower)
        result = np.clip(result, self.lower, self.upper)
        return result

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved = trial_fitness < fitness[:len(trial_fitness)]
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[:len(trials)][improved] = trials[improved]
        new_fitness[:len(trials)][improved] = trial_fitness[improved]
        return new_population, new_fitness, improved

    def _update_strategy_success(self, improved_mask):
        for s in range(self.n_strategies):
            count_s = np.sum((self.strategy_idx == s) & improved_mask)
            self.strategy_success_count[s] += count_s
            self.strategy_attempt_count[s] += np.sum(self.strategy_idx == s)
            if count_s > 0:
                self.strategy_success_fitness[s] = (
                    0.5 * self.strategy_success_fitness[s] + 0.5 * count_s
                )

    def _adapt_strategy_probabilities(self):
        success_rates = np.zeros(self.n_strategies)
        valid_strategies = self.strategy_attempt_count > 0
        success_rates[valid_strategies] = (
            self.strategy_success_count[valid_strategies] / 
            self.strategy_attempt_count[valid_strategies]
        )

        best_fitness = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.min(self.archive) if self.archive else 0.0

        ref_fitness = getattr(self, 'reference_fitness', 1e-3)
        threshold = ref_fitness * 1.5 if ref_fitness > 0 else 1e-4

        if best_fitness < threshold:
            orig_prob = 1.0 / self.n_strategies
            adaptive_prob = success_rates / (np.sum(success_rates) + 1e-10)
            alpha = 0.85
            probs = alpha * orig_prob + (1 - alpha) * adaptive_prob
        else:
            temperature = 0.3
            exp_rates = np.exp(success_rates / (temperature + 1e-10))
            probs = exp_rates / (np.sum(exp_rates) + 1e-10)

        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (np.sum(probs) + 1e-10)

        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_F_CR_jade_style(self, population, fitness, trials, trial_fitness, improved_mask):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved_trials = trials[improved_mask]
        improved_parents = population[:len(trials)][improved_mask]

        if len(improved_trials) > 0:
            fi_values = np.abs(improved_trials - improved_parents)
            fi_values = fi_values / (np.linalg.norm(fi_values, axis=1, keepdims=True) + 1e-10)
            self.archive.extend(improved_trials.tolist())
            if len(self.archive) > self.archive_max_size:
                remove_n = len(self.archive) - self.archive_max_size
                self.archive = self.archive[remove_n:]

        for i in range(self.NP):
            if improved_mask[i]:
                self.F[i] = self.rng.uniform(0.1, 0.9)
            else:
                self.F[i] = 0.9 * self.F[i] + 0.1 * 0.5

            if improved_mask[i]:
                self.CR[i] = self.rng.beta(1.0, 0.5)
            else:
                self.CR[i] = 0.9 * self.CR[i] + 0.1 * 0.5
            self.CR[i] = np.clip(self.CR[i], 0.0, 1.0)

    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return float(np.mean(distances))

    def _check_stagnation(self, fitness):
        current_best = float(np.min(fitness))
        if current_best < self.last_best_fitness - 1e-10:
            self.generation_without_improvement = 0
            self.last_best_fitness = current_best
            return False
        else:
            self.generation_without_improvement += 1
            return self.generation_without_improvement > 50

    def _reinitialize_stagnant_subpopulation(self, population, fitness):
        n_replace = max(5, self.NP // 10)
        replace_idx = self.rng.choice(self.NP, size=n_replace, replace=False)
        lo, hi = self.lower, self.upper
        new_individuals = lo + (hi - lo) * self.rng.uniform(size=(n_replace, self.dim))
        population[replace_idx] = new_individuals
        fitness[replace_idx] = np.inf
        self.generation_without_improvement = 0
        return population, fitness

    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        population = self._clip_reflective(population)
        fitness = self._evaluate_batch(population, func)

        if len(fitness) == 0:
            return float(np.inf), np.zeros(self.dim)

        self.population_fitness = fitness.copy()

        best_idx = int(np.argmin(fitness))
        f_opt = float(fitness[best_idx])
        x_opt = population[best_idx].copy()

        while not stopping_condition():
            trials = self._mutate_batch(population, fitness)
            
            # Store pre-crossover state for reward computation
            pre_best = float(np.min(fitness))
            
            trials = self._crossover_batch(population, trials)
            trials = self._clip_reflective(trials)
            trial_fitness = self._evaluate_batch(trials, func)

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]

            if len(trial_fitness) == 0:
                break

            if stopping_condition():
                # Still update best before breaking
                for i in range(len(trial_fitness)):
                    if trial_fitness[i] < f_opt:
                        f_opt = float(trial_fitness[i])
                        x_opt = trials[i].copy()
                break

            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            self.population_fitness = fitness.copy()

            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < f_opt:
                f_opt = float(fitness[current_best_idx])
                x_opt = population[current_best_idx].copy()

            # Update crossover operator reward
            n_improved = int(np.sum(improved_mask))
            n_total = len(improved_mask)
            post_best = float(np.min(fitness))
            
            # Fitness improvement signal (relative)
            if abs(pre_best) > 1e-30:
                fit_improvement = float(max(0.0, (pre_best - post_best) / (abs(pre_best) + 1e-30)))
            else:
                fit_improvement = float(max(0.0, pre_best - post_best))
            fit_improvement = float(np.clip(fit_improvement, 0.0, 1.0))
            
            # Track improvement rate for variant_06 style
            self.last_improvement_rate = float(n_improved) / float(max(n_total, 1))
            
            self._update_crossover_reward(
                self.current_cx_op, n_improved, n_total, fit_improvement
            )

            self._update_strategy_success(improved_mask)
            self._adapt_strategy_probabilities()
            self._adapt_F_CR_jade_style(population, fitness, trials, trial_fitness, improved_mask)

            div = self._compute_diversity(population)
            self.diversity_history.append(div)

            if self._check_stagnation(fitness):
                population, fitness = self._reinitialize_stagnant_subpopulation(population, fitness)
                self.population_fitness = fitness.copy()
                self.archive = []
                self.F[:] = 0.5
                self.CR[:] = 0.5

        return f_opt, x_opt

```