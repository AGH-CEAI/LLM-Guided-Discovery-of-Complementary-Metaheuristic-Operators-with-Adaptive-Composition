Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_adapt_parameters` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.464372e+01            -inf                    -inf                    6.531174e-06            -inf                    1.148441e-07            -inf                    8.034503e+01            6.800615e-06            -inf                    
1      1.063610e+01            -inf                    -inf                    3.851221e+00            -inf                    3.760932e+00            -inf                    1.601639e+02            2.951661e+00            -inf                    
2      1.377142e+02            -inf                    -inf                    6.770466e-02            -inf                    7.425561e-05            -inf                    6.201834e+02            1.525569e-02            -inf                    
3      3.628204e+01            -inf                    -inf                    1.445496e+01            -inf                    2.769512e+00            -inf                    2.993560e+02            1.532402e+01            -inf                    
4      2.969893e+00            -inf                    -inf                    1.193775e+00            -inf                    9.949669e-01            -inf                    5.609651e+00            1.147820e+00            -inf                    
5      1.896783e+05            -inf                    -inf                    4.592725e+03            -inf                    2.702739e+01            -inf                    3.023241e+09            1.997983e+03            -inf                    
6      5.951470e+02            -inf                    -inf                    6.423956e+02            -inf                    1.258695e+02            -inf                    8.346179e+03            5.744652e+02            -inf                    
7      3.513064e+01            -inf                    -inf                    1.037684e+01            -inf                    5.661689e+00            -inf                    3.958342e+02            1.039837e+01            -inf                    
8      1.065077e+01            -inf                    -inf                    3.258303e+00            -inf                    2.364168e+00            -inf                    5.104909e+01            3.326670e+00            -inf                    
9      4.974743e+01            -inf                    -inf                    4.606677e+01            -inf                    4.033458e+01            -inf                    3.526673e+02            5.866562e+01            -inf                    
10     2.367942e+01            -inf                    -inf                    6.317357e+00            -inf                    1.263143e+00            -inf                    2.921634e+02            5.840498e+00            -inf                    
11     6.824612e+02            -inf                    -inf                    3.793368e+02            -inf                    3.546620e+02            -inf                    1.679467e+03            3.628874e+02            -inf                    
12     2.248419e+01            -inf                    -inf                    7.114137e+00            -inf                    2.525282e+00            -inf                    2.040802e+02            7.332795e+00            -inf                    
13     2.785285e+01            -inf                    -inf                    3.072475e+01            -inf                    3.141164e+01            -inf                    5.608928e+01            3.260805e+01            -inf                    
14     6.197612e+00            -inf                    -inf                    3.746483e+00            -inf                    7.261424e+00            -inf                    1.886749e+01            3.968990e+00            -inf                    
15     3.902992e+00            -inf                    -inf                    3.546728e+00            -inf                    3.614938e+00            -inf                    4.369516e+00            3.525292e+00            -inf                    
16     4.036631e+03            -inf                    -inf                    4.356665e+03            -inf                    4.857326e+03            -inf                    2.464682e+04            6.422847e+03            -inf                    
17     1.768199e+05            -inf                    -inf                    7.098542e+04            -inf                    6.409501e+04            -inf                    2.945056e+05            7.948742e+04            -inf                    
18     7.583969e+01            -inf                    -inf                    5.572893e+01            -inf                    5.790753e+01            -inf                    9.213958e+01            5.819878e+01            -inf                    
19     1.655039e+02            -inf                    -inf                    1.210844e+02            -inf                    1.462088e+02            -inf                    5.023833e+02            1.551565e+02            -inf                    
20     2.018937e+01            -inf                    -inf                    2.098714e+01            -inf                    2.043047e+01            -inf                    5.675808e+01            2.157428e+01            -inf                    
21     5.887309e+00            -inf                    -inf                    5.766769e+00            -inf                    5.707348e+00            -inf                    5.921891e+00            5.718762e+00            -inf                    
22     1.720340e+01            -inf                    -inf                    1.283293e+01            -inf                    1.334288e+01            -inf                    2.248617e+01            1.244170e+01            -inf                    
23     3.758941e+01            -inf                    -inf                    4.373341e+01            -inf                    4.747748e+01            -inf                    1.355386e+02            4.111100e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_catF_idea_0.py  (error=1.148441e-07)
Task  1: variant_09_catA_idea_0.py  (error=2.951661e+00)
Task  2: variant_06_catF_idea_0.py  (error=7.425561e-05)
Task  3: variant_06_catF_idea_0.py  (error=2.769512e+00)
Task  4: variant_06_catF_idea_0.py  (error=9.949669e-01)
Task  5: variant_06_catF_idea_0.py  (error=2.702739e+01)
Task  6: variant_06_catF_idea_0.py  (error=1.258695e+02)
Task  7: variant_06_catF_idea_0.py  (error=5.661689e+00)
Task  8: variant_06_catF_idea_0.py  (error=2.364168e+00)
Task  9: variant_06_catF_idea_0.py  (error=4.033458e+01)
Task 10: variant_06_catF_idea_0.py  (error=1.263143e+00)
Task 11: variant_06_catF_idea_0.py  (error=3.546620e+02)
Task 12: variant_06_catF_idea_0.py  (error=2.525282e+00)
Task 13: original.py  (error=2.785285e+01)
Task 14: variant_03_catC_idea_0.py  (error=3.746483e+00)
Task 15: variant_09_catA_idea_0.py  (error=3.525292e+00)
Task 16: original.py  (error=4.036631e+03)
Task 17: variant_06_catF_idea_0.py  (error=6.409501e+04)
Task 18: variant_03_catC_idea_0.py  (error=5.572893e+01)
Task 19: variant_03_catC_idea_0.py  (error=1.210844e+02)
Task 20: original.py  (error=2.018937e+01)
Task 21: variant_06_catF_idea_0.py  (error=5.707348e+00)
Task 22: variant_09_catA_idea_0.py  (error=1.244170e+01)
Task 23: original.py  (error=3.758941e+01)

WIN COUNTS:
  variant_06_catF_idea_0.py: 14 wins
  original.py: 4 wins
  variant_09_catA_idea_0.py: 3 wins
  variant_03_catC_idea_0.py: 3 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_03_catC_idea_0.py (3 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using entropy of the fitness distribution.

        Category C: Information-theoretic / distributional.
        Treats fitness values as a probability distribution and uses Shannon entropy
        to detect exploration (high entropy, diverse fitness) vs. exploitation
        (low entropy, converged fitness) regimes. KL divergence from a reference
        distribution quantifies how far the current regime has shifted.
        """
        # Compute entropy of fitness distribution via kernel density estimation proxy
        fitness = np.array([f for f, _ in self.F_success_history[-self.adaptation_window:]] +
                           [np.mean([f for f, _ in self.F_success_history[-self.adaptation_window:]])])

        if len(fitness) < 2 or np.all(np.isinf(fitness)):
            return

        # Normalize to probability-like distribution using softmax (temperature = 1)
        fitness_min = np.nanmin(fitness)
        fitness_max = np.nanmax(fitness)
        if fitness_max - fitness_min < 1e-10:
            return

        # Discretize fitness range into bins for entropy estimation
        n_bins = min(10, max(3, len(fitness) // 2))
        bin_edges = np.linspace(fitness_min - 0.5 * (fitness_max - fitness_min),
                                fitness_max + 0.5 * (fitness_max - fitness_min),
                                n_bins + 1)
        bin_counts, _ = np.histogram(fitness, bins=bin_edges)
        bin_probs = bin_counts / (np.sum(bin_counts) + 1e-10)

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(np.where(bin_probs > 0, bin_probs * np.log(bin_probs + 1e-10), 0))
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / (max_entropy + 1e-10)

        # Initialize entropy history if needed
        if not hasattr(self, 'fitness_entropy_history'):
            self.fitness_entropy_history = []
        if not hasattr(self, 'last_F_for_entropy'):
            self.last_F_for_entropy = self.F

        self.fitness_entropy_history.append(normalized_entropy)
        if len(self.fitness_entropy_history) > self.adaptation_window:
            self.fitness_entropy_history.pop(0)

        if len(self.fitness_entropy_history) >= 3:
            recent_entropies = self.fitness_entropy_history[-3:]
            entropy_change = recent_entropies[-1] - recent_entropies[0]

            # Reference distribution: uniform over bins (maximum entropy)
            uniform_probs = np.ones(n_bins) / n_bins

            # KL divergence from uniform reference: D_KL(P || U)
            # High KL means distribution is concentrated (exploitation regime)
            kl_div = np.sum(np.where(uniform_probs > 0,
                                      uniform_probs * np.log(uniform_probs / (bin_probs + 1e-10),
                                                              uniform_probs),
                                      0))

            # Adaptive scaling based on entropy regime
            if recent_entropies[-1] > 0.5:
                # High entropy: exploration regime, increase F
                self.F = np.clip(self.F * (1.0 + 0.15 * entropy_change), 0.1, 1.5)
                self.Cr = np.clip(self.Cr * (1.0 - 0.05 * entropy_change), 0.1, 0.9)
            else:
                # Low entropy / low KL: exploitation regime, decrease F
                self.F = np.clip(self.F * (1.0 - 0.10 * abs(entropy_change)), 0.1, 1.5)
                self.Cr = np.clip(self.Cr * (1.0 + 0.08 * abs(entropy_change)), 0.1, 0.9)

            # Momentum term: smooth adaptation based on entropy trend
            momentum = 0.7
            target_F = self.F
            self.F = np.clip(momentum * self.last_F_for_entropy + (1 - momentum) * target_F, 0.1, 1.5)
            self.last_F_for_entropy = self.F
```

# --- From variant_06_catF_idea_0.py (14 wins) ---
```python
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
```

# --- From variant_09_catA_idea_0.py (3 wins) ---
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr based on geometric layout of population.

        Category A: Uses axis-aligned spread and k-NN structure instead of
        fitness-based success history to determine exploration vs exploitation.
        """
        pop = self.population if hasattr(self, 'population') else None
        if pop is None or len(pop) < 3:
            return

        # Compute axis-aligned spread per dimension (sum of ranges)
        dim_mins = np.min(pop, axis=0)
        dim_maxs = np.max(pop, axis=0)
        dim_ranges = dim_maxs - dim_mins
        total_spread = np.sum(dim_ranges)

        # Normalize by search space size for scale-invariance
        search_space_size = (self.upper - self.lower) * self.dim
        normalized_spread = total_spread / (search_space_size + 1e-30)

        # Compute k-NN distances for each individual (k=3)
        k = min(3, len(pop) - 1)
        knn_distances = np.zeros(len(pop))
        for i in range(len(pop)):
            dists = np.linalg.norm(pop - pop[i], axis=1)
            dists[i] = np.inf  # Exclude self
            sorted_dists = np.sort(dists)
            knn_distances[i] = np.mean(sorted_dists[:k])

        mean_knn = np.mean(knn_distances)
        knn_normalized = mean_knn / (search_space_size ** 0.5 + 1e-30)

        # Geometric adaptation rules:
        # Low spread = tightly clustered = need MORE exploration (higher F)
        # High spread = well-dispersed = need exploitation (lower F)
        # High k-NN distance = sparse local neighborhoods = higher Cr for mixing
        # Low k-NN distance = dense neighborhoods = lower Cr for local search

        if normalized_spread < 0.01:  # Very tight clustering
            self.F = np.clip(self.F * 1.15, 0.1, 2.0)
        elif normalized_spread > 0.5:  # Well dispersed
            self.F = np.clip(self.F * 0.9, 0.1, 2.0)

        if knn_normalized > 0.3:  # Sparse neighborhoods
            self.Cr = np.clip(self.Cr * 1.08, 0.1, 0.9)
        elif knn_normalized < 0.1:  # Dense neighborhoods
            self.Cr = np.clip(self.Cr * 0.92, 0.1, 0.9)
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
        """Adapt F and Cr based on recent success history."""
        success_rate = np.mean(improved_mask)
        
        self.F_success_history.append((success_rate, F_used))
        self.Cr_success_history.append((success_rate, Cr_used))
        
        if len(self.F_success_history) > self.adaptation_window:
            self.F_success_history.pop(0)
            self.Cr_success_history.pop(0)
        
        if len(self.F_success_history) >= 5:
            recent_success = [s for s, _ in self.F_success_history[-5:]]
            avg_success = np.mean(recent_success)
            
            if avg_success > 0.3:
                self.F = np.clip(self.F * 1.1, 0.1, 1.5)
                self.Cr = np.clip(self.Cr * 0.95, 0.1, 0.9)
            elif avg_success < 0.1:
                self.F = np.clip(self.F * 0.9, 0.1, 1.5)
                self.Cr = np.clip(self.Cr * 1.05, 0.1, 0.9)
    
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