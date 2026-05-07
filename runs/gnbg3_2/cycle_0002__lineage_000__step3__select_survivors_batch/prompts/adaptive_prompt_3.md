Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_select_survivors_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.148441e-07            1.054599e-07            3.261666e-06            4.191318e+01            1.285100e+01            2.197803e-07            2.270567e-01            1.524012e+00            nan                     nan                     
1      3.760932e+00            2.020029e+00            1.945994e+00            8.258721e+01            4.601916e+01            9.450920e-01            6.880247e+00            1.751784e+01            1.294764e+02            nan                     
2      7.425561e-05            4.888388e-05            8.764126e-02            2.357514e+02            1.257859e+02            1.966562e-03            3.742386e-01            1.837029e+01            nan                     nan                     
3      2.769512e+00            2.748174e+00            7.342023e+00            2.047955e+02            1.244708e+02            3.131049e+00            3.690043e+00            6.926784e+01            1.885537e+02            nan                     
4      9.949669e-01            8.394468e-01            1.445349e+00            4.711940e+00            3.535131e+00            1.126300e+00            nan                     2.392038e+00            nan                     nan                     
5      2.702739e+01            2.604877e+01            6.576858e+02            2.246101e+08            4.397511e+07            2.372299e+01            3.938043e+02            1.680896e+06            nan                     nan                     
6      1.258695e+02            3.681777e+02            6.747795e+02            2.279008e+03            1.163017e+03            3.715149e+01            6.676300e+02            9.014338e+02            1.906643e+03            nan                     
7      5.661689e+00            6.535947e+00            8.936841e+00            2.293517e+02            1.389749e+02            4.755174e+00            3.178290e+00            4.466418e+01            nan                     nan                     
8      2.364168e+00            1.805747e+00            2.763904e+00            3.731582e+01            2.679606e+01            1.861033e+00            3.394214e+00            1.494291e+01            3.949562e+01            nan                     
9      4.033458e+01            4.385684e+01            4.227840e+01            1.595096e+02            1.241975e+02            2.212173e+01            5.925302e+01            1.082884e+02            nan                     nan                     
10     1.263143e+00            1.249134e+00            7.556082e+00            2.700813e+02            1.338593e+02            1.677033e+02            1.052292e+01            1.902024e+02            nan                     nan                     
11     3.546620e+02            4.057737e+02            4.178103e+02            1.131651e+03            1.203207e+03            7.624137e+02            6.246874e+02            1.141621e+03            nan                     nan                     
12     2.525282e+00            2.617421e+00            9.227668e+00            2.102716e+02            1.544788e+02            1.296471e+02            1.637095e+01            1.505196e+02            nan                     nan                     
13     3.141164e+01            2.991983e+01            3.060785e+01            4.945484e+01            3.986167e+01            3.907137e+01            3.751897e+01            4.954196e+01            3.589911e+01            nan                     
14     7.261424e+00            6.935853e+00            7.148490e+00            1.568325e+01            1.534728e+01            1.167636e+01            3.914172e+00            1.557102e+01            6.234638e+00            nan                     
15     3.614938e+00            3.637790e+00            3.600417e+00            4.170568e+00            4.233187e+00            4.034358e+00            nan                     4.114949e+00            3.924353e+00            nan                     
16     4.857326e+03            7.286114e+03            7.045414e+03            1.271740e+04            8.751564e+03            8.784588e+03            8.779350e+03            1.405124e+04            nan                     nan                     
17     6.409501e+04            8.205852e+04            8.498014e+04            3.228671e+05            2.443450e+05            2.523217e+05            2.055685e+05            2.740631e+05            nan                     nan                     
18     5.790753e+01            5.777520e+01            5.922628e+01            9.621567e+01            8.895986e+01            7.434270e+01            7.795145e+01            9.232106e+01            8.236587e+01            nan                     
19     1.462088e+02            1.143359e+02            1.386624e+02            3.562938e+02            3.400940e+02            2.766973e+02            1.015093e+02            4.279128e+02            1.669844e+02            nan                     
20     2.043047e+01            2.008039e+01            2.018356e+01            5.726737e+01            4.198096e+01            4.029573e+01            2.962243e+01            5.148135e+01            4.316005e+01            nan                     
21     5.707348e+00            5.692552e+00            5.752914e+00            6.022506e+00            5.991663e+00            5.986396e+00            nan                     5.998327e+00            5.909841e+00            nan                     
22     1.334288e+01            1.331859e+01            1.368918e+01            2.039182e+01            2.112953e+01            1.692293e+01            1.066085e+01            2.027211e+01            nan                     nan                     
23     4.747748e+01            5.138967e+01            5.152073e+01            9.106447e+01            8.092691e+01            5.699087e+01            5.788362e+01            1.123396e+02            nan                     nan                     

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_catA_idea_0.py  (error=1.054599e-07)
Task  1: variant_05_catE_idea_0.py  (error=9.450920e-01)
Task  2: variant_01_catA_idea_0.py  (error=4.888388e-05)
Task  3: variant_01_catA_idea_0.py  (error=2.748174e+00)
Task  4: variant_01_catA_idea_0.py  (error=8.394468e-01)
Task  5: variant_05_catE_idea_0.py  (error=2.372299e+01)
Task  6: variant_05_catE_idea_0.py  (error=3.715149e+01)
Task  7: variant_06_catF_idea_0.py  (error=3.178290e+00)
Task  8: variant_01_catA_idea_0.py  (error=1.805747e+00)
Task  9: variant_05_catE_idea_0.py  (error=2.212173e+01)
Task 10: variant_01_catA_idea_0.py  (error=1.249134e+00)
Task 11: original.py  (error=3.546620e+02)
Task 12: original.py  (error=2.525282e+00)
Task 13: variant_01_catA_idea_0.py  (error=2.991983e+01)
Task 14: variant_06_catF_idea_0.py  (error=3.914172e+00)
Task 15: variant_02_catB_idea_0.py  (error=3.600417e+00)
Task 16: original.py  (error=4.857326e+03)
Task 17: original.py  (error=6.409501e+04)
Task 18: variant_01_catA_idea_0.py  (error=5.777520e+01)
Task 19: variant_06_catF_idea_0.py  (error=1.015093e+02)
Task 20: variant_01_catA_idea_0.py  (error=2.008039e+01)
Task 21: variant_01_catA_idea_0.py  (error=5.692552e+00)
Task 22: variant_06_catF_idea_0.py  (error=1.066085e+01)
Task 23: original.py  (error=4.747748e+01)

WIN COUNTS:
  variant_01_catA_idea_0.py: 10 wins
  original.py: 5 wins
  variant_05_catE_idea_0.py: 4 wins
  variant_06_catF_idea_0.py: 4 wins
  variant_02_catB_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (10 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with geometric diversity maintenance via pairwise distances."""
        improved_mask = trial_fitness < fitness

        new_population = population.copy()
        new_fitness = fitness.copy()

        # Accept fitness improvements unconditionally
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]

        # For non-improving trials, apply geometric diversity selection
        not_improved_mask = ~improved_mask
        if np.any(not_improved_mask):
            # Compute average pairwise distances within population
            # This establishes the characteristic geometric scale
            all_pairwise = np.linalg.norm(
                population[:, np.newaxis, :] - population[np.newaxis, :, :],
                axis=2
            )
            np.fill_diagonal(all_pairwise, np.inf)
            avg_dist_to_others = np.mean(all_pairwise, axis=1)

            # For non-improving trials, compute distance to corresponding population members
            trial_to_member_dist = np.linalg.norm(
                trials[not_improved_mask] - population[not_improved_mask],
                axis=1
            )

            # Accept non-improving trials that maintain better geometric separation
            # Accept if: distance(trial, member) > avg distance(member, others)
            geometric_acceptance = trial_to_member_dist > avg_dist_to_others[not_improved_mask]

            indices_to_update = np.where(not_improved_mask)[0][geometric_acceptance]
            new_population[indices_to_update] = trials[indices_to_update]
            new_fitness[indices_to_update] = trial_fitness[indices_to_update]

        return new_population, new_fitness, improved_mask
```

# --- From variant_02_catB_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Spectral selection: use SVD condition number to preserve exploration subspace."""
        improved_mask = trial_fitness < fitness

        new_population = population.copy()
        new_fitness = fitness.copy()

        # Apply greedy selection first
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]

        # Compute condition number via SVD of population matrix
        centered = new_population - np.mean(new_population, axis=0)

        try:
            # SVD gives us the effective dimensionality of the search distribution
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)

            # Condition number: ratio of largest to smallest singular value
            # High condition number = population collapsed to a subspace
            eps = 1e-12
            cond_number = s[0] / max(s[-1], eps)

            # Threshold for detecting subspace collapse (ill-conditioning)
            cond_threshold = 50.0

            if cond_number > cond_threshold and len(trials) > 0:
                # Identify minor singular directions (where population has little variance)
                minor_threshold_idx = max(0, len(s) // 3)  # Bottom third of singular values
                minor_directions = Vt[minor_threshold_idx:]  # Shape: (k, dim), k = num minor dirs

                if minor_directions.shape[0] > 0:
                    # Project all trial vectors onto minor singular directions
                    trial_projections = trials @ minor_directions.T  # (np, k)
                    trial_minor_energy = np.sum(trial_projections ** 2, axis=1)  # (np,)

                    # Also compute current population projections for comparison
                    current_projections = new_population @ minor_directions.T
                    current_minor_energy = np.sum(current_projections ** 2, axis=1)

                    # Find trials that add MORE minor-direction energy than their replaced counterparts
                    improvement_in_minor = trial_minor_energy - current_minor_energy

                    # Select up to 2 candidates that maximally restore minor-direction spread
                    # Only consider those that were NOT already selected (to add new diversity)
                    not_improved = ~improved_mask
                    if np.any(not_improved):
                        candidates = np.where(not_improved)[0]
                        candidate_improvement = improvement_in_minor[candidates]

                        # Select top candidates that expand the minor subspace
                        n_select = min(2, len(candidates))
                        top_indices = candidates[np.argsort(candidate_improvement)[-n_select:]]

                        # Only replace if they genuinely expand the subspace
                        for idx in top_indices:
                            if improvement_in_minor[idx] > 0.01 * np.mean(s):
                                new_population[idx] = trials[idx]
                                new_fitness[idx] = trial_fitness[idx]

        except Exception:
            # Fallback: just return greedy selection on any numerical issue
            pass

        return new_population, new_fitness, improved_mask
```

# --- From variant_05_catE_idea_0.py (4 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Graph-connectivity-aware survivor selection using k-NN topology."""
        new_population = population.copy()
        new_fitness = fitness.copy()

        pop_size = len(population)
        combined = np.vstack([population, trials])

        # Build k-NN graph on combined population to assess structural connectivity
        k = min(5, pop_size - 1)

        # Compute pairwise Euclidean distances for the combined set
        # Vectorized for speed: broadcast to (2N, 2N, D), then reduce
        diffs = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]  # (2N, 2N, D)
        dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))  # (2N, 2N)

        # For each individual, compute mean distance to its k-nearest neighbors
        # (k+1 because the individual itself has distance 0)
        sorted_indices = np.argpartition(dist_matrix, k + 1, axis=1)
        knn_connectivity = np.zeros(2 * pop_size)
        for i in range(2 * pop_size):
            neighbor_dists = dist_matrix[i, sorted_indices[i, :k + 1]]
            knn_connectivity[i] = np.mean(neighbor_dists)

        pop_conn = knn_connectivity[:pop_size]           # connectivity of current population
        trial_conn = knn_connectivity[pop_size:]         # connectivity of trial individuals

        improved_mask = trial_fitness < fitness

        # --- PHASE 1: Clear fitness improvements ---
        # Always accept trials that strictly improve fitness
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]

        # --- PHASE 2: Connectivity-preserving replacement for near-ties ---
        # When trial ≈ parent in fitness, prefer the one that maintains better
        # graph structure (lower knn_connectivity = denser local neighborhood)
        not_improved = ~improved_mask
        if np.any(not_improved):
            fitness_diffs = fitness - trial_fitness
            max_diff = np.max(fitness_diffs[not_improved]) + 1e-10

            # "Near-tie" threshold: within 20% of the worst-case gap
            near_tie_threshold = 0.2 * max_diff
            near_tie_mask = not_improved & (fitness_diffs < near_tie_threshold)

            if np.any(near_tie_mask):
                idx_nt = np.where(near_tie_mask)[0]

                # Compute connectivity advantage for each near-tie pair
                conn_advantage = pop_conn[idx_nt] - trial_conn[idx_nt]  # positive = trial is better

                # Accept trial if it improves connectivity by at least 5%
                connectivity_threshold = 0.05 * pop_conn[idx_nt]
                accept_connectivity = conn_advantage > connectivity_threshold

                accept_idx = idx_nt[accept_connectivity]
                new_population[accept_idx] = trials[accept_idx]
                new_fitness[accept_idx] = trial_fitness[accept_idx]

                # Mark accepted indices as improved for adaptation tracking
                improved_mask[accept_idx] = True

        return new_population, new_fitness, improved_mask
```

# --- From variant_06_catF_idea_0.py (4 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Temporal-aware selection using EMA of fitness and rate-of-improvement tracking."""

        # Initialize temporal state on first call
        if not hasattr(self, 'fitness_ema'):
            self.fitness_ema = fitness.copy()
            self.fitness_ema_alpha = 0.3
            self.prev_fitness = fitness.copy()
            self.improvement_streak = np.zeros(len(fitness))
            self.generation_age = np.zeros(len(fitness))

        # Track generation age (survival time in population)
        self.generation_age += 1

        # Compute rate of change (velocity) of current fitness
        fitness_velocity = fitness - self.prev_fitness  # Negative means improving

        # Adaptive EMA alpha: consistently improving individuals get more responsive EMA
        improvement_boost = np.clip(self.improvement_streak / 10.0, 0, 0.2)
        adaptive_alpha = self.fitness_ema_alpha - improvement_boost
        adaptive_alpha = np.clip(adaptive_alpha, 0.1, 0.5)

        # Update EMA of current fitness (per-individual)
        new_ema = np.zeros_like(fitness)
        for i in range(len(fitness)):
            new_ema[i] = adaptive_alpha[i] * fitness[i] + (1 - adaptive_alpha[i]) * self.fitness_ema[i]

        # Evaluate trial against EMA (smoothed reference) rather than raw fitness
        ema_improved_mask = trial_fitness < self.fitness_ema

        # Compute projected EMA for trials
        trial_ema = np.array([adaptive_alpha[i] * trial_fitness[i] + 
                              (1 - adaptive_alpha[i]) * self.fitness_ema[i] 
                              for i in range(len(trial_fitness))])

        # Velocity bonus: reward individuals showing consistent improvement trend
        # Negative velocity = fitness decreasing = improving
        velocity_bonus = np.zeros(len(fitness))
        improving_mask = fitness_velocity < 0
        velocity_bonus[improving_mask] = 0.1 * np.abs(fitness_velocity[improving_mask])
        velocity_bonus = np.clip(velocity_bonus, 0, 0.5)

        # Age penalty: reduce selection pressure for older individuals to maintain diversity
        age_penalty = np.clip(self.generation_age / 50.0, 0, 0.3)

        # Combined selection: EMA comparison with velocity bonus and age penalty
        combined_score_current = self.fitness_ema - velocity_bonus + age_penalty
        combined_score_trial = trial_ema

        temporal_select_mask = combined_score_trial < combined_score_current

        # Fallback: also accept if raw fitness significantly improves (ensures no regression)
        raw_improved_mask = trial_fitness < fitness

        # Combined decision: accept if temporal trend improves OR raw fitness strictly improves
        accept_mask = temporal_select_mask | raw_improved_mask

        # Build new population
        new_population = population.copy()
        new_fitness = fitness.copy()

        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]

        # Update temporal state for accepted individuals
        self.fitness_ema[accept_mask] = trial_ema[accept_mask]
        self.improvement_streak[accept_mask] += 1
        self.improvement_streak[~accept_mask] = np.maximum(0, self.improvement_streak[~accept_mask] - 1)
        self.generation_age[accept_mask] = 0  # Reset age on acceptance

        # Update previous fitness for next generation
        self.prev_fitness = fitness.copy()

        return new_population, new_fitness, accept_mask
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