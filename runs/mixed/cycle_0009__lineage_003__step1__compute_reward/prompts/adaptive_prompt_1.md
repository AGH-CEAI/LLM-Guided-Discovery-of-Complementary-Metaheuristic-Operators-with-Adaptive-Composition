Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_compute_reward` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            
1      2.941785e-01            2.828998e-01            2.967831e-01            -inf                    3.078954e-01            3.242510e-01            3.107648e-01            2.929355e-01            3.123893e-01            -inf                    2.910515e-01            
2      1.000000e-08            1.021689e+01            1.281306e-08            -inf                    1.087192e-08            1.045620e-08            2.225165e-08            1.000000e-08            1.021814e-08            -inf                    1.000000e-08            
3      1.703349e-08            2.187139e-08            1.468767e-08            -inf                    2.761960e-08            2.006430e-08            9.597050e-08            1.051414e-08            1.028834e-08            -inf                    1.000000e-08            
4      1.204317e+00            1.101689e+00            1.252182e+00            -inf                    1.343051e+00            1.198444e+00            1.248525e+00            1.170562e+00            1.157831e+00            -inf                    1.226938e+00            
5      1.501900e+00            1.295882e+00            1.385913e+00            -inf                    1.439344e+00            1.459059e+00            1.455809e+00            1.430282e+00            1.383674e+00            -inf                    1.346869e+00            
6      1.000000e-08            9.374310e-08            1.179020e-08            -inf                    1.000000e-08            1.000000e-08            2.300297e-08            1.245750e-08            1.139500e-08            -inf                    1.076540e-08            
7      1.000000e-08            2.402554e-06            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            3.769253e-08            1.695900e-08            1.000000e-08            -inf                    1.000000e-08            
8      3.310473e+02            2.895201e+02            1.554537e+02            -inf                    2.527107e+02            2.734519e+02            3.371635e+02            5.737179e+02            3.437839e+02            -inf                    1.299027e+02            
9      1.897525e-08            5.568361e-07            1.265943e-08            -inf                    1.000000e-08            1.115748e-08            1.231844e-07            2.765891e-08            1.764384e-08            -inf                    2.384375e-08            
10     1.024218e-07            9.222515e-05            4.596390e-08            -inf                    1.588282e-08            8.170489e-08            4.570318e-07            8.080086e-08            9.027586e-08            -inf                    1.282646e-07            
11     1.069938e-07            1.078561e-05            5.801250e-08            -inf                    1.953474e-08            2.382231e-08            4.431365e-07            9.895465e-08            1.000000e-08            -inf                    1.402698e-07            
12     7.820426e+02            6.557185e+02            1.025388e+03            -inf                    1.209552e+03            1.087264e+03            1.253874e+03            8.353453e+02            1.560364e+03            -inf                    5.597792e+02            
13     5.305453e+00            6.165634e+01            1.638204e+01            -inf                    1.157933e+01            3.752186e+01            1.702915e+01            3.683611e+00            9.224911e+00            -inf                    6.769114e+00            
14     5.503642e+00            5.288946e+00            5.227158e+00            -inf                    5.302182e+00            5.416323e+00            5.590710e+00            5.564536e+00            5.555929e+00            -inf                    5.460213e+00            
15     2.000000e-08            5.296000e+02            4.000000e-08            -inf                    5.537009e+02            4.000000e-08            1.333333e-08            5.899305e+02            5.759675e+02            -inf                    5.759675e+02            
16     6.002686e+02            6.571420e+02            6.025624e+02            -inf                    5.991351e+02            6.041790e+02            5.883932e+02            6.003256e+02            6.001851e+02            -inf                    5.696677e+02            
17     2.788688e-08            5.339485e+02            5.438415e-08            -inf                    2.000000e-08            1.333333e-08            5.296000e+02            5.638249e-07            1.136610e-06            -inf                    2.000000e-08            
18     5.759675e+02            6.211892e-06            3.114140e-08            -inf                    5.712419e+02            5.296000e+02            4.897851e-06            1.418323e-05            8.634073e-05            -inf                    1.060763e-05            
19     1.560077e-03            1.425840e-03            2.211467e-03            -inf                    1.814647e-03            1.262274e-03            3.257843e-03            1.942592e-03            1.718688e-03            -inf                    1.755650e-03            
20     5.000000e+00            5.001162e+00            5.000009e+00            -inf                    5.000002e+00            5.000000e+00            5.000191e+00            5.000018e+00            5.000000e+00            -inf                    5.000000e+00            
21     5.000011e+01            5.003487e+01            5.000252e+01            -inf                    5.000041e+01            5.001118e+01            5.002223e+01            5.000012e+01            5.000036e+01            -inf                    5.000041e+01            
22     1.904850e-05            3.088471e-05            1.633454e-05            -inf                    1.684516e-05            1.721766e-05            4.948732e-05            1.501512e-04            2.765975e-05            -inf                    1.449680e-05            
23     1.173479e+01            2.744325e+01            2.308197e+01            -inf                    2.301610e+01            2.531917e+01            2.418858e+01            1.650208e+01            1.197024e+01            -inf                    5.590039e+00            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_01_idea_0.py  (error=2.828998e-01)
Task  2: SKIPPED (trivial)
Task  3: SKIPPED (trivial)
Task  4: variant_01_idea_0.py  (error=1.101689e+00)
Task  5: variant_01_idea_0.py  (error=1.295882e+00)
Task  6: SKIPPED (trivial)
Task  7: SKIPPED (trivial)
Task  8: variant_10_idea_0.py  (error=1.299027e+02)
Task  9: SKIPPED (trivial)
Task 10: SKIPPED (trivial)
Task 11: SKIPPED (trivial)
Task 12: variant_10_idea_0.py  (error=5.597792e+02)
Task 13: variant_07_idea_0.py  (error=3.683611e+00)
Task 14: variant_02_idea_0.py  (error=5.227158e+00)
Task 15: variant_06_idea_0.py  (error=1.333333e-08)
Task 16: variant_10_idea_0.py  (error=5.696677e+02)
Task 17: SKIPPED (trivial)
Task 18: variant_02_idea_0.py  (error=3.114140e-08)
Task 19: variant_05_idea_0.py  (error=1.262274e-03)
Task 20: variant_08_idea_0.py  (error=5.000000e+00)
Task 21: original.py  (error=5.000011e+01)
Task 22: variant_10_idea_0.py  (error=1.449680e-05)
Task 23: variant_10_idea_0.py  (error=5.590039e+00)

WIN COUNTS:
  variant_10_idea_0.py: 5 wins
  variant_01_idea_0.py: 3 wins
  variant_02_idea_0.py: 2 wins
  variant_07_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins
  original.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute rank-based reward with success rate and relative improvement"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        n_improved = int(np.sum(improvements > 0))
        n_total = len(fitness)

        if n_total == 0:
            return 0.0

        # Success rate component (primary signal)
        success_rate = n_improved / float(n_total)

        # Relative improvement: normalize by current best fitness for scale invariance
        current_best = float(np.min(fitness))
        if abs(current_best) > 1e-10 and np.isfinite(current_best):
            # Relative improvement of best individual
            best_idx = int(np.argmin(fitness))
            rel_improvement = improvements[best_idx] / (abs(current_best) + 1.0)
        else:
            rel_improvement = 0.0

        # Rank-based component: use ordinal position change
        # Lower rank = better fitness
        rank_fitness = np.argsort(np.argsort(fitness))
        rank_trial = np.argsort(np.argsort(trial_fitness))
        rank_improvement = np.mean(rank_fitness - rank_trial) / float(n_total)

        # Combine components: success rate is primary, augmented by relative and rank signals
        reward = 0.6 * success_rate + 0.25 * np.clip(rel_improvement, -1.0, 1.0) + 0.15 * np.clip(rank_improvement, -1.0, 1.0)

        # Penalize complete failure more strongly
        if n_improved == 0:
            reward = -0.1

        return float(np.clip(reward, -1.0, 1.0))
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
            """Compute reward based on relative improvement quality (log-scaled)"""
            # Guard against invalid values
            fitness = np.clip(fitness, -1e50, 1e50)
            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

            improvements = fitness - trial_fitness
            n_improved = int(np.sum(improvements > 0))
            total_improved = float(n_improved)

            if n_improved == 0:
                return -0.1

            # Compute relative improvement (fractional gain)
            relative_imp = improvements[improvements > 0] / (np.abs(fitness[improvements > 0]) + 1e-10)

            # Log-scaled reward: small gains on hard problems valued similarly to large gains on easy ones
            log_rewards = np.log1p(relative_imp)
            avg_log_reward = float(np.mean(log_rewards))

            # Consistency bonus: reward operators that improve many individuals
            consistency_ratio = total_improved / max(float(len(fitness)), 1.0)
            consistency_bonus = 0.15 * consistency_ratio

            # Combined reward: log-scale captures difficulty-normalized progress
            reward = avg_log_reward + consistency_bonus

            return float(np.clip(reward, -0.5, 1.0))
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
        """Log-scaled consistency reward: penalize stagnation, reward per-individual improvement"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness

        # Compute per-individual log-scaled improvement rewards
        reward_components = []

        for i in range(len(improvements)):
            imp = improvements[i]
            if imp > 1e-10:
                # Log-scaled reward: compress huge improvements, boost consistent small ones
                # Using log(1 + imp) gives diminishing returns for large imp
                # Adding small baseline ensures tiny improvements still register
                log_reward = np.log1p(imp) / 20.0
                reward_components.append(log_reward)
            elif imp < -1e-10:
                # Penalize degradation (but not as harshly as -0.01 flat penalty)
                reward_components.append(-0.05 * np.log1p(-imp) / 20.0)
            else:
                # No change: small penalty to encourage seeking improvements
                reward_components.append(-0.005)

        if len(reward_components) == 0:
            return -0.1

        # Return mean of per-individual rewards (rewards consistency over total magnitude)
        reward = float(np.mean(reward_components))

        return float(np.clip(reward, -1.0, 1.0))
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on best individual improvement + rank-based robustness"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        # Track improvement of best individual (primary signal for escaping traps)
        best_f = float(np.min(fitness))
        best_trial_f = float(np.min(trial_fitness))
        best_improvement = best_f - best_trial_f

        # Rank-based component: more robust to outliers than raw values
        fitness_sorted_idx = np.argsort(fitness)
        trial_sorted_idx = np.argsort(trial_fitness)
        best_rank = float(np.argmin(fitness_sorted_idx == np.argmin(fitness)))
        trial_best_rank = float(np.argmin(trial_sorted_idx == np.argmin(trial_fitness)))

        # Normalize ranks to [0, 1] range
        max_rank = max(float(len(fitness) - 1), 1.0)
        rank_component = 0.3 * (1.0 - (trial_best_rank / max_rank))

        # Stagnation detection: if best didn't improve, apply strong penalty
        if best_improvement <= 0:
            # Stagnation penalty (more aggressive to break out of traps)
            reward = -0.15
        else:
            # Progressive reward based on best improvement magnitude
            # Scale by population size for dimension-invariance
            scale_factor = max(float(len(fitness)), 1.0)
            improvement_component = 0.7 * min(float(best_improvement) / (scale_factor + 1e-10), 1.0)
            reward = improvement_component + rank_component

        return float(np.clip(reward, -1.0, 1.0))
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
            """Compute reward based on logarithmic improvement toward target (1e-08)"""
            # Guard against invalid values
            fitness = np.clip(fitness, -1e50, 1e50)
            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

            improvements = fitness - trial_fitness
            n_improved = int(np.sum(improvements > 0))

            if n_improved == 0:
                return -0.01

            # Log-scale reward: reward is proportional to decades closed toward target
            # This prioritizes hard tasks (error ~1e+2) over easy tasks (error ~1e-6)
            target = 1e-8
            log_fitness = np.log10(np.maximum(np.abs(fitness), target) + 1e-100)
            log_trial = np.log10(np.maximum(np.abs(trial_fitness), target) + 1e-100)

            # Log improvement: positive when closing distance to target
            log_improvements = log_trial - log_fitness
            log_improvements[improvements <= 0] = 0.0

            # Sum of decades closed, weighted by population fraction
            total_log_improvement = float(np.sum(log_improvements))
            n_dims = float(len(fitness))

            # Scale: ~2 decades of improvement → reward = 1.0
            reward = total_log_improvement / (2.0 * n_dims / max(n_improved, 1))

            return float(np.clip(reward, -1.0, 1.0))
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward with stagnation-aware bimodal strategy"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        # Compute best fitness improvement rate
        best_before = float(np.min(fitness))
        best_after = float(np.min(trial_fitness))
        best_delta = best_before - best_after  # positive = improvement

        # Compute diversity of trial solutions (exploration signal)
        # Use mean distance of trials from population centroid
        pop_centroid = np.mean(fitness)  # weighted by fitness similarity
        trial_centroid = np.mean(trial_fitness)
        # Diversity = spread of trial fitness values
        trial_spread = float(np.std(trial_fitness[np.isfinite(trial_fitness)])) if np.any(np.isfinite(trial_fitness)) else 0.0
        pop_spread = float(np.std(fitness[np.isfinite(fitness)])) if np.any(np.isfinite(fitness)) else 1.0
        diversity_ratio = trial_spread / (pop_spread + 1e-10)

        # Compute population-level diversity (geometric diversity of solutions)
        finite_trials = trial_fitness[np.isfinite(trial_fitness)]
        if len(finite_trials) > 1:
            diversity_score = float(np.std(finite_trials)) / (float(np.abs(np.mean(finite_trials))) + 1e-10)
        else:
            diversity_score = 0.0

        # Detect stagnation: best solution not improving
        is_stagnant = (best_delta < 1e-6 * (abs(best_before) + 1.0)) or (best_delta <= 0)

        if is_stagnant:
            # Bimodal: when stagnant, reward EXPLORATION
            # Higher diversity = better at escaping local optima
            reward = float(np.clip(0.3 * diversity_ratio + 0.7 * diversity_score, -0.5, 1.0))
        else:
            # Bimodal: when improving, reward EXPLOITATION
            if n_improved == 0:
                reward = -0.05
            else:
                # Normalize by population spread to reward bold improvements
                norm_improvement = total_improvement / max(float(n_improved), 1.0)
                pop_scale = pop_spread + abs(best_before) * 0.1 + 1.0
                reward = float(np.clip(norm_improvement / pop_scale, -0.1, 1.0))

        return float(np.clip(reward, -1.0, 1.0))
```

# --- From variant_10_idea_0.py (5 wins) ---
```python
def _compute_reward(self, fitness, trial_fitness):
            """Compute reward based on relative improvement, prioritizing hard unsolved tasks"""
            # Guard against invalid values
            fitness = np.clip(fitness, -1e50, 1e50)
            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

            improvements = fitness - trial_fitness
            n_improved = int(np.sum(improvements > 0))

            if n_improved == 0:
                # Penalty scales with error magnitude - don't give up on hard tasks
                current_error = float(np.mean(np.abs(fitness)))
                penalty = -0.01 * (1.0 + np.log1p(max(current_error, 1.0)))
                return float(np.clip(penalty, -1.0, 0.0))

            total_improvement = float(np.sum(improvements[improvements > 0]))
            current_error = float(np.mean(np.abs(trial_fitness)))

            # Logarithmic relative improvement for large errors (hard unsolved tasks)
            # This gives higher rewards for proportional progress on Tasks 12, 16, 8, etc.
            if current_error > 1e-02:
                # Log scale: 10x improvement gets ~2.3 reward, 100x gets ~4.6
                reward = np.log1p(total_improvement) / (np.log1p(current_error) + 1e-10)
                reward = float(np.clip(reward, -0.5, 2.0))
            else:
                # Linear scale near target: reward fine-grained progress on Tasks 3, 15, 18
                reward = total_improvement / max(current_error, 1e-15)
                reward = float(np.clip(reward, -0.5, 3.0))

            return float(np.clip(reward, -1.0, 1.0))
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


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation strategy works best during optimization.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(kwargs.get('NP', 8 * dim), 300)
        self.min_pop_size = 20
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        # Adaptive parameters
        self.F = 0.7
        self.CR = 0.5
        self.p_best_rate = 0.1

        # Cultural memory
        self.cultural_memory_size = min(30, self.NP)
        self.cultural_memory = np.zeros((self.cultural_memory_size, dim))
        self.cultural_weights = np.ones(self.cultural_memory_size)
        self.culture_idx = 0

        # Local search
        self.local_search_interval = kwargs.get('local_search_interval', 7)
        self.local_radius = kwargs.get('local_radius', 2.0)
        self.local_max_iter = 10

        # Diversity and stagnation
        self.stagnation_count = 0
        self.stagnation_limit = kwargs.get('stagnation_limit', 60)
        self.diversity_threshold = 0.1
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        self.improvement_buffer = []

        # === Adaptive Operator Selection (Thompson Sampling) ===
        self.num_operators = 4
        self.operator_names = ['original', 'variant_01', 'variant_07', 'variant_09']
        # Beta distribution parameters for Thompson Sampling
        self.operator_alpha = np.ones(self.num_operators)
        self.operator_beta = np.ones(self.num_operators)
        # Sliding window reward tracking
        self.reward_window_size = 30
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        self.current_operator_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []

        # Archive for variant_01 (JADE-style)
        self.archive = None
        self._pending_archive_additions = None

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on improvement quality"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            reward = -0.01
        else:
            # Normalize by population size and number of improvements
            reward = total_improvement / max(float(n_improved), 1.0)

        return float(np.clip(reward, -1.0, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(reward)

        # Maintain sliding window
        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        # Update Beta distribution parameters based on recent performance
        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = np.mean(recent)

            # Update alpha (successes + 1) and beta (failures + 1)
            self.operator_alpha[operator_idx] = 1.0 + success_rate * 5.0 + avg_reward * 2.0
            self.operator_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0

            # Ensure valid parameters
            self.operator_alpha[operator_idx] = float(np.clip(self.operator_alpha[operator_idx], 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(self.operator_beta[operator_idx], 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        # Guard against invalid initial fitness
        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self._select_operator_thompson()

        while not stopping_condition():
            mutants = self._mutate_current_to_pbest_with_culture(population, fitness)
            mutants = np.clip(mutants, self.lower, self.upper)

            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            # Guard against invalid trial fitness
            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            # Compute reward before selection
            reward = self._compute_reward(fitness, trial_fitness)

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            fitness = np.clip(fitness, -1e50, 1e50)

            best_idx = np.argmin(fitness)
            current_best = float(fitness[best_idx])
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Update operator performance
            self._update_operator_rewards(self.current_operator_idx, reward)

            # Update archive if using variant_01
            if self.current_operator_idx == 1 and self.archive is not None and len(self.archive) > 0:
                self._update_archive(population, fitness, trials, trial_fitness)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            # Operator switching with minimum generations between switches
            if self.generation - self.last_operator_switch_gen >= self.operator_min_switch_gens:
                new_op = self._select_operator_thompson()
                if new_op != self.current_operator_idx:
                    self.current_operator_idx = new_op
                    self.last_operator_switch_gen = self.generation
                    self.operator_switches.append((self.generation, new_op))

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

    def _initialize_population_lhs(self):
        samples = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
        for d in range(self.dim):
            edges = np.linspace(self.lower[d], self.upper[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = edges[perm] + np.random.uniform(0, edges[1] - edges[0], self.NP)
        return self._clip_to_bounds_batch(samples)

    def _clip_to_bounds_batch(self, population):
        range_width = self.upper - self.lower
        clipped = np.clip(population, self.lower, self.upper)
        overflow = population - clipped
        reflected = clipped - overflow
        reflected = np.where(population > self.upper,
                            self.upper - ((population - self.upper) % (2 * range_width + 1e-10)),
                            reflected)
        reflected = np.where(population < self.lower,
                            self.lower + ((self.lower - population) % (2 * range_width + 1e-10)),
                            reflected)
        return np.clip(reflected, self.lower, self.upper)

    def _eval_wrapper_batch(self, population, func):
        result = func(population)
        if len(result) < len(population):
            result = np.pad(result, (0, len(population) - len(result)), constant_values=np.inf)
        return result

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
        """Dispatch to selected mutation strategy"""
        if self.current_operator_idx == 0:
            return self._mutate_original(population, fitness)
        elif self.current_operator_idx == 1:
            return self._mutate_variant01(population, fitness)
        elif self.current_operator_idx == 2:
            return self._mutate_variant07(population, fitness)
        else:
            return self._mutate_variant09(population, fitness)

    def _mutate_original(self, population, fitness):
        """Original mutation strategy: current-to-pbest/1 with cultural memory"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant01(self, population, fitness):
        """Variant 01: JADE mutation with archive and bounded random F"""
        # Initialize archive if needed
        if self.archive is None:
            self.archive = np.zeros((0, self.dim))

        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        p_best_indices = sorted_idx[:n_best]
        p_best_idx = p_best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        # Generate random indices r1, r2
        r1 = np.random.randint(0, self.NP, size=self.NP)
        r2 = np.zeros(self.NP, dtype=int)
        for i in range(self.NP):
            while True:
                r2[i] = np.random.randint(0, self.NP)
                if r2[i] != i and r2[i] != r1[i]:
                    break

        # JADE mutation with archive
        archive_size = len(self.archive)
        total_size = self.NP + archive_size

        if archive_size > 0:
            all_indices = np.concatenate([np.arange(self.NP), self.archive])
            r3_indices = np.random.randint(0, total_size, size=self.NP)
            r3 = np.where(r3_indices < self.NP, r3_indices, -(r3_indices - self.NP + 1))

            for i in range(self.NP):
                if r3[i] == i:
                    r3[i] = (i + 1) % self.NP
            r3_pop = np.where(r3 >= 0, population[r3], self.archive[-(r3 + 1)])
        else:
            r3 = np.random.randint(0, self.NP, size=self.NP)
            for i in range(self.NP):
                while r3[i] == i or r3[i] == r1[i]:
                    r3[i] = (r3[i] + 1) % self.NP
            r3_pop = population[r3]

        # Bounded random F per individual
        F_i = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)

        donors = population + F_i[:, np.newaxis] * (p_best - population) + F_i[:, np.newaxis] * (r3_pop - population[r1])
        return np.clip(donors, self.lower, self.upper)

    def _update_archive(self, population, fitness, trials, trial_fitness):
        """Update JADE archive with rejected solutions"""
        rejected = population[trial_fitness >= fitness]
        if len(rejected) > 0:
            max_archive = self.NP
            if len(self.archive) + len(rejected) > max_archive:
                indices = np.random.choice(len(rejected), max_archive - len(self.archive), replace=False)
                rejected = rejected[indices]
            self.archive = np.vstack([self.archive, rejected])[-max_archive:]

    def _mutate_variant07(self, population, fitness):
        """Variant 07: GLOBAL best, adaptive F, escalating perturbation"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        global_best_idx = best_indices[0]
        global_best = population[global_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        # Adaptive F when stagnant
        F_adapted = self.F
        if self.stagnation_count > 10:
            F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count - 10)), 2.0)

        base = population + F_adapted * (global_best - population) + F_adapted * (population[r1] - population[r2])

        # Escalating random perturbation when stagnant
        if self.stagnation_count > 5:
            perturbation_scale = min(0.3 * np.log1p(self.stagnation_count), 2.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            base = base + random_perturbation

        # Cultural memory blending
        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant09(self, population, fitness):
        """Variant 09: Multi-Donor Composite Mutation with Diversity-Weighted Selection"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        # Compute centroid of top 20% as exploration target
        top_percentile = max(1, self.NP // 5)
        top_indices = sorted_idx[:top_percentile]
        centroid = np.mean(population[top_indices], axis=0)

        # Generate THREE candidate donors per individual
        # Candidate 1: Best-directed (exploitation)
        donor_best = population + self.F * (p_best - population)
        # Candidate 2: Centroid-directed (diversification)
        donor_centroid = population + self.F * (centroid - population)
        # Candidate 3: Standard rand/diff (random exploration)
        donor_rand = population + self.F * (population[r1] - population[r2])

        # Diversity-weighted composite (no extra fitness evals needed)
        # Higher diversity contribution = more different from parent = better explorer
        dist_best = np.linalg.norm(donor_best - population, axis=1) + 1e-10
        dist_centroid = np.linalg.norm(donor_centroid - population, axis=1) + 1e-10
        dist_rand = np.linalg.norm(donor_rand - population, axis=1) + 1e-10

        # Normalize diversity scores
        total_dist = dist_best + dist_centroid + dist_rand + 1e-10
        w_best = dist_best / total_dist
        w_centroid = dist_centroid / total_dist
        w_rand = dist_rand / total_dist

        # Adaptive weight adjustment based on stagnation
        if self.stagnation_count > 10:
            # When stuck: boost centroid exploration, reduce best-pull
            w_best = np.clip(w_best * 0.5, 0.1, 0.6)
            w_centroid = np.clip(w_centroid * 1.5, 0.2, 0.6)
            w_rand = np.clip(w_rand * 1.2, 0.1, 0.4)
            # Renormalize
            total_w = w_best + w_centroid + w_rand
            w_best /= total_w
            w_centroid /= total_w
            w_rand /= total_w

        # Composite donors
        donors = (w_best[:, np.newaxis] * donor_best +
                  w_centroid[:, np.newaxis] * donor_centroid +
                  w_rand[:, np.newaxis] * donor_rand)

        return np.clip(donors, self.lower, self.upper)

    def _crossover_binomial_batch(self, population, donors):
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        trials = np.where(mask, donors, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        new_population = np.copy(population)
        new_population[better] = trials[better]
        new_fitness = np.copy(fitness)
        new_fitness[better] = trial_fitness[better]
        return new_population, new_fitness

    def _update_cultural_memory_on_improvement(self, best_individual, best_mutant):
        if self.stagnation_count == 0:
            pattern = best_mutant - best_individual
            self.cultural_memory[self.culture_idx] = pattern
            self.cultural_weights[self.culture_idx] = 1.0 / (1.0 + self.stagnation_count)
            self.culture_idx = (self.culture_idx + 1) % self.cultural_memory_size
            decay = 0.95
            self.cultural_weights *= decay

    def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        # Track weighted improvement rate for parameter adaptation
        if not hasattr(self, '_adapt_weighted_improvement'):
            self._adapt_weighted_improvement = 0.0
            self._adapt_fitness_spread = 1.0
            self._adapt_history_F = []
            self._adapt_history_CR = []

        # Compute directional improvement: is best fitness improving?
        current_best = float(np.min(fitness))
        if hasattr(self, '_prev_best_fitness') and np.isfinite(self._prev_best_fitness):
            direction = self._prev_best_fitness - current_best
            # Weighted moving average with exponential decay
            decay = 0.7
            self._adapt_weighted_improvement = decay * self._adapt_weighted_improvement + (1 - decay) * direction
        self._prev_best_fitness = current_best

        # Track fitness spread (convergence indicator)
        finite_fitness = fitness[np.isfinite(fitness)]
        if len(finite_fitness) > 1:
            spread = float(np.std(finite_fitness)) + 1e-10
            self._adapt_fitness_spread = 0.9 * self._adapt_fitness_spread + 0.1 * spread

        # Normalize improvement by spread to get adaptive signal
        norm_signal = self._adapt_weighted_improvement / (self._adapt_fitness_spread + 1e-10)

        # Adaptive F: exploit (lower F) when improving, explore (higher F) when stagnant
        if norm_signal > 0.1:
            # Making progress: focus on exploitation with finer steps
            F_base = 0.4 + 0.3 * np.random.rand()
            F_scale = 0.05
        elif norm_signal < -0.1:
            # Stagnant: increase exploration
            F_base = 0.7 + 0.4 * np.random.rand()
            F_scale = 0.15
        else:
            # Neutral: balanced approach
            F_base = 0.5 + 0.3 * np.random.rand()
            F_scale = 0.1

        F_candidate = F_base + np.random.randn() * F_scale
        new_F = float(np.clip(F_candidate, 0.1, 2.0))

        # Adaptive CR: higher CR when improving (combine more from mutant), lower when stagnant
        if norm_signal > 0.1:
            CR_base = 0.7 + 0.2 * np.random.rand()
        elif norm_signal < -0.1:
            CR_base = 0.3 + 0.2 * np.random.rand()
        else:
            CR_base = 0.5 + 0.2 * np.random.rand()

        CR_candidate = CR_base + np.random.randn() * 0.1
        new_CR = float(np.clip(CR_candidate, 0.05, 0.98))

        # Momentum: blend with history to reduce oscillation
        if len(self._adapt_history_F) > 0:
            momentum = 0.3
            new_F = momentum * np.mean(self._adapt_history_F[-5:]) + (1 - momentum) * new_F
            new_CR = momentum * np.mean(self._adapt_history_CR[-5:]) + (1 - momentum) * new_CR

        self.F = float(np.clip(new_F, 0.1, 2.0))
        self.CR = float(np.clip(new_CR, 0.05, 0.98))

        # Track history
        self._adapt_history_F.append(self.F)
        self._adapt_history_CR.append(self.CR)
        if len(self._adapt_history_F) > 20:
            self._adapt_history_F.pop(0)
            self._adapt_history_CR.pop(0)

        # Adapt p_best_rate based on stagnation
        if self.stagnation_count > 15:
            delta = np.random.uniform(0.02, 0.08)
            self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.05, 0.4))
        elif self.stagnation_count > 5:
            if np.random.rand() < 0.3:
                self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.03, 0.03), 0.05, 0.3))

    def _apply_adaptive_local_search(self, func):
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        for _ in range(self.local_max_iter):
            if radius < 1e-6:
                break

            directions = np.random.randn(self.dim)
            directions = directions / (np.linalg.norm(directions) + 1e-10)
            candidates = np.clip(center + radius * directions, self.lower, self.upper)
            candidates = np.vstack([candidates, np.clip(center - radius * directions, self.lower, self.upper)])
            candidates = np.vstack([candidates, center])

            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = np.argmin(fit_candidates)
            best_local_fitness = float(fit_candidates[best_local_idx])

            if best_local_fitness < current_fitness:
                center = candidates[best_local_idx].copy()
                current_fitness = best_local_fitness
                radius = min(radius * 1.5, 10.0)
            else:
                radius *= 0.4

        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = center.copy()
            self.local_radius = min(radius * 1.2, 10.0)
        else:
            self.local_radius = max(radius * 0.8, 0.1)

    def _compute_diversity_batch(self, population):
        centroids = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroids, axis=1)
        return float(np.mean(distances))

    def _check_diversity_low(self, population):
        diversity = self._compute_diversity_batch(population)
        return (diversity < self.diversity_threshold or
                self.stagnation_count > self.stagnation_limit)

    def _reinitialize_population(self, population, fitness):
        if self.best_solution is not None:
            best_idx = np.argmin(fitness)
            population[0] = self.best_solution.copy()
            fitness[0] = self.best_fitness

        n_random = max(self.min_pop_size, self.NP // 2)
        new_part = self._initialize_population_lhs()[:n_random]
        population[1:1 + n_random] = new_part
        fitness[1:1 + n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5

```