Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_clip_to_bounds_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
1      2.993975e-01            3.012459e-01            3.086560e-01            2.987519e-01            2.906146e-01            3.156356e-01            2.916094e-01            2.948974e-01            2.927017e-01            2.923650e-01            -inf                    
2      1.000000e-08            1.000000e-08            1.000000e-08            1.034863e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
3      1.842024e-08            1.341243e-08            1.255016e-08            1.406981e-08            1.227100e-08            1.268734e-08            1.114666e-08            1.317354e-08            1.385192e-08            1.415737e-08            -inf                    
4      1.186997e+00            1.189112e+00            1.131531e+00            1.184569e+00            1.167064e+00            1.166310e+00            1.087253e+00            1.271182e+00            1.191347e+00            1.207701e+00            -inf                    
5      1.393678e+00            1.362810e+00            1.412482e+00            1.437663e+00            1.357725e+00            1.452087e+00            1.394513e+00            1.368333e+00            1.419255e+00            1.429822e+00            -inf                    
6      1.000000e-08            1.000000e-08            1.594030e-08            1.040126e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
7      1.000000e-08            1.000000e-08            1.047104e-08            3.117809e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.062118e-08            1.000000e-08            1.000000e-08            -inf                    
8      1.814578e+02            1.308395e+02            1.814524e+02            1.938554e+02            2.205301e+02            2.078318e+02            1.160176e+02            2.471092e+02            1.818906e+02            1.409755e+02            -inf                    
9      1.691951e-08            2.328877e-08            1.549210e-08            2.095032e-08            2.237078e-08            1.339111e-08            1.732601e-08            3.399330e-08            2.823951e-08            6.074920e-08            -inf                    
10     9.298431e-08            1.452542e-07            8.920823e-08            1.415558e-07            6.667736e-08            1.124807e-07            1.625773e-07            6.112256e-08            1.460676e-07            6.566468e-08            -inf                    
11     5.436087e-08            8.502628e-08            9.596979e-08            5.450063e-08            3.737931e-08            7.241505e-08            8.320203e-08            4.950728e-08            3.496610e-08            9.423531e-08            -inf                    
12     7.187390e+02            9.857480e+02            5.965569e+02            4.102919e+02            1.206747e+03            2.791471e+02            1.084369e+03            3.548259e+02            5.056759e+02            1.104739e+03            -inf                    
13     1.014243e+01            6.396333e+00            2.997651e+00            3.955159e+00            6.019431e+00            6.953742e+00            4.205156e+00            5.689597e+00            4.754760e+00            6.779866e+00            -inf                    
14     5.374886e+00            4.935246e+00            5.142607e+00            5.257774e+00            5.339594e+00            5.332461e+00            5.695335e+00            5.364613e+00            5.127371e+00            5.011072e+00            -inf                    
15     2.000000e-08            1.333333e-08            3.999962e-08            4.000000e-08            1.333333e-08            5.296000e+02            4.000000e-08            2.000000e-08            5.456251e+02            4.000000e-08            -inf                    
16     6.003248e+02            6.003486e+02            5.835098e+02            6.009145e+02            6.002953e+02            5.788043e+02            5.918933e+02            6.006821e+02            6.002452e+02            6.003017e+02            -inf                    
17     3.999974e-08            1.400314e-05            5.456251e+02            1.158259e-07            4.000000e-08            2.000000e-08            4.000000e-08            4.000000e-08            5.413824e+02            3.999892e-08            -inf                    
18     1.171192e-06            5.301609e+02            2.278251e-05            2.428690e-07            5.537009e+02            6.521976e-06            1.002773e-05            5.759675e+02            4.000000e-08            4.422283e-06            -inf                    
19     1.536927e-03            1.407321e-03            1.382452e-03            1.249081e-03            1.878030e-03            1.441953e-03            1.316195e-03            1.340354e-03            1.583958e-03            1.426309e-03            -inf                    
20     5.000000e+00            5.000000e+00            5.000001e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            -inf                    
21     5.000024e+01            5.000019e+01            5.000025e+01            5.000015e+01            5.000017e+01            5.000022e+01            5.000007e+01            5.000049e+01            5.000033e+01            5.000018e+01            -inf                    
22     1.338140e-05            1.404629e-05            1.011030e-05            1.689783e-05            2.260400e-05            2.827827e-05            1.544615e-05            2.723873e-05            1.527570e-05            1.406277e-05            -inf                    
23     1.434931e+01            1.265466e+01            1.174898e+01            6.335226e+00            1.297968e+01            9.776670e+00            1.241471e+01            1.206891e+01            8.960968e+00            1.373978e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_04_idea_0.py  (error=2.906146e-01)
Task  2: SKIPPED (trivial)
Task  3: SKIPPED (trivial)
Task  4: variant_06_idea_0.py  (error=1.087253e+00)
Task  5: variant_04_idea_0.py  (error=1.357725e+00)
Task  6: SKIPPED (trivial)
Task  7: SKIPPED (trivial)
Task  8: variant_06_idea_0.py  (error=1.160176e+02)
Task  9: SKIPPED (trivial)
Task 10: variant_07_idea_0.py  (error=6.112256e-08)
Task 11: SKIPPED (trivial)
Task 12: variant_05_idea_0.py  (error=2.791471e+02)
Task 13: variant_02_idea_0.py  (error=2.997651e+00)
Task 14: variant_01_idea_0.py  (error=4.935246e+00)
Task 15: SKIPPED (trivial)
Task 16: variant_05_idea_0.py  (error=5.788043e+02)
Task 17: SKIPPED (trivial)
Task 18: variant_08_idea_0.py  (error=4.000000e-08)
Task 19: variant_03_idea_0.py  (error=1.249081e-03)
Task 20: variant_07_idea_0.py  (error=5.000000e+00)
Task 21: variant_06_idea_0.py  (error=5.000007e+01)
Task 22: variant_02_idea_0.py  (error=1.011030e-05)
Task 23: variant_03_idea_0.py  (error=6.335226e+00)

WIN COUNTS:
  variant_06_idea_0.py: 3 wins
  variant_04_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_05_idea_0.py: 2 wins
  variant_02_idea_0.py: 2 wins
  variant_03_idea_0.py: 2 wins
  variant_01_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        result = np.copy(population)

        # Reflect values above upper bound
        above_upper = result > self.upper
        if np.any(above_upper):
            reflected = 2.0 * self.upper - result[above_upper]
            result[above_upper] = np.where(reflected >= self.lower, reflected,
                                            2.0 * self.lower - reflected)

        # Reflect values below lower bound
        below_lower = result < self.lower
        if np.any(below_lower):
            reflected = 2.0 * self.lower - result[below_lower]
            result[below_lower] = np.where(reflected <= self.upper, reflected,
                                            2.0 * self.upper - reflected)

        return result
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        lower = self.lower
        upper = self.upper

        # Guard against degenerate bounds
        if np.any(upper <= lower):
            return np.clip(population, lower, upper)

        result = population.copy()
        range_size = upper - lower

        # Determine boundary mode based on stagnation
        # Low stagnation: use reflection (maintain direction)
        # High stagnation: use wrap (escape boundary traps)
        if self.stagnation_count < 10:
            # Reflection mode: bounce off boundaries
            below = result < lower
            above = result > upper

            # Reflect below-bound values
            reflected_below = lower + (lower - result)
            reflected_below = np.clip(reflected_below, lower, upper)
            result = np.where(below, reflected_below, result)

            # Reflect above-bound values
            reflected_above = upper - (result - upper)
            reflected_above = np.clip(reflected_above, lower, upper)
            result = np.where(above, reflected_above, result)

            # Second-order reflection for extreme overshoots
            still_below = result < lower
            still_above = result > upper
            result = np.where(still_below, lower + (lower - result) * 0.5, result)
            result = np.where(still_above, upper - (result - upper) * 0.5, result)
        else:
            # Wrap mode: periodic boundary for escaping boundary traps
            # This helps when population is stuck at bounds during stagnation
            result = lower + (result - lower) % (range_size + 1e-10)

        # Final safety clip for any remaining edge cases
        result = np.clip(result, lower, upper)

        # Handle NaN/Inf if present
        invalid = ~np.isfinite(result)
        if np.any(invalid):
            result[invalid] = (lower + upper)[invalid] * 0.5

        return result
```

# --- From variant_03_idea_0.py (2 wins) ---
```python
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
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        reflected = np.copy(population)
        # Adaptive damping: less damping when stagnant (allow bigger jumps to escape)
        stagnation_factor = min(1.0 + 0.5 * self.stagnation_count, 2.5)
        damping = 1.0 / stagnation_factor
        # Upper bound: reflect with damped bounce
        upper_bounce = self.upper - (population - self.upper) * damping
        reflected = np.where(population > self.upper, upper_bounce, reflected)
        # Lower bound: reflect with damped bounce
        lower_bounce = self.lower + (self.lower - population) * damping
        reflected = np.where(population < self.lower, lower_bounce, reflected)
        # Safety clip for extreme outliers
        return np.clip(reflected, self.lower, self.upper)
```

# --- From variant_05_idea_0.py (2 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        reflected = np.copy(population)
        # Reflect values above upper bound
        above_upper = population > self.upper
        if np.any(above_upper):
            excess = population[above_upper] - self.upper
            reflected[above_upper] = self.upper - 0.7 * excess
        # Reflect values below lower bound
        below_lower = population < self.lower
        if np.any(below_lower):
            deficit = self.lower - population[below_lower]
            reflected[below_lower] = self.lower + 0.7 * deficit
        return reflected
```

# --- From variant_06_idea_0.py (3 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        # Reflection-based boundary handling: bounce back instead of hard clip
        # Preserves directional information for continued exploration
        result = np.copy(population)

        # Handle upper bound violations with reflection
        upper_mask = result > self.upper
        if np.any(upper_mask):
            overflow = result[upper_mask] - self.upper
            result[upper_mask] = self.upper - overflow

        # Handle lower bound violations with reflection
        lower_mask = result < self.lower
        if np.any(lower_mask):
            underflow = self.lower - result[lower_mask]
            result[lower_mask] = self.lower + underflow

        # Final safety clip for any remaining boundary violations
        return np.clip(result, self.lower, self.upper)
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        result = np.copy(population)

        # Compute how far each dimension exceeds bounds
        below_mask = result < self.lower
        above_mask = result > self.upper

        # Calculate distance beyond bounds
        below_dist = np.where(below_mask, self.lower - result, 0.0)
        above_dist = np.where(above_mask, result - self.upper, 0.0)

        # Adaptive boundary softness: increase during stagnation
        stagnation_factor = min(self.stagnation_count / 30.0, 1.0)
        generation_factor = min(self.generation / 100.0, 0.3)
        boundary_softness = 0.5 + 0.5 * stagnation_factor + generation_factor

        # Probabilistic reflection with adaptive probability
        for _ in range(3):  # Max 3 reflection bounces
            any_outside = np.any(below_mask) or np.any(above_mask)
            if not any_outside:
                break

            # Reflect below bounds
            reflect_below = below_mask & (np.random.rand(*result.shape) < boundary_softness)
            result[reflect_below] = 2 * self.lower[reflect_below] - result[reflect_below]

            # Reflect above bounds
            reflect_above = above_mask & (np.random.rand(*result.shape) < boundary_softness)
            result[reflect_above] = 2 * self.upper[reflect_above] - result[reflect_above]

            # Re-evaluate masks after reflection
            below_mask = result < self.lower
            above_mask = result > self.upper

        # Final hard clip as safety net
        result = np.clip(result, self.lower, self.upper)
        return result
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _clip_to_bounds_batch(self, population):
        clipped = np.clip(population, self.lower, self.upper)
        # Detect boundary hits and apply reflective perturbation
        hit_lower = population < self.lower
        hit_upper = population > self.upper
        if not (np.any(hit_lower) or np.any(hit_upper)):
            return clipped
        # Reflect with small random jitter to escape boundary traps
        jitter = np.random.randn(*population.shape) * 0.01 * (self.upper - self.lower)
        reflected = np.where(hit_lower, self.lower + (self.lower - population[hit_lower]) + jitter[hit_lower], clipped)
        reflected = np.where(hit_upper, self.upper - (population[hit_upper] - self.upper) + jitter[hit_upper], reflected)
        return np.clip(reflected, self.lower, self.upper)
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
        return np.clip(population, self.lower, self.upper)

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