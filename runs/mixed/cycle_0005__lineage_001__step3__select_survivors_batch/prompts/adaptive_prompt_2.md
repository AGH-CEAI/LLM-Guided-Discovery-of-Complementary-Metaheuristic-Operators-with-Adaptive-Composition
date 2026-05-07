Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_survivors_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.116253e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            
1      3.412855e-01            4.017599e-01            3.396568e-01            3.679633e-01            -inf                    3.314122e-01            3.358835e-01            3.414086e-01            3.416804e-01            -inf                    3.647062e-01            
2      6.066678e-08            9.540718e-04            6.797238e-04            1.497880e-03            -inf                    4.874097e-04            3.856215e-05            4.982706e-04            4.637175e-08            -inf                    2.831632e-03            
3      3.051545e-08            3.527194e-07            3.300296e-08            8.435383e-08            -inf                    3.801848e-07            2.239276e-08            2.196777e-08            1.415231e-08            -inf                    1.171778e-07            
4      1.304372e+00            8.898417e-01            1.371314e+00            1.880375e+00            -inf                    1.461140e+00            1.363687e+00            1.284123e+00            1.372084e+00            -inf                    1.884168e+00            
5      1.367386e+00            1.202403e+00            1.547840e+00            1.779064e+00            -inf                    1.718372e+00            1.460519e+00            1.396796e+00            1.496681e+00            -inf                    1.749407e+00            
6      1.321751e-07            1.284211e-03            2.131029e-03            3.416913e-02            -inf                    2.229744e+01            1.533664e-03            1.836512e-03            2.915557e-08            -inf                    6.716093e-02            
7      2.769454e-07            1.395962e-03            2.033952e-03            3.235140e-02            -inf                    4.608770e+00            1.666169e-03            1.858525e-03            2.883751e-08            -inf                    8.014091e-02            
8      1.172169e+03            4.475248e+02            8.853475e+02            9.162377e+02            -inf                    1.548247e+03            7.507704e+02            1.046870e+03            9.588735e+02            -inf                    1.328391e+03            
9      3.694661e-06            1.348413e-03            4.685156e-03            3.464620e-02            -inf                    3.170663e+03            1.714524e-03            1.684795e-03            6.379882e-07            -inf                    1.194101e-01            
10     3.598147e-05            1.608503e-03            4.524820e-02            2.922018e-02            -inf                    3.181190e+03            3.081667e-03            1.696131e-03            1.905821e-05            -inf                    1.987957e-01            
11     3.221561e-05            1.311891e-03            1.466028e-02            4.052506e-02            -inf                    5.849350e+03            1.607385e-03            1.923789e-03            1.460776e-05            -inf                    2.060240e-01            
12     1.611585e+03            1.085874e+01            1.679400e+03            1.879940e+03            -inf                    1.308461e+03            1.963854e+03            1.863621e+03            1.459752e+03            -inf                    1.882868e+03            
13     4.590017e+01            4.176509e+00            4.916943e+02            2.932853e+01            -inf                    9.103015e+03            4.112222e+01            2.268900e+01            6.985510e+01            -inf                    2.249538e+01            
14     5.856714e+00            5.090433e+00            5.906740e+00            5.821006e+00            -inf                    5.873006e+00            5.955115e+00            5.917983e+00            6.017281e+00            -inf                    6.035321e+00            
15     1.333333e-08            2.000000e-08            5.413824e+02            1.333333e-08            -inf                    2.000000e-08            1.000000e-08            1.000000e-08            4.000000e-08            -inf                    2.000000e-08            
16     6.001000e+02            6.001551e+02            5.942752e+02            6.001055e+02            -inf                    6.317443e+02            6.001048e+02            5.864106e+02            6.001000e+02            -inf                    5.784193e+02            
17     1.849436e-06            5.665954e+02            6.024033e+01            7.839852e-02            -inf                    7.518992e+03            5.059171e-02            2.681373e+02            5.296000e+02            -inf                    1.484287e+00            
18     2.723466e-04            5.283309e-03            5.828240e+01            5.929995e-01            -inf                    8.305103e+03            1.489913e-01            5.537659e+02            1.250741e-01            -inf                    5.315591e+02            
19     1.823489e-02            1.885075e-01            1.591025e+00            3.003186e-01            -inf                    4.240174e+00            4.555513e-01            1.174475e-01            1.216173e-02            -inf                    3.877690e+00            
20     5.000000e+00            5.020919e+00            5.028791e+00            5.054347e+00            -inf                    5.024995e+00            5.014612e+00            5.021367e+00            5.000000e+00            -inf                    5.465572e+00            
21     5.000111e+01            5.000913e+01            1.365401e+02            5.002617e+01            -inf                    1.125109e+03            5.005987e+01            5.050893e+01            5.001925e+01            -inf                    5.177102e+01            
22     9.574088e-04            6.168438e-02            1.243456e+00            1.309923e-01            -inf                    1.004601e+01            1.173690e-01            6.430373e-02            8.082007e-04            -inf                    2.269697e+00            
23     2.510144e+00            6.568349e+00            3.262019e+01            2.404835e+00            -inf                    6.055143e+01            2.297419e+00            3.212546e+00            9.573346e+00            -inf                    2.844644e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_05_idea_0.py  (error=3.314122e-01)
Task  2: variant_08_idea_0.py  (error=4.637175e-08)
Task  3: SKIPPED (trivial)
Task  4: variant_01_idea_0.py  (error=8.898417e-01)
Task  5: variant_01_idea_0.py  (error=1.202403e+00)
Task  6: variant_08_idea_0.py  (error=2.915557e-08)
Task  7: variant_08_idea_0.py  (error=2.883751e-08)
Task  8: variant_01_idea_0.py  (error=4.475248e+02)
Task  9: variant_08_idea_0.py  (error=6.379882e-07)
Task 10: variant_08_idea_0.py  (error=1.905821e-05)
Task 11: variant_08_idea_0.py  (error=1.460776e-05)
Task 12: variant_01_idea_0.py  (error=1.085874e+01)
Task 13: variant_01_idea_0.py  (error=4.176509e+00)
Task 14: variant_01_idea_0.py  (error=5.090433e+00)
Task 15: SKIPPED (trivial)
Task 16: variant_10_idea_0.py  (error=5.784193e+02)
Task 17: original.py  (error=1.849436e-06)
Task 18: original.py  (error=2.723466e-04)
Task 19: variant_08_idea_0.py  (error=1.216173e-02)
Task 20: variant_08_idea_0.py  (error=5.000000e+00)
Task 21: original.py  (error=5.000111e+01)
Task 22: variant_08_idea_0.py  (error=8.082007e-04)
Task 23: variant_06_idea_0.py  (error=2.297419e+00)

WIN COUNTS:
  variant_08_idea_0.py: 9 wins
  variant_01_idea_0.py: 6 wins
  original.py: 3 wins
  variant_05_idea_0.py: 1 wins
  variant_10_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (6 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        # Guard against empty or malformed inputs
        if population.shape[0] == 0 or trials.shape[0] == 0:
            return np.copy(population), np.copy(fitness)

        # Ensure matching dimensions
        np_pop = min(population.shape[0], trials.shape[0])
        pop_subset = population[:np_pop]
        fit_subset = fitness[:np_pop]
        trial_subset = trials[:np_pop]
        trial_fit_subset = trial_fitness[:np_pop]

        # Combine parent and trial pools
        combined_pop = np.vstack([pop_subset, trial_subset])
        combined_fit = np.concatenate([fit_subset, trial_fit_subset])

        # Guard against invalid fitness values
        combined_fit = np.clip(combined_fit, -1e50, 1e50)

        # Select best NP individuals (μ+λ style)
        if len(combined_fit) > np_pop:
            # Use argpartition for O(n) partial sort, then sort the selected k
            selected_indices = np.argpartition(combined_fit, np_pop)[:np_pop]
            selected_indices = selected_indices[np.argsort(combined_fit[selected_indices])]
        else:
            selected_indices = np.argsort(combined_fit)[:np_pop]

        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        new_population[:np_pop] = combined_pop[selected_indices]
        new_fitness[:np_pop] = combined_fit[selected_indices]

        return new_population, new_fitness
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        # Compute diversity contribution: average distance to other population members
        def diversity_contribution(solution, pop):
                if len(pop) < 2:
                    return 0.0
                dists = np.linalg.norm(pop - solution, axis=1)
                return float(np.mean(dists))
        candidates_pop = np.vstack([population, trials])
        candidates_fit = np.concatenate([fitness, trial_fitness])
        is_trial = np.concatenate([np.zeros(len(population), dtype=bool),
                                   np.ones(len(trials), dtype=bool)])

        # Compute diversity contributions for all candidates
        div_contribs = np.array([diversity_contribution(candidates_pop[i], candidates_pop)
                                 for i in range(len(candidates_pop))])

        # Normalize diversity contributions
        div_min, div_max = np.min(div_contribs), np.max(div_contribs)
        if div_max > div_min + 1e-10:
            div_normalized = (div_contribs - div_min) / (div_max - div_min)
        else:
            div_normalized = np.ones_like(div_contribs) * 0.5

        # Compute fitness scores (lower fitness = better)
        fit_min, fit_max = np.min(candidates_fit), np.max(candidates_fit)
        if fit_max > fit_min + 1e-10:
            fitness_score = 1.0 - (candidates_fit - fit_min) / (fit_max - fit_min)
        else:
            fitness_score = np.ones_like(candidates_fit) * 0.5

        # Adaptive diversity weight: increases when stagnant
        diversity_weight = min(0.4, 0.1 + 0.02 * self.stagnation_count)

        # Combined score: balance fitness vs diversity
        combined_score = (1.0 - diversity_weight) * fitness_score + diversity_weight * div_normalized

        # Select: for each pair (original, trial), pick the one with higher combined score
        n = len(population)
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)

        for i in range(n):
            orig_idx = i
            trial_idx = n + i
            if combined_score[trial_idx] > combined_score[orig_idx]:
                new_population[i] = candidates_pop[trial_idx]
                new_fitness[i] = candidates_fit[trial_idx]
            else:
                new_population[i] = candidates_pop[orig_idx]
                new_fitness[i] = candidates_fit[orig_idx]

        return new_population, new_fitness
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        # Track global best separately for deterministic elitism
        best_idx = np.argmin(fitness)
        global_best = population[best_idx].copy()
        global_best_fit = float(fitness[best_idx])

        # Adaptive temperature based on stagnation
        # High stagnation -> high temperature -> accept worse solutions (explore)
        # Low stagnation -> low temperature -> greedy selection (exploit)
        if self.stagnation_count > 20:
            temperature = min(0.5, 0.01 * (self.stagnation_count - 20))
        elif self.stagnation_count > 10:
            temperature = 0.01
        else:
            temperature = 0.001

        new_population = np.copy(population)
        new_fitness = np.copy(fitness)

        for i in range(len(population)):
            delta = float(fitness[i]) - float(trial_fitness[i])
            if delta > 0:
                # Trial is better: always accept
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
            elif self.stagnation_count > 15 and temperature > 1e-6:
                # Trial is worse but we're stagnant: Boltzmann acceptance
                # Higher temperature = more likely to accept worse solutions
                accept_prob = np.exp(delta / temperature)
                if np.random.random() < accept_prob:
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_fitness[i]
            # else: greedy rejection (standard DE behavior)

        # Deterministic elitism: always preserve global best
        new_best_idx = np.argmin(new_fitness)
        if float(new_fitness[new_best_idx]) > global_best_fit:
            new_population[new_best_idx] = global_best
            new_fitness[new_best_idx] = global_best_fit

        return new_population, new_fitness
```

# --- From variant_08_idea_0.py (9 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        NP = len(population)

        # Initialize age tracking if needed
        if not hasattr(self, 'individual_ages'):
            self.individual_ages = np.zeros(NP)

        # Increment ages of all existing individuals
        self.individual_ages += 1

        # Elitism: always keep the single best individual unchanged
        best_idx = int(np.argmin(fitness))

        # Compute improvement scores
        improvements = fitness - trial_fitness
        is_better = improvements > 0

        # Create replacement mask and tracking arrays
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        new_ages = np.copy(self.individual_ages)

        # For individuals that improve: compute replacement score
        # Score = improvement + age_bonus (favor replacing older individuals)
        age_bonus = 0.01 * self.individual_ages  # small bonus for older individuals
        replacement_scores = np.where(is_better, improvements + age_bonus, -np.inf)

        # Find best replacement for each trial
        for i in range(NP):
            if i == best_idx:
                continue  # Skip elitist
            if replacement_scores[i] > 0:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
                new_ages[i] = 0  # Reset age on successful replacement

        # Periodically inject diversity by replacing worst 20% with random
        # but only if stagnation is high (adaptive diversity injection)
        if self.stagnation_count > 15:
            n_replace = max(1, NP // 5)
            worst_indices = np.argsort(new_fitness)[-n_replace:]
            for idx in worst_indices:
                if idx != best_idx:
                    # Generate random individual near best but with perturbation
                    random_ind = self.best_solution + np.random.randn(self.dim) * (10.0 + self.stagnation_count * 0.1)
                    random_ind = np.clip(random_ind, self.lower, self.upper)
                    new_population[idx] = random_ind
                    new_fitness[idx] = np.inf
                    new_ages[idx] = 0

        self.individual_ages = new_ages
        return new_population, new_fitness
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        # Boltzmann selection: probabilistic acceptance of worse solutions
        # Temperature adapts to stagnation — high temp = more exploration

        # Adaptive temperature based on stagnation state
        if self.stagnation_count > 30:
            temperature = 2.0 + 0.5 * min(self.stagnation_count - 30, 150)
        elif self.stagnation_count > 15:
            temperature = 0.5 + 0.2 * (self.stagnation_count - 15)
        elif self.stagnation_count > 5:
            temperature = 0.2 + 0.15 * (self.stagnation_count - 5)
        else:
            temperature = 0.1

        temperature = max(float(temperature), 0.01)

        # Compute fitness delta (positive = trial is worse)
        fitness_delta = trial_fitness - fitness

        # Boltzmann acceptance probability
        # Better or equal trials (delta <= 0): accept with P = 1.0
        # Worse trials (delta > 0): P = exp(-delta / T)
        # Clamp exponent for numerical stability
        exponent = np.clip(-fitness_delta / temperature, -700, 700)
        accept_prob = np.where(fitness_delta <= 0, 1.0, np.exp(exponent))

        # Probabilistic selection
        roll = np.random.rand(len(fitness))
        accept = roll < accept_prob

        # Build new population/fitness
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        new_population[accept] = trials[accept]
        new_fitness[accept] = trial_fitness[accept]

        return new_population, new_fitness
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
        """Variant 09: Adaptive F based on improvement ratio, direction weight"""
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

        if np.isfinite(self.best_fitness) and self.generation > 5:
            current_best = np.min(fitness)
            denom = abs(self.best_fitness) + 1e-30
            improvement_ratio = (current_best - self.best_fitness) / denom

            F_escape = np.clip(self.F * (1.0 + max(0, -improvement_ratio) * 0.5), 0.5, 2.0)
            w_hist = np.clip(0.7 + max(0, -improvement_ratio) * 0.2, 0.5, 0.9)
            w_curr = 1.0 - w_hist

            if np.any(self.cultural_weights > 0):
                weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
                cultural_centroid = np.dot(weights_norm, self.cultural_memory)
                target = w_hist * self.best_solution + w_curr * p_best + 0.1 * cultural_centroid
            else:
                target = w_hist * self.best_solution + w_curr * p_best

            donors = population + F_escape * (target - population) + self.F * (population[r1] - population[r2])
        else:
            donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

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
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:self.NP // 2]

        F_candidates = np.random.uniform(0.4, 1.0, size=5)
        self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))

        if np.random.rand() < 0.2:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3))

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