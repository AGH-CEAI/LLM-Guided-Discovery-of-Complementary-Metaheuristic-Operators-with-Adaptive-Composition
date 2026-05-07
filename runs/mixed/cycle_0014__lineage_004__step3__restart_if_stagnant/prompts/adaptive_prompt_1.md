Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_restart_if_stagnant` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
1      2.405273e-01            2.396729e-01            -inf                    2.411137e-01            2.410497e-01            -inf                    2.409868e-01            2.396583e-01            2.403560e-01            2.382908e-01            -inf                    
2      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
3      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
4      1.304970e+00            1.322168e+00            -inf                    1.315694e+00            1.317730e+00            -inf                    1.306964e+00            1.271732e+00            1.257137e+00            1.332805e+00            -inf                    
5      1.184654e+00            1.137583e+00            -inf                    1.082852e+00            9.120955e-01            -inf                    1.083795e+00            1.112377e+00            1.060923e+00            1.223325e+00            -inf                    
6      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
7      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
8      2.205380e+03            1.760602e+03            -inf                    2.381166e+03            1.612531e+03            -inf                    1.815976e+03            1.995864e+03            2.215105e+03            2.099298e+03            -inf                    
9      1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
10     1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
11     1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
12     3.198235e+03            3.503283e+03            -inf                    3.714154e+03            2.949819e+03            -inf                    3.658873e+03            3.174389e+03            3.352726e+03            2.935574e+03            -inf                    
13     2.816932e+00            4.202225e+00            -inf                    4.199048e+00            6.112800e+00            -inf                    5.118498e+00            5.115677e+00            2.374593e+00            2.986537e+00            -inf                    
14     6.484174e+00            6.480373e+00            -inf                    6.792771e+00            6.649082e+00            -inf                    6.809244e+00            6.667780e+00            6.717693e+00            6.731580e+00            -inf                    
15     1.000000e-08            4.000000e-08            -inf                    4.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            4.000000e-08            2.000000e-08            -inf                    
16     1.333333e-08            4.000000e-08            -inf                    5.622669e+02            2.000000e-08            -inf                    4.000000e-08            5.584808e+02            5.416962e+02            4.000000e-08            -inf                    
17     1.000000e-08            5.296000e+02            -inf                    1.000000e-08            1.333333e-08            -inf                    4.000000e-08            1.333333e-08            1.000000e-08            1.333333e-08            -inf                    
18     1.333333e-08            1.333333e-08            -inf                    1.000000e-08            2.000000e-08            -inf                    5.296000e+02            1.000000e-08            2.000000e-08            1.000000e-08            -inf                    
19     7.582603e-04            7.883852e-04            -inf                    7.721457e-04            7.671971e-04            -inf                    7.647745e-04            7.771200e-04            7.518487e-04            7.635316e-04            -inf                    
20     5.000001e+00            5.000001e+00            -inf                    5.000001e+00            5.000001e+00            -inf                    5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            -inf                    
21     5.000000e+01            5.000000e+01            -inf                    5.000000e+01            5.000000e+01            -inf                    5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            -inf                    
22     9.854523e-06            1.033526e-05            -inf                    1.004567e-05            1.003676e-05            -inf                    9.824621e-06            9.796888e-06            9.737058e-06            1.002404e-05            -inf                    
23     7.022101e+00            8.079797e+00            -inf                    1.124934e+01            5.604793e+00            -inf                    5.046628e+00            5.534014e+00            7.241226e+00            6.511168e+00            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_09_idea_0.py  (error=2.382908e-01)
Task  2: SKIPPED (trivial)
Task  3: SKIPPED (trivial)
Task  4: variant_08_idea_0.py  (error=1.257137e+00)
Task  5: variant_04_idea_0.py  (error=9.120955e-01)
Task  6: SKIPPED (trivial)
Task  7: SKIPPED (trivial)
Task  8: variant_04_idea_0.py  (error=1.612531e+03)
Task  9: SKIPPED (trivial)
Task 10: SKIPPED (trivial)
Task 11: SKIPPED (trivial)
Task 12: variant_09_idea_0.py  (error=2.935574e+03)
Task 13: variant_08_idea_0.py  (error=2.374593e+00)
Task 14: variant_01_idea_0.py  (error=6.480373e+00)
Task 15: SKIPPED (trivial)
Task 16: SKIPPED (trivial)
Task 17: SKIPPED (trivial)
Task 18: SKIPPED (trivial)
Task 19: variant_08_idea_0.py  (error=7.518487e-04)
Task 20: variant_03_idea_0.py  (error=5.000001e+00)
Task 21: variant_07_idea_0.py  (error=5.000000e+01)
Task 22: variant_08_idea_0.py  (error=9.737058e-06)
Task 23: variant_06_idea_0.py  (error=5.046628e+00)

WIN COUNTS:
  variant_08_idea_0.py: 4 wins
  variant_09_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  variant_01_idea_0.py: 1 wins
  variant_03_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()

        # Preserve the best individual
        best_idx = np.argmin(fitness)

        # Replace worst half with completely new random solutions
        n_replace = NP // 2
        worst_indices = np.argpartition(fitness, n_replace)[:n_replace]

        for idx in worst_indices:
            new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)

        # Ensure best individual is retained
        new_pop[best_idx] = x_best.copy()

        return new_pop
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        dim = self.dim
        lower, upper = self.lower, self.upper

        # Strategy 1: Random uniform restart (1/3 of replacements)
        n_random = NP // 3
        random_indices = np.random.choice(NP, n_random, replace=False)
        for idx in random_indices:
            new_pop[idx] = np.random.uniform(lower, upper, dim)

        # Strategy 2: Opposition-based restart (1/3 of replacements)
        n_oppose = NP // 3
        oppose_indices = np.random.choice([i for i in range(NP) if i not in random_indices], 
                                           min(n_oppose, NP - n_random), replace=False)
        for idx in oppose_indices:
            # Reflect through search space center
            new_pop[idx] = (lower + upper) / 2.0 + (upper - lower) / 2.0 * np.random.uniform(-1, 1, dim)

        # Strategy 3: Latin Hypercube Sampling for remaining
        remaining = [i for i in range(NP) if i not in random_indices and i not in oppose_indices]
        if len(remaining) > 0:
            n_lhs = len(remaining)
            # Generate LHS samples in [0, 1] and scale to bounds
            lhs_samples = np.zeros((n_lhs, dim))
            for d in range(dim):
                samples = np.linspace(0, 1, n_lhs + 1)[:-1] + np.random.uniform(0, 1/n_lhs, n_lhs)
                lhs_samples[:, d] = samples
            np.random.shuffle(lhs_samples)
            lhs_scaled = lower + lhs_samples * (upper - lower)
            for i, idx in enumerate(remaining):
                new_pop[idx] = lhs_scaled[i]

        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
            NP = len(population)
            new_pop = population.copy()

            # Determine how many to reinitialize randomly vs perturb
            # Full random = aggressive diversity for escaping bad local optima
            # Perturbed = maintain some exploitation of good region
            random_count = NP // 2
            perturb_count = NP - random_count

            # Half: complete reinitialization across entire search space
            for i in range(random_count):
                new_pop[i] = np.random.uniform(self.lower, self.upper, self.dim)

            # Half: Gaussian perturbation around best with adaptive sigma
            # Sigma scales with search range to ensure meaningful displacement
            sigma = (self.upper - self.lower) * 0.3
            for i in range(random_count, NP):
                new_pop[i] = x_best + np.random.normal(0, sigma, self.dim)

            new_pop = self._clip_to_bounds_batch(new_pop)
            return new_pop
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
            NP = len(population)
            new_pop = population.copy()

            # Compute stagnation factor: increases with stagnation count
            stagnation_gen = getattr(self, 'stagnation', 0)
            distance_scale = 1.0 + min(stagnation_gen / 20.0, 5.0)  # Cap at 6x

            # Replace ALL individuals (not just worst 1/3) for full diversity
            for idx in range(NP):
                # Generate random direction vector
                direction = np.random.randn(self.dim)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-15:
                    direction = direction / direction_norm

                # Use tanh to map to [0,1] then scale to bounds - pushes to BOUNDARIES
                boundary_t = np.tanh(np.random.uniform(-3, 3, self.dim))  # ~[-1,1]

                # Blend: some near-best, some at boundaries, some in between
                blend = np.random.random()
                if blend < 0.3:
                    # 30%: Near best with hyperbolic expansion
                    offset = distance_scale * 50.0 * boundary_t
                    new_pop[idx] = x_best + offset
                elif blend < 0.7:
                    # 40%: Fully boundary-focused exploration
                    range_val = self.upper - self.lower
                    boundary_pos = (self.lower + self.upper) / 2 + boundary_t * range_val / 2
                    new_pop[idx] = boundary_pos
                else:
                    # 30%: Random in full space
                    new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)

            new_pop = self._clip_to_bounds_batch(new_pop)
            return new_pop
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        dim = self.dim
        new_pop = population.copy()

        # Compute population statistics for adaptive scaling
        centroid = np.mean(population, axis=0)
        pop_spread = np.std(population, axis=0) + 1e-10
        bound_range = self.upper - self.lower

        # Replace worst 50% with diverse strategies
        worst_count = NP // 2
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]

        for i, idx in enumerate(worst_indices):
            strategy = i % 3
            if strategy == 0:
                # Opposition-based: reflect x_best around centroid
                # This explores the opposite side of the search space
                reflection = 2.0 * centroid - x_best
                new_pop[idx] = reflection + np.random.uniform(-0.2 * bound_range, 0.2 * bound_range, dim)
            elif strategy == 1:
                # Space-filling: Halton quasi-random sequence (base 2)
                # Provides better dispersion than uniform random
                halton = np.array([(i + 1) / (dim ** k) % 1.0 for k in range(1, dim + 1)])
                new_pop[idx] = self.lower + halton * bound_range
            else:
                # Heavy-tailed: Cauchy perturbation for long jumps
                # Escapes local optima via occasional very large steps
                cauchy_scale = pop_spread * 2.0
                new_pop[idx] = x_best + np.random.standard_cauchy(dim) * cauchy_scale

        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```

# --- From variant_08_idea_0.py (4 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()

        # Adaptive perturbation scale based on error magnitude
        # For huge errors (~1e+03), use large perturbations; for smaller errors (~1e-01), use finer ones
        f_best = np.min(fitness)
        error_scale = max(1.0, np.log10(max(f_best, 1e-10) + 1e-10))
        perturbation_range = np.clip(50.0 * error_scale, 10.0, 500.0)

        # Replace worst third with error-scaled perturbations around best
        worst_count = NP // 3
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]

        for idx in worst_indices:
            # Adaptive per-dimension scaling: larger for high-error tasks, smaller for near-solved
            scale_factor = perturbation_range * np.random.uniform(0.5, 1.5, self.dim)
            new_pop[idx] = x_best + np.random.uniform(-scale_factor, scale_factor, self.dim)

        # Inject 20% completely random solutions for diversity (critical for escaping traps)
        random_count = max(1, NP // 5)
        random_indices = np.random.choice(NP, random_count, replace=False)
        for idx in random_indices:
            new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)

        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP, dim = population.shape
        new_pop = population.copy()

        # Determine how many worst individuals to replace
        worst_count = max(1, NP // 3)
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]

        # Compute search space center and range
        center = (self.lower + self.upper) / 2.0
        range_val = self.upper - self.lower

        # Compute current diversity to adaptively scale exploration
        current_diversity = self._compute_diversity(population)
        diversity_ratio = np.clip(current_diversity / (range_val * 0.1), 0.1, 2.0)

        for idx in worst_indices:
            # Strategy selection: 50% opposition-based, 30% adaptive scaled perturbation, 20% Latin hypercube
            strategy = np.random.random()

            if strategy < 0.5:
                # Opposition-based: mirror the individual across search space center
                # This explores the "opposite side" of the space, escaping local basins
                opp_point = self.lower + self.upper - population[idx]
                # Blend with random exploration (30% random influence)
                new_pop[idx] = 0.7 * opp_point + 0.3 * np.random.uniform(self.lower, self.upper, dim)

            elif strategy < 0.8:
                # Adaptive scaled perturbation around best (larger scale for hard problems)
                # Scale exploration radius based on current best error magnitude
                scale = max(1.0, min(range_val * 0.5, 50.0 * diversity_ratio))
                new_pop[idx] = x_best + np.random.uniform(-scale, scale, dim)

            else:
                # Latin hypercube-inspired: stratified sampling in reduced subspace
                # Focus exploration on promising hyper-rectangles
                subspace_lower = np.minimum(population[idx], x_best) - range_val * 0.1
                subspace_upper = np.maximum(population[idx], x_best) + range_val * 0.1
                subspace_lower = np.clip(subspace_lower, self.lower, self.upper)
                subspace_upper = np.clip(subspace_upper, self.lower, self.upper)

                # Generate sample in reduced subspace
                u = np.random.random(dim)
                new_pop[idx] = subspace_lower + u * (subspace_upper - subspace_lower)

        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
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


class CulturalDEWithAdaptiveArchive:
    """
    Cultural Differential Evolution with Adaptive Archive, Velocity-Guided
    Mutation, and Targeted Local Search.
    
    Key innovations:
    - Current-to-pbest/mean mutation with cultural guidance
    - Velocity-based momentum for exploration continuity
    - Adaptive F/CR based on exponential moving average success
    - Bounded archive for diversity maintenance
    - Stagnation-triggered local search on best candidate
    - Diversity-aware restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(4 * dim, 60), 300)
        self.lower = -100.0
        self.upper = 100.0
        self.p_best_frac = 0.15
        self.archive_max = self.NP // 2
        self.F_init = 0.6
        self.CR_init = 0.85
        self.F_min, self.F_max = 0.3, 1.2
        self.CR_min, self.CR_max = 0.3, 0.95
        self.local_search_interval = 50
        self.local_search_radius = 0.1
        self.diversity_threshold = 1e-6
        self.stagnation_limit = 80
        
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = self._eval_wrapper(func, population)
        archive = self._init_archive(population, fitness)
        velocity = self._init_velocity()
        F, CR = self._init_parameters()
        F_ema, CR_ema = F, CR
        success_F, success_CR = [], []
        best_idx = self._argmin(fitness)
        f_best, x_best = fitness[best_idx], population[best_idx].copy()
        stagnation = 0
        gen = 0
        
        while not stopping_condition():
            trials, velocity = self._build_trials_batched(population, velocity, F, CR, archive, fitness)
            trial_fit = self._eval_wrapper(func, trials)
            
            if len(trial_fit) < len(trials):
                valid = len(trial_fit)
                trials, trial_fit = trials[:valid], trial_fit[:valid]
                if valid == 0:
                    break
            
            if stopping_condition():
                break
            
            population, fitness, archive, success_F, success_CR = self._select_survivors_batch(
                population, fitness, trials, trial_fit, archive, success_F, success_CR
            )
            
            F, CR, F_ema, CR_ema = self._adapt_parameters_batch(
                F, CR, F_ema, CR_ema, success_F, success_CR
            )
            
            new_best_idx = self._argmin(fitness)
            if fitness[new_best_idx] < f_best - 1e-12:
                f_best, x_best = fitness[new_best_idx], population[new_best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1
            
            velocity = self._update_velocity_batch(population, velocity, x_best)
            
            if stagnation >= self.local_search_interval:
                x_best, f_best = self._apply_adaptive_local_search(
                    func, x_best, f_best, self.local_search_radius
                )
                stagnation = 0
            
            if self._compute_diversity(population) < self.diversity_threshold:
                population = self._restart_if_stagnant(population, fitness, x_best)
                velocity = self._init_velocity()
            
            gen += 1
        
        return f_best, x_best
    
    def _initialize_population(self):
        return np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
    
    def _init_archive(self, population, fitness):
        top_k = min(self.archive_max, self.NP // 4)
        top_indices = np.argpartition(fitness, top_k)[:top_k]
        return population[top_indices].copy()
    
    def _init_velocity(self):
        range_val = self.upper - self.lower
        return np.random.uniform(-0.1 * range_val, 0.1 * range_val, (self.NP, self.dim))
    
    def _init_parameters(self):
        return self.F_init, self.CR_init
    
    def _eval_wrapper(self, func, population):
        clipped = self._clip_to_bounds_batch(population)
        return func(clipped)
    
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.lower, self.upper)
    
    def _argmin(self, arr):
        return int(np.argmin(arr))
    
    def _build_trials_batched(self, population, velocity, F, CR, archive, fitness):
        mutated = self._mutate_current_to_pbest_with_culture(
            population, fitness, archive, F
        )
        vel_scaled = velocity * 0.1 * np.random.uniform(0.8, 1.2)
        mutated = mutated + vel_scaled
        trials = self._crossover_batch(population, mutated, CR)
        new_velocity = mutated - population
        return trials, new_velocity
    
    def _mutate_current_to_pbest_with_culture(self, population, fitness, archive, F):
        NP, dim = population.shape
        pbest_count = max(1, int(self.p_best_frac * NP))
        pbest_indices = np.argpartition(fitness, pbest_count)[:pbest_count]
        pbest = population[np.random.choice(pbest_indices)]
        
        r1_idx = np.random.choice(NP, NP, replace=True)
        r2_idx = np.random.choice(NP, NP, replace=True)
        while np.any(r1_idx == np.arange(NP)) or np.any(r2_idx == np.arange(NP)):
            r1_idx = np.random.choice(NP, NP, replace=True)
            r2_idx = np.random.choice(NP, NP, replace=True)
        
        r1, r2 = population[r1_idx], population[r2_idx]
        
        if len(archive) > 0:
            arch_idx = np.random.choice(len(archive), NP, replace=True)
            r3 = archive[arch_idx]
        else:
            r3 = population[np.random.choice(NP, NP, replace=True)]
        
        mean_target = np.mean(population, axis=0)
        cultural_weight = np.random.uniform(0.0, 0.3, (NP, 1))
        
        mutated = population + F * (pbest - population) + F * (r1 - r2) + F * cultural_weight * (mean_target - population)
        mutated = self._clip_to_bounds_batch(mutated)
        return mutated
    
    def _crossover_batch(self, target, donor, CR):
        NP, dim = target.shape
        j_rand = np.random.randint(0, dim, NP)
        mask = np.random.random((NP, dim)) < CR
        mask[np.arange(NP), j_rand] = True
        trial = np.where(mask, donor, target)
        return trial
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fit, 
                                 archive, success_F, success_CR):
        NP = len(population)
        improved = trial_fit < fitness
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        improve_idx = np.where(improved)[0]
        for idx in improve_idx:
            new_pop[idx] = trials[idx]
            new_fit[idx] = trial_fit[idx]
        
        for idx in improve_idx:
            F_i = np.random.uniform(self.F_min, self.F_max)
            CR_i = np.random.uniform(self.CR_min, self.CR_max)
            success_F.append(F_i)
            success_CR.append(CR_i)
        
        new_archive = self._update_archive_batch(archive, new_pop[improve_idx], 
                                                  new_fit[improve_idx])
        
        if len(success_F) > 20:
            trim = len(success_F) - 20
            success_F = success_F[trim:]
            success_CR = success_CR[trim:]
        
        return new_pop, new_fit, new_archive, success_F, success_CR
    
    def _update_archive_batch(self, archive, new_solutions, new_fitness):
        if len(new_solutions) == 0:
            return archive

        # Combine archive and new solutions
        if len(archive) > 0:
            combined = np.vstack([archive, new_solutions])
            combined_fit = np.concatenate([np.full(len(archive), -np.inf), new_fitness])
        else:
            combined = new_solutions
            combined_fit = new_fitness.copy()

        # Remove near-duplicates (keep the better one)
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]

        n = len(combined)
        if n <= self.archive_max:
            return combined

        # Compute minimum distance to nearest neighbor for diversity metric
        n_combined = len(combined)
        min_dists = np.full(n_combined, np.inf)

        # Vectorized distance computation (batch for efficiency)
        chunk_size = 500
        for i in range(0, n_combined, chunk_size):
            end_i = min(i + chunk_size, n_combined)
            chunk = combined[i:end_i]  # (chunk_size, dim)
            # Compute distances to all solutions
            diffs = combined[np.newaxis, :, :] - chunk[:, np.newaxis, :]  # (chunk, n, dim)
            dists = np.linalg.norm(diffs, axis=2)  # (chunk, n)
            np.fill_diagonal(dists[i:i+chunk_size, :][:end_i-i], np.inf)  # Mask self-distances
            min_dists[i:end_i] = np.min(dists, axis=1)

        # Normalize fitness (lower is better, so negate for proper scoring)
        fit_min, fit_max = np.min(combined_fit), np.max(combined_fit)
        fit_range = fit_max - fit_min + 1e-15
        fitness_score = 1.0 - (combined_fit - fit_min) / fit_range  # Higher = better

        # Normalize diversity (higher distance = better diversity)
        dist_max = np.max(min_dists) + 1e-15
        diversity_score = min_dists / dist_max

        # Composite: balance fitness (70%) and diversity (30%)
        # For hard problems, diversity helps escape local optima
        composite = 0.7 * fitness_score + 0.3 * diversity_score

        # Select top archive_max by composite score
        top_indices = np.argsort(composite)[-self.archive_max:]

        return combined[top_indices]
    
    def _find_unique_solutions(self, solutions, tol=1e-3):
        if len(solutions) <= 1:
            return np.ones(len(solutions), dtype=bool)
        
        n = len(solutions)
        is_unique = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_unique[i]:
                continue
            diffs = np.abs(solutions[i] - solutions[i+1:]) if i+1 < n else np.array([])
            if len(diffs) > 0 and np.any(np.all(diffs < tol, axis=1)):
                is_unique[i+1:] = False
        return is_unique
    
    def _adapt_parameters_batch(self, F, CR, F_ema, CR_ema, success_F, success_CR):
        if len(success_F) >= 5:
            F_ema = 0.9 * F_ema + 0.1 * np.mean(success_F[-10:])
            CR_ema = 0.9 * CR_ema + 0.1 * np.mean(success_CR[-10:])
            F = np.clip(F_ema + np.random.uniform(-0.1, 0.1), self.F_min, self.F_max)
            CR = np.clip(CR_ema + np.random.uniform(-0.1, 0.1), self.CR_min, self.CR_max)
        return F, CR, F_ema, CR_ema
    
    def _update_velocity_batch(self, population, velocity, x_best):
        NP = len(population)
        inertia = 0.7
        cognitive = 1.5
        social = 1.5
        
        r1 = np.random.uniform(0, 1, (NP, self.dim))
        r2 = np.random.uniform(0, 1, (NP, self.dim))
        
        new_vel = inertia * velocity + \
                  cognitive * r1 * (x_best - population) + \
                  social * r2 * (np.mean(population, axis=0) - population)
        
        max_vel = (self.upper - self.lower) * 0.2
        new_vel = np.clip(new_vel, -max_vel, max_vel)
        return new_vel
    
    def _apply_adaptive_local_search(self, func, x_best, f_best, radius):
        history = [x_best.copy()]
        current = x_best.copy()
        current_f = f_best
        
        for _ in range(3):
            step = np.random.uniform(-radius, radius, self.dim)
            candidate = self._clip_to_bounds_batch(current + step)
            cand_f = self._eval_wrapper(func, candidate.reshape(1, -1))[0]
            
            if cand_f < current_f:
                current = candidate
                current_f = cand_f
                radius *= 1.2
            else:
                radius *= 0.5
            
            history.append(current.copy())
        
        if current_f < f_best:
            return current, current_f
        return x_best, f_best
    
    def _compute_diversity(self, population):
        if len(population) < 2:
            return 1.0
        
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return float(np.mean(distances))
    
    def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        worst_count = NP // 3
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        for idx in worst_indices:
            new_pop[idx] = x_best + np.random.uniform(-10, 10, self.dim)
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop

```