Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_argmin` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
1      2.405273e-01            2.410607e-01            2.399614e-01            2.396865e-01            2.390084e-01            2.400856e-01            2.408493e-01            2.405611e-01            2.404180e-01            2.398474e-01            2.396020e-01            
2      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
3      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
4      1.304970e+00            1.328285e+00            1.328953e+00            1.293601e+00            1.303107e+00            1.296741e+00            1.292055e+00            1.309726e+00            1.343998e+00            1.309519e+00            1.324142e+00            
5      1.184654e+00            1.102611e+00            1.183565e+00            1.255556e+00            1.087159e+00            1.144721e+00            1.094757e+00            1.154319e+00            1.117699e+00            1.032566e+00            1.112486e+00            
6      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
7      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
8      2.205380e+03            2.368407e+03            2.080384e+03            2.336041e+03            2.435388e+03            2.065553e+03            2.175951e+03            2.207047e+03            1.945175e+03            2.163130e+03            2.440954e+03            
9      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
10     1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
11     1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
12     3.198235e+03            3.917044e+03            3.408511e+03            3.418227e+03            3.174155e+03            3.280719e+03            3.366627e+03            3.015158e+03            3.230437e+03            3.036571e+03            3.898903e+03            
13     2.816932e+00            2.546996e+00            3.842872e+00            1.723547e+00            2.580281e+00            3.165459e+00            3.270504e+00            2.755677e+00            4.304234e+00            4.183596e+00            3.570053e+00            
14     6.484174e+00            6.708724e+00            6.767630e+00            6.627303e+00            6.700846e+00            6.607247e+00            6.682524e+00            6.628327e+00            6.782296e+00            6.692351e+00            6.669045e+00            
15     1.000000e-08            4.000000e-08            1.333333e-08            2.000000e-08            5.296000e+02            5.296000e+02            4.000000e-08            1.333333e-08            1.000000e-08            1.000000e-08            5.296000e+02            
16     1.333333e-08            2.000000e-08            5.341478e+02            5.364051e+02            1.000000e-08            4.000000e-08            2.000000e-08            4.000000e-08            5.510091e+02            2.000000e-08            1.000000e-08            
17     1.000000e-08            1.000000e-08            5.296000e+02            4.000000e-08            1.000000e-08            1.333333e-08            1.333333e-08            1.333333e-08            1.333333e-08            1.000000e-08            1.000000e-08            
18     1.333333e-08            1.000000e-08            1.333333e-08            1.333333e-08            1.000000e-08            1.000000e-08            4.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
19     7.582603e-04            7.530646e-04            7.633537e-04            7.722906e-04            7.476600e-04            7.564756e-04            7.738327e-04            7.759856e-04            7.736877e-04            7.465563e-04            7.531627e-04            
20     5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            5.000001e+00            
21     5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            5.000000e+01            
22     9.854523e-06            1.013329e-05            9.647989e-06            1.002854e-05            9.664049e-06            9.836413e-06            9.573252e-06            9.757447e-06            1.019072e-05            1.038647e-05            9.642060e-06            
23     7.022101e+00            6.392110e+00            9.499190e+00            8.545636e+00            9.768562e+00            1.032189e+01            6.570302e+00            6.513302e+00            9.106549e+00            1.034417e+01            1.028835e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_04_idea_0.py  (error=2.390084e-01)
Task  2: SKIPPED (trivial)
Task  3: SKIPPED (trivial)
Task  4: variant_06_idea_0.py  (error=1.292055e+00)
Task  5: variant_09_idea_0.py  (error=1.032566e+00)
Task  6: SKIPPED (trivial)
Task  7: SKIPPED (trivial)
Task  8: variant_08_idea_0.py  (error=1.945175e+03)
Task  9: SKIPPED (trivial)
Task 10: SKIPPED (trivial)
Task 11: SKIPPED (trivial)
Task 12: variant_07_idea_0.py  (error=3.015158e+03)
Task 13: variant_03_idea_0.py  (error=1.723547e+00)
Task 14: original.py  (error=6.484174e+00)
Task 15: SKIPPED (trivial)
Task 16: SKIPPED (trivial)
Task 17: SKIPPED (trivial)
Task 18: SKIPPED (trivial)
Task 19: variant_09_idea_0.py  (error=7.465563e-04)
Task 20: variant_07_idea_0.py  (error=5.000001e+00)
Task 21: variant_09_idea_0.py  (error=5.000000e+01)
Task 22: variant_06_idea_0.py  (error=9.573252e-06)
Task 23: variant_01_idea_0.py  (error=6.392110e+00)

WIN COUNTS:
  variant_09_idea_0.py: 3 wins
  variant_06_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_04_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins
  variant_03_idea_0.py: 1 wins
  original.py: 1 wins
  variant_01_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _argmin(self, arr):
        if len(arr) == 0:
            return None
        # np.nanargmin returns the index of the minimum value, ignoring NaNs
        # This is robust to common numerical issues in metaheuristic fitness evaluation
        idx = np.nanargmin(arr)
        return int(idx)
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _argmin(self, arr):
        arr = np.asarray(arr)
        median_val = np.median(arr)
        median_mask = arr <= median_val

        if np.any(median_mask):
            candidates = np.where(median_mask)[0]
            if len(candidates) == 1:
                return int(candidates[0])
            # Among below-median candidates, pick most central (diverse) one
            sub_arr = arr[candidates]
            best_sub_idx = int(np.argmin(sub_arr))
            return int(candidates[best_sub_idx])
        else:
            return int(np.argmin(arr))
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _argmin(self, arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return 0
            if arr.size == 1:
                return 0

            # Compute diversity: distance from centroid (normalized)
            # We'll use population stored in self if available, otherwise just the array
            # Since we don't have direct access to population here, use arr statistics
            centroid = np.mean(arr)
            distances = np.abs(arr - centroid)
            max_dist = np.max(distances) + 1e-15
            diversity = distances / max_dist

            # Normalize fitness (0=worst, 1=best)
            arr_min, arr_max = np.min(arr), np.max(arr)
            arr_range = arr_max - arr_min + 1e-15
            fitness_norm = 1.0 - (arr - arr_min) / arr_range  # Lower arr = higher fitness_norm

            # Composite score: 70% fitness, 30% diversity
            composite = 0.7 * fitness_norm + 0.3 * diversity

            # Small probability of random selection for exploration
            if np.random.random() < 0.05:
                return int(np.random.randint(len(arr)))

            return int(np.argmax(composite))
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _argmin(self, arr):
        if len(arr) == 0:
            return 0
        if len(arr) == 1:
            return 0

        # 1. Fitness ranking (lower fitness = better = higher score)
        fitness_ranks = np.argsort(np.argsort(arr))  # 0 = best rank
        fitness_score = 1.0 - fitness_ranks / (len(arr) - 1)  # normalize to [0, 1]

        # 2. Diversity contribution: farther from centroid = more valuable
        pop = getattr(self, '_current_pop', None)
        if pop is not None and len(pop) == len(arr):
            centroid = np.mean(pop, axis=0)
            dists = np.linalg.norm(pop - centroid, axis=1)
            max_dist = np.max(dists) + 1e-15
            diversity_score = dists / max_dist
        else:
            diversity_score = np.ones(len(arr)) * 0.5

        # 3. Adaptive composite weights: exploration-biased early, exploitation later
        gen = getattr(self, '_generation', 0)
        adapt = min(0.6 + 0.3 * np.tanh(gen / 100), 0.9)  # [0.6, 0.9]

        # Composite score
        composite = (adapt * fitness_score + 
                     (1 - adapt) * 0.5 * diversity_score)

        # 4. Probabilistic selection from top candidates (diversify selection pressure)
        k = max(3, len(arr) // 4)
        top_indices = np.argsort(composite)[-k:]
        top_probs = composite[top_indices]
        top_probs = np.clip(top_probs, 1e-15, None)
        top_probs = top_probs / np.sum(top_probs)

        return int(np.random.choice(top_indices, p=top_probs))
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _argmin(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0

        best_idx = int(np.argmin(arr))
        best_val = arr[best_idx]

        # Compute performance differences from best
        perf_diff = arr - best_val
        max_diff = np.max(perf_diff)

        # Handle edge case: all individuals identical
        if max_diff < 1e-15:
            return best_idx

        # Normalize performance differences to [0, 1]
        norm_perf = perf_diff / (max_diff + 1e-15)

        # Adaptive temperature: high variance → high temp (explore), low variance → low temp (exploit)
        variance = np.var(arr)
        temp = 0.1 + 0.5 * np.tanh(variance * 1e-6)

        # Compute selection probabilities via softmax
        exponent = -norm_perf / (temp + 1e-15)
        exponent -= np.max(exponent)  # Numerical stability
        probs = np.exp(exponent)
        probs = probs / (np.sum(probs) + 1e-15)

        # Final selection: weighted combination of best and probabilistic
        # 70% chance to pick the true best, 30% probabilistic exploration
        if np.random.random() < 0.7:
            return best_idx
        else:
            return int(np.random.choice(len(arr), p=probs))
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _argmin(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return int(np.argmin(arr))

        # Temperature controls exploration vs exploitation
        # Low temp = greedy (only top candidates), High temp = exploratory (any candidate possible)
        if not hasattr(self, '_argmin_temp'):
            self._argmin_temp = 0.5

        # Adapt temperature based on population diversity
        fitness_std = np.std(arr)
        if fitness_std < 1e-6:
            # Low diversity: increase temperature to encourage exploration
            self._argmin_temp = min(3.0, self._argmin_temp * 1.2)
        else:
            # Good diversity: can afford greedy selection
            self._argmin_temp = max(0.1, self._argmin_temp * 0.95)

        # Rank-based selection: ranks 0=best, n-1=worst
        ranks = np.argsort(np.argsort(arr))
        n = len(arr)

        # Exponential weights favoring better ranks, modulated by temperature
        weights = np.exp(-ranks / self._argmin_temp)
        probs = weights / np.sum(weights)

        # Probabilistic selection instead of deterministic argmin
        selected_idx = np.random.choice(n, p=probs)
        return int(selected_idx)
```

# --- From variant_09_idea_0.py (3 wins) ---
```python
def _argmin(self, arr):
        if len(arr) == 0:
            return 0
        if len(arr) == 1:
            return 0

        arr = np.asarray(arr)
        best_idx = int(np.argmin(arr))
        best_val = arr[best_idx]

        # Exponential fitness score (emphasizes better solutions)
        worst_val = np.max(arr)
        fit_range = worst_val - best_val + 1e-30
        fit_score = np.exp(-2.0 * (arr - best_val) / fit_range)

        # Diversity score based on distance from best solution
        # Higher when farther from best (exploration incentive)
        # Note: population stored in self._last_pop if available, otherwise uniform
        try:
            pop = getattr(self, '_last_pop', None)
            if pop is not None and len(pop) == len(arr):
                dists = np.linalg.norm(pop - pop[best_idx], axis=1)
                max_dist = np.max(dists) + 1e-30
                div_score = dists / max_dist
            else:
                div_score = np.ones(len(arr)) * 0.5
        except:
            div_score = np.ones(len(arr)) * 0.5

        # Composite: 70% fitness + 30% diversity
        composite = 0.7 * fit_score + 0.3 * div_score

        return int(np.argmax(composite))
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