This is iteration 7 of 10. Propose ONE replacement implementation for `_update_archive_batch`.

Requirements:
- Keep the EXACT function signature: `def _update_archive_batch(self, archive, new_solutions, new_fitness):`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that reaches error <= 1e-08 on EACH task.
- Tasks marked UNSOLVED below are critical gaps. The bigger the remaining
  error, the higher the priority — your variant should specifically target
  the WORST unsolved tasks at the top of the priority list.

TASK COVERAGE SUMMARY: 12 SOLVED (<= 1e-08), 12 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 12: *** UNSOLVED *** — best=2.562e+03 by variant_01_idea_0.py      (target=1e-08, ~+11.4 decades above target)
  Task  8: *** UNSOLVED *** — best=2.034e+03 by variant_06_idea_0.py      (target=1e-08, ~+11.3 decades above target)
  Task 21: *** UNSOLVED *** — best=5.000e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.7 decades above target)
  Task 14: *** UNSOLVED *** — best=6.537e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 23: *** UNSOLVED *** — best=5.755e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 20: *** UNSOLVED *** — best=5.000e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 13: *** UNSOLVED *** — best=1.654e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.2 decades above target)
  Task  4: *** UNSOLVED *** — best=1.289e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.1 decades above target)
  Task  5: *** UNSOLVED *** — best=1.069e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task  1: *** UNSOLVED *** — best=2.388e-01 by variant_03_idea_0.py      (target=1e-08, ~+7.4 decades above target)
  Task 19: *** UNSOLVED *** — best=7.227e-04 by variant_03_idea_0.py      (target=1e-08, ~+4.9 decades above target)
  Task 22: *** UNSOLVED *** — best=9.649e-06 by variant_06_idea_0.py      (target=1e-08, ~+3.0 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  0: SOLVED  — best=1.000e-08 by original.py                   
  Task  2: SOLVED  — best=1.000e-08 by original.py                   
  Task  3: SOLVED  — best=1.000e-08 by original.py                   
  Task  6: SOLVED  — best=1.000e-08 by original.py                   
  Task  7: SOLVED  — best=1.000e-08 by original.py                   
  Task  9: SOLVED  — best=1.000e-08 by original.py                   
  Task 10: SOLVED  — best=1.000e-08 by original.py                   
  Task 11: SOLVED  — best=1.000e-08 by original.py                   
  Task 15: SOLVED  — best=1.000e-08 by variant_04_idea_0.py          
  Task 16: SOLVED  — best=1.000e-08 by variant_01_idea_0.py          
  Task 17: SOLVED  — best=1.000e-08 by variant_01_idea_0.py          
  Task 18: SOLVED  — best=1.000e-08 by variant_02_idea_0.py          

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  
---------------------------------------------------------------------------------------------------------------------------------------------------------------
0    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
1    2.3995e-01            2.4072e-01            2.3998e-01            2.3879e-01            2.4065e-01            2.3971e-01            2.3965e-01            
2    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
3    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
4    1.3061e+00            1.3098e+00            1.3595e+00            1.3038e+00            1.3045e+00            1.2894e+00            1.3029e+00            
5    1.1292e+00            1.0689e+00            1.1244e+00            1.1559e+00            1.1448e+00            1.2269e+00            1.0730e+00            
6    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
7    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
8    2.3540e+03            2.2010e+03            2.2992e+03            2.4401e+03            2.2779e+03            2.2977e+03            2.0344e+03            
9    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
10   1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
11   1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
12   4.0164e+03            2.5620e+03            3.7984e+03            3.6548e+03            3.3126e+03            3.3310e+03            3.7435e+03            
13   2.0589e+00            2.5854e+00            1.6539e+00            2.8915e+00            2.2829e+00            8.9360e+00            2.7506e+00            
14   6.5370e+00            6.6924e+00            6.7141e+00            6.5986e+00            6.5584e+00            6.5921e+00            6.6745e+00            
15   2.0000e-08            5.2960e+02            4.0000e-08            2.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
16   1.3333e-08            1.0000e-08            1.3333e-08            2.0000e-08            1.3333e-08            2.0000e-08            2.0000e-08            
17   4.0000e-08            1.0000e-08            1.0000e-08            2.0000e-08            1.3333e-08            4.0000e-08            1.0000e-08            
18   2.0000e-08            5.2960e+02            1.0000e-08            1.3333e-08            5.2960e+02            4.0000e-08            1.3333e-08            
19   7.5431e-04            7.6055e-04            7.6440e-04            7.2274e-04            7.5614e-04            7.6043e-04            7.6137e-04            
20   5.0000e+00            5.0000e+00            5.0000e+00            5.0000e+00            5.0000e+00            5.0000e+00            5.0000e+00            
21   5.0000e+01            5.0000e+01            5.0000e+01            5.0000e+01            5.0000e+01            5.0000e+01            5.0000e+01            
22   9.9801e-06            1.0142e-05            1.0364e-05            1.0128e-05            9.9962e-06            9.9067e-06            9.6493e-06            
23   7.7392e+00            5.7553e+00            8.4606e+00            9.2355e+00            7.7248e+00            8.3854e+00            7.0076e+00            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 12: best error so far = 2.562e+03  (target = 1e-08)
  Task  8: best error so far = 2.034e+03  (target = 1e-08)
  Task 21: best error so far = 5.000e+01  (target = 1e-08)
  Task 14: best error so far = 6.537e+00  (target = 1e-08)
  Task 23: best error so far = 5.755e+00  (target = 1e-08)
  Task 20: best error so far = 5.000e+00  (target = 1e-08)
  Task 13: best error so far = 1.654e+00  (target = 1e-08)
  Task  4: best error so far = 1.289e+00  (target = 1e-08)
  ... and 4 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to crush the error on the WORST unsolved
tasks above. It is acceptable — even expected — for the new variant to be
worse than existing variants on already-SOLVED tasks; the adaptive selector
will handle that. Reason explicitly about the priority targets:
1. What property of those WORST unsolved tasks (multimodality, ill-conditioning,
   separability, ruggedness, noise, deceptive local optima, narrow basins,
   non-separable rotation, etc.) is preventing existing operators from reaching
   1e-08? Use the per-task error magnitudes as evidence — errors
   stuck at ~1e+1 vs ~1e-3 vs ~1e-6 imply different failure modes.
2. What specific mechanism in your proposed operator is designed to break
   through that exact obstacle and push the error several orders of magnitude
   lower?
3. Why is this approach fundamentally different from the prior variants —
   especially from whichever variant currently holds the best (but still
   insufficient) error on the priority tasks?

Current implementation:
```python
def _update_archive_batch(self, archive, new_solutions, new_fitness):
        if len(new_solutions) == 0:
            return archive
        
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        if len(combined) > self.archive_max:
            top_keep = min(self.archive_max // 2, len(new_solutions))
            arch_keep = self.archive_max - top_keep
            
            new_sorted = np.argsort(new_fitness)[:top_keep]
            arch_sorted = np.argsort(combined_fit[:len(archive)])[-arch_keep:] if len(archive) > 0 else []
            
            selected = np.concatenate([new_sorted, arch_sorted])
            return combined[selected]
        
        return combined
```

Full algorithm for context:
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
        
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        if len(combined) > self.archive_max:
            top_keep = min(self.archive_max // 2, len(new_solutions))
            arch_keep = self.archive_max - top_keep
            
            new_sorted = np.argsort(new_fitness)[:top_keep]
            arch_sorted = np.argsort(combined_fit[:len(archive)])[-arch_keep:] if len(archive) > 0 else []
            
            selected = np.concatenate([new_sorted, arch_sorted])
            return combined[selected]
        
        return combined
    
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

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _update_archive_batch(self, ...):
    ...
```