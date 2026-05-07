Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_mutate_variant09` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
1      3.393864e-01            3.446129e-01            3.414115e-01            3.410222e-01            2.993975e-01            3.468266e-01            3.378797e-01            3.417211e-01            3.417572e-01            3.456027e-01            3.441418e-01            
2      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.354538e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
3      1.226875e-08            1.129575e-08            1.499663e-08            1.360413e-08            1.842024e-08            1.183730e-08            1.107408e-08            1.149316e-08            1.167544e-08            1.277355e-08            1.565858e-08            
4      7.732072e-01            9.586650e-01            1.130396e+00            9.796331e-01            1.186997e+00            1.097192e+00            8.058172e-01            9.457579e-01            9.200554e-01            9.748747e-01            1.087945e+00            
5      1.226328e+00            1.270757e+00            1.314398e+00            1.240592e+00            1.393678e+00            1.401337e+00            1.247250e+00            1.299048e+00            1.263792e+00            1.320073e+00            1.330747e+00            
6      1.000000e-08            1.981043e-08            1.070099e-08            1.814098e-08            1.000000e-08            1.809409e-08            2.845253e-08            1.940469e-08            1.795329e-08            1.519245e-08            2.491146e-08            
7      1.000000e-08            1.866271e-08            1.196691e-08            1.410399e-08            1.000000e-08            2.443606e-08            2.323132e-08            1.359140e-08            1.839288e-08            1.790346e-08            4.292629e-08            
8      8.458875e+02            9.082000e+02            6.402548e+02            1.033607e+03            1.814578e+02            6.941208e+02            7.513810e+02            6.866529e+02            9.742923e+02            7.761358e+02            7.485731e+02            
9      8.968550e-08            2.517819e-07            6.769865e-08            2.334473e-07            1.691951e-08            1.772219e-07            3.687793e-07            1.818331e-07            2.454622e-07            2.189413e-07            2.839123e-07            
10     1.703054e-06            3.165810e-06            7.715096e-07            2.134035e-06            9.298431e-08            2.293146e-06            2.756204e-06            2.291905e-06            2.296353e-06            1.829856e-06            3.392862e-06            
11     1.644196e-06            1.718118e-06            8.354081e-07            2.071484e-06            5.436087e-08            1.718569e-06            3.060661e-06            2.758970e-06            2.630498e-06            2.322696e-06            2.956471e-06            
12     1.212734e+03            1.526950e+03            1.518708e+03            1.416646e+03            7.187390e+02            1.486949e+03            1.456219e+03            1.417571e+03            1.215784e+03            1.147374e+03            1.474952e+03            
13     3.307582e+00            5.327230e+00            3.868087e+00            5.550159e+00            1.014243e+01            5.892586e+00            4.058534e+00            5.290358e+00            2.581111e+00            6.011375e+00            5.043276e+00            
14     5.830920e+00            6.021552e+00            5.919212e+00            5.948942e+00            5.374886e+00            5.940691e+00            5.844284e+00            5.982599e+00            5.845318e+00            5.936725e+00            5.860275e+00            
15     1.333333e-08            4.000000e-08            5.456251e+02            1.333333e-08            2.000000e-08            1.333333e-08            4.000000e-08            4.000000e-08            1.333333e-08            2.000000e-08            1.333333e-08            
16     5.871148e+02            5.690374e+02            5.612245e+02            6.001000e+02            6.003248e+02            5.887510e+02            6.001190e+02            5.963164e+02            5.814787e+02            5.466702e+02            5.813303e+02            
17     4.997928e-07            2.000000e-08            1.333333e-08            4.000000e-08            3.999974e-08            1.295286e-08            1.018610e-08            1.000000e-08            5.296000e+02            3.999995e-08            5.537009e+02            
18     1.631953e-06            4.226899e-04            5.581398e+02            4.506963e+01            1.171192e-06            2.682939e-01            5.347703e+02            5.296000e+02            4.295718e-05            3.602078e-04            6.617569e-05            
19     2.843944e-03            3.721759e-03            3.610710e-03            3.511472e-03            1.536927e-03            3.644690e-03            4.345760e-03            3.788611e-03            3.978058e-03            3.690911e-03            3.696011e-03            
20     5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            5.000000e+00            
21     5.000025e+01            5.000022e+01            5.000017e+01            5.000025e+01            5.000024e+01            5.000023e+01            5.000033e+01            5.000025e+01            5.000026e+01            5.000023e+01            5.000021e+01            
22     3.641920e-05            5.091376e-05            4.631183e-05            5.097834e-05            1.338140e-05            5.439791e-05            5.210537e-05            4.521647e-05            4.652468e-05            5.172513e-05            5.558222e-05            
23     4.017523e+00            3.775840e+00            7.140451e+00            9.332856e+00            1.434931e+01            1.001662e+01            6.998039e+00            6.048866e+00            7.421220e+00            5.823950e+00            3.400966e+00            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_04_idea_0.py  (error=2.993975e-01)
Task  2: SKIPPED (trivial)
Task  3: SKIPPED (trivial)
Task  4: original.py  (error=7.732072e-01)
Task  5: original.py  (error=1.226328e+00)
Task  6: SKIPPED (trivial)
Task  7: SKIPPED (trivial)
Task  8: variant_04_idea_0.py  (error=1.814578e+02)
Task  9: variant_04_idea_0.py  (error=1.691951e-08)
Task 10: variant_04_idea_0.py  (error=9.298431e-08)
Task 11: variant_04_idea_0.py  (error=5.436087e-08)
Task 12: variant_04_idea_0.py  (error=7.187390e+02)
Task 13: variant_08_idea_0.py  (error=2.581111e+00)
Task 14: variant_04_idea_0.py  (error=5.374886e+00)
Task 15: SKIPPED (trivial)
Task 16: variant_09_idea_0.py  (error=5.466702e+02)
Task 17: SKIPPED (trivial)
Task 18: variant_04_idea_0.py  (error=1.171192e-06)
Task 19: variant_04_idea_0.py  (error=1.536927e-03)
Task 20: variant_09_idea_0.py  (error=5.000000e+00)
Task 21: variant_02_idea_0.py  (error=5.000017e+01)
Task 22: variant_04_idea_0.py  (error=1.338140e-05)
Task 23: variant_10_idea_0.py  (error=3.400966e+00)

WIN COUNTS:
  variant_04_idea_0.py: 10 wins
  original.py: 2 wins
  variant_09_idea_0.py: 2 wins
  variant_08_idea_0.py: 1 wins
  variant_02_idea_0.py: 1 wins
  variant_10_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (1 wins) ---
```python
def _mutate_variant09(self, population, fitness):
        """Variant 09: Mean-Reversion Opposition DE — escape local optima via centroid pull and opposition"""
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

        # Compute population centroid (geometric center of search)
        centroid = np.mean(population, axis=0)

        # Compute search-space center and opposition vector
        search_center = 0.5 * (self.lower + self.upper)
        opposition_vector = 2.0 * search_center - population

        # Adaptive stagnation detection
        is_stagnant = self.stagnation_count > 10
        stagnation_factor = min(self.stagnation_count / 50.0, 1.0) if is_stagnant else 0.0

        # Mean-reversion weight: increases when stagnant to pull away from converged cluster
        w_mean = 0.1 + 0.4 * stagnation_factor

        # Opposition weight: only significant when stagnant (escape to opposite side)
        w_opp = 0.3 * stagnation_factor

        # Standard DE components (always present for exploitation)
        w_best = 0.5
        w_diff = 0.5

        # Build composite donor
        base_direction = w_best * (p_best - population) + w_diff * (population[r1] - population[r2])
        mean_direction = w_mean * (centroid - population)
        opp_direction = w_opp * (opposition_vector - population)

        # Combine with adaptive F
        F_base = self.F
        F_escape = np.clip(F_base * (1.0 + stagnation_factor * 0.5), 0.5, 2.0)

        donors = population + F_escape * (base_direction + mean_direction + opp_direction)

        # Clamp to bounds and return
        return np.clip(donors, self.lower, self.upper)
```

# --- From variant_04_idea_0.py (10 wins) ---
```python
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
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _mutate_variant09(self, population, fitness):
        """Variant 09: Centroid Repulsion — escape local optima by mutating away from population center"""
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

        # Compute centroid (center of mass of current population)
        centroid = np.mean(population, axis=0)

        # Compute search space diameter for scale normalization
        space_diameter = np.linalg.norm(self.upper - self.lower) + 1e-10

        # Distance from centroid to best individual (normalized)
        centroid_to_best = self.best_solution - centroid
        dist_centroid_best = np.linalg.norm(centroid_to_best) + 1e-10

        # Anti-centroid: reflect best through centroid to explore opposite side
        # anti_centroid = centroid - (best - centroid) = 2*centroid - best
        anti_centroid = 2.0 * centroid - self.best_solution

        # Stagnation-driven scale: larger when stuck, to escape basin of attraction
        stagnation = float(self.stagnation_count)
        repulsion_scale = 1.5 + min(stagnation * 0.05, 2.0)  # range [1.5, 3.5]
        repulsion_scale = min(repulsion_scale, 3.5)

        # Compute repulsion vector: move FROM centroid toward anti-centroid
        # The farther the best is from centroid, the stronger the repulsion
        repulsion_direction = anti_centroid - centroid
        repulsion_norm = np.linalg.norm(repulsion_direction) + 1e-10
        unit_repulsion = repulsion_direction / repulsion_norm

        # Target = centroid + scaled repulsion vector
        target = centroid + unit_repulsion * dist_centroid_best * repulsion_scale

        # Clip target to bounds (edge case: handle out-of-bounds reflection)
        target = np.clip(target, self.lower, self.upper)

        # Large perturbation scale for exploration
        F_large = np.clip(0.8 + min(stagnation * 0.03, 1.0), 0.8, 2.0)
        F_secondary = np.clip(self.F + 0.2, 0.5, 2.0)

        # Primary mutation: move toward anti-centroid (away from best)
        donors = population + F_large * (target - population) + F_secondary * (population[r1] - population[r2])

        # Additional diversity injection when stagnant
        if stagnation > 10:
            diversity_noise = np.random.randn(self.NP, self.dim)
            diversity_scale = min(0.2 * np.log1p(stagnation), 2.0)
            donors = donors + diversity_scale * diversity_noise

        return np.clip(donors, self.lower, self.upper)
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _mutate_variant09(self, population, fitness):
        """Variant 09: Opposition-Based DE with centroid reflection and restart injection"""
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

        # Standard DE mutation
        base_mutants = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Opposition learning: reflect around centroid
        centroid = np.mean(population, axis=0)
        lower_arr = np.array(self.lower)
        upper_arr = np.array(self.upper)
        range_arr = upper_arr - lower_arr

        # Opposition candidates (mirrored across search space)
        opposition = lower_arr + upper_arr - base_mutants

        # Blend DE mutant with opposition candidate
        alpha = np.random.rand(self.NP, 1)
        donors = alpha * base_mutants + (1 - alpha) * opposition

        # Inject random restart when severely stagnant
        if self.stagnation_count > 20:
            n_inject = max(1, self.NP // 4)
            inject_idx = np.random.randint(0, self.NP, size=n_inject)
            inject = np.random.uniform(self.lower, self.upper, (n_inject, self.dim))
            donors[inject_idx] = inject

        # Dimension-wise perturbation for escaping local traps
        if self.stagnation_count > 10:
            perturb_mask = np.random.rand(self.NP, self.dim) < 0.3
            perturb = np.random.randn(self.NP, self.dim) * 0.5 * (self.stagnation_count / 50.0)
            donors = donors + perturb_mask * perturb

        return np.clip(donors, self.lower, self.upper)
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _mutate_variant09(self, population, fitness):
        """Variant 09: Opposition-guided mutation with restart and niching"""
        pop_dim = population.shape

        # Check if we need a restart (severe stagnation or trapped)
        needs_restart = (self.stagnation_count > 40 or 
                         np.min(fitness) > 1e2)  # Stuck in bad region

        if needs_restart and self.best_solution is not None:
            # Generate opposition population
            opp_pop = self.lower + self.upper - population

            # Blend best solution with random direction to escape basin
            escape_dir = np.random.randn(self.NP, self.dim)
            escape_dir = escape_dir / (np.linalg.norm(escape_dir, axis=1, keepdims=True) + 1e-10)
            escape_scale = np.random.uniform(10, 50, size=(self.NP, 1))
            escape_pop = np.clip(self.best_solution + escape_dir * escape_scale, self.lower, self.upper)

            # Mix: keep some current, some opposition, some escape
            selector = np.random.randint(0, 3, size=self.NP)
            restart_pop = np.where(selector[:, None] == 0, population,
                                  np.where(selector[:, None] == 1, opp_pop, escape_pop))

            # Fitness-based niche selection: keep diverse solutions
            sorted_idx = np.argsort(fitness)
            n_keep = max(5, self.NP // 4)
            keep_indices = sorted_idx[:n_keep]

            # Fill rest with opposition/escape and random
            fill_pop = np.vstack([
                restart_pop[keep_indices],
                self.lower + np.random.rand(self.NP - n_keep, self.dim) * (self.upper - self.lower)
            ])
            population = np.clip(fill_pop, self.lower, self.upper)
            fitness = np.full(self.NP, np.inf)
            self.stagnation_count = 0

        # Main mutation: current-to-niche-best with opposition learning
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        # Find fitness niche: select from similar-fitness region
        best_fitness_val = fitness[sorted_idx[0]]
        niche_mask = fitness < (best_fitness_val * 1.5 + 1.0)
        niche_indices = np.where(niche_mask)[0]
        if len(niche_indices) < 2:
            niche_indices = best_indices

        # p_best from niche
        p_best_idx = niche_indices[np.random.randint(0, len(niche_indices), size=self.NP)]
        p_best = population[p_best_idx]

        # Random indices with opposition bias
        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        # Opposition vectors: generate opposite of random individuals
        opp_r1 = self.lower + self.upper - population[r1]
        opp_r2 = self.lower + self.upper - population[r2]

        # Adaptive F with higher values for exploration
        if self.stagnation_count > 10:
            F_adapt = np.clip(self.F * (1.0 + 0.3 * np.log1p(self.stagnation_count)), 0.5, 2.5)
        else:
            F_adapt = np.clip(self.F, 0.5, 1.5)

        # Mutation with opposition and niche guidance
        base_mutation = p_best - population
        diff_opp = opp_r1 - opp_r2
        diff_current = population[r1] - population[r2]

        # Blend: 60% niche-guided, 40% opposition-based difference
        donors = population + F_adapt * base_mutation + 0.5 * F_adapt * (0.6 * diff_current + 0.4 * diff_opp)

        # Add small opposition perturbation to some individuals
        opp_perturb = np.random.rand(self.NP, self.dim) < 0.2
        perturb_scale = np.random.uniform(1, 10, size=(self.NP, self.dim))
        opp_offset = (self.lower + self.upper - donors) * opp_perturb * perturb_scale * 0.1
        donors = donors + opp_offset

        return np.clip(donors, self.lower, self.upper)
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