Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_mutate_original` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            
1      2.967831e-01            3.241690e-01            3.354342e-01            3.210596e-01            3.163415e-01            3.248423e-01            3.020910e-01            3.104310e-01            3.094552e-01            -inf                    3.056220e-01            
2      1.281306e-08            2.011832e-06            5.041594e-06            9.699072e-05            4.494207e-04            3.111049e-01            6.360139e-08            2.925565e-08            1.817825e-05            -inf                    1.659374e-08            
3      1.468767e-08            4.526173e-08            3.337150e-08            5.625029e-07            9.225188e-08            1.371468e-07            4.859560e-08            1.122616e-08            4.181612e-08            -inf                    1.805775e-08            
4      1.252182e+00            1.216853e+00            1.250306e+00            1.305995e+00            1.335968e+00            1.271334e+00            1.198602e+00            1.166004e+00            1.249588e+00            -inf                    1.180103e+00            
5      1.385913e+00            1.412975e+00            1.367421e+00            1.446211e+00            1.454110e+00            1.591848e+00            1.471609e+00            1.466333e+00            1.523034e+00            -inf                    1.366570e+00            
6      1.179020e-08            5.382230e-08            1.512199e-07            7.650547e-07            6.152148e-07            6.801804e-07            1.272084e-07            1.143742e-08            1.100706e-06            -inf                    9.341861e-08            
7      1.000000e-08            1.371442e-07            7.905038e-07            1.296524e-06            2.873445e-07            8.674064e-07            4.526636e-08            1.000000e-08            4.513795e-07            -inf                    4.929045e-08            
8      1.554537e+02            1.814810e+02            1.644161e+02            3.815329e+02            4.137697e+02            6.215381e+02            1.634794e+02            1.760312e+02            1.313697e+02            -inf                    2.400764e+02            
9      1.265943e-08            2.321066e-07            3.263891e-06            1.371707e-06            1.631794e-06            1.099904e-06            1.470564e-07            2.073560e-08            9.724379e-07            -inf                    3.482064e-07            
10     4.596390e-08            3.797882e-07            7.384104e-06            1.748550e-06            9.942048e-07            3.087990e-06            3.567856e-07            4.603129e-08            1.323010e-06            -inf                    7.662762e-07            
11     5.801250e-08            1.282033e-06            2.270886e-06            1.914910e-06            1.670385e-06            2.264890e-06            3.713872e-07            9.466488e-08            1.152847e-06            -inf                    6.363910e-07            
12     1.025388e+03            1.353573e+03            5.045737e+02            1.723341e+03            1.554036e+03            1.377244e+03            3.194486e+02            8.631746e+02            6.911890e+02            -inf                    9.612354e+02            
13     1.638204e+01            2.739200e+01            2.088401e+01            6.495055e+01            2.853869e+01            3.847935e+01            1.452931e+01            9.777935e+00            2.687196e+01            -inf                    3.338639e+01            
14     5.227158e+00            5.547821e+00            5.639340e+00            5.770460e+00            5.855832e+00            5.711092e+00            5.269800e+00            5.523172e+00            5.229387e+00            -inf                    5.219805e+00            
15     4.000000e-08            4.000000e-08            5.456251e+02            2.000000e-08            2.000000e-08            5.899305e+02            1.000000e-08            4.000000e-08            2.000000e-08            -inf                    5.413824e+02            
16     6.025624e+02            5.739237e+02            6.015765e+02            6.526214e+02            6.197990e+02            6.244991e+02            5.962677e+02            5.522655e+02            5.993723e+02            -inf                    5.995921e+02            
17     5.438415e-08            2.060202e-08            5.413824e+02            3.996403e-08            5.537009e+02            4.196171e-05            4.930516e-07            1.333333e-08            2.994189e-07            -inf                    2.324064e-08            
18     3.114140e-08            5.581397e+02            8.857154e-07            1.344447e-05            1.953533e-05            6.821000e+02            1.066717e-05            2.539885e-06            5.759675e+02            -inf                    7.419589e-06            
19     2.211467e-03            5.511453e-03            1.330109e-02            1.430552e-02            1.525494e-02            1.092801e-02            8.060029e-03            2.938141e-03            1.234023e-02            -inf                    5.379198e-03            
20     5.000009e+00            5.000067e+00            5.000814e+00            5.001656e+00            5.003135e+00            5.000000e+00            5.000309e+00            5.000240e+00            5.002358e+00            -inf                    5.000000e+00            
21     5.000252e+01            5.000916e+01            5.000273e+01            5.006639e+01            5.000463e+01            5.001462e+01            5.001724e+01            5.002988e+01            5.002231e+01            -inf                    5.001482e+01            
22     1.633454e-05            1.220041e-04            5.552740e-04            1.869463e-04            1.765595e-04            1.625706e-04            1.325582e-04            2.919156e-05            1.292369e-04            -inf                    8.656952e-05            
23     2.308197e+01            2.378763e+01            1.653324e+01            2.253776e+01            2.645418e+01            2.255322e+01            1.860939e+01            2.379111e+01            2.700449e+01            -inf                    2.368887e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: original.py  (error=2.967831e-01)
Task  2: original.py  (error=1.281306e-08)
Task  3: SKIPPED (trivial)
Task  4: variant_07_idea_0.py  (error=1.166004e+00)
Task  5: variant_10_idea_0.py  (error=1.366570e+00)
Task  6: variant_07_idea_0.py  (error=1.143742e-08)
Task  7: original.py  (error=1.000000e-08)
Task  8: variant_08_idea_0.py  (error=1.313697e+02)
Task  9: original.py  (error=1.265943e-08)
Task 10: original.py  (error=4.596390e-08)
Task 11: original.py  (error=5.801250e-08)
Task 12: variant_06_idea_0.py  (error=3.194486e+02)
Task 13: variant_07_idea_0.py  (error=9.777935e+00)
Task 14: variant_10_idea_0.py  (error=5.219805e+00)
Task 15: SKIPPED (trivial)
Task 16: variant_07_idea_0.py  (error=5.522655e+02)
Task 17: variant_07_idea_0.py  (error=1.333333e-08)
Task 18: original.py  (error=3.114140e-08)
Task 19: original.py  (error=2.211467e-03)
Task 20: variant_05_idea_0.py  (error=5.000000e+00)
Task 21: original.py  (error=5.000252e+01)
Task 22: original.py  (error=1.633454e-05)
Task 23: variant_02_idea_0.py  (error=1.653324e+01)

WIN COUNTS:
  original.py: 10 wins
  variant_07_idea_0.py: 5 wins
  variant_10_idea_0.py: 2 wins
  variant_08_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins
  variant_02_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (1 wins) ---
```python
def _mutate_original(self, population, fitness):
        """Historical Best-Guided Mutation with Convergence-Aware Exploration"""
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

        # Historical best guidance (NEW: uses global best from ALL generations)
        if self.best_solution is not None:
            hist_best = self.best_solution
            # Compute distance-based weights: closer individuals get stronger pull
            hist_dist = np.linalg.norm(population - hist_best, axis=1) + 1e-10
            hist_weights = 1.0 / hist_dist
            hist_weights = hist_weights / (np.sum(hist_weights) + 1e-10)
            # Direction toward historical best, weighted by convergence level
            pop_center = np.mean(population, axis=0)
            convergence = np.linalg.norm(population - pop_center, axis=1) / (np.linalg.norm(hist_best - pop_center) + 1e-10)
            convergence = np.clip(convergence, 0, 1)
            # Stronger pull when more converged (population clustered away from best)
            hist_strength = 0.3 * (1.0 - convergence)
            hist_direction = (hist_best - population)
            base = base + hist_strength[:, np.newaxis] * hist_direction

        # Convergence-aware perturbation (NEW: escapes local optima)
        pop_center = np.mean(population, axis=0)
        pop_dispersion = np.linalg.norm(population - pop_center, axis=1)
        avg_dispersion = np.mean(pop_dispersion) + 1e-10
        normalized_dispersion = pop_dispersion / avg_dispersion

        # When dispersion is low (population clustered), add stronger perturbation
        low_disp_mask = normalized_dispersion < 0.5
        if np.any(low_disp_mask):
            perturbation_scale = 0.5 * self.F * (1.0 - normalized_dispersion[low_disp_mask])
            perturbation = np.random.randn(np.sum(low_disp_mask), self.dim) * perturbation_scale[:, np.newaxis]
            base[low_disp_mask] = base[low_disp_mask] + perturbation

        return np.clip(base, self.lower, self.upper)
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _mutate_original(self, population, fitness):
        """Opposition-Guided DE with Adaptive Reflection and Quasi-Opposition"""
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

        # Standard base mutation
        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Opposition-based learning: reflect solutions across search space center
        # Adaptive reflection probability (high at start, decays over generations)
        decay_rate = 0.01
        reflect_prob = max(0.05, 0.8 * np.exp(-decay_rate * self.generation))

        # Search space center
        center = (self.lower + self.upper) / 2.0

        # Generate opposite candidates (full reflection)
        opposite_full = 2.0 * center - base

        # Generate quasi-opposition candidates (between center and opposite)
        quasi_opposite = np.random.uniform(center, opposite_full)

        # Generate quasi-reflected (between center and base)
        quasi_reflected = np.random.uniform(center, base)

        # Dimension-wise random walk for additional exploration
        dim_perturb = np.random.randn(self.NP, self.dim) * (0.1 + 0.3 * np.exp(-0.005 * self.generation))

        # Decide which candidates to use per individual
        rand_mask = np.random.rand(self.NP, self.dim)

        # Apply different strategies to different dimensions for diversity
        donors = base.copy()
        reflect_mask = np.random.rand(self.NP) < reflect_prob
        n_reflect = int(np.sum(reflect_mask))

        if n_reflect > 0:
            reflect_indices = np.where(reflect_mask)[0]
            # Half use full opposition, half use quasi-opposition
            half = n_reflect // 2
            full_opp_indices = reflect_indices[:half]
            quasi_opp_indices = reflect_indices[half:]

            donors[full_opp_indices] = opposite_full[full_opp_indices]
            donors[quasi_opp_indices] = quasi_opposite[quasi_opp_indices]

        # Apply dimension-wise perturbation to non-reflected individuals
        non_reflect_indices = np.where(~reflect_mask)[0]
        if len(non_reflect_indices) > 0:
            donors[non_reflect_indices] = donors[non_reflect_indices] + dim_perturb[non_reflect_indices]
            # Mix with quasi-reflected for some individuals
            mix_mask = np.random.rand(len(non_reflect_indices)) < 0.3
            mix_indices = non_reflect_indices[mix_mask]
            if len(mix_indices) > 0:
                donors[mix_indices] = 0.7 * donors[mix_indices] + 0.3 * quasi_reflected[mix_indices]

        # Blend with cultural memory if available
        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * donors + 0.15 * (population + cultural_offset)

        return np.clip(donors, self.lower, self.upper)
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _mutate_original(self, population, fitness):
        """Opposition-Guided Escape: basin-hopping via dynamic opposition vectors"""
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

        # Compute centroid for opposition reference
        centroid = np.mean(population, axis=0)

        # Adaptive F: larger when stagnant to escape basins
        if self.stagnation_count > 10:
            F_escape = min(self.F * (1.0 + 0.5 * np.log1p(self.stagnation_count)), 2.5)
        else:
            F_escape = self.F

        # Standard mutation direction (exploitation)
        mutation_dir = p_best - population + F_escape * (population[r1] - population[r2])

        # Opposition direction: reflect around centroid (exploration)
        # Opposition of x is x_opp = centroid + (centroid - x) = 2*centroid - x
        opposition_vector = 2.0 * centroid - population
        opposition_dir = opposition_vector - population + F_escape * (population[r2] - population[r1])

        # Adaptive selection: more opposition when stagnant
        # Start conservative, increase opposition probability with stagnation
        base_opp_prob = 0.1
        stagnation_boost = min(0.6, 0.05 * self.stagnation_count)
        opp_prob = np.clip(base_opp_prob + stagnation_boost, 0.0, 0.7)

        use_opposition = np.random.rand(self.NP) < opp_prob

        # Compute donors for both directions
        donor_exploit = population + mutation_dir
        donor_explore = population + opposition_dir

        # Select based on adaptive probability
        donors = np.where(use_opposition[:, np.newaxis], donor_explore, donor_exploit)

        # Small perturbation when very stagnant (adds randomization for escape)
        if self.stagnation_count > 20:
            perturbation_scale = min(0.5 * np.log1p(self.stagnation_count - 20), 5.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            donors = donors + random_perturbation

        return np.clip(donors, self.lower, self.upper)
```

# --- From variant_07_idea_0.py (5 wins) ---
```python
def _mutate_original(self, population, fitness):
        """Restart-based mutation: aggressive escape from local optima via random restarts"""
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

        if self.stagnation_count > 20:
            restart_prob = min(0.50 + 0.05 * (self.stagnation_count - 20), 0.95)
            restart_mask = np.random.rand(self.NP) < restart_prob
            n_restart = int(np.sum(restart_mask))
            if n_restart > 0:
                restart_vectors = np.random.uniform(self.lower, self.upper, (n_restart, self.dim))
                base[restart_mask] = restart_vectors

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
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _mutate_original(self, population, fitness):
        """Reflective Diversification: escape local optima by bouncing away from population centroid"""
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

        # Compute population centroid and direction vectors from centroid
        centroid = np.mean(population, axis=0)
        diff_from_centroid = population - centroid
        dist_from_centroid = np.linalg.norm(diff_from_centroid, axis=1, keepdims=True) + 1e-10
        direction_from_centroid = diff_from_centroid / dist_from_centroid

        # Reflected direction (opposite to cluster direction)
        reflected_direction = -direction_from_centroid

        # Compute reflection strength based on stagnation and convergence
        convergence_ratio = np.std(population, axis=0) / (self.upper - self.lower + 1e-10)
        avg_convergence = np.mean(convergence_ratio)

        if self.stagnation_count > 10 or avg_convergence < 0.05:
            # Strong reflection when stuck or highly converged
            reflection_strength = min(0.5, 0.25 + 0.025 * self.stagnation_count)
        elif self.stagnation_count > 3:
            # Moderate reflection when starting to stagnate
            reflection_strength = 0.15 + 0.02 * self.stagnation_count
        else:
            reflection_strength = 0.0

        if reflection_strength > 0:
            # Scale jump based on search space and distance from centroid
            jump_scale = (self.upper - self.lower) * 0.15
            reflected_points = centroid + reflected_direction * jump_scale
            donors = (1 - reflection_strength) * base + reflection_strength * reflected_points
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _mutate_original(self, population, fitness):
        """Restart-Driven Opposition-Based Mutation: escape local optima via reflection"""
        bounds_center = (self.lower + self.upper) / 2.0

        # Detect deep stagnation: stuck far from target
        is_stuck = self.stagnation_count > 15

        if is_stuck:
            # Opposition-based exploration: reflect population across bounds center
            # This jumps to the opposite side of the search space
            opposition_pop = 2.0 * bounds_center - population
            opposition_pop = np.clip(opposition_pop, self.lower, self.upper)

            # Mutate from opposition points instead of current population
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

            # Mutation from OPPOSITION population with amplified F
            F_oppose = min(self.F * 1.5, 2.0)
            opp_base = opposition_pop + F_oppose * (p_best - opposition_pop) + F_oppose * (population[r1] - population[r2])

            # Blend: 60% opposition-mutated, 40% standard mutation from current
            F_std = self.F
            std_base = population + F_std * (p_best - population) + F_std * (population[r1] - population[r2])

            donors = 0.6 * opp_base + 0.4 * std_base
            return np.clip(donors, self.lower, self.upper)

        # Standard mutation (same as original for non-stuck state)
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