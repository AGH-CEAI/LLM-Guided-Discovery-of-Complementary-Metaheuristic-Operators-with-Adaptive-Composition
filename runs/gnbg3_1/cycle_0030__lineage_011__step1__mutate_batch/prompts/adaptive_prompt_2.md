Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_mutate_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            6.059212e-05            6.421560e+00            -inf                    5.091016e+01            -inf                    4.897218e+00            3.104189e-08            6.703423e+01            1.450268e-06            1.217231e-06            
1      2.043329e+00            4.717760e-01            2.966793e+00            -inf                    1.382677e+02            -inf                    1.814725e+01            1.166227e-03            1.359447e+02            9.699020e-02            1.316910e+02            
2      1.000000e-08            6.871660e-04            1.658793e+01            -inf                    4.913239e+02            -inf                    4.205085e+01            8.932702e-06            5.868584e+02            1.740962e-05            4.996486e+02            
3      2.929132e+00            2.453443e+00            1.805901e+01            -inf                    2.588817e+02            -inf                    8.726664e+01            2.293823e+00            2.658088e+02            2.108943e+00            2.523114e+02            
4      9.657071e-02            1.756614e-01            1.799772e+00            -inf                    4.927396e+00            -inf                    3.336667e+00            5.475169e-02            5.310885e+00            1.028081e-01            5.290883e-01            
5      1.342840e+02            1.696219e+01            6.553810e+03            -inf                    9.808691e+08            -inf                    1.036404e+07            5.927262e+01            2.152322e+09            1.247347e+01            1.251092e+09            
6      2.825724e+01            3.786234e+01            6.136826e+02            -inf                    3.376227e+03            -inf                    1.769607e+03            3.072392e+01            6.963770e+03            3.297309e+01            5.260121e+03            
7      3.244975e+00            2.109467e+00            1.179580e+01            -inf                    3.100786e+02            -inf                    8.718978e+01            2.981565e+00            3.388146e+02            5.215588e+00            2.961531e+02            
8      8.358741e-01            1.512754e+00            6.010097e+00            -inf                    4.316119e+01            -inf                    1.645755e+01            1.285245e-01            4.715753e+01            3.228489e-01            2.130665e+00            
9      8.591809e+00            7.865182e+00            4.884621e+01            -inf                    1.580577e+02            -inf                    4.142996e+02            8.570147e+00            3.649332e+02            8.066583e+00            2.779786e+02            
10     5.245463e-02            2.715878e-02            2.360102e+02            -inf                    1.153748e+02            -inf                    5.560444e-01            1.166087e-04            2.969450e+02            1.106871e-04            2.878793e+02            
11     2.155157e+02            7.936606e+02            9.991132e+02            -inf                    4.397727e+02            -inf                    1.830116e+03            3.303010e+01            1.306768e+03            2.182715e+01            1.369845e+03            
12     5.573247e-02            1.469102e+00            1.773082e+02            -inf                    4.243010e+01            -inf                    3.204801e+00            8.737248e-04            2.043001e+02            8.220374e-04            2.075901e+02            
13     2.379330e+01            5.095833e+00            4.806223e+01            -inf                    3.286955e+01            -inf                    1.742654e+01            2.683066e+00            4.816098e+01            3.966225e+00            2.646256e+01            
14     4.087047e+00            1.658781e+01            6.404767e+00            -inf                    3.383554e+00            -inf                    2.195381e+01            9.805063e+00            1.813595e+01            1.109632e+01            4.182729e+00            
15     3.334464e+00            4.058125e+00            4.008101e+00            -inf                    3.551687e+00            -inf                    4.462534e+00            2.584492e+00            4.282461e+00            2.166626e+00            3.166207e+00            
16     4.368825e+03            1.857277e+03            1.515445e+04            -inf                    1.066062e+04            -inf                    4.094022e+04            1.304408e+03            2.464874e+04            5.270509e+02            1.737027e+04            
17     3.742525e+04            2.270455e+03            2.634667e+05            -inf                    7.555722e+04            -inf                    8.249085e+03            3.462031e+02            1.968803e+05            1.753715e+03            2.390797e+05            
18     4.092172e+01            3.016554e+01            8.607188e+01            -inf                    5.774182e+01            -inf                    2.766105e+01            1.620901e+01            8.159503e+01            2.592206e+01            3.989959e+01            
19     6.321246e+01            4.220661e+02            2.252600e+02            -inf                    9.664392e+01            -inf                    5.955518e+02            3.237361e+02            4.824417e+02            1.999644e+02            4.452918e+02            
20     2.121709e+01            2.553967e+01            4.568554e+01            -inf                    2.194210e+01            -inf                    6.728673e+01            1.692127e+01            5.967654e+01            1.745096e+01            2.103809e+01            
21     4.739355e+00            5.796760e+00            5.876472e+00            -inf                    5.702047e+00            -inf                    6.006465e+00            5.157240e+00            5.916530e+00            5.384616e+00            4.698278e+00            
22     1.219917e+01            2.037750e+01            1.735212e+01            -inf                    1.354180e+01            -inf                    2.395330e+01            1.815550e+01            2.173765e+01            1.500024e+01            1.131067e+01            
23     5.242432e+01            8.159832e+01            5.491082e+01            -inf                    4.137700e+01            -inf                    1.659652e+02            5.514728e+01            1.338597e+02            4.851446e+01            1.130095e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.000000e-08)
Task  1: variant_07_idea_0.py  (error=1.166227e-03)
Task  2: original.py  (error=1.000000e-08)
Task  3: variant_09_idea_0.py  (error=2.108943e+00)
Task  4: variant_07_idea_0.py  (error=5.475169e-02)
Task  5: variant_09_idea_0.py  (error=1.247347e+01)
Task  6: original.py  (error=2.825724e+01)
Task  7: variant_01_idea_0.py  (error=2.109467e+00)
Task  8: variant_07_idea_0.py  (error=1.285245e-01)
Task  9: variant_01_idea_0.py  (error=7.865182e+00)
Task 10: variant_09_idea_0.py  (error=1.106871e-04)
Task 11: variant_09_idea_0.py  (error=2.182715e+01)
Task 12: variant_09_idea_0.py  (error=8.220374e-04)
Task 13: variant_07_idea_0.py  (error=2.683066e+00)
Task 14: variant_04_idea_0.py  (error=3.383554e+00)
Task 15: variant_09_idea_0.py  (error=2.166626e+00)
Task 16: variant_09_idea_0.py  (error=5.270509e+02)
Task 17: variant_07_idea_0.py  (error=3.462031e+02)
Task 18: variant_07_idea_0.py  (error=1.620901e+01)
Task 19: original.py  (error=6.321246e+01)
Task 20: variant_07_idea_0.py  (error=1.692127e+01)
Task 21: variant_10_idea_0.py  (error=4.698278e+00)
Task 22: variant_10_idea_0.py  (error=1.131067e+01)
Task 23: variant_04_idea_0.py  (error=4.137700e+01)

WIN COUNTS:
  variant_07_idea_0.py: 7 wins
  variant_09_idea_0.py: 7 wins
  original.py: 4 wins
  variant_01_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  variant_10_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _mutate_batch(self, population, fitness, selected_operator):
            """Generate mutant vectors using current-to-rand/1 with optional archive."""
            np_pop = len(population)
            dim = self.dim

            # Vectorized index selection using np.random.choice on flattened indices
            all_indices = np.arange(np_pop)

            # Efficiently sample 4 unique indices per individual (avoiding current index i)
            # Use broadcasting trick: create (np_pop, np_pop) difference matrix approach
            r = np.random.randint(0, np_pop, size=(np_pop, 4))

            # Ensure uniqueness per row (simple rejection for small populations)
            for i in range(np_pop):
                while len(np.unique(r[i])) < 4:
                    r[i] = np.random.randint(0, np_pop, size=4)

            r1, r2, r3, r4 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

            # Build archive from current population (union with existing if applicable)
            if self.archive is None:
                self.archive = population.copy()
            else:
                # Keep archive bounded
                if len(self.archive) > self.archive_max_size:
                    self.archive = self.archive[np.random.choice(
                        len(self.archive), self.archive_max_size, replace=False)]
                self.archive = np.vstack([self.archive, population])
                if len(self.archive) > self.archive_max_size * 2:
                    self.archive = self.archive[np.random.choice(
                        len(self.archive), self.archive_max_size, replace=False)]

            # Sample from archive for additional diversity (with fallback)
            archive_size = len(self.archive)
            if archive_size > np_pop:
                r_archive = np.random.randint(0, archive_size, size=np_pop)
                archive_donors = self.archive[r_archive]
            else:
                archive_donors = population  # Fallback to population if archive small

            # Get pbest indices (top p% where p varies per strategy)
            sorted_idx = np.argsort(fitness)
            p = 0.1 + 0.1 * selected_operator  # Vary pbest percentage by operator
            pbest_count = max(1, int(np_pop * p))
            pbest_idx = sorted_idx[:pbest_count]

            # Strategy-specific mutation with current-to-rand/1 as primary
            mutants = np.empty_like(population)

            # Adaptive F: use success history if available, otherwise base value
            f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
            f_values = np.clip(f_values, 0.1, 2.0)

            if selected_operator == 0:  # Current-to-rand/1 (rotationally invariant)
                # m = x_i + F*(x_r1 - x_i) + F*(x_r2 - x_r3)
                mutants = (population + f_values[:, np.newaxis] * 
                          (population[r1] - population + population[r2] - population[r3]))

            elif selected_operator == 1:  # Best/2 with archive
                best_idx = sorted_idx[0]
                mutants = (population[best_idx] + f_values[:, np.newaxis] * 
                          (population[r1] - population[r2] + archive_donors - population[r3]))

            elif selected_operator == 2:  # Rand/1 with pbest injection
                pbest = population[np.random.choice(pbest_idx, np_pop)]
                mutants = (population[r1] + f_values[:, np.newaxis] * 
                          (pbest - population[r2] + population[r3] - population[r4]))

            else:  # Current-to-pbest/1 with archive
                pbest = population[np.random.choice(pbest_idx, np_pop)]
                mutants = (population + f_values[:, np.newaxis] * 
                          (pbest - population[r1] + archive_donors - population[r2]))

            # Clip to bounds
            mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

            return mutants
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using covariance-adaptive mutation with archive."""
        np_pop = len(population)
        dim = self.dim

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Maintain archive of best solutions
        if not hasattr(self, 'mutation_archive'):
            self.mutation_archive = population[fitness.argsort()[:min(10, np_pop)]].copy()
            self._archive_fitness = np.sort(fitness)[:min(10, np_pop)]

        # Update archive with current best solutions
        n_archive = len(self.mutation_archive)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < self._archive_fitness[0]:
            if n_archive < 50:
                self.mutation_archive = np.vstack([self.mutation_archive, population[current_best_idx]])
                self._archive_fitness = np.append(self._archive_fitness, fitness[current_best_idx])
            else:
                worst_arch_idx = np.argmax(self._archive_fitness)
                self.mutation_archive[worst_arch_idx] = population[current_best_idx]
                self._archive_fitness[worst_arch_idx] = fitness[current_best_idx]

        # Compute covariance matrix of population for adaptive direction
        mean_pop = np.mean(population, axis=0)
        centered = population - mean_pop
        cov_matrix = np.cov(centered.T)

        # Regularize for numerical stability
        cov_matrix += np.eye(dim) * 1e-8

        # Eigendecomposition with fallback
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(dim)
            eigenvectors = np.eye(dim)

        # Normalize eigenvalues for scaling
        eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-12)
        sqrt_eigen = np.sqrt(eigenvalues_norm)

        # Sample random indices for base vectors
        r1 = np.random.randint(0, np_pop, size=np_pop)
        r2 = np.random.randint(0, n_archive, size=np_pop)

        # Adaptive F parameter
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)

        # Strategy 0: Covariance-adaptive with archive guidance
        if selected_operator == 0:
            base_pop = population[r1]
            target_archive = self.mutation_archive[r2]

            # Compute scaled direction using covariance eigendirections
            diff = target_archive - base_pop
            scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T

            # Add correlated noise along principal axes
            noise = np.random.randn(np_pop, dim)
            noise = noise @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
            noise *= 0.1 * f_values[:, np.newaxis]

            mutants = base_pop + f_values[:, np.newaxis] * scaled_diff + noise

        # Strategy 1: Archive-driven exploration with population diversity
        elif selected_operator == 1:
            r3 = np.random.randint(0, np_pop, size=np_pop)
            base_pop = population[r1]
            archive_pop = self.mutation_archive[r2]
            third_pop = population[r3]

            # Weighted combination emphasizing archive
            archive_weight = 0.7
            mutants = (archive_pop * archive_weight + 
                      base_pop * (1 - archive_weight) +
                      f_values[:, np.newaxis] * (third_pop - base_pop))

        # Strategy 2: Dimension-wise adaptive mutation
        elif selected_operator == 2:
            r3 = np.random.randint(0, np_pop, size=np_pop)
            base_pop = population[r1]
            archive_pop = self.mutation_archive[r2]
            third_pop = population[r3]

            # Per-dimension eigenvalue-weighted mutation
            dim_weights = sqrt_eigen / (np.sum(sqrt_eigen) + 1e-12)
            dim_weights = dim_weights / (np.max(dim_weights) + 1e-12)

            mutants = (base_pop + 
                      f_values[:, np.newaxis] * dim_weights[np.newaxis, :] * (archive_pop - third_pop))

        # Strategy 3: Covariance-based jumping with archive exploitation
        else:
            r3 = np.random.randint(0, np_pop, size=np_pop)
            base_pop = population[r1]
            archive_pop = self.mutation_archive[r2]
            third_pop = population[r3]

            # Large covariance jump toward archive
            diff = archive_pop - base_pop
            scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T

            # Secondary perturbation
            perp_diff = third_pop - base_pop

            mutants = (base_pop + 
                      f_values[:, np.newaxis] * scaled_diff +
                      0.3 * f_values[:, np.newaxis] * perp_diff)

        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
```

# --- From variant_07_idea_0.py (7 wins) ---
```python
def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using multi-strategy ensemble with fitness-weighted selection."""
        np_pop = len(population)

        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)

        # Get sorted indices for various selection schemes
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
        worst_idx = sorted_idx[-1]

        # Compute fitness landscape properties for adaptive strategy selection
        f_min, f_max = np.min(fitness), np.max(fitness)
        f_range = max(f_max - f_min, 1e-10)
        f_variance = np.var(fitness)
        best_worst_ratio = (f_min + 1e-10) / (f_max + 1e-10)

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Efficient batch index sampling
        valid_mask = np.ones((np_pop, np_pop), dtype=bool)
        np.fill_diagonal(valid_mask, False)

        for i in range(np_pop):
            valid_mask[i, i] = False

        # Sample r1, r2, r3 for each individual
        r1 = np.array([np.random.choice(np.where(valid_mask[i])[0]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) 
                       for i in range(np_pop)])

        # Adaptive F with dynamic bounds based on landscape
        base_f = self.f_base * (1.0 + 0.5 * np.log1p(f_variance) / (np_pop + 1))
        f_values = base_f + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.05, 2.5)

        # Strategy 0: Random direction (rand/3/bin - different from rand/1)
        mutant_rand = population[r1] + f_values[:, np.newaxis] * (
            population[r2] - population[r3])

        # Strategy 1: Gradient-following (current-to-pbest/2 with better indices)
        pbest = population[np.random.choice(pbest_idx, np_pop)]
        mutant_gradient = population + f_values[:, np.newaxis] * (
            pbest - population[r1] + population[r2] - population[r3])

        # Strategy 2: Opposition-based (toward better, away from worst)
        best_idx = sorted_idx[0]
        mutant_oppose = population[r1] + f_values[:, np.newaxis] * (
            population[best_idx] - population[r2] + population[r3] - population[r1])

        # Strategy 3: Dimension-wise精英 (elite-guided with dimensional diversity)
        elite = population[sorted_idx[:max(1, int(np_pop * 0.05))]]  # Top 5%
        elite_choice = elite[np.random.randint(0, len(elite), size=np_pop)]
        mutant_elite = elite_choice + f_values[:, np.newaxis] * (
            population[r1] - population[r2])

        # Compute strategy selection probabilities based on fitness landscape
        # High variance = exploration needed, low ratio = stuck in local optima
        explore_weight = min(0.5, f_variance / (f_range + 1e-10))
        exploit_weight = 1.0 - explore_weight

        # Probabilities: [rand, gradient, oppose, elite]
        if best_worst_ratio < 0.1:  # Stuck - favor exploration
            probs = np.array([0.35, 0.25, 0.25, 0.15])
        elif best_worst_ratio > 0.5:  # Good progress - balance
            probs = np.array([0.2, 0.35, 0.15, 0.3])
        else:  # Normal - moderate exploration
            probs = np.array([0.3, 0.3, 0.2, 0.2])

        # Normalize probabilities
        probs = probs / probs.sum()

        # Generate random choices for each individual
        strategy_choices = np.random.choice(4, size=np_pop, p=probs)

        # Assemble final mutants based on strategy choices
        for i in range(np_pop):
            if strategy_choices[i] == 0:
                mutants[i] = mutant_rand[i]
            elif strategy_choices[i] == 1:
                mutants[i] = mutant_gradient[i]
            elif strategy_choices[i] == 2:
                mutants[i] = mutant_oppose[i]
            else:
                mutants[i] = mutant_elite[i]

        # Clip to bounds with numerical safety
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        # Handle any NaN/Inf from numerical issues
        nan_mask = np.any(np.isnan(mutants) | np.isinf(mutants), axis=1)
        if np.any(nan_mask):
            mutants[nan_mask] = population[nan_mask] + np.random.uniform(
                -0.1, 0.1, (np.sum(nan_mask), self.dim))
            mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
```

# --- From variant_09_idea_0.py (7 wins) ---
```python
def _mutate_batch(self, population, fitness, selected_operator):
        """Ring-Guided Adaptive Mutation with structured neighborhood exploration."""
        np_pop = len(population)

        # Ring topology indices (now actively used!)
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Sample indices for mutation (excluding current)
        r1 = np.array([np.random.choice(np.delete(np.arange(np_pop), i)) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) for i in range(np_pop)])
        r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) for i in range(np_pop)])

        # Adaptive F values
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.3, 1.5)

        # For each individual, compute ring-guided direction
        for i in range(np_pop):
            # Compare fitness of ring neighbors
            f_curr = fitness[i]
            f_prev = fitness[ring_prev[i]]
            f_next = fitness[ring_next[i]]

            # Direction toward better neighbor (valley climbing)
            if f_prev <= f_next and f_prev < f_curr:
                # Move toward previous neighbor
                ring_dir = population[ring_prev[i]] - population[i]
                ring_weight = 0.5
            elif f_next < f_prev and f_next < f_curr:
                # Move toward next neighbor
                ring_dir = population[ring_next[i]] - population[i]
                ring_weight = 0.5
            else:
                # Current is best among ring - use inter-neighbor direction for exploration
                ring_dir = population[ring_next[i]] - population[ring_prev[i]]
                ring_weight = 0.3

            # Apply mutation based on selected operator
            if selected_operator == 0:  # rand/1 with ring guidance
                base = population[r1[i]]
                de_vec = population[r2[i]] - population[r3[i]]
                mutants[i] = base + f_values[i] * de_vec + ring_weight * ring_dir

            elif selected_operator == 1:  # current-to-ring-best with ring guidance
                base = population[i]
                de_vec = population[r1[i]] - population[r2[i]]
                # Move toward better of two ring neighbors
                if f_prev <= f_next:
                    target = population[ring_prev[i]]
                else:
                    target = population[ring_next[i]]
                mutants[i] = base + f_values[i] * de_vec + 0.7 * (target - base) + 0.3 * ring_dir

            elif selected_operator == 2:  # ring-assisted current-to-pbest
                sorted_idx = np.argsort(fitness)
                pbest_idx = sorted_idx[0]
                base = population[i]
                pbest_vec = population[pbest_idx] - population[i]
                de_vec = population[r1[i]] - population[r2[i]]
                mutants[i] = base + f_values[i] * de_vec + 0.5 * pbest_vec + 0.2 * ring_dir

            else:  # ring-guided rand-to-best
                best_idx = np.argmin(fitness)
                base = population[r1[i]]
                best_vec = population[best_idx] - population[r1[i]]
                de_vec = population[r2[i]] - population[r3[i]]
                mutants[i] = base + f_values[i] * de_vec + 0.4 * best_vec + 0.3 * ring_dir

        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using archive-guided multi-scale mutation."""
        np_pop = len(population)
        dim = population.shape[1]

        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)

        # Get sorted indices by fitness for pbest selection
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Maintain archive of best solutions encountered
        if not hasattr(self, 'mutation_archive') or self.mutation_archive is None:
            self.mutation_archive = population[sorted_idx[:min(10, np_pop)]].copy()
        else:
            # Merge current best solutions into archive
            combined = np.vstack([self.mutation_archive, population[sorted_idx[:5]]])
            combined_fitness = np.array(
                list(np.zeros(len(self.mutation_archive)) * 0.1) + 
                list(fitness[sorted_idx[:5]])
            )
            top_indices = np.argsort(combined_fitness)[:min(15, len(combined))]
            self.mutation_archive = combined[top_indices]

        # Sample indices for mutation
        r1 = np.array([np.delete(np.arange(np_pop), i) for i in range(np_pop)])
        r1 = np.array([np.random.choice(r1[i]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])

        # Archive indices
        archive_size = len(self.mutation_archive)
        arch_idx = np.random.randint(0, archive_size, size=np_pop)

        # Multi-scale F: small, medium, large for different exploration phases
        # Use different scales based on a rotating pattern and adaptation
        if not hasattr(self, 'mutation_scale_mode'):
            self.mutation_scale_mode = 0
        self.mutation_scale_mode = (self.mutation_scale_mode + 1) % 3

        if self.mutation_scale_mode == 0:
            f_values = self.f_base * np.ones(np_pop) * 0.4  # Small scale
        elif self.mutation_scale_mode == 1:
            f_values = self.f_base * np.ones(np_pop) * 0.8  # Medium scale
        else:
            f_values = self.f_base * np.ones(np_pop) * 1.5  # Large scale

        # Add individual variation
        f_values = f_values + 0.1 * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)

        # Apply archive-guided mutation strategy
        if selected_operator == 0:  # Archive-rand/1
            mutants = (self.mutation_archive[arch_idx] + f_values[:, np.newaxis] * 
                      (population[r1] - population[r2]))
        elif selected_operator == 1:  # Archive-to-best/1
            best_idx = np.argmin(fitness)
            mutants = (self.mutation_archive[arch_idx] + f_values[:, np.newaxis] * 
                      (population[best_idx] - self.mutation_archive[arch_idx] + 
                       population[r1] - population[r2]))
        elif selected_operator == 2:  # Current-to-archive-pbest/1
            # Use archive member as pbest
            arch_pbest = self.mutation_archive[arch_idx]
            mutants = (population + f_values[:, np.newaxis] * 
                      (arch_pbest - population[r1] + population[r1] - population[r2]))
        else:  # Archive-rand/2
            arch_idx2 = np.random.randint(0, archive_size, size=np_pop)
            mutants = (self.mutation_archive[arch_idx] + f_values[:, np.newaxis] * 
                      (population[r1] - population[r2] + 
                       population[r2] - self.mutation_archive[arch_idx2]))

        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
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


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Success-Based Operator Pool.
    
    Key features:
    - Pool of DE mutation strategies selected via success history (UCB-based)
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5*dim (balance between exploration and efficiency)
        self.np = min(max(5 * dim, 20), 200)
        
        # Mutation strategies pool
        self.mutation_strategies = ['rand', 'best', 'current_to_pbest', 'rand_to_best']
        self.n_strategies = len(self.mutation_strategies)
        
        # Success history for operator selection (UCB)
        self.operator_rewards = np.zeros(self.n_strategies)
        self.operator_counts = np.ones(self.n_strategies)
        self.ucb_c = 2.0  # Exploration weight for UCB
        
        # DE parameters
        self.cr = 0.9
        self.f_base = 0.5
        self.f_adapt = 0.1
        
        # Local search parameters
        self.local_search_interval = 5  # Apply LS every N generations
        self.local_search_budget = 0.02  # Fraction of func budget for LS
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE (optional, bounded)
        self.archive = None
        self.archive_max_size = self.np
        
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.np, self.dim))
        
        # LHS for better spread
        grid = np.linspace(0, 1, self.np + 1)[:-1] + np.random.uniform(0, 1/self.np, self.np)
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.lower_bound + (self.upper_bound - self.lower_bound) * grid[perm]
        
        # Evaluate initial population
        fitness = func(pop)
        fitness = self._handle_budget_truncation(fitness, pop)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            # Truncate population to match fitness
            actual_len = len(fitness)
            # Keep first `actual_len` individuals
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Select mutation operator using UCB-based operator selection."""
        # UCB formula
        ucb_values = (self.operator_rewards / np.maximum(self.operator_counts, 1) + 
                      self.ucb_c * np.sqrt(np.log(sum(self.operator_counts)) / 
                      np.maximum(self.operator_counts, 1)))
        
        # Select operator for each individual (with small random perturbation)
        selected = np.argmax(ucb_values) + np.random.randint(-1, 2)
        selected = np.clip(selected, 0, self.n_strategies - 1)
        return selected
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using selected DE mutation strategy."""
        np_pop = len(population)
        
        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        
        # Get sorted indices by fitness for pbest selection
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
        
        # Initialize mutant population
        mutants = np.empty_like(population)
        
        # Sample indices for mutation
        r1 = np.array([np.delete(np.arange(np_pop), i) for i in range(np_pop)])
        r1 = np.array([np.random.choice(r1[i]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        
        # Adaptive F with slight individual variation
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)
        
        # Apply selected mutation strategy
        if selected_operator == 0:  # rand/1
            mutants = (population[r1] + f_values[:, np.newaxis] * 
                      (population[r2] - population[np.roll(r1, -1)]))
        elif selected_operator == 1:  # best/1
            best_idx = np.argmin(fitness)
            mutants = (population[best_idx] + f_values[:, np.newaxis] * 
                      (population[r1] - population[r2]))
        elif selected_operator == 2:  # current-to-pbest/1
            pbest = population[np.random.choice(pbest_idx, np_pop)]
            mutants = (population + f_values[:, np.newaxis] * 
                      (pbest - population[r1] + population[r1] - population[r2]))
        else:  # rand-to-best/1
            best_idx = np.argmin(fitness)
            mutants = ((population[r1] + f_values[:, np.newaxis] * 
                       (population[best_idx] - population[r1] + population[r2] - population[r1])))
        
        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        np_pop = len(population)
        
        # Random crossover mask
        cr_mask = np.random.rand(np_pop, self.dim) < self.cr
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        cr_mask[np.arange(np_pop), j_rand] = True
        
        # Create trial vectors
        trials = np.where(cr_mask, mutants, population)
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Selection: greedy replacement based on fitness."""
        # Better fitness wins (assuming minimization)
        better_mask = trial_fitness < fitness
        
        # Update population and fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness, better_mask
    
    def _apply_local_refinement(self, population, fitness, func):
        """Apply Nelder-Mead-like local search to top individuals."""
        if len(population) < self.dim + 1:
            return population, fitness
        
        n_elite = min(3, len(population))  # Apply to top 3
        
        for i in range(n_elite):
            # Get current best position
            x_best = population[i].copy()
            f_best = fitness[i]
            
            # Build simplex around current best
            simplex = np.tile(x_best, (self.dim + 1, 1))
            simplex[0] = x_best
            
            # Add small perturbations for other vertices
            step_size = 0.5
            for j in range(1, self.dim + 1):
                simplex[j] = x_best.copy()
                simplex[j, (j-1) % self.dim] += step_size * (self.upper_bound - self.lower_bound)
                simplex[j] = np.clip(simplex[j], self.lower_bound, self.upper_bound)
            
            # Evaluate simplex
            simplex_fitness = func(simplex)
            simplex_fitness = self._handle_budget_truncation(simplex_fitness, simplex)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            # Simple Nelder-Mead iteration
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            # Sort simplex
            sort_idx = np.argsort(simplex_fitness)
            simplex = simplex[sort_idx]
            simplex_fitness = simplex_fitness[sort_idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
            f_r = func(x_r.reshape(1, -1))[0]
            
            if f_r < simplex_fitness[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                f_e = func(x_e.reshape(1, -1))[0]
                
                if f_e < f_r:
                    new_point, new_f = x_e, f_e
                else:
                    new_point, new_f = x_r, f_r
            elif f_r < simplex_fitness[-2]:
                new_point, new_f = x_r, f_r
            else:
                # Shrink
                new_point = simplex[0] + rho * (simplex[-1] - simplex[0])
                new_point = np.clip(new_point, self.lower_bound, self.upper_bound)
                new_f = func(new_point.reshape(1, -1))[0]
            
            # Update if improvement found
            if new_f < f_best:
                population[i] = new_point
                fitness[i] = new_f
        
        return population, fitness
    
    def _update_operator_rewards(self, selected_operator, success_mask):
        """Update operator rewards based on success rate."""
        success_rate = np.mean(success_mask)
        
        # Reward successful operators
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        # Decay rewards of non-selected operators slightly
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
    
    def _adapt_step_size(self, success_rate):
        """Adapt F and CR based on recent success."""
        if success_rate > 0.2:
            # Good success: maintain or slightly increase exploration
            self.f_adapt = min(0.5, self.f_adapt * 1.1)
        elif success_rate < 0.05:
            # Poor success: reduce step size for exploitation
            self.f_adapt = max(0.01, self.f_adapt * 0.9)
        
        # Adapt CR: increase if stuck, decrease if unstable
        if success_rate > 0.3:
            self.cr = min(0.95, self.cr * 1.01)
        elif success_rate < 0.1:
            self.cr = max(0.5, self.cr * 0.99)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Average of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        avg_distance = np.mean(distances[mask])
        
        return avg_distance
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
        self.best_history.append(current_best)
        
        if len(self.best_history) > self.stagnation_limit:
            self.best_history.pop(0)
        
        if len(self.best_history) >= self.stagnation_limit:
            improvement = self.best_history[-1] - self.best_history[0]
            if abs(improvement) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            return self.stagnation_counter >= 2
        
        return False
    
    def _restart_if_needed(self, population, fitness, func, force=False):
        """Restart population if stagnation or diversity loss detected."""
        diversity = self._compute_diversity(population)
        
        if force or diversity < self.diversity_threshold or self.stagnation_counter >= 2:
            # Reinitialize portion of population
            n_reinit = int(0.7 * self.np)
            reinit_idx = np.argsort(fitness)[-n_reinit:]  # Replace worst
            
            # Generate new individuals using LHS
            for d in range(self.dim):
                grid = np.linspace(0, 1, n_reinit + 1)[:-1] + np.random.uniform(0, 1/n_reinit, n_reinit)
                perm = np.random.permutation(n_reinit)
                population[reinit_idx, d] = (self.lower_bound + 
                    (self.upper_bound - self.lower_bound) * grid[perm])
            
            # Evaluate reinitialized portion
            new_fitness = func(population[reinit_idx])
            if len(new_fitness) < len(reinit_idx):
                new_fitness = np.pad(new_fitness, (0, len(reinit_idx) - len(new_fitness)), 
                                     constant_values=np.inf)
            fitness[reinit_idx] = new_fitness
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            self.best_history = []
            
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Ensure all individuals are within bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        population, fitness = self._initialize_population(func)
        
        # Check budget after initialization
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return fitness[best_idx], population[best_idx]
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator
            selected_operator = self._select_mutation_operator_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip trials to bounds
            trials = self._clip_to_bounds(trials)
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget truncation
            if len(trial_fitness) < len(trials):
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                population = population[:valid_trials]
                fitness = fitness[:valid_trials]
                self.np = valid_trials
                if self.np < 10:
                    break
            
            # Check stopping after partial evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update operator rewards
            self._update_operator_rewards(selected_operator, success_mask)
            
            # Adapt step size
            success_rate = np.mean(success_mask)
            self._adapt_step_size(success_rate)
            
            # Periodic local search
            local_search_counter += 1
            if local_search_counter >= self.local_search_interval:
                population, fitness = self._apply_local_refinement(population, fitness, func)
                local_search_counter = 0
                
                if stopping_condition():
                    break
            
            # Check stagnation
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            # Periodic restart check based on diversity
            if generation % 10 == 0:
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            generation += 1
        
        # Return best solution
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]

```