Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_mutate_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.725137e-08            5.304555e-08            1.000000e-08            6.178220e-03            1.000000e-08            1.000000e-08            
1      1.000000e-08            1.000000e-08            1.000000e-08            8.035082e-03            1.000000e-08            1.225584e-02            1.825009e+00            2.659393e-07            5.464834e+01            5.241179e-08            6.372329e-08            
2      1.000000e-08            1.000000e-08            1.000000e-08            1.080767e+00            1.000000e-08            6.065223e-06            7.677709e-01            1.000000e-08            1.558493e-02            1.000000e-08            1.000000e-08            
3      1.399473e-04            1.384495e-04            1.690693e-04            3.094134e+00            1.598384e-04            2.521738e+00            3.658637e+00            3.464925e-01            1.279674e+02            1.142323e-01            1.664747e-04            
4      1.000000e-08            1.000000e-08            1.000000e-08            9.409396e-01            1.333333e-08            9.842234e-01            8.464810e-01            1.999998e-08            1.209008e+00            1.000000e-08            1.501311e-02            
5      1.000000e-08            1.000000e-08            1.000000e-08            9.794067e+01            1.000000e-08            6.697747e+00            2.474664e+02            1.000000e-08            3.713143e+07            1.000000e-08            1.000000e-08            
6      7.968360e-08            6.539288e-08            1.489385e-02            2.737529e+01            7.276807e-08            2.986947e+01            2.171971e+02            2.721913e+01            3.774804e+01            1.510013e+01            4.081135e-05            
7      4.318590e-06            4.489137e-06            4.854553e-06            2.881852e+00            5.194480e-06            6.247278e+00            6.946533e+00            2.433955e-03            1.701995e+02            3.030861e-04            5.533477e-06            
8      1.000000e-08            8.173265e-04            8.289457e-04            2.626109e+00            8.061120e-04            2.012890e+00            2.664474e+00            4.426633e-03            1.274668e+01            1.096927e-03            9.627963e-04            
9      2.136244e+00            1.926003e+00            3.405462e+00            6.982182e+00            2.375953e-04            7.447416e+00            1.094260e+01            7.009345e+00            7.789834e+00            6.276498e+00            2.618485e+00            
10     1.271581e-02            2.922923e-02            2.623682e-01            1.070421e+00            1.547250e-01            6.522998e+00            7.168903e+00            6.165420e+00            3.970431e-05            1.000000e-08            5.119401e-03            
11     3.377357e+00            2.411133e+00            2.630729e+00            3.021465e+01            4.005068e+00            1.466253e+01            5.718198e+01            3.165081e+01            1.933666e+02            3.295273e+01            2.879097e+00            
12     1.742651e-01            2.747914e-01            1.177325e+00            3.621433e+00            8.427980e-01            3.700986e-01            5.698790e-01            1.276122e-05            1.470469e-04            3.556914e-01            8.549443e-02            
13     1.687388e+00            1.722417e+00            1.347124e+00            3.234431e+00            1.453923e+00            1.931102e+01            2.344342e+01            2.223558e+01            1.462726e+01            2.460989e+00            1.514826e+00            
14     2.458867e+00            2.483077e+00            2.448906e+00            1.313117e+01            2.468490e+00            3.454044e+00            3.593193e+00            3.397022e+00            3.604497e+00            2.796439e+00            2.428704e+00            
15     1.182693e+00            1.095388e+00            1.227785e+00            2.554137e+00            1.284880e+00            2.753225e+00            2.640679e+00            2.963980e+00            3.016932e+00            2.743388e+00            1.121709e+00            
16     1.745701e+02            1.590299e+02            1.691933e+02            4.786246e+02            9.475952e+01            6.696879e+02            6.620856e+02            9.654665e+02            6.975690e+02            4.976578e+02            3.810561e+02            
17     3.120103e+00            3.120103e+00            3.120103e+00            6.923332e+00            3.120103e+00            5.171136e+03            1.544480e+04            2.526836e+04            2.316078e+04            2.146876e+03            3.120103e+00            
18     1.025985e+01            2.672276e+00            2.712789e+00            1.706095e+01            2.672823e+00            3.296501e+01            3.430297e+01            3.651996e+01            3.597672e+01            2.409405e+01            1.183896e+01            
19     8.625591e+00            6.443069e+00            1.068644e+01            2.216574e+02            7.733162e+00            4.159477e+01            4.312070e+01            1.467560e+02            4.672654e+01            3.080284e+01            1.285128e+01            
20     1.845448e+01            1.997614e+01            3.225020e+01            3.296090e+01            3.150424e+01            2.184760e+01            2.195562e+01            2.276178e+01            1.641839e+01            1.938337e+01            1.927286e+01            
21     4.403171e+00            4.191873e+00            4.376351e+00            5.528278e+00            4.235524e+00            4.650215e+00            4.573974e+00            4.594090e+00            4.622474e+00            4.513123e+00            4.499793e+00            
22     2.305814e+00            2.204834e+00            2.291752e+00            1.583514e+01            2.409073e+00            1.159846e+01            1.159855e+01            1.124559e+01            1.006624e+01            1.159943e+01            2.409416e+00            
23     1.928706e+01            2.591798e+01            1.488166e+01            7.345059e+01            3.511363e+01            2.725807e+01            2.458822e+01            3.394232e+01            3.179921e+01            2.602098e+01            2.295393e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: SKIPPED (trivial)
Task  2: SKIPPED (trivial)
Task  3: variant_01_idea_0.py  (error=1.384495e-04)
Task  4: SKIPPED (trivial)
Task  5: SKIPPED (trivial)
Task  6: variant_01_idea_0.py  (error=6.539288e-08)
Task  7: original.py  (error=4.318590e-06)
Task  8: original.py  (error=1.000000e-08)
Task  9: variant_04_idea_0.py  (error=2.375953e-04)
Task 10: variant_09_idea_0.py  (error=1.000000e-08)
Task 11: variant_01_idea_0.py  (error=2.411133e+00)
Task 12: variant_07_idea_0.py  (error=1.276122e-05)
Task 13: variant_02_idea_0.py  (error=1.347124e+00)
Task 14: variant_10_idea_0.py  (error=2.428704e+00)
Task 15: variant_01_idea_0.py  (error=1.095388e+00)
Task 16: variant_04_idea_0.py  (error=9.475952e+01)
Task 17: original.py  (error=3.120103e+00)
Task 18: variant_01_idea_0.py  (error=2.672276e+00)
Task 19: variant_01_idea_0.py  (error=6.443069e+00)
Task 20: variant_08_idea_0.py  (error=1.641839e+01)
Task 21: variant_01_idea_0.py  (error=4.191873e+00)
Task 22: variant_01_idea_0.py  (error=2.204834e+00)
Task 23: variant_02_idea_0.py  (error=1.488166e+01)

WIN COUNTS:
  variant_01_idea_0.py: 8 wins
  original.py: 3 wins
  variant_04_idea_0.py: 2 wins
  variant_02_idea_0.py: 2 wins
  variant_09_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins
  variant_10_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (8 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using current-to-pbest/1 with rank-based pbest selection."""
        n = len(population)
        dim = self.dim

        # Sort by fitness for p-best selection
        sorted_idx = np.argsort(fitness)

        # Adaptive p: use strategy_mask to vary greediness
        # mask=True -> smaller p (greedier), mask=False -> larger p (more explorative)
        p_greedy = max(2, int(np.ceil(0.05 * n)))
        p_explore = max(2, int(np.ceil(0.25 * n)))

        pbest_idx = np.empty(n, dtype=int)
        for i in range(n):
            if strategy_mask[i]:
                pbest_idx[i] = sorted_idx[np.random.randint(0, p_greedy)]
            else:
                pbest_idx[i] = sorted_idx[np.random.randint(0, p_explore)]

        # Select r1 != i
        r1 = np.empty(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, n)
                if idx != i:
                    r1[i] = idx
                    break

        # Union of population and archive for r2
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Select r2 from union, r2 != i and r2 != r1
        r2 = np.empty(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        f_col = f_values[:, np.newaxis]

        # current-to-pbest/1: v_i = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2_union)
        mutants = (
            population
            + f_col * (population[pbest_idx] - population)
            + f_col * (population[r1] - union[r2])
        )

        return mutants
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim

        sorted_idx = np.argsort(fitness)
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)

        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        r1 = self._random_indices_not_equal(n, np.arange(n))

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(n, dtype=int)
        r3 = np.zeros(n, dtype=int)
        r4 = np.zeros(n, dtype=int)
        for i in range(n):
            used = {i, r1[i]}
            while True:
                idx = np.random.randint(0, union_size)
                if idx not in used:
                    r2[i] = idx
                    used.add(idx)
                    break
            while True:
                idx = np.random.randint(0, union_size)
                if idx not in used:
                    r3[i] = idx
                    used.add(idx)
                    break
            while True:
                idx = np.random.randint(0, union_size)
                if idx not in used:
                    r4[i] = idx
                    break

        # Rank-based F scaling: worse individuals get larger F
        rank_ratio = ranks / max(n - 1, 1)  # 0=best, 1=worst
        f_scaled = f_values * (0.5 + 1.0 * rank_ratio)  # range [0.5F, 1.5F]
        f_scaled = np.clip(f_scaled, 0.1, 1.5)
        f_col = f_scaled[:, np.newaxis]

        mutants = np.empty((n, dim))

        third = n // 3

        # Strategy 1: Top third — current-to-pbest/1 (exploitation)
        elite_mask = ranks < third
        if np.any(elite_mask):
            idx = np.where(elite_mask)[0]
            mutants[idx] = (population[idx]
                + f_col[idx] * (population[pbest_idx[idx]] - population[idx])
                + f_col[idx] * (population[r1[idx]] - union[r2[idx]]))

        # Strategy 2: Middle third — current-to-pbest/2 (balanced)
        mid_mask = (ranks >= third) & (ranks < 2 * third)
        if np.any(mid_mask):
            idx = np.where(mid_mask)[0]
            mutants[idx] = (population[idx]
                + f_col[idx] * (population[pbest_idx[idx]] - population[idx])
                + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                + 0.5 * f_col[idx] * (union[r3[idx]] - union[r4[idx]]))

        # Strategy 3: Worst third — random base with large steps (exploration)
        worst_mask = ranks >= 2 * third
        if np.any(worst_mask):
            idx = np.where(worst_mask)[0]
            r_base = np.random.randint(0, n, size=len(idx))
            # Ensure r_base != idx
            same = r_base == idx
            r_base[same] = (r_base[same] + 1) % n
            # Use larger F for exploration
            f_explore = np.clip(f_scaled[idx] * 1.5, 0.4, 2.0)[:, np.newaxis]
            mutants[idx] = (population[r_base]
                + f_explore * (population[pbest_idx[idx]] - union[r2[idx]])
                + f_explore * (population[r1[idx]] - union[r3[idx]]))

        return mutants
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim

        sorted_idx = np.argsort(fitness)
        # Rank-based scaling: worst individuals get larger F multipliers
        ranks = np.empty(n, dtype=float)
        ranks[sorted_idx] = np.arange(n, dtype=float)
        rank_scale = 0.5 + 0.5 * (ranks / max(n - 1, 1))  # range [0.5, 1.0]

        # Best individual
        best_idx = sorted_idx[0]

        # p-best for diversity
        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Union for donor selection
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Select 4 distinct random indices for two difference vectors
        r1 = np.empty(n, dtype=int)
        r2 = np.empty(n, dtype=int)
        r3 = np.empty(n, dtype=int)
        r4 = np.empty(n, dtype=int)
        for i in range(n):
            candidates = list(range(n))
            candidates.remove(i)
            chosen = np.random.choice(candidates, size=min(2, len(candidates)), replace=False)
            r1[i] = chosen[0]
            r3[i] = chosen[1] if len(chosen) > 1 else chosen[0]
            # r2, r4 from union
            attempts = 0
            while attempts < 100:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break
                attempts += 1
            else:
                r2[i] = np.random.randint(0, union_size)
            attempts = 0
            while attempts < 100:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r3[i] and idx != r2[i]:
                    r4[i] = idx
                    break
                attempts += 1
            else:
                r4[i] = np.random.randint(0, union_size)

        f_col = f_values[:, np.newaxis]
        rs = rank_scale[:, np.newaxis]

        mutants = np.empty((n, dim))

        # Strategy 1 (mask True): current-to-best/2 with rank scaling
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            mutants[idx1] = (
                population[idx1]
                + rs[idx1] * f_col[idx1] * (population[best_idx] - population[idx1])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
                + 0.5 * f_col[idx1] * (population[r3[idx1]] - union[r4[idx1]])
            )

        # Strategy 2 (mask False): rand-to-pbest/2 with rank scaling
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            mutants[idx2] = (
                population[r_base]
                + rs[idx2] * f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
                + 0.5 * f_col[idx2] * (population[r3[idx2]] - union[r4[idx2]])
            )

        return mutants
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim

        # Rank-based adaptive F scaling: worse individuals get larger F
        sorted_idx = np.argsort(fitness)
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)
        rank_ratio = ranks / max(n - 1, 1)  # 0=best, 1=worst
        # Scale F: best get F*0.5, worst get F*1.5
        f_scaled = f_values * (0.5 + rank_ratio)
        f_scaled = np.clip(f_scaled, 0.1, 1.5)
        f_col = f_scaled[:, np.newaxis]

        # Compute covariance-based rotation (use top 50% of population)
        n_elite = max(4, n // 2)
        elite_pop = population[sorted_idx[:n_elite]]
        try:
            cov = np.cov(elite_pop.T) + 1e-10 * np.eye(dim)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_eigvals = np.sqrt(eigvals / np.max(eigvals))  # normalize
            transform = eigvecs * sqrt_eigvals[np.newaxis, :]  # dim x dim
            inv_transform = np.linalg.inv(transform + 1e-12 * np.eye(dim))
        except:
            transform = np.eye(dim)
            inv_transform = np.eye(dim)

        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Generate 4 distinct random indices
        indices = np.arange(n)
        r1 = self._random_indices_not_equal(n, indices)
        r2 = np.array([np.random.randint(0, union_size) for _ in range(n)])
        for i in range(n):
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, union_size)
        r3 = self._random_indices_not_equal(n, indices)
        for i in range(n):
            while r3[i] == r1[i]:
                r3[i] = np.random.randint(0, n)

        mutants = np.empty((n, dim))

        # Strategy 1 (mask=True): covariance-rotated current-to-pbest with extra diff
        m1 = strategy_mask
        if np.any(m1):
            idx = np.where(m1)[0]
            diff1 = population[pbest_idx[idx]] - population[idx]
            diff2 = population[r1[idx]] - union[r2[idx]]
            diff3 = population[r3[idx]] - population[idx]
            # Rotate differences through eigenspace
            rot_diff = (diff1 + 0.5 * diff2) @ inv_transform @ transform
            mutants[idx] = population[idx] + f_col[idx] * rot_diff + 0.3 * f_col[idx] * diff3

        # Strategy 2 (mask=False): rand-based with covariance perturbation
        m2 = ~strategy_mask
        if np.any(m2):
            idx = np.where(m2)[0]
            r_base = self._random_indices_not_equal(n, indices)[idx]
            diff1 = population[pbest_idx[idx]] - population[r_base]
            diff2 = population[r1[idx]] - union[r2[idx]]
            # Add Gaussian perturbation in eigenspace
            noise = np.random.normal(0, 0.1, (len(idx), dim)) @ transform
            mutants[idx] = population[r_base] + f_col[idx] * (diff1 + diff2) + noise

        return mutants
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim
        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        sorted_idx = np.argsort(fitness)

        # Compute rank-weighted centroid of top half
        top_k = max(3, n // 3)
        top_pop = population[sorted_idx[:top_k]]
        # Rank weights: best gets highest weight
        weights = np.log(top_k + 0.5) - np.log(np.arange(1, top_k + 1))
        weights /= weights.sum()
        centroid = weights @ top_pop

        # Compute covariance and eigenvectors for rotation-aware mutation
        try:
            diff = top_pop - centroid
            cov = (diff * weights[:, np.newaxis]).T @ diff
            cov += 1e-10 * np.eye(dim)  # regularize
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            # Scaling matrix: sqrt of eigenvalues
            D = np.sqrt(eigvals)
        except:
            eigvecs = np.eye(dim)
            D = np.ones(dim)

        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Generate 3 distinct random indices per individual
        r1 = self._random_indices_not_equal(n, np.arange(n))
        r2 = self._random_indices_not_equal(n, r1)
        # Ensure r2 != i
        for i in range(n):
            while r2[i] == i:
                r2[i] = np.random.randint(0, n)
        r3 = self._random_indices_not_equal(n, r2)
        for i in range(n):
            while r3[i] == i or r3[i] == r1[i]:
                r3[i] = np.random.randint(0, n)

        # Strategy 1 (mask=True): Eigenvector-rotated centroid mutation
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            # Direction toward centroid + eigenvector-rotated difference
            diff_vec = population[r1[idx1]] - population[r2[idx1]]
            # Rotate difference vector through eigenbasis
            rotated = (diff_vec @ eigvecs) * (D / (D.mean() + 1e-30))
            rotated = rotated @ eigvecs.T

            mutants[idx1] = (
                population[idx1]
                + f_col[idx1] * (centroid - population[idx1])
                + 0.5 * f_col[idx1] * rotated
            )

        # Strategy 2 (mask=False): pbest with double difference vectors
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            mutants[idx2] = (
                population[idx2]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[idx2])
                + 0.5 * f_col[idx2] * (population[r1[idx2]] - population[r2[idx2]])
                + 0.5 * f_col[idx2] * (population[r2[idx2]] - population[r3[idx2]])
            )

        return mutants
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim
        sorted_idx = np.argsort(fitness)
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)

        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        def rand_idx_not(n_pool, *excludes):
                result = np.random.randint(0, n_pool, size=n)
                for _ in range(20):
                    bad = np.zeros(n, dtype=bool)
                    for ex in excludes:
                        bad |= (result == ex)
                    if not np.any(bad):
                        break
                    result[bad] = np.random.randint(0, n_pool, size=np.sum(bad))
                return result

        arange_n = np.arange(n)
        r1 = rand_idx_not(n, arange_n)
        r2 = rand_idx_not(union_size, arange_n, r1)
        r3 = rand_idx_not(n, arange_n, r1)
        r4 = rand_idx_not(n, arange_n, r1, r3)
        r5 = rand_idx_not(n, arange_n, r1, r3, r4)

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        third = n // 3
        # Top third (best fitness, low rank): current-to-pbest/1 exploitation
        mask_exploit = ranks < third
        # Middle third: trigonometric mutation
        mask_trig = (ranks >= third) & (ranks < 2 * third)
        # Bottom third (worst fitness): rand/2 with amplified F
        mask_explore = ranks >= 2 * third

        if np.any(mask_exploit):
            idx = np.where(mask_exploit)[0]
            mutants[idx] = (population[idx]
                            + f_col[idx] * (population[pbest_idx[idx]] - population[idx])
                            + f_col[idx] * (population[r1[idx]] - union[r2[idx]]))

        if np.any(mask_trig):
            idx = np.where(mask_trig)[0]
            p1, p2, p3 = population[r1[idx]], population[r3[idx]], population[r4[idx]]
            f1, f2, f3 = fitness[r1[idx]], fitness[r3[idx]], fitness[r4[idx]]
            psum = np.abs(f1) + np.abs(f2) + np.abs(f3) + 1e-30
            w1 = (np.abs(f1) / psum)[:, np.newaxis]
            w2 = (np.abs(f2) / psum)[:, np.newaxis]
            w3 = (np.abs(f3) / psum)[:, np.newaxis]
            centroid = (p1 + p2 + p3) / 3.0
            mutants[idx] = centroid + f_col[idx] * (p1 - p2) + f_col[idx] * (p2 - p3)

        if np.any(mask_explore):
            idx = np.where(mask_explore)[0]
            F_big = np.clip(f_col[idx] * 1.5, 0.4, 1.5)
            mutants[idx] = (population[r3[idx]]
                            + F_big * (population[r4[idx]] - population[r5[idx]])
                            + F_big * (population[r1[idx]] - union[r2[idx]]))

        return mutants
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim

        sorted_idx = np.argsort(fitness)

        # Weighted centroid of top individuals (like CMA-ES)
        n_top = max(3, n // 4)
        top_idx = sorted_idx[:n_top]
        top_pop = population[top_idx]

        # Log-linear weights for weighted recombination
        weights = np.log(n_top + 0.5) - np.log(np.arange(1, n_top + 1))
        weights = weights / np.sum(weights)
        weighted_mean = np.sum(weights[:, np.newaxis] * top_pop, axis=0)

        # Compute covariance-based directions for rotation invariance
        if dim <= 100 and n_top >= dim // 2:
            diffs = top_pop - weighted_mean
            cov = np.dot((weights[:, np.newaxis] * diffs).T, diffs)
            cov += 1e-10 * np.eye(dim)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except:
                sqrt_cov = None
        else:
            sqrt_cov = None

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        r1 = self._random_indices_not_equal(n, np.arange(n))

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.array([np.random.choice([j for j in range(union_size) if j != i and j != r1[i]]) for i in range(n)])

        for i in range(n):
            if strategy_mask[i]:
                # Strategy 1: Weighted mean guidance + eigenvector perturbation
                base = population[i] + f_values[i] * (weighted_mean - population[i])
                diff = f_values[i] * (population[r1[i]] - union[r2[i]])

                if sqrt_cov is not None and np.random.random() < 0.4:
                    z = np.random.standard_normal(dim)
                    eigen_perturb = 0.1 * f_values[i] * sqrt_cov @ z
                    mutants[i] = base + diff + eigen_perturb
                else:
                    mutants[i] = base + diff
            else:
                # Strategy 2: current-to-pbest with double difference vectors
                r3 = np.random.randint(0, n)
                while r3 == i or r3 == r1[i]:
                    r3 = np.random.randint(0, n)
                r4 = np.random.randint(0, union_size)
                while r4 == i or r4 == r1[i] or r4 == r2[i]:
                    r4 = np.random.randint(0, union_size)

                mutants[i] = (
                    population[i]
                    + f_values[i] * (population[pbest_idx[i]] - population[i])
                    + f_values[i] * (population[r1[i]] - union[r2[i]])
                    + 0.5 * f_values[i] * (population[r3] - union[r4])
                )

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


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, and intelligent restart mechanism.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(8 * dim, 300)
        self.archive_max = self.np_size
        self.memory_size = 5
        self.p_best_rate = 0.15
        self.restart_threshold = 50
        self.min_pop_size = max(4, dim // 2)

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        population, fitness = self._initialize_population(func, stopping_condition)
        if stopping_condition():
            return self.f_opt, self.x_opt

        self._update_best(population, fitness)
        archive = np.empty((0, self.dim))
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        prev_best_fitness = self.f_opt

        while not stopping_condition():
            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch()

            # Generate mutant vectors
            mutants = self._mutate_batch(population, fitness, archive, f_values, strategy_mask)

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values)

            # Clip to bounds
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                self._safe_update_best(trials, trial_fitness)
                break

            # Handle truncated results
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                f_values = f_values[:len(trial_fitness)]
                cr_values = cr_values[:len(trial_fitness)]

            valid = ~np.isnan(trial_fitness)
            
            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Update parameter memory
            memory_f, memory_cr, memory_idx = self._update_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta
            )

            self._update_best(population, fitness)

            # Detect stagnation
            stagnant = self._detect_stagnation(prev_best_fitness)
            if stagnant:
                population, fitness, archive, memory_f, memory_cr, memory_idx = (
                    self._restart(func, stopping_condition, population, fitness)
                )
                if stopping_condition():
                    break

            prev_best_fitness = self.f_opt
            self.generation += 1

            # Periodically try opposition-based learning
            if self.generation % 25 == 0 and not stopping_condition():
                population, fitness = self._opposition_based_jump(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
                self._update_best(population, fitness)

        return self.f_opt, self.x_opt

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values for entire population from parameter memory."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Cauchy distribution for F
        f_values = np.zeros(n)
        for i in range(n):
            while True:
                f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    break
            f_values[i] = min(f_val, 1.0)

        # Normal distribution for CR
        cr_values = np.clip(
            np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0
        )

        return f_values, cr_values

    def _select_strategies_batch(self):
        """Select mutation strategy for each individual. Returns boolean mask."""
        # True = current-to-pbest/1, False = rand-to-pbest/1
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using mixed strategies for the whole population."""
        n = len(population)
        dim = self.dim

        # Sort by fitness for p-best selection
        sorted_idx = np.argsort(fitness)
        p = max(2, int(np.ceil(self.p_best_rate * n)))

        # Select p-best indices for each individual
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Select r1 != i
        r1 = self._random_indices_not_equal(n, np.arange(n))

        # Union of population and archive for r2
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Select r2 from union, r2 != i and r2 != r1
        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        # Strategy 1: current-to-pbest/1 (for strategy_mask == True)
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            mutants[idx1] = (
                population[idx1]
                + f_col[idx1] * (population[pbest_idx[idx1]] - population[idx1])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            )

        # Strategy 2: rand-to-pbest/1 (for strategy_mask == False)
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            mutants[idx2] = (
                population[r_base]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            )

        return mutants

    def _random_indices_not_equal(self, n, exclude_indices):
        """Generate random indices in [0, n) that differ from exclude_indices."""
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to whole population at once."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        # Ensure at least one dimension is from mutant
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds with midpoint reflection."""
        # For out-of-bounds, reflect towards the center
        too_low = trials < self.lb
        too_high = trials > self.ub
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """Greedy selection: replace if trial is better. Track successful params."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        improved = np.zeros(n, dtype=bool)
        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    improved[i] = True
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))
                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    def _update_memory(self, memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta):
        """Update parameter memory using weighted Lehmer mean."""
        if len(success_f) == 0:
            return memory_f, memory_cr, memory_idx

        weights = success_delta / (np.sum(success_delta) + 1e-30)

        # Weighted Lehmer mean for F
        lehmer_f = np.sum(weights * success_f ** 2) / (np.sum(weights * success_f) + 1e-30)
        memory_f[memory_idx] = lehmer_f

        # Weighted arithmetic mean for CR
        mean_cr = np.sum(weights * success_cr)
        memory_cr[memory_idx] = mean_cr

        memory_idx = (memory_idx + 1) % self.memory_size
        return memory_f, memory_cr, memory_idx

    def _detect_stagnation(self, prev_best_fitness):
        """Detect if the algorithm is stagnating."""
        improvement = prev_best_fitness - self.f_opt
        if improvement < 1e-12 * (abs(prev_best_fitness) + 1e-30):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        return self.stagnation_counter >= self.restart_threshold

    def _restart(self, func, stopping_condition, population, fitness):
        """Partial restart: keep best individuals, reinitialize rest with opposition."""
        self.stagnation_counter = 0
        n = len(population)
        keep_count = max(2, n // 5)

        sorted_idx = np.argsort(fitness)
        elite = population[sorted_idx[:keep_count]].copy()
        elite_fit = fitness[sorted_idx[:keep_count]].copy()

        reinit_count = n - keep_count
        # Generate new individuals using quasi-opposition
        new_pop = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

        # Quasi-opposition: reflect around population center
        center = np.mean(elite, axis=0)
        opp_pop = 2.0 * center - new_pop
        opp_pop = np.clip(opp_pop, self.lb, self.ub)

        # Choose random mix of new and opposition
        mix_mask = np.random.random(reinit_count) < 0.5
        combined_new = np.where(mix_mask[:, np.newaxis], opp_pop, new_pop)

        if stopping_condition():
            population = np.vstack([elite, combined_new])
            fitness_new = np.full(reinit_count, np.inf)
            return population, np.concatenate([elite_fit, fitness_new]), np.empty((0, self.dim)), np.full(self.memory_size, 0.5), np.full(self.memory_size, 0.5), 0

        combined_new = self._clip_to_bounds(combined_new)
        new_fitness = func(combined_new)
        if len(new_fitness) < len(combined_new):
            combined_new = combined_new[:len(new_fitness)]

        self._safe_update_best(combined_new, new_fitness)

        population = np.vstack([elite, combined_new[:len(new_fitness)]])
        fitness_all = np.concatenate([elite_fit, new_fitness])

        # Reset memory
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        archive = np.empty((0, self.dim))

        self.np_size = len(population)

        return population, fitness_all, archive, memory_f, memory_cr, memory_idx

    def _opposition_based_jump(self, population, fitness, func, stopping_condition):
        """Apply opposition-based learning to diversify population."""
        center = (self.lb + self.ub) / 2.0
        opp_population = 2.0 * center - population
        opp_population = self._clip_to_bounds(opp_population)

        # Add random perturbation
        noise = np.random.normal(0, 1.0, opp_population.shape)
        opp_population = self._clip_to_bounds(opp_population + noise)

        if stopping_condition():
            return population, fitness

        opp_fitness = func(opp_population)
        if len(opp_fitness) < len(opp_population):
            opp_population = opp_population[:len(opp_fitness)]

        self._safe_update_best(opp_population, opp_fitness)

        # Combine and select best
        n = min(len(population), len(opp_population))
        combined_pop = np.vstack([population[:n], opp_population[:n]])
        combined_fit = np.concatenate([fitness[:n], opp_fitness[:n]])

        sorted_idx = np.argsort(combined_fit)
        best_idx = sorted_idx[:len(population)]
        return combined_pop[best_idx], combined_fit[best_idx]

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance."""
        center = np.mean(population, axis=0)
        diffs = population - center
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _update_best(self, population, fitness):
        """Update global best from population."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fitness = fitness[valid]
        valid_pop = population[valid]
        best_idx = np.argmin(valid_fitness)
        if valid_fitness[best_idx] < self.f_opt:
            self.f_opt = valid_fitness[best_idx]
            self.x_opt = valid_pop[best_idx].copy()

    def _safe_update_best(self, candidates, fitnesses):
        """Safely update best considering possible NaN values."""
        if len(fitnesses) == 0:
            return
        valid = ~np.isnan(fitnesses)
        if not np.any(valid):
            return
        best_idx = np.nanargmin(fitnesses)
        if fitnesses[best_idx] < self.f_opt:
            self.f_opt = fitnesses[best_idx]
            self.x_opt = candidates[best_idx].copy()

```