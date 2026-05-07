Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_crossover_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.306928e-07            1.000000e-08            1.000000e-08            1.253159e-07            1.000000e-08            
1      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.768956e-07            1.000000e-08            1.000000e-08            2.674316e-07            1.000000e-08            
2      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.855520e-05            1.000000e-08            1.000000e-08            2.819313e-05            1.000000e-08            
3      1.344828e-04            1.357881e-04            1.415489e-04            1.355933e-04            1.102607e-04            1.365132e-04            3.988872e-04            1.354678e-04            1.362536e-04            3.857588e-04            1.368410e-04            
4      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.576141e-02            1.000000e-08            1.000000e-08            2.533776e-02            1.000000e-08            
5      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
6      5.686047e-08            5.974204e-08            6.517197e-08            5.646290e-08            5.309008e-08            5.888095e-08            4.819384e-07            5.787710e-08            5.943892e-08            4.759501e-07            5.423835e-08            
7      4.035845e-06            4.038241e-06            3.953975e-06            3.904187e-06            3.258552e-06            4.055592e-06            1.661266e-05            3.839289e-06            4.065606e-06            1.631265e-05            3.635213e-06            
8      1.000000e-08            7.759357e-04            7.767974e-04            1.000000e-08            4.954112e-04            1.000000e-08            1.986894e-03            1.000000e-08            1.000000e-08            1.925027e-03            1.000000e-08            
9      1.956082e-04            2.013222e-04            7.639845e-01            1.880236e-04            1.879435e-04            1.904162e-04            5.711282e-04            1.879105e-04            1.856006e-04            5.621534e-04            1.853720e-04            
10     1.000000e-08            7.004172e-02            2.114495e-01            5.297420e+00            8.654810e-01            5.112682e+00            8.574689e-05            1.715403e+00            4.066701e+00            8.639419e+00            6.312037e-08            
11     2.634005e+00            5.055654e+00            2.442142e+00            3.963401e+00            6.254046e+00            2.184922e+00            7.946997e-04            1.709049e+00            2.165568e+00            5.177399e+00            5.304964e+00            
12     4.285775e-06            8.670848e-01            4.853947e+00            2.432111e+00            2.044736e+01            1.759309e+00            8.011108e-01            1.142061e+00            1.913606e+00            5.821395e+00            2.302134e+00            
13     1.197145e+00            1.468883e+00            1.866631e+00            9.775052e-01            1.060971e+00            1.822645e+00            9.310118e-03            1.417117e+00            8.869435e-01            2.062206e+00            1.689209e+00            
14     2.471461e+00            2.458370e+00            2.404274e+00            2.503045e+00            3.781372e+00            2.524163e+00            2.119800e+00            2.442272e+00            2.531631e+00            2.461994e+00            2.537771e+00            
15     1.148720e+00            4.058167e+00            1.188380e+00            1.225800e+00            2.048802e+00            1.151277e+00            1.157390e+00            1.215601e+00            1.123645e+00            2.060913e+00            3.840785e+00            
16     9.685850e+01            9.341721e+01            1.590338e+02            7.190045e+01            1.028816e-02            6.886914e+01            2.000000e-08            6.144863e+01            3.602215e+01            8.484765e+01            5.078633e+02            
17     3.120103e+00            3.333229e+00            3.120103e+00            3.120103e+00            3.121234e+00            3.120103e+00            3.120103e+00            3.120103e+00            3.120103e+00            1.190803e+01            3.120103e+00            
18     2.671167e+00            3.214714e+00            2.674484e+00            2.671447e+00            2.861721e+00            2.671443e+00            2.674529e+00            2.671167e+00            2.671167e+00            1.162148e+01            1.728148e+01            
19     6.741957e+00            7.098984e+00            6.225411e+00            5.628353e+00            4.403799e+01            5.621800e+00            5.467755e+00            5.394836e+00            5.502478e+00            5.420582e+00            4.641883e+01            
20     1.213484e+01            3.529543e+01            5.951233e+00            1.050946e+01            9.681522e+00            1.208691e+01            3.724148e+00            1.164180e+01            7.380285e+00            1.491840e+01            2.024786e+01            
21     4.403188e+00            4.187991e+00            4.218819e+00            4.169173e+00            4.739677e+00            4.173623e+00            4.169258e+00            4.172085e+00            4.164387e+00            4.186142e+00            4.425364e+00            
22     2.305725e+00            2.454629e+00            2.238697e+00            2.285756e+00            8.410901e+00            2.202039e+00            2.516287e+00            2.253636e+00            2.207582e+00            2.247729e+00            5.306654e+00            
23     1.565188e+01            2.596405e+01            2.137501e+01            9.374212e+00            4.719652e+01            8.756051e+00            3.193673e+01            8.859309e+00            1.235298e+01            7.246895e+00            3.139066e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: SKIPPED (trivial)
Task  2: SKIPPED (trivial)
Task  3: variant_04_idea_0.py  (error=1.102607e-04)
Task  4: SKIPPED (trivial)
Task  5: SKIPPED (trivial)
Task  6: SKIPPED (trivial)
Task  7: variant_04_idea_0.py  (error=3.258552e-06)
Task  8: SKIPPED (trivial)
Task  9: variant_10_idea_0.py  (error=1.853720e-04)
Task 10: original.py  (error=1.000000e-08)
Task 11: variant_06_idea_0.py  (error=7.946997e-04)
Task 12: original.py  (error=4.285775e-06)
Task 13: variant_06_idea_0.py  (error=9.310118e-03)
Task 14: variant_06_idea_0.py  (error=2.119800e+00)
Task 15: variant_08_idea_0.py  (error=1.123645e+00)
Task 16: variant_06_idea_0.py  (error=2.000000e-08)
Task 17: original.py  (error=3.120103e+00)
Task 18: original.py  (error=2.671167e+00)
Task 19: variant_07_idea_0.py  (error=5.394836e+00)
Task 20: variant_06_idea_0.py  (error=3.724148e+00)
Task 21: variant_08_idea_0.py  (error=4.164387e+00)
Task 22: variant_05_idea_0.py  (error=2.202039e+00)
Task 23: variant_09_idea_0.py  (error=7.246895e+00)

WIN COUNTS:
  variant_06_idea_0.py: 5 wins
  original.py: 4 wins
  variant_04_idea_0.py: 2 wins
  variant_08_idea_0.py: 2 wins
  variant_10_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_04_idea_0.py (2 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Covariance-rotated exponential crossover for non-separable problems."""
        n, dim = population.shape

        # Estimate rotation from population covariance
        try:
            if dim <= 200 and n >= dim // 2:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Regularize
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Rotate population and mutants into eigenspace
                pop_rot = np.dot(population - center, eigenvectors) 
                mut_rot = np.dot(mutants - center, eigenvectors)
                use_rotation = True
            else:
                use_rotation = False
        except Exception:
            use_rotation = False

        if use_rotation:
            # Exponential crossover in rotated space
            trials_rot = pop_rot.copy()
            for i in range(n):
                cr = cr_values[i]
                # Start from a random dimension
                j_start = np.random.randint(0, dim)
                L = 0
                while L < dim and (np.random.random() < cr or L == 0):
                    j = (j_start + L) % dim
                    trials_rot[i, j] = mut_rot[i, j]
                    L += 1

            # Rotate back to original space
            trials = np.dot(trials_rot, eigenvectors.T) + center

            # With some probability, mix with standard binomial for diversity
            mix_prob = 0.2
            use_standard = np.random.random(n) < mix_prob
            if np.any(use_standard):
                idx = np.where(use_standard)[0]
                rand_matrix = np.random.random((len(idx), dim))
                cr_matrix = cr_values[idx, np.newaxis]
                j_rand = np.random.randint(0, dim, size=len(idx))
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(len(idx)), j_rand] = True
                standard_trials = np.where(cross_mask, mutants[idx], population[idx])
                trials[idx] = standard_trials
        else:
            # Fallback: exponential crossover (better for non-separable than binomial)
            trials = population.copy()
            for i in range(n):
                cr = cr_values[i]
                j_start = np.random.randint(0, dim)
                L = 0
                while L < dim and (np.random.random() < cr or L == 0):
                    j = (j_start + L) % dim
                    trials[i, j] = mutants[i, j]
                    L += 1

        return trials
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-based rotation-invariant crossover."""
        n, dim = population.shape

        # Compute covariance structure of population for rotation-invariant crossover
        if n > dim + 1 and dim <= 100:
            try:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Regularize
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                eigenvalues = np.maximum(eigenvalues, 1e-20)

                # Transform population and mutants into eigenspace
                pop_eigen = np.dot(population - center, eigenvectors)
                mut_eigen = np.dot(mutants - center, eigenvectors)

                # Perform binomial crossover in eigenspace
                rand_matrix = np.random.random((n, dim))
                cr_matrix = cr_values[:, np.newaxis]
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(n), j_rand] = True

                trial_eigen = np.where(cross_mask, mut_eigen, pop_eigen)

                # Transform back to original space
                trials = np.dot(trial_eigen, eigenvectors.T) + center

                # For a fraction, also do a small step along weak eigenvectors (exploit narrow valleys)
                explore_frac = 0.15
                explore_mask = np.random.random(n) < explore_frac
                if np.any(explore_mask):
                    n_exp = np.sum(explore_mask)
                    # Perturb along the smallest eigenvalue directions (narrow valleys)
                    weak_dims = min(max(1, dim // 4), dim)
                    perturbation_eigen = np.zeros((n_exp, dim))
                    scales = np.sqrt(eigenvalues[:weak_dims])
                    perturbation_eigen[:, :weak_dims] = np.random.normal(0, 1, (n_exp, weak_dims)) * scales * 0.1
                    perturbation = np.dot(perturbation_eigen, eigenvectors.T)
                    trials[explore_mask] = trials[explore_mask] + perturbation

                trials = np.clip(trials, self.lb, self.ub)
                return trials

            except (np.linalg.LinAlgError, ValueError):
                pass

        # Fallback: standard binomial crossover
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True
        trials = np.where(cross_mask, mutants, population)
        return trials
```

# --- From variant_06_idea_0.py (5 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-guided rotational crossover for non-separable problems."""
        n, dim = population.shape

        if dim < 3 or n < dim + 1:
            # Fall back to standard binomial for very small cases
            rand_matrix = np.random.random((n, dim))
            cr_matrix = cr_values[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)
            return trials

        # Compute population covariance and eigenvectors
        try:
            center = np.mean(population, axis=0)
            centered = population - center
            cov = np.dot(centered.T, centered) / max(n - 1, 1)
            # Regularize
            cov += 1e-10 * np.eye(dim)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # eigenvectors columns are eigenvecs, sorted ascending
        except:
            # Fallback to identity if decomposition fails
            eigenvectors = np.eye(dim)

        # Transform population and mutants into eigenvector space
        pop_rot = population @ eigenvectors
        mut_rot = mutants @ eigenvectors

        # Perform crossover in rotated space with block-based transfer
        trials_rot = pop_rot.copy()
        cr_matrix = cr_values[:, np.newaxis]

        for i in range(n):
            cr = cr_values[i]
            # Adaptive block size: higher CR = larger blocks
            block_size = max(1, int(np.ceil(cr * dim * 0.5)))

            # Start position for block transfer (random)
            start = np.random.randint(0, dim)

            # Create mask: contiguous block in eigenvector space
            mask = np.zeros(dim, dtype=bool)
            indices = np.arange(start, start + block_size) % dim
            mask[indices] = True

            # Also add random scattered dimensions (binomial-style)
            rand_vals = np.random.random(dim)
            scatter_mask = rand_vals < (cr * 0.3)
            mask = mask | scatter_mask

            # Ensure at least one dimension from mutant
            if not np.any(mask):
                mask[np.random.randint(0, dim)] = True

            trials_rot[i, mask] = mut_rot[i, mask]

        # Transform back to original space
        trials = trials_rot @ eigenvectors.T

        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)

        return trials
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-based crossover in rotated coordinate space for non-separable problems."""
        n, dim = population.shape
        cr_matrix = cr_values[:, np.newaxis]

        # Compute population covariance eigenvectors for rotation
        try:
            if dim <= 200 and n >= dim // 2:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Regularize
                cov += np.eye(dim) * 1e-10
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Sort by eigenvalue descending
                idx_sort = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx_sort]

                # Rotate population and mutants into eigenvector space
                pop_rot = np.dot(population - center, eigenvectors)
                mut_rot = np.dot(mutants - center, eigenvectors)

                # Apply crossover in rotated space with block segments
                rand_matrix = np.random.random((n, dim))
                cross_mask = rand_matrix < cr_matrix

                # Ensure at least one dimension from mutant
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask[np.arange(n), j_rand] = True

                # For 30% of population, use contiguous block crossover (exponential-like)
                block_mask = np.random.random(n) < 0.3
                if np.any(block_mask):
                    block_idx = np.where(block_mask)[0]
                    for i in block_idx:
                        start = np.random.randint(0, dim)
                        L = max(1, int(cr_values[i] * dim))
                        L = min(L, dim)
                        cross_mask[i, :] = False
                        for k in range(L):
                            cross_mask[i, (start + k) % dim] = True

                trials_rot = np.where(cross_mask, mut_rot, pop_rot)

                # Rotate back to original space
                trials = np.dot(trials_rot, eigenvectors.T) + center
            else:
                raise ValueError("Skip rotation for high dim")

        except (np.linalg.LinAlgError, ValueError):
            # Fallback: standard binomial crossover
            rand_matrix = np.random.random((n, dim))
            cross_mask = rand_matrix < cr_matrix
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)

        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)

        return trials
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-based rotational crossover for non-separable problems."""
        n, dim = population.shape

        # Compute population covariance and its eigenvectors
        try:
            if n > dim + 1:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Regularize
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # eigenvectors columns are the principal axes

                # Rotate population and mutants into eigenspace
                pop_rot = np.dot(population - center, eigenvectors)
                mut_rot = np.dot(mutants - center, eigenvectors)

                # Perform binomial crossover in rotated space
                rand_matrix = np.random.random((n, dim))
                cr_matrix = cr_values[:, np.newaxis]
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(n), j_rand] = True

                trials_rot = np.where(cross_mask, mut_rot, pop_rot)

                # Rotate back to original space
                trials = np.dot(trials_rot, eigenvectors.T) + center

                # For a fraction of individuals, also try block crossover
                # along top eigenvectors (most variance directions)
                block_mask = np.random.random(n) < 0.15
                if np.any(block_mask) and dim > 2:
                    block_idx = np.where(block_mask)[0]
                    # Swap top-k eigenvector components as a block
                    k = max(1, dim // 3)
                    # Sort eigenvalues descending
                    top_k = np.argsort(eigenvalues)[-k:]
                    for i in block_idx:
                        trial_rot_i = pop_rot[i].copy()
                        trial_rot_i[top_k] = mut_rot[i][top_k]
                        trials[i] = np.dot(trial_rot_i, eigenvectors.T) + center

                # Clip to bounds
                trials = np.clip(trials, self.lb, self.ub)
                return trials

            else:
                raise ValueError("Not enough population for covariance")

        except (np.linalg.LinAlgError, ValueError):
            # Fallback to standard binomial crossover
            rand_matrix = np.random.random((n, dim))
            cr_matrix = cr_values[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)
            return trials
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-guided rotational exponential crossover."""
        n, dim = population.shape

        # Compute covariance-based rotation if population is large enough
        try:
            if n > dim + 1 and dim <= 100:
                centered = population - np.mean(population, axis=0)
                cov = np.dot(centered.T, centered) / (n - 1)
                # Regularize
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Rotate population and mutants into eigenvector space
                pop_rot = np.dot(population, eigenvectors)
                mut_rot = np.dot(mutants, eigenvectors)
            else:
                pop_rot = population.copy()
                mut_rot = mutants.copy()
                eigenvectors = None
        except (np.linalg.LinAlgError, ValueError):
            pop_rot = population.copy()
            mut_rot = mutants.copy()
            eigenvectors = None

        # Exponential crossover in rotated space with boosted CR
        boosted_cr = np.clip(cr_values * 1.3 + 0.1, 0.1, 1.0)
        trials_rot = pop_rot.copy()

        for i in range(n):
            # Start position for exponential crossover
            L = np.random.randint(0, dim)
            j = L
            count = 0
            while True:
                trials_rot[i, j] = mut_rot[i, j]
                count += 1
                j = (j + 1) % dim
                if np.random.random() >= boosted_cr[i] or count >= dim:
                    break

        # Ensure at least a minimum number of dimensions come from mutant
        min_dims = max(1, dim // 4)
        for i in range(n):
            changed = np.sum(trials_rot[i] != pop_rot[i])
            if changed < min_dims:
                extra_dims = np.random.choice(dim, size=min_dims - int(changed), replace=False)
                trials_rot[i, extra_dims] = mut_rot[i, extra_dims]

        # Rotate back to original space
        if eigenvectors is not None:
            trials = np.dot(trials_rot, eigenvectors.T)
        else:
            trials = trials_rot

        return trials
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Block-sectored crossover with random injection for escaping local optima."""
        n, dim = population.shape
        trials = population.copy()

        for i in range(n):
            cr = cr_values[i]

            # Random injection: with small probability, inject random dimensions
            inject_random = np.random.random() < 0.05

            if dim <= 4:
                # For very low dims, use standard binomial
                rand_vals = np.random.random(dim)
                j_rand = np.random.randint(0, dim)
                mask = rand_vals < cr
                mask[j_rand] = True
                trials[i] = np.where(mask, mutants[i], population[i])
            else:
                # Create random block structure
                # Number of blocks scales with sqrt(dim), minimum 2
                n_blocks = max(2, int(np.sqrt(dim)))

                # Random permutation of dimensions, then split into blocks
                perm = np.random.permutation(dim)
                block_sizes = np.diff(np.sort(np.random.choice(range(1, dim), size=n_blocks - 1, replace=False)))
                block_sizes = np.concatenate([[np.sort(np.random.choice(range(1, dim), size=n_blocks - 1, replace=False))[0]], 
                                               block_sizes,
                                               [dim - np.sort(np.random.choice(range(1, dim), size=n_blocks - 1, replace=False))[-1]]])

                # Simpler: just use equal-ish blocks
                splits = np.array_split(perm, n_blocks)

                # Ensure at least one block comes from mutant
                forced_block = np.random.randint(0, len(splits))

                for b_idx, block_dims in enumerate(splits):
                    if len(block_dims) == 0:
                        continue
                    # Each block is taken entirely from mutant or parent
                    take_from_mutant = (np.random.random() < cr) or (b_idx == forced_block)

                    if take_from_mutant:
                        if inject_random and np.random.random() < 0.3:
                            # Inject random values in this block
                            trials[i, block_dims] = np.random.uniform(self.lb, self.ub, size=len(block_dims))
                        else:
                            trials[i, block_dims] = mutants[i, block_dims]
                    # else: keep from population (already copied)

                # Additional: with small prob, do a local perturbation around best known
                if np.random.random() < 0.03 and self.x_opt is not None:
                    scale = 0.01 * (self.ub - self.lb) * np.random.random()
                    perturbation = np.random.normal(0, scale, dim)
                    candidate = self.x_opt + perturbation
                    candidate = np.clip(candidate, self.lb, self.ub)
                    trials[i] = candidate

        return trials
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
        """Generate F and CR values with generation-adaptive scaling for better convergence."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Adaptive scale factor: starts at 0.1, decreases over generations for finer tuning
        # but never goes below 0.02 to maintain some exploration
        base_scale = 0.1
        generation = getattr(self, 'generation', 0)
        scale_f = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))
        scale_cr = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))

        # Vectorized Cauchy distribution for F with rejection sampling
        mu_f = memory_f[indices]
        f_values = np.zeros(n)
        remaining = np.ones(n, dtype=bool)
        max_attempts = 100
        attempt = 0

        while np.any(remaining) and attempt < max_attempts:
            count = np.sum(remaining)
            # Cauchy samples: tan(pi * (U - 0.5))
            u = np.random.uniform(0, 1, size=count)
            cauchy_samples = np.tan(np.pi * (u - 0.5))
            candidates = mu_f[remaining] + scale_f * cauchy_samples

            # Accept those > 0
            valid = candidates > 0
            idx_remaining = np.where(remaining)[0]
            accepted = idx_remaining[valid]
            f_values[accepted] = np.minimum(candidates[valid], 1.0)
            remaining[accepted] = False
            attempt += 1

        # Fallback for any remaining (shouldn't happen but be safe)
        if np.any(remaining):
            f_values[remaining] = np.clip(mu_f[remaining], 0.01, 1.0)

        # Normal distribution for CR with adaptive scale
        cr_values = np.random.normal(memory_cr[indices], scale_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        # For some fraction of population, use more explorative parameters
        # This helps escape local optima on multimodal functions
        explore_mask = np.random.random(n) < 0.1
        if np.any(explore_mask):
            n_explore = np.sum(explore_mask)
            # Use larger F for exploration
            f_values[explore_mask] = np.clip(
                np.random.uniform(0.5, 1.0, size=n_explore), 0.1, 1.0
            )
            # Use higher CR for exploration (more dimensions from donor)
            cr_values[explore_mask] = np.clip(
                np.random.uniform(0.8, 1.0, size=n_explore), 0.0, 1.0
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