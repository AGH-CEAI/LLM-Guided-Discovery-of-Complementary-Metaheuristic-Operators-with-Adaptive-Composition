Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_initialize_population` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.470222e-07            2.087880e-07            -inf                    -inf                    2.106107e-07            3.050332e-07            2.947851e-07            3.172987e-07            2.464603e-07            2.123628e-07            3.889932e-07            
1      3.667557e-07            3.347736e-07            -inf                    -inf                    3.381691e-07            2.621405e-07            2.972758e-07            3.271177e-07            3.453528e-07            4.289398e-07            3.293850e-07            
2      3.913127e-05            3.304066e-05            -inf                    -inf                    5.187307e-05            2.852312e-05            5.222623e-05            5.754510e-05            3.635203e-05            8.038704e-05            3.517787e-05            
3      3.792322e-04            3.460789e-04            -inf                    -inf                    3.537984e-04            3.355650e-04            3.360592e-04            3.392619e-04            3.705954e-04            3.586223e-04            3.369948e-04            
4      2.326852e-02            2.293327e-02            -inf                    -inf                    2.262452e-02            2.258993e-02            2.294223e-02            2.326478e-02            2.393397e-02            2.341620e-02            2.235998e-02            
5      1.000000e-08            1.000000e-08            -inf                    -inf                    1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
6      2.714417e-05            8.045036e-06            -inf                    -inf                    7.625389e-06            4.736406e-06            2.701036e-05            5.821367e-06            5.265582e-06            5.105519e-06            2.507161e-05            
7      1.408268e-05            1.875416e-05            -inf                    -inf                    1.340659e-05            1.487825e-05            1.352571e-05            1.455551e-05            1.408639e-05            1.538029e-05            1.430479e-05            
8      1.809649e-03            1.838619e-03            -inf                    -inf                    1.819817e-03            1.867952e-03            1.848936e-03            1.862179e-03            1.841776e-03            1.772596e-03            1.815130e-03            
9      6.721897e-04            1.071172e-03            -inf                    -inf                    5.052951e-04            5.389909e-04            4.971897e-04            5.072498e-04            6.729192e-04            1.104985e-03            5.024399e-04            
10     2.046021e+01            3.538303e+01            -inf                    -inf                    2.542812e+01            4.085753e+01            1.630461e+01            2.279003e+01            2.340872e+01            2.935419e+01            2.469574e+01            
11     1.233027e+01            1.494772e+01            -inf                    -inf                    3.656023e+01            1.554158e+01            1.234505e+01            1.893963e+01            1.334735e+01            2.236565e+01            1.063520e+01            
12     1.653521e+01            2.460376e+01            -inf                    -inf                    3.223591e+01            2.579708e+01            5.106039e+01            1.951073e+01            2.141769e+01            4.243413e+01            2.029709e+01            
13     2.284005e+00            2.592183e+00            -inf                    -inf                    2.302204e+00            2.484917e+00            3.273734e+00            1.887542e+00            2.869833e+00            2.481289e+00            2.327139e+00            
14     2.685189e+00            2.534967e+00            -inf                    -inf                    2.604051e+00            2.621019e+00            2.627048e+00            2.648299e+00            2.640305e+00            2.637045e+00            2.626833e+00            
15     2.738212e+00            2.657564e+00            -inf                    -inf                    2.806633e+00            2.784894e+00            2.879227e+00            2.875859e+00            2.869528e+00            2.878975e+00            2.761383e+00            
16     6.636941e+01            5.956657e+01            -inf                    -inf                    7.462048e+01            7.632607e+01            8.228094e+01            6.336439e+01            5.172241e+01            9.674740e+01            7.418037e+01            
17     4.519592e+01            5.038487e+01            -inf                    -inf                    4.308132e+02            3.084443e+02            1.399057e+03            6.123907e+02            7.442387e+01            4.071857e+02            3.469990e+01            
18     2.090077e+01            1.645000e+01            -inf                    -inf                    2.393315e+01            2.432822e+01            2.146938e+01            1.810116e+01            2.296373e+01            2.611903e+01            1.644406e+01            
19     1.112357e+01            1.143209e+01            -inf                    -inf                    1.751881e+01            1.305747e+01            1.286588e+01            1.277771e+01            1.131994e+01            1.676850e+01            1.389265e+01            
20     2.220254e+01            2.016178e+01            -inf                    -inf                    2.159167e+01            2.014284e+01            2.034513e+01            2.143025e+01            2.165091e+01            2.272130e+01            1.999074e+01            
21     4.533599e+00            4.304315e+00            -inf                    -inf                    4.305596e+00            4.347641e+00            4.417090e+00            4.409289e+00            4.297318e+00            4.320375e+00            4.480502e+00            
22     1.041344e+01            1.079122e+01            -inf                    -inf                    7.998550e+00            9.973087e+00            9.825799e+00            1.059331e+01            1.042633e+01            9.253215e+00            9.746747e+00            
23     1.862680e+01            1.955697e+01            -inf                    -inf                    1.091683e+01            1.967754e+01            1.852721e+01            1.775598e+01            1.926942e+01            1.919696e+01            2.013631e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_idea_0.py  (error=2.087880e-07)
Task  1: variant_05_idea_0.py  (error=2.621405e-07)
Task  2: variant_05_idea_0.py  (error=2.852312e-05)
Task  3: variant_05_idea_0.py  (error=3.355650e-04)
Task  4: variant_10_idea_0.py  (error=2.235998e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_05_idea_0.py  (error=4.736406e-06)
Task  7: variant_04_idea_0.py  (error=1.340659e-05)
Task  8: variant_09_idea_0.py  (error=1.772596e-03)
Task  9: variant_06_idea_0.py  (error=4.971897e-04)
Task 10: variant_06_idea_0.py  (error=1.630461e+01)
Task 11: variant_10_idea_0.py  (error=1.063520e+01)
Task 12: original.py  (error=1.653521e+01)
Task 13: variant_07_idea_0.py  (error=1.887542e+00)
Task 14: variant_01_idea_0.py  (error=2.534967e+00)
Task 15: variant_01_idea_0.py  (error=2.657564e+00)
Task 16: variant_08_idea_0.py  (error=5.172241e+01)
Task 17: variant_10_idea_0.py  (error=3.469990e+01)
Task 18: variant_10_idea_0.py  (error=1.644406e+01)
Task 19: original.py  (error=1.112357e+01)
Task 20: variant_10_idea_0.py  (error=1.999074e+01)
Task 21: variant_08_idea_0.py  (error=4.297318e+00)
Task 22: variant_04_idea_0.py  (error=7.998550e+00)
Task 23: variant_04_idea_0.py  (error=1.091683e+01)

WIN COUNTS:
  variant_10_idea_0.py: 5 wins
  variant_05_idea_0.py: 4 wins
  variant_01_idea_0.py: 3 wins
  variant_04_idea_0.py: 3 wins
  variant_06_idea_0.py: 2 wins
  original.py: 2 wins
  variant_08_idea_0.py: 2 wins
  variant_09_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling with Gaussian perturbation."""
        # Ensure bounds are numpy arrays
        lb = np.asarray(self.lb, dtype=np.float64)
        ub = np.asarray(self.ub, dtype=np.float64)

        # Handle edge case of zero range
        range_vec = ub - lb
        range_vec = np.where(range_vec < 1e-15, 1.0, range_vec)

        # Latin Hypercube Sampling for space-filling coverage
        try:
            from scipy.stats import qmc
            # Use Halton sequence as base for LHS (more robust than random LHS)
            sampler = qmc.Halton(d=self.dim, scramble=True)
            samples = sampler.random(self.NP)

            # Apply inverse CDF transform for better uniformity
            from scipy import stats
            samples = stats.norm.ppf(np.clip(samples, 1e-10, 1 - 1e-10))

            # Scale to bounds using center and range
            center = (ub + lb) / 2.0
            half_range = range_vec / 2.0

            # Map from N(0,1) to [lb, ub] - use moderate sigma for exploration
            sigma_init = 0.5  # 50% of half-range as initial step size
            samples = center + samples * half_range * sigma_init

        except Exception:
            # Fallback: stratified grid with jitter
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(lb[d], ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                # Place points at bin centers with uniform jitter
                samples[:, d] = bins[:-1] + bin_width * (0.5 + (np.random.random(self.NP) - 0.5) * 0.8)

        # Clip to bounds ensuring numerical stability
        samples = np.clip(samples, lb, ub)

        # Ensure no duplicate individuals (replace duplicates with perturbed versions)
        unique_mask = np.ones(self.NP, dtype=bool)
        for i in range(self.NP):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, self.NP):
                if unique_mask[j]:
                    dist = np.linalg.norm(samples[i] - samples[j])
                    if dist < 1e-8 * np.mean(range_vec):
                        # Add small perturbation to avoid duplicate
                        samples[j] += np.random.randn(self.dim) * range_vec * 0.01
                        samples[j] = np.clip(samples[j], lb, ub)
                        unique_mask[j] = False

        self.population = samples

        # Initialize CMA-ES state variables
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        # Covariance matrix with numerical safety
        cov_matrix = np.cov(self.population.T)
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            cov_matrix = np.eye(self.dim) * (np.mean(range_vec) / 6.0) ** 2
        min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eig < 1e-10:
            cov_matrix += (1e-8 - min_eig) * np.eye(self.dim)
        self.C = cov_matrix

        # Step size initialization based on problem scale
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = np.clip(self.sigma, 1e-10, np.mean(range_vec) / 3.0)

        # Path vectors
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        # Evaluate initial population
        self.fitness = self.func(self.population)

        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        # Reset counters
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_04_idea_0.py (3 wins) ---
```python
def _initialize_population(self):
        """Initialize population using LHS with elite-targeted adaptive resampling."""
        try:
            from scipy.stats import qmc
            # Latin Hypercube Sampling: better space-filling than Sobol for escaping local optima
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            # Map from [0,1]^dim to [lb, ub]
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            # Fallback: stratified sampling with jitter
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        self.population = samples.copy()

        # Evaluate initial LHS population
        init_fitness = self.func(self.population)
        init_fitness = np.asarray(init_fitness).flatten()

        # For high-error tasks: perform adaptive resampling around elite regions
        # This helps escape deceptive local optima that trap Sobol-based initialization
        if np.min(init_fitness) > 1.0:  # High error detected
            n_elite = max(3, self.NP // 16)
            elite_indices = np.argsort(init_fitness)[:n_elite]
            elite_samples = self.population[elite_indices]

            # Generate additional samples around each elite region
            n_resample = self.NP // 4
            resample_per_elite = max(1, n_resample // n_elite)

            resampled = []
            for elite in elite_samples:
                # Scale spread by distance to bounds for bounded optima
                spread = 0.15 * (self.ub - self.lb)
                spread = np.clip(spread, 1.0, 50.0)

                for _ in range(resample_per_elite):
                    candidate = elite + np.random.randn(self.dim) * spread
                    candidate = np.clip(candidate, self.lb, self.ub)
                    resampled.append(candidate)

            resampled = np.array(resampled[:n_resample])

            # Evaluate resampled candidates
            if len(resampled) > 0:
                resample_fitness = np.asarray(self.func(resampled)).flatten()

                # Keep all original samples plus best resampled candidates
                combined_pop = np.vstack([self.population, resampled])
                combined_fit = np.concatenate([init_fitness, resample_fitness])

                # Select best NP individuals (elitist selection)
                keep_indices = np.argsort(combined_fit)[:self.NP]
                self.population = combined_pop[keep_indices]
                init_fitness = combined_fit[keep_indices]

        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = init_fitness[:self.NP]

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_05_idea_0.py (4 wins) ---
```python
def _initialize_population(self):
        """Initialize population using boundary-layer adaptive sampling with multi-seed starts."""
        # Phase 1: Generate base samples using Halton quasi-random sequence
        try:
            from scipy.stats import qmc
            # Use Halton (more robust than Sobol for varying dimensions)
            sampler = qmc.Halton(self.dim, scramble=True)
            base_samples = sampler.random(self.NP // 2)
            base_samples = qmc.scale(base_samples, self.lb, self.ub)
        except Exception:
            base_samples = np.random.uniform(self.lb, self.ub, (self.NP // 2, self.dim))

        # Phase 2: Generate boundary-layer samples (global optima often at boundaries)
        boundary_ratio = 0.3  # 30% of samples near boundaries
        n_boundary = max(4, int(self.NP * boundary_ratio))
        boundary_samples = np.zeros((n_boundary, self.dim))

        for i in range(n_boundary):
            # Sample from corners/edges with Gaussian falloff
            boundary_layer_width = 0.15 * (self.ub - self.lb)
            for d in range(self.dim):
                # Choose boundary side (lower or upper) with probability based on position
                side = np.random.choice([0, 1])
                if side == 0:
                    boundary_samples[i, d] = self.lb[d] + np.random.exponential(boundary_layer_width[d])
                else:
                    boundary_samples[i, d] = self.ub[d] - np.random.exponential(boundary_layer_width[d])
            boundary_samples[i] = np.clip(boundary_samples[i], self.lb, self.ub)

        # Phase 3: Multi-seed starting points (escape local optima traps)
        n_seeds = max(2, self.dim // 4)
        seed_samples = np.zeros((n_seeds, self.dim))
        for i in range(n_seeds):
            # Start from random boundary point and spread locally
            seed_center = boundary_samples[i % len(boundary_samples)].copy()
            spread = 0.2 * (self.ub - self.lb)
            seed_samples[i] = seed_center + np.random.randn(self.dim) * spread
            seed_samples[i] = np.clip(seed_samples[i], self.lb, self.ub)

        # Combine all samples
        remaining = self.NP - len(base_samples) - n_boundary - n_seeds
        if remaining > 0:
            extra_samples = np.random.uniform(self.lb, self.ub, (remaining, self.dim))
            samples = np.vstack([base_samples, boundary_samples, seed_samples, extra_samples])
        else:
            samples = np.vstack([base_samples, boundary_samples, seed_samples])

        # Ensure we have exactly NP samples
        if len(samples) > self.NP:
            samples = samples[:self.NP]
        elif len(samples) < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - len(samples), self.dim))
            samples = np.vstack([samples, extra])

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _initialize_population(self):
        """Initialize population using fitness-informed Latin Hypercube with adaptive clustering."""
        # Phase 1: Generate space-filling LHS samples
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        # Phase 2: Evaluate initial samples to understand fitness landscape
        init_fitness = self.func(samples)

        # Phase 3: Adaptive clustering based on problem difficulty
        # For high-dimensional or potentially multimodal problems, use more clusters
        n_clusters = min(max(3, self.dim // 10), 10, self.NP // 15)

        if n_clusters >= 2 and self.NP > n_clusters * 3:
            try:
                from scipy.cluster.vq import kmeans2
                # Cluster positions, then analyze cluster fitness
                centroids, labels = kmeans2(samples, n_clusters, minit='points')

                # Compute mean fitness per cluster
                cluster_fitness = np.full(n_clusters, np.inf)
                for c in range(n_clusters):
                    mask = labels == c
                    if np.sum(mask) > 0:
                        cluster_fitness[c] = np.mean(init_fitness[mask])

                # Identify best clusters (potential basins of attraction)
                best_clusters = np.argsort(cluster_fitness)[:max(1, n_clusters // 2)]

                # Build new population: 60% from best clusters, 40% uniform LHS
                n_from_best = int(0.6 * self.NP)
                n_uniform = self.NP - n_from_best

                # Sample from best clusters weighted by inverse fitness
                inv_fitness = 1.0 / (cluster_fitness[best_clusters] - np.min(cluster_fitness[best_clusters]) + 1e-6)
                cluster_probs = inv_fitness / np.sum(inv_fitness)

                new_pop = []
                for _ in range(n_from_best):
                    c = np.random.choice(best_clusters, p=cluster_probs)
                    mask = labels == c
                    idx = np.random.choice(np.where(mask)[0])
                    # Add small perturbation around cluster member
                    new_pop.append(samples[idx] + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb) / 100)

                # Fill remaining with uniform LHS samples
                indices = np.random.choice(self.NP, n_uniform, replace=False)
                for idx in indices:
                    new_pop.append(samples[idx])

                samples = np.array(new_pop)
            except Exception:
                pass  # Keep original LHS samples

        # Clip any out-of-bounds values
        samples = np.clip(samples, self.lb, self.ub)

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling with bounded random walk exploration."""
        # Latin Hypercube Sampling for better stratified coverage across dimensions
        # vs Sobol's low-discrepancy which may cluster in certain regions
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            # Fallback: stratified sampling with jitter
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            # Shuffle each dimension independently
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        # Bounded random walk exploration: perturb samples to escape
        # potential deceptive local optima regions in worst tasks (errors ~10-60)
        walk_scale = 0.15 * (self.ub - self.lb)
        walk_scale = np.maximum(walk_scale, 1e-6)
        num_walk = max(1, self.NP // 5)
        walk_perturbation = np.random.randn(num_walk, self.dim) * walk_scale
        samples[:num_walk] = samples[:num_walk] + walk_perturbation

        # Additional extreme exploration: uniform random samples in outer bounds
        num_extreme = max(1, self.NP // 10)
        extreme_samples = np.random.uniform(self.lb, self.ub, (num_extreme, self.dim))
        samples[num_walk:num_walk + num_extreme] = extreme_samples

        # Clip all samples to bounds
        samples = np.clip(samples, self.lb, self.ub)

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T)
        if self.C.size == 1:
            self.C = np.array([[max(self.C.item(), 1e-8)]])
        self.C += 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = max(self.sigma, 1e-8)
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
def _initialize_population(self):
        """Initialize population using adaptive multi-method sampling with boundary focus."""
        samples_list = []

        # Method 1: Sobol quasi-random (60% of population)
        n_sobol = int(0.6 * self.NP)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            sobol_samples = sampler.random(n_sobol)
            sobol_samples = qmc.scale(sobol_samples, self.lb, self.ub)
            samples_list.append(sobol_samples)
        except Exception:
            pass

        # Method 2: Latin Hypercube for better dimension-wise independence (25% of population)
        n_lhs = int(0.25 * self.NP)
        try:
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            lhs_samples = sampler.random(n_lhs)
            lhs_samples = qmc.scale(lhs_samples, self.lb, self.ub)
            samples_list.append(lhs_samples)
        except Exception:
            pass

        # Method 3: Boundary-focused sampling with adaptive density (15% of population)
        # For ill-conditioned/multimodal tasks, optima often lie near boundaries
        n_boundary = self.NP - sum(len(s) for s in samples_list)
        if n_boundary > 0:
            boundary_samples = np.zeros((n_boundary, self.dim))
            # Use power-law scaling to concentrate samples near boundaries
            # pow=3 creates strong concentration near edges
            u = np.random.random((n_boundary, self.dim))
            power = 3.0
            scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
            # Randomly choose which dimensions get boundary treatment
            dim_mask = np.random.random(self.dim) < 0.3
            boundary_samples[:, dim_mask] = scaled[:, dim_mask]
            boundary_samples[:, ~dim_mask] = u[:, ~dim_mask]
            # Scale to bounds
            for d in range(self.dim):
                boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])
            samples_list.append(boundary_samples)

        # Combine all samples
        if samples_list:
            samples = np.vstack(samples_list)
        else:
            # Ultimate fallback: simple random
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        # Ensure exact size and numerical robustness
        if samples.shape[0] > self.NP:
            samples = samples[:self.NP]
        elif samples.shape[0] < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
            samples = np.vstack([samples, extra])

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _initialize_population(self):
        """Initialize population using Latin Hypercube + fitness-guided niching with Cauchy perturbations."""
        import numpy as np

        # Use Latin Hypercube for better space-filling than Sobol
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        # Evaluate initial samples
        init_fitness = self.func(samples)

        # Identify top solutions as niche centers
        sorted_indices = np.argsort(init_fitness)
        num_elites = max(1, self.NP // 10)
        elite_indices = sorted_indices[:num_elites]

        # Create niches around elite solutions
        num_niches = min(num_elites, 4)
        samples_per_niche = self.NP // num_niches

        population = []

        for niche_id in range(num_niches):
            elite_idx = elite_indices[niche_id % len(elite_indices)]
            center = samples[elite_idx]

            # Add elite solution
            population.append(center)

            # Generate Cauchy perturbations around niche center (heavy-tailed)
            remaining = samples_per_niche - 1
            for _ in range(remaining):
                scale = 0.2 * (self.ub - self.lb)
                perturbation = np.random.standard_cauchy(self.dim) * scale
                candidate = center + perturbation
                candidate = np.clip(candidate, self.lb, self.ub)
                population.append(candidate)

        # Fill remaining slots with LHS samples
        remaining_slots = self.NP - len(population)
        if remaining_slots > 0:
            extra = sampler.random(remaining_slots)
            extra = qmc.scale(extra, self.lb, self.ub)
            population.extend(extra)

        self.population = np.array(population[:self.NP])
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
```

# --- From variant_10_idea_0.py (5 wins) ---
```python
def _initialize_population(self):
        """Initialize population with restart-guided exploration bias."""
        # Standard quasi-random base (same as original)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        # Restart-guided anti-centralization: bias away from previously explored regions
        if hasattr(self, 'restart_archive') and len(self.restart_archive) >= 3:
            archive_array = np.array(self.restart_archive)
            archive_centroid = np.mean(archive_array, axis=0)

            # Compute typical spread scale of archive
            distances = np.linalg.norm(archive_array - archive_centroid, axis=1)
            escape_scale = max(np.median(distances), 1.0)

            # Bias samples away from archive centroid
            for i in range(self.NP):
                to_archive = samples[i] - archive_centroid
                dist = np.linalg.norm(to_archive)
                if dist < escape_scale * 0.8:
                    # Push away from archive with Gaussian perturbation
                    direction = to_archive / (dist + 1e-10)
                    push_dist = escape_scale * (0.5 + 0.5 * np.random.random())
                    samples[i] = samples[i] + direction * push_dist
                    samples[i] += np.random.randn(self.dim) * escape_scale * 0.3

            # Clip to bounds
            samples = np.clip(samples, self.lb, self.ub)

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
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


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Sobol quasi-random sequences."""
        # Generate Sobol sequence samples (quasi-random, low-discrepancy)
        # Falls back to stratified sampling if Sobol unavailable
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            # Map from [0,1]^dim to [lb, ub]
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            # Fallback: stratified sampling with jitter
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            # Shuffle each dimension independently
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        # More efficient than eigendecomposition for sampling
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        # Sample from standard normal and transform
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivors_batch(self):
        """Select survivors via elitist (mu, lambda)-selection."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = self.fitness[i] - self.trial_fitness[i]
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Initialize cumulative tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay to accumulated reward (recent rewards weighted more)
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Log-scaled improvement reward (more sensitive to small improvements)
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Explicit stagnation detection
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Exploration bonus when stagnant and diversity is low
        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        # Update cumulative weighted reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize by decay sum to get comparable reward values
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_08()
        else:
            self._adapt_covariance_variant_09()
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_01(self):
        """Archive-guided covariance perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Initialize archive for tracking distinct best solutions
        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        # Add current best to archive if distinct enough
        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for arch_pos in self.archive_positions:
            dist = np.linalg.norm(self.x_opt - arch_pos)
            if dist < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.archive_positions) < 10:
            self.archive_positions.append(self.x_opt.copy())
            self.archive_fitness.append(self.f_opt)
            self.archive_generations.append(self.generation)
        elif is_distinct and len(self.archive_positions) >= 10:
            # Replace worst entry
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        # Compute diversity and condition metrics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        # Compute archive-based escape direction
        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            # Direction from mean toward archive centroid, weighted by diversity
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                # Also consider directions to individual archive members
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        # Perturb along diverse directions
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            # Add global exploration along principal axes
            escape_perturb += 0.1 * np.eye(self.dim)

        # Adaptive learning rates with exploration boost when trapped
        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with archive-based perturbation
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Inject escape perturbation when trapped
        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        # Full restart on extreme ill-conditioning
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_06(self):
        """Diversity-sensitive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
        
        if is_converging:
            scale = 10.0
        else:
            scale = 1.0 + 5.0 * min(rel_var, 0.1)
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Evolution path update (unchanged)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with positive/negative decomposition
        rank_mu_pos = np.zeros((self.dim, self.dim))
        rank_mu_neg = np.zeros((self.dim, self.dim))

        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            weight = self.weights[i]
            outer = np.outer(diff, diff)

            if weight > 0:
                rank_mu_pos += weight * outer
            else:
                rank_mu_neg += abs(weight) * outer

        # Normalize by sum of positive and negative weights separately
        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        # Combine positive and negative updates (active CMA-ES core)
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking for escaping local optima."""
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt - self.f_opt
        self.prev_f_opt = self.f_opt

        fitness_gradient = max(abs(delta_f_opt), 1e-15)
        temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
        self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * self.exploration_temp
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute diversity using percentile-robust and condition-aware metrics.

        Key differences from ESS-based approach:
        - Uses MAX-normalized spread (catches catastrophic collapse better)
        - Uses fitness percentile range (detects local optima trapping)
        - Uses condition number directly (detects rank deficiency)
        - Uses population spread ratio (detects severe elongation)

        This makes the metric more sensitive to extreme states that cause
        catastrophic failure on multimodal/ill-conditioned tasks.
        """
        # Max-normalized spread: captures catastrophic collapse better than trace
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        spread_norm = np.clip(spread_norm, 0.0, 2.0)

        # Fitness percentile range: detects local optima trapping
        sorted_fit = np.sort(self.fitness)
        fit_range = sorted_fit[-1] - sorted_fit[0]
        fit_median = sorted_fit[len(sorted_fit) // 2]
        fit_p10 = sorted_fit[max(0, len(sorted_fit) // 10 - 1)]
        fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
        fit_percentile_range = np.clip(fit_percentile_range, 0.0, 1.0)

        # Condition number: directly detects rank deficiency
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        cond_norm = 1.0 / np.log1p(cond)
        cond_norm = np.clip(cond_norm, 0.0, 1.0)

        # Spread ratio: detects severe elongation (ratio of max to min axis lengths)
        diag = np.diag(self.C)
        diag = np.maximum(diag, 1e-15)
        max_var = np.max(diag)
        min_var = np.min(diag)
        spread_ratio = min_var / max_var
        spread_ratio = np.clip(spread_ratio, 0.0, 1.0)

        # Weighted combination: emphasize condition and spread ratio for ill-conditioned tasks
        diversity = (
            0.25 * spread_norm +
            0.35 * fit_percentile_range +
            0.25 * cond_norm +
            0.15 * spread_ratio
        )

        return float(np.clip(diversity, 1e-15, None))
    
    def _check_stagnation(self):
        """Track stagnation counter with scale-adaptive threshold.

        Key insight: The original fixed threshold (1e-12) is orders of magnitude
        too small for high-error tasks (error ~10-500). For Task 17 (error ~538),
        even a 0.001 improvement is meaningful, but 1e-12 is numerically invisible.

        This version uses: threshold = max(1e-6 * |f_opt|, 1e-10)
        - High-error tasks (|f_opt|~100): threshold ~1e-4 → proper stagnation detection
        - Low-error tasks (|f_opt|~1e-8): threshold ~1e-10 → fine-grained detection
        """
        improvement = self.f_opt_prev - self.f_opt

        # Adaptive threshold: scale with problem difficulty
        base_scale = max(abs(self.f_opt), 1.0)
        rel_threshold = 1e-6 * base_scale
        threshold = max(rel_threshold, 1e-10)

        if improvement > threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Scale max_stagnation with problem difficulty for harder tasks
        if abs(self.f_opt) > 10.0:
            self.max_stagnation = 150 + self.dim * 5
        elif abs(self.f_opt) > 1.0:
            self.max_stagnation = 100 + self.dim * 4
        else:
            self.max_stagnation = 50 + self.dim * 3

        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0

```