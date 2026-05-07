Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_restart_if_needed` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      2.464603e-07            2.371489e-07            5.527066e-06            2.076796e-07            2.412081e-07            1.005623e-07            8.985376e-08            6.429383e+01            4.946170e+01            2.369871e-07            -inf                    
1      3.453528e-07            3.878479e-07            3.037698e-07            3.346036e-07            3.478475e-07            2.056079e-07            1.557806e-07            1.278415e+02            1.004777e+02            3.292309e-07            -inf                    
2      3.635203e-05            5.858800e-05            7.333424e-05            5.296906e-05            2.958227e-05            5.727224e-05            2.585734e-05            5.332244e+02            4.607252e+02            7.946541e-05            -inf                    
3      3.705954e-04            3.729492e-04            3.661809e-04            3.785732e-04            3.633143e-04            3.530052e-04            3.351154e-04            2.479892e+02            2.182300e+02            3.487393e-04            -inf                    
4      2.393397e-02            2.303957e-02            2.291095e-02            2.308844e-02            2.328036e-02            2.302459e-02            2.370888e-02            5.351027e+00            4.919575e+00            2.369124e-02            -inf                    
5      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.210689e+09            7.336402e+08            1.000000e-08            -inf                    
6      5.265582e-06            5.954771e-06            4.531955e-06            5.972208e-06            5.872990e-06            2.617975e-05            5.322803e-07            6.704648e+03            4.862059e+03            6.520497e-06            -inf                    
7      1.408639e-05            1.428271e-05            1.413138e-05            1.378288e-05            1.390606e-05            1.476735e-05            1.278602e-05            3.036107e+02            2.611877e+02            1.449338e-05            -inf                    
8      1.841776e-03            1.777485e-03            1.886100e-03            1.819825e-03            1.736399e-03            1.845126e-03            1.789464e-03            4.414471e+01            3.923191e+01            1.782365e-03            -inf                    
9      6.729192e-04            5.151177e-04            5.233912e-04            5.061522e-04            4.744363e+00            4.985284e-04            9.676610e-04            3.378667e+02            2.894318e+02            5.245387e-04            -inf                    
10     2.340872e+01            5.231700e+01            2.464477e+01            2.308725e+01            2.272314e+01            5.026029e+01            4.702060e+01            3.524234e+02            3.331272e+02            1.701618e+01            -inf                    
11     1.334735e+01            1.003871e+02            3.811569e+01            8.419700e+00            8.383939e+00            7.663316e+01            1.003145e+02            1.443115e+03            1.460112e+03            8.931974e+00            -inf                    
12     2.141769e+01            5.550755e+01            1.969229e+01            1.476585e+01            1.933307e+01            3.778415e+01            3.903508e+01            2.575411e+02            2.519139e+02            1.546733e+01            -inf                    
13     2.869833e+00            1.635198e+01            2.540902e+00            2.175422e+00            2.641985e+00            5.748419e+00            1.154982e+01            4.799125e+01            4.431977e+01            2.752610e+00            -inf                    
14     2.640305e+00            3.064878e+00            2.632634e+00            2.654684e+00            2.671798e+00            2.787882e+00            3.291813e+00            1.718815e+01            1.597889e+01            2.664143e+00            -inf                    
15     2.869528e+00            3.507459e+00            2.828631e+00            2.711022e+00            2.791877e+00            3.220616e+00            3.492463e+00            4.285570e+00            4.306597e+00            2.755349e+00            -inf                    
16     5.172241e+01            9.249155e+01            7.184235e+01            5.246275e+01            6.497418e+01            6.153959e+01            9.627048e+01            2.101371e+04            1.327673e+04            5.705870e+01            -inf                    
17     7.442387e+01            4.389991e+03            4.232936e+02            2.514159e+02            3.248636e+02            5.494063e+03            1.014248e+04            2.242098e+05            2.271223e+05            4.924543e+02            -inf                    
18     2.296373e+01            2.938399e+01            1.883197e+01            2.157034e+01            1.722373e+01            2.683295e+01            2.938709e+01            8.367971e+01            8.892885e+01            1.914747e+01            -inf                    
19     1.131994e+01            2.788075e+01            1.354067e+01            1.684425e+01            1.248548e+01            2.171952e+01            1.782438e+01            4.270973e+02            3.699636e+02            1.093114e+01            -inf                    
20     2.165091e+01            2.782288e+01            2.158526e+01            2.140953e+01            2.122661e+01            2.231282e+01            2.513251e+01            6.134705e+01            6.032139e+01            2.017428e+01            -inf                    
21     4.297318e+00            4.677710e+00            4.386348e+00            4.546450e+00            4.357257e+00            4.587614e+00            4.691443e+00            5.900600e+00            5.906077e+00            4.369911e+00            -inf                    
22     1.042633e+01            1.245316e+01            1.030514e+01            9.985439e+00            8.600159e+00            1.149048e+01            1.255102e+01            2.183612e+01            2.198957e+01            9.809853e+00            -inf                    
23     1.926942e+01            2.039177e+01            1.719295e+01            1.572643e+01            1.622518e+01            1.846809e+01            1.966288e+01            1.198175e+02            1.138020e+02            1.604958e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_idea_0.py  (error=8.985376e-08)
Task  1: variant_06_idea_0.py  (error=1.557806e-07)
Task  2: variant_06_idea_0.py  (error=2.585734e-05)
Task  3: variant_06_idea_0.py  (error=3.351154e-04)
Task  4: variant_02_idea_0.py  (error=2.291095e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_06_idea_0.py  (error=5.322803e-07)
Task  7: variant_06_idea_0.py  (error=1.278602e-05)
Task  8: variant_04_idea_0.py  (error=1.736399e-03)
Task  9: variant_05_idea_0.py  (error=4.985284e-04)
Task 10: variant_09_idea_0.py  (error=1.701618e+01)
Task 11: variant_04_idea_0.py  (error=8.383939e+00)
Task 12: variant_03_idea_0.py  (error=1.476585e+01)
Task 13: variant_03_idea_0.py  (error=2.175422e+00)
Task 14: variant_02_idea_0.py  (error=2.632634e+00)
Task 15: variant_03_idea_0.py  (error=2.711022e+00)
Task 16: original.py  (error=5.172241e+01)
Task 17: original.py  (error=7.442387e+01)
Task 18: variant_04_idea_0.py  (error=1.722373e+01)
Task 19: variant_09_idea_0.py  (error=1.093114e+01)
Task 20: variant_09_idea_0.py  (error=2.017428e+01)
Task 21: original.py  (error=4.297318e+00)
Task 22: variant_04_idea_0.py  (error=8.600159e+00)
Task 23: variant_03_idea_0.py  (error=1.572643e+01)

WIN COUNTS:
  variant_06_idea_0.py: 6 wins
  variant_04_idea_0.py: 4 wins
  variant_03_idea_0.py: 4 wins
  variant_09_idea_0.py: 3 wins
  original.py: 3 wins
  variant_02_idea_0.py: 2 wins
  variant_05_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (2 wins) ---
```python
def _restart_if_needed(self):
        """Restart population using archive-guided diversity with adaptive intensity.

        For high-error tasks (10-75 range), the current single-elite restart is
        insufficient because it discards all learned information. This version:
        1. Uses multiple archive solutions (if available) as seeds
        2. Scales restart intensity with stagnation severity and error magnitude
        3. Samples around archive solutions proportionally to their fitness
        4. Adds boundary-focused exploration for severe trapping
        """
        diversity = self._compute_diversity()

        # Determine restart intensity: 0=normal restart, 1=aggressive escape
        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
        error_magnitude = np.log1p(max(abs(self.f_opt), 1.0))
        intensity = np.clip(0.3 * stagnation_ratio + 0.1 * error_magnitude / 10.0, 0.0, 1.0)

        should_restart = (self.stagnation_counter > self.max_stagnation or 
                          diversity < self.min_diversity or 
                          np.any(np.isnan(self.C)))

        if not should_restart:
            return

        # Collect archive solutions if available
        archive_solutions = []
        if hasattr(self, 'archive_positions') and len(self.archive_positions) >= 2:
            # Sort archive by fitness
            sorted_indices = np.argsort(self.archive_fitness)
            for idx in sorted_indices[:min(5, len(self.archive_positions))]:
                archive_solutions.append(self.archive_positions[idx].copy())

        # Get current best
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        # Reinitialize base population
        self._initialize_population()

        # Determine how many slots to fill with archive-guided vs random samples
        n_archive = int(np.floor(self.NP * intensity * 0.5))
        n_archive = min(n_archive, len(archive_solutions), self.NP - 1)

        # Fill with archive-guided samples (proportional to fitness rank)
        if n_archive > 0 and archive_solutions:
            for i in range(n_archive):
                if i >= len(archive_solutions):
                    break
                # Sample around archive solution with adaptive spread
                spread = self.sigma * (2.0 + 3.0 * intensity)
                sample = archive_solutions[i] + np.random.randn(self.dim) * spread
                sample = self._clip_to_bounds(sample)
                self.population[i + 1] = sample

        # Fill remaining with boundary-focused exploration
        n_boundary = self.NP - 1 - n_archive
        if n_boundary > 0:
            # Power-law sampling for strong boundary focus on high-intensity restarts
            power = 2.0 + 3.0 * intensity
            u = np.random.random((n_boundary, self.dim))
            boundary_samples = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)

            # Apply to random subset of dimensions (more for aggressive restarts)
            dim_fraction = 0.2 + 0.4 * intensity
            dim_mask = np.random.random(self.dim) < dim_fraction
            for d in range(self.dim):
                if not dim_mask[d]:
                    boundary_samples[:, d] = np.random.uniform(0, 1, n_boundary)

            # Scale to bounds
            for d in range(self.dim):
                boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])

            start_idx = 1 + n_archive
            self.population[start_idx:start_idx + n_boundary] = boundary_samples

        # Restore elite as first individual
        self.population[0] = elite
        self.fitness[0] = elite_fit

        # Update state
        self.f_opt = float(np.asarray(elite_fit).flatten()[0])
        self.x_opt = elite.copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0

        # Reset sigma and covariance for aggressive restarts
        if intensity > 0.5:
            self.sigma = max(self.sigma, 0.5 * (self.ub[0] - self.lb[0]) / 6.0)
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
```

# --- From variant_03_idea_0.py (4 wins) ---
```python
def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost.

        Key improvements for high-error tasks:
        - Preserve multiple diverse elites (not just 1) based on problem difficulty
        - Inject elites at evenly spaced positions for diverse coverage
        - Inject additional random solutions on subsequent restarts to break cycles
        """
        diversity = self._compute_diversity()

        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):

            # Save best and top diverse solutions before reinitialization
            sorted_indices = np.argsort(self.fitness)
            num_elites = min(5, self.NP // 4)
            elite_solutions = self.population[sorted_indices[:num_elites]].copy()
            elite_fitness = self.fitness[sorted_indices[:num_elites]].copy()

            self._initialize_population()

            # Restore best solution
            self.population[0] = elite_solutions[0]
            self.fitness[0] = elite_fitness[0]
            self.f_opt = float(np.asarray(elite_fitness[0]).flatten()[0])
            self.x_opt = elite_solutions[0].copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0

            # Inject remaining elites at evenly spaced positions for diversity
            for i in range(1, num_elites):
                idx = (i * self.NP) // num_elites
                if idx < self.NP:
                    self.population[idx] = elite_solutions[i]
                    self.fitness[idx] = elite_fitness[i]

            # On subsequent restarts, inject random diversity to break cycles
            if hasattr(self, 'restart_counter'):
                self.restart_counter += 1
                if self.restart_counter > 1:
                    # Inject random solutions in latter half of population
                    n_diverse = min(10, self.NP // 4)
                    for _ in range(n_diverse):
                        idx = np.random.randint(self.NP // 4, self.NP)
                        self.population[idx] = np.random.uniform(self.lb, self.ub)
            else:
                self.restart_counter = 1
```

# --- From variant_04_idea_0.py (4 wins) ---
```python
def _restart_if_needed(self):
        """Restart population using covariance-directed multi-point sampling with archive injection."""
        diversity = self._compute_diversity()

        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):

            # Preserve elite
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]

            # Get eigendecomposition of current covariance for directional restart
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-15)
            except np.linalg.LinAlgError:
                eigvals = np.ones(self.dim)
                eigvecs = np.eye(self.dim)

            # Generate restart points along principal axes
            num_restart_points = min(5, self.NP // 10)
            restart_points = []

            # Point 1: Elite (already saved)
            restart_points.append(elite.copy())

            # Points along top eigendirections (exploit known geometry)
            for i in range(num_restart_points - 1):
                direction = eigvecs[:, -(i + 1)]
                scale = np.sqrt(eigvals[-(i + 1)]) * 2.0
                # Alternate between positive and negative directions
                sign = 1.0 if i % 2 == 0 else -1.0
                offset = sign * scale * direction
                restart_point = self._clip_to_bounds(elite + offset * (0.3 + 0.7 * np.random.random()))
                restart_points.append(restart_point)

            # Build new population: inject restart points, fill rest with quasi-random
            new_population = []
            for pt in restart_points:
                new_population.append(pt)

            # Fill remaining with boundary-focused Latin Hypercube for diversity
            remaining = self.NP - len(new_population)
            if remaining > 0:
                try:
                    from scipy.stats import qmc
                    sampler = qmc.LatinHypercube(self.dim, scramble=True)
                    lhs_samples = qmc.scale(sampler.random(remaining), self.lb, self.ub)
                    new_population.extend([lhs_samples[i] for i in range(remaining)])
                except Exception:
                    # Fallback to uniform
                    for _ in range(remaining):
                        new_population.append(np.random.uniform(self.lb, self.ub))

            # Reinitialize population and inject restart points
            self._initialize_population()

            # Overwrite first positions with our curated restart points
            for i, pt in enumerate(restart_points):
                if i < self.NP:
                    self.population[i] = pt

            # Re-evaluate fitness for all individuals
            self.fitness = self.func(self.population)

            # Update best with elite
            self.population[0] = elite
            self.fitness[0] = elite_fit

            best_new_idx = np.argmin(self.fitness)
            self.f_opt = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
            self.x_opt = self.population[best_new_idx].copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _restart_if_needed(self):
        """Restart population with elite-directed focused reinitialization."""
        diversity = self._compute_diversity()

        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):

            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]

            # Compute condition number to detect ill-conditioned state
            eigvals = np.linalg.eigvalsh(self.C)
            eig_min = np.min(eigvals)
            eig_max = np.max(eigvals)
            cond = eig_max / max(eig_min, 1e-15)

            # Determine restart strategy based on condition and stagnation severity
            is_ill_conditioned = cond > 1e6
            is_deeply_stagnant = self.stagnation_counter > 2 * self.max_stagnation

            # Compute current step size scale
            sigma_scale = max(self.sigma, 1e-6)

            # Strategy 1: Elite-directed restart (default for multimodal)
            # Sample from anisotropic distribution around elite using current covariance
            if not is_ill_conditioned:
                # Use Cholesky of current covariance for correlated sampling
                try:
                    L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
                except np.linalg.LinAlgError:
                    diag_C = np.diag(self.C)
                    diag_C = np.maximum(diag_C, 1e-10)
                    L = np.diag(np.sqrt(diag_C))

                # Scale factor: larger for deeply stagnant, smaller for moderate
                if is_deeply_stagnant:
                    exploration_radius = 5.0 * sigma_scale
                else:
                    exploration_radius = 2.0 * sigma_scale

                # Sample new population around elite
                n_new = self.NP - 1
                z = np.random.randn(n_new, self.dim)
                new_pop = elite + exploration_radius * (z @ L.T)
                new_pop = self._clip_to_bounds(new_pop)

                # Strategy 2: Hyper-sphere restart (for ill-conditioned CMA state)
            else:
                # Reset covariance to spherical for numerical stability
                self.C = np.eye(self.dim) * np.mean(eigvals)
                self.pc = np.zeros(self.dim)
                self.L = None

                # Sample uniformly in hyper-sphere around elite
                n_new = self.NP - 1
                exploration_radius = max(10.0 * sigma_scale, 0.1 * (self.ub[0] - self.lb[0]))
                u = np.random.randn(n_new, self.dim)
                norms = np.linalg.norm(u, axis=1, keepdims=True)
                # Random radii for uniform hyper-sphere sampling
                radii = np.random.random((n_new, 1)) ** (1.0 / self.dim)
                u = u / (norms + 1e-10) * radii * exploration_radius
                new_pop = elite + u
                new_pop = self._clip_to_bounds(new_pop)

            # Reinitialize rest of population with boundary-focused sampling
            n_remain = self.NP - 1 - len(new_pop)
            if n_remain > 0:
                # Use boundary-focused sampling for multimodal landscapes
                boundary_ratio = 0.4 if is_deeply_stagnant else 0.25
                n_boundary = int(boundary_ratio * n_remain)
                n_random = n_remain - n_boundary

                remainder = []
                if n_random > 0:
                    random_samples = np.random.uniform(self.lb, self.ub, (n_random, self.dim))
                    remainder.append(random_samples)

                if n_boundary > 0:
                    u = np.random.random((n_boundary, self.dim))
                    power = 3.0
                    scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
                    dim_mask = np.random.random(self.dim) < 0.3
                    boundary = np.zeros((n_boundary, self.dim))
                    boundary[:, dim_mask] = scaled[:, dim_mask]
                    boundary[:, ~dim_mask] = u[:, ~dim_mask]
                    for d in range(self.dim):
                        boundary[:, d] = self.lb[d] + boundary[:, d] * (self.ub[d] - self.lb[d])
                    remainder.append(boundary)

                if remainder:
                    new_pop = np.vstack([new_pop] + remainder)

            # Ensure exact population size
            if len(new_pop) < self.NP:
                extra = np.random.uniform(self.lb, self.ub, (self.NP - len(new_pop), self.dim))
                new_pop = np.vstack([new_pop, extra])
            elif len(new_pop) > self.NP:
                new_pop = new_pop[:self.NP]

            self.population = new_pop
            self.mean = np.mean(self.population, axis=0)
            self.old_mean = self.mean.copy()

            # Keep current covariance structure (don't reset C for non-ill-conditioned cases)
            if not is_ill_conditioned:
                # Dampen covariance slightly to allow re-exploration
                self.C = 0.9 * self.C + 0.1 * np.eye(self.dim) * np.trace(self.C) / self.dim
            # For ill-conditioned case, C was already reset to spherical above

            self.sigma = np.clip(self.sigma * 1.5, 1e-6, 10.0)
            self.ps = np.zeros(self.dim)
            self.pc = np.zeros(self.dim)

            self.fitness = self.func(self.population)

            # Insert elite at best position
            best_new_idx = np.argmin(self.fitness)
            if self.fitness[best_new_idx] > elite_fit:
                # Replace worst individual with elite
                worst_idx = np.argmax(self.fitness)
                self.population[worst_idx] = elite
                self.fitness[worst_idx] = elite_fit

            # Update best
            best_idx = np.argmin(self.fitness)
            self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
            self.x_opt = self.population[best_idx].copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```

# --- From variant_06_idea_0.py (6 wins) ---
```python
def _restart_if_needed(self):
        """Restart with covariance-preserving re-seeding around elite for ill-conditioned tasks."""
        diversity = self._compute_diversity()

        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):

            # Save elite and current state
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]

            # Preserve covariance structure but increase variance for exploration
            # Get eigenvalues and eigenvectors of current covariance
            eigvals, eigvecs = np.linalg.eigh(self.C)

            # Increase variance along principal axes (1.5x for mild, 3x for severe stagnation)
            stagnation_severity = min(self.stagnation_counter / (2.0 * self.max_stagnation), 1.0)
            variance_scale = 1.5 + 1.5 * stagnation_severity

            # Reconstruct covariance with scaled eigenvalues
            scaled_eigvals = eigvals * variance_scale
            C_scaled = eigvecs @ np.diag(scaled_eigvals) @ eigvecs.T
            C_scaled = self._ensure_positive_definite(C_scaled)

            # Generate new population centered on elite using scaled covariance
            n_elite_preserve = min(5, self.NP // 10)
            n_scaled = self.NP - n_elite_preserve

            # Ensure positive definiteness for Cholesky
            min_eig = np.min(np.linalg.eigvalsh(C_scaled))
            if min_eig < 1e-8:
                C_scaled += (1e-7 - min_eig) * np.eye(self.dim)

            try:
                L = np.linalg.cholesky(C_scaled)
            except np.linalg.LinAlgError:
                diag_C = np.diag(C_scaled)
                diag_C = np.maximum(diag_C, 1e-10)
                L = np.diag(np.sqrt(diag_C))

            # Sample around elite
            z = np.random.randn(n_scaled, self.dim)
            samples = elite + self.sigma * (z @ L.T)
            samples = self._clip_to_bounds(samples)

            # Add diverse variants of elite (perturb along eigenvectors)
            for i in range(n_elite_preserve):
                # Generate perturbation along principal components
                pert_z = np.random.randn(self.dim)
                pert = self.sigma * 2.0 * (pert_z @ L.T)
                sample = elite + pert
                samples = np.vstack([samples, self._clip_to_bounds(sample)])

            # Fill remaining with boundary-focused diversity
            n_boundary = self.NP - len(samples)
            if n_boundary > 0:
                u = np.random.random((n_boundary, self.dim))
                power = 3.0
                boundary = np.where(
                    np.random.random((n_boundary, self.dim)) > 0.5,
                    np.power(u, power),
                    1.0 - np.power(u, power)
                )
                dim_mask = np.random.random((n_boundary, self.dim)) < 0.3
                boundary = np.where(dim_mask, boundary, u)
                for d in range(self.dim):
                    boundary[:, d] = self.lb[d] + boundary[:, d] * (self.ub[d] - self.lb[d])
                samples = np.vstack([samples, boundary])

            # Ensure exact size
            if samples.shape[0] > self.NP:
                samples = samples[:self.NP]
            elif samples.shape[0] < self.NP:
                extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
                samples = np.vstack([samples, extra])

            # Apply new population and state
            self.population = samples
            self.mean = np.mean(self.population, axis=0)
            self.old_mean = self.mean.copy()

            # Update covariance with scaled version (preserves structure, increases spread)
            self.C = C_scaled
            self.sigma = np.clip(self.sigma * 1.5, 1e-10, 10.0)

            self.fitness = self.func(self.population)

            # Restore elite
            self.population[0] = elite
            self.fitness[0] = elite_fit

            # Update best
            best_new_idx = np.argmin(self.fitness)
            if self.fitness[best_new_idx] < elite_fit:
                self.f_opt = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
                self.x_opt = self.population[best_new_idx].copy()
            else:
                self.f_opt = float(np.asarray(elite_fit).flatten()[0])
                self.x_opt = elite.copy()

            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
            self.generation = 0
            self.pc = np.zeros(self.dim)
            self._reset_operator_state()
```

# --- From variant_09_idea_0.py (3 wins) ---
```python
def _restart_if_needed(self):
        """Adaptive restart with hyperbolic step boosting for high-error tasks."""
        diversity = self._compute_diversity()

        if not (self.stagnation_counter > self.max_stagnation or 
                diversity < self.min_diversity or 
                np.any(np.isnan(self.C))):
            return

        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        # Compute error magnitude for adaptive perturbation sizing
        error_mag = max(abs(elite_fit), 1.0)
        # Hyperbolic scaling: large errors get larger perturbations
        # log scale bridges 1e-8 (target) to 1e+2 (worst tasks)
        log_error = np.log10(error_mag + 1e-15)
        target_log = -8.0
        decades_from_target = max(log_error - target_log, 0.0)

        # Adaptive perturbation radius: scales with problem difficulty
        bound_range = max(self.ub[0] - self.lb[0], 1e-6)
        # Perturbation grows exponentially with decades from target
        # 0 decades → 0.1% of range; 10 decades → 30% of range
        perturb_factor = np.tanh(decades_from_target / 5.0) * 0.3
        perturb_radius = perturb_factor * bound_range
        perturb_radius = np.clip(perturb_radius, 1e-6, bound_range * 0.5)

        # Generate new population centered on elite with adaptive Gaussian perturbation
        new_pop = np.zeros((self.NP, self.dim))
        for i in range(self.NP):
            # Gaussian perturbation around elite, scaled by problem difficulty
            if i == 0:
                new_pop[i] = elite.copy()
            else:
                # Use isotropic Gaussian with adaptive radius
                new_pop[i] = elite + np.random.randn(self.dim) * perturb_radius

        # Clip to bounds
        new_pop = self._clip_to_bounds(new_pop)

        # Update state: keep new population, recompute covariance
        self.population = new_pop
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        # Recompute covariance from new population (learns new structure)
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.C = self._ensure_positive_definite(self.C)

        # Adapt sigma based on perturbation radius
        self.sigma = np.clip(perturb_radius / 3.0, 1e-10, 10.0)

        # Reset evolution paths for fresh covariance adaptation
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.L = None

        # Evaluate new population
        self.fitness = self.func(self.population)

        # Update best
        new_best_idx = np.argmin(self.fitness)
        new_best_fit = float(np.asarray(self.fitness[new_best_idx]).flatten()[0])
        if new_best_fit < elite_fit:
            self.f_opt = new_best_fit
            self.x_opt = self.population[new_best_idx].copy()
        else:
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()

        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
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