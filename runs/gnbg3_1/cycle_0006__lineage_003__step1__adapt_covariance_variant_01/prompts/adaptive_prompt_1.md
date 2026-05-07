Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_01` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.550935e-02            5.234138e-02            5.749018e-02            5.643633e-02            5.746581e-02            5.509309e-02            5.792037e-02            5.747688e-02            5.748195e-02            5.774076e-02            5.314496e-02            
1      4.445035e-02            5.553358e-02            4.572951e-02            4.420793e-02            4.367142e-02            4.305392e-02            4.466615e-02            4.553484e-02            4.251895e-02            4.460519e-02            4.447001e-02            
2      1.493582e-01            5.814459e+00            1.533103e-01            1.503147e-01            1.529214e-01            1.465711e-01            2.059457e-01            1.512804e-01            2.857355e+00            2.142327e-01            2.877613e-01            
3      2.608754e-01            3.051276e+00            2.511658e-01            1.333002e-01            1.843699e-01            1.328460e-01            1.759578e-01            1.484936e-01            1.768164e-01            1.795324e-01            1.347502e-01            
4      5.181382e-01            1.073770e+00            5.179979e-01            5.090979e-01            5.158287e-01            4.990893e-01            5.221431e-01            5.226853e-01            5.735940e-01            5.205686e-01            5.231861e-01            
5      1.208062e-04            1.190394e-03            1.323643e-04            1.365615e-04            1.242989e-04            8.305665e-05            1.325261e-04            1.465418e-04            1.138733e-04            9.116106e-05            9.226954e-05            
6      3.018473e-02            3.422135e-02            1.524981e-02            1.572636e-02            1.515402e-02            1.636728e-02            2.425400e-02            1.547102e-02            2.172253e-02            2.071783e-02            4.082087e-02            
7      1.535840e-01            2.441236e+00            2.617999e-01            9.176093e-02            7.963824e-02            7.456005e-02            8.368419e-02            8.612958e-02            1.194705e-01            8.815794e-02            8.003612e-02            
8      2.568061e-01            2.118693e+00            2.597854e-01            2.660873e-01            2.539921e-01            2.402331e-01            2.530114e-01            4.237315e-01            4.450694e-01            3.208090e-01            2.510861e-01            
9      6.916313e-02            7.896205e+00            6.883953e-02            8.114958e-02            9.463418e-02            6.928331e-02            6.867017e-02            7.166003e-02            9.948686e-02            1.396055e-01            7.031523e-02            
10     2.054512e+01            7.579219e+00            1.332966e+01            3.506015e+01            2.720847e+01            3.200037e+01            1.454249e+01            1.734653e+01            2.832885e+01            2.892463e+01            1.880487e+01            
11     2.991904e+01            4.300809e+01            1.924741e+01            6.218171e+01            9.706274e+01            9.915354e+01            5.110470e+01            3.001541e+01            3.253423e+01            3.069354e+01            1.598485e+01            
12     1.723485e+01            3.822727e+00            2.389250e+01            2.175194e+01            3.970809e+01            3.154604e+01            3.442962e+01            2.579893e+01            1.789085e+01            2.130501e+01            1.843518e+01            
13     3.404627e+00            6.062204e+00            2.372590e+00            2.982746e+00            4.093003e+00            3.237193e+00            3.494470e+00            3.230465e+00            4.032756e+00            1.777210e+00            2.988347e+00            
14     2.802552e+00            2.849513e+00            2.802123e+00            2.894263e+00            2.924838e+00            2.652539e+00            2.719990e+00            2.899168e+00            2.801109e+00            2.658638e+00            2.851132e+00            
15     2.654716e+00            2.881282e+00            2.749994e+00            2.764871e+00            3.084573e+00            3.127939e+00            2.799252e+00            2.823979e+00            2.679090e+00            2.767333e+00            2.814486e+00            
16     9.645892e+01            1.276569e+02            1.281612e+02            9.495318e+01            1.029337e+02            5.931626e+01            7.649978e+01            9.132225e+01            1.667583e+02            8.805262e+01            1.072903e+02            
17     6.137306e+02            1.528766e+03            7.814082e+01            1.736049e+02            1.240111e+03            4.791311e+01            5.738738e+02            2.727473e+02            5.081446e+02            1.318441e+02            4.180814e+01            
18     2.099258e+01            2.461760e+01            1.533610e+01            1.914256e+01            2.351484e+01            2.418687e+01            2.092699e+01            2.178945e+01            2.049236e+01            1.877158e+01            1.701105e+01            
19     2.839440e+01            2.026653e+01            2.054681e+01            2.082755e+01            2.174646e+01            3.526032e+01            2.271889e+01            2.018883e+01            1.678179e+01            2.423776e+01            1.652616e+01            
20     2.071565e+01            2.119530e+01            2.137434e+01            2.081158e+01            2.055053e+01            2.134231e+01            1.960324e+01            2.086631e+01            2.170841e+01            2.142338e+01            2.055197e+01            
21     4.593500e+00            4.561472e+00            4.544214e+00            4.419889e+00            4.517527e+00            4.647178e+00            4.391829e+00            4.390866e+00            4.468449e+00            4.382143e+00            4.509719e+00            
22     9.626746e+00            8.195009e+00            8.183190e+00            9.050166e+00            8.921639e+00            1.041272e+01            7.853926e+00            8.504260e+00            9.395398e+00            8.123051e+00            8.070405e+00            
23     2.228208e+01            3.055695e+01            2.198691e+01            2.188586e+01            2.658804e+01            2.058046e+01            1.768222e+01            1.642883e+01            1.914557e+01            1.792819e+01            1.977124e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_idea_0.py  (error=5.234138e-02)
Task  1: variant_08_idea_0.py  (error=4.251895e-02)
Task  2: variant_05_idea_0.py  (error=1.465711e-01)
Task  3: variant_05_idea_0.py  (error=1.328460e-01)
Task  4: variant_05_idea_0.py  (error=4.990893e-01)
Task  5: variant_05_idea_0.py  (error=8.305665e-05)
Task  6: variant_04_idea_0.py  (error=1.515402e-02)
Task  7: variant_05_idea_0.py  (error=7.456005e-02)
Task  8: variant_05_idea_0.py  (error=2.402331e-01)
Task  9: variant_06_idea_0.py  (error=6.867017e-02)
Task 10: variant_01_idea_0.py  (error=7.579219e+00)
Task 11: variant_10_idea_0.py  (error=1.598485e+01)
Task 12: variant_01_idea_0.py  (error=3.822727e+00)
Task 13: variant_09_idea_0.py  (error=1.777210e+00)
Task 14: variant_05_idea_0.py  (error=2.652539e+00)
Task 15: original.py  (error=2.654716e+00)
Task 16: variant_05_idea_0.py  (error=5.931626e+01)
Task 17: variant_10_idea_0.py  (error=4.180814e+01)
Task 18: variant_02_idea_0.py  (error=1.533610e+01)
Task 19: variant_10_idea_0.py  (error=1.652616e+01)
Task 20: variant_06_idea_0.py  (error=1.960324e+01)
Task 21: variant_09_idea_0.py  (error=4.382143e+00)
Task 22: variant_06_idea_0.py  (error=7.853926e+00)
Task 23: variant_07_idea_0.py  (error=1.642883e+01)

WIN COUNTS:
  variant_05_idea_0.py: 8 wins
  variant_01_idea_0.py: 3 wins
  variant_06_idea_0.py: 3 wins
  variant_10_idea_0.py: 3 wins
  variant_09_idea_0.py: 2 wins
  variant_08_idea_0.py: 1 wins
  variant_04_idea_0.py: 1 wins
  original.py: 1 wins
  variant_02_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Diagonal-only (sep-CMA) covariance adaptation with per-dimension learning rates."""
        # Initialize diagonal covariance if needed
        if not hasattr(self, 'diag_C') or self.diag_C is None:
            self.diag_C = np.ones(self.dim)

        # Compute mean shift in normalized space
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Per-dimension rank-1 update using evolution path
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Separate learning rates for rank-1 and rank-mu updates
        cc_1 = 1.0 / (self.dim + 2.0)
        cc_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)

        # Rank-1 update for diagonal (squared values)
        rank_one_contrib = self.pc ** 2

        # Rank-mu update for diagonal
        rank_mu_contrib = np.zeros(self.dim)
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu_contrib += self.weights[i] * (diff ** 2)
            total_weight += abs(self.weights[i])

        if total_weight > 0:
            rank_mu_contrib /= total_weight

        # Adaptive learning rate based on dimensionality
        dim_adapt = np.log(self.dim + 1.0) / np.log(100.0)
        cc_1_adapt = cc_1 * (0.5 + 0.5 * dim_adapt)
        cc_mu_adapt = cc_mu * (1.5 - 0.5 * dim_adapt)

        # Clip learning rates for stability
        cc_1_adapt = np.clip(cc_1_adapt, 1e-10, 0.5)
        cc_mu_adapt = np.clip(cc_mu_adapt, 1e-10, 0.5)

        # Update diagonal covariance
        self.diag_C = (1.0 - cc_1_adapt - cc_mu_adapt) * self.diag_C
        self.diag_C += cc_1_adapt * rank_one_contrib
        self.diag_C += cc_mu_adapt * rank_mu_contrib

        # Ensure positive diagonal with minimum variance
        min_var = 1e-10 * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        self.diag_C = np.maximum(self.diag_C, min_var)

        # Build full covariance matrix from diagonal
        self.C = np.diag(self.diag_C)

        # Update Cholesky factor for sampling (diagonal case)
        self.L = np.diag(np.sqrt(self.diag_C))
```

# --- From variant_02_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Eigendecomposition-based anisotropic expansion for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += abs(w)

        if total_weight > 0:
            rank_mu /= total_weight

        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu * np.sum(self.weights[:self.mu]))

        # Detect narrow covariance (high condition number) indicating potential local optima trap
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
            cond = np.max(eigvals) / np.min(eigvals)

            # Threshold for "too narrow" covariance - tune based on problem scale
            narrow_threshold = 1e5

            if cond > narrow_threshold:
                # Anisotropic expansion: scale up smallest eigenvalues significantly
                min_eig = np.min(eigvals)
                expansion_factor = 10.0  # Strong expansion to escape local basin

                # Create expansion vector: scale smallest eigvals more
                expansion = np.ones_like(eigvals)
                expansion[eigvals < np.median(eigvals)] = expansion_factor

                new_eigvals = eigvals * expansion
                new_eigvals = np.maximum(new_eigvals, min_eig * 0.1)

                # Reconstruct covariance with expanded eigenvalues
                self.C = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
                self.C = 0.5 * (self.C + self.C.T)

            # Also trigger expansion if population diversity is very low
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity_ratio = pop_variance / (expected_var + 1e-10)

            if diversity_ratio < 1e-4:
                # Broad isotropic expansion when diversity collapses
                min_eig = np.min(eigvals)
                self.C += 0.5 * min_eig * np.eye(self.dim)

        except np.linalg.LinAlgError:
            pass

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Eigenspace-adaptive covariance with explosive exploration."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Eigendecomposition of current covariance
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
        except np.linalg.LinAlgError:
            self.C = 1e-6 * np.eye(self.dim)
            return

        # Project mean change into eigenspace
        z_1 = eigvecs.T @ y_mean
        z_1_norm = np.linalg.norm(z_1)
        if z_1_norm > 1e-10:
            z_1 = z_1 / z_1_norm

        # Adaptive learning rates based on eigenvalue spread
        cond_C = np.max(eigvals) / np.min(eigvals)
        cond_log = np.log1p(cond_C)

        # More aggressive rates for ill-conditioned problems (priority tasks)
        ccov_1 = min(2.0 / (self.dim + 2.0), 0.3)
        ccov_mu = min(self.mueff / (self.dim + 2.0) / (self.dim + 4.0), 0.2)

        # Rank-one update in eigenspace
        rank_one = z_1 ** 2

        # Rank-mu update in eigenspace
        rank_mu = np.zeros(self.dim)
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            z_i = eigvecs.T @ diff
            rank_mu += self.weights[i] * (z_i ** 2)

        # Update eigenvalues directly
        eigvals_new = (1.0 - ccov_1 - ccov_mu) * eigvals
        eigvals_new += ccov_1 * z_1_norm * z_1_norm * eigvals * rank_one
        eigvals_new += ccov_mu * eigvals * rank_mu

        # Explosive exploration: elongate axes for ill-conditioned tasks
        if cond_log > 2.0:
            elongation = min(cond_log * 0.3, 2.0)
            max_idx = np.argmax(eigvals)
            eigvals_new[max_idx] *= (1.0 + elongation)

        # Clamp eigenvalues for numerical stability
        eigvals_new = np.clip(eigvals_new, 1e-14, 1e+10)

        # Reconstruct covariance matrix
        self.C = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)

        # Condition number control
        new_cond = np.max(eigvals_new) / np.min(eigvals_new)
        if new_cond > 1e+8:
            min_eig = np.min(eigvals_new)
            max_eig = min_eig * 1e+8
            eigvals_new = np.clip(eigvals_new, None, max_eig)
            self.C = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
```

# --- From variant_05_idea_0.py (8 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Aggressive restart-driven covariance with stagnation escape."""
        stagnation_threshold = max(20, self.dim * 2)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        if self.stagnation_counter > stagnation_threshold:
            explosion_factor = min(50.0, 1.0 + self.stagnation_counter * 0.5)
            self.sigma *= explosion_factor
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)

            self.C *= explosion_factor

            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

            if diversity < 0.1:
                self.C += 0.5 * np.diag(np.abs(np.diag(self.C)) + 1e-8)

            ccov_escape = min(0.8, self.ccov * explosion_factor)
            self.C = ((1.0 - ccov_escape) * self.C + 
                      ccov_escape * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_escape * 2.0 * rank_mu)

            self.stagnation_counter = 0
        else:
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_06_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Stagnation-triggered eigenvalue perturbation for escaping local optima."""
        # Detect stagnation by monitoring fitness improvement
        if not hasattr(self, 'stagnation_history'):
            self.stagnation_history = []

        self.stagnation_history.append(self.f_opt)
        if len(self.stagnation_history) > 20:
            self.stagnation_history.pop(0)

        # Compute stagnation metrics
        is_stagnant = False
        if len(self.stagnation_history) >= 20:
            recent_improvement = self.stagnation_history[0] - self.stagnation_history[-1]
            relative_improvement = recent_improvement / (abs(self.stagnation_history[0]) + 1e-10)
            is_stagnant = (relative_improvement < 0.01) or (self.stagnation_counter > self.max_stagnation // 2)

        # Evolution path update
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Stagnation-triggered perturbation along minor eigenvectors
        if is_stagnant:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-15)

                # Perturb along 3 minor eigenvectors (highest uncertainty directions)
                num_perturb = min(3, self.dim)
                for k in range(num_perturb):
                    minor_dir = eigvecs[:, k]
                    scale = np.sqrt(eigvals[k]) * self.sigma * 0.3
                    perturbation = minor_dir * np.random.randn() * scale
                    self.mean = self.mean + perturbation

                # Reset evolution path to encourage new direction
                self.pc *= 0.1

                # Inject fresh diversity into population
                for i in range(min(10, self.NP)):
                    idx = self.NP - 1 - i
                    random_dir = np.random.randn(self.dim)
                    random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-15)
                    self.population[idx] = self._clip_to_bounds(
                        self.mean + random_dir * np.random.uniform(0.5, 2.0) * self.sigma * np.sqrt(self.dim)
                    )
            except np.linalg.LinAlgError:
                pass

        # Rank-one and rank-μ updates
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adaptive learning rate based on stagnation
        if is_stagnant:
            ccov_effective = min(self.ccov * 3.0, 0.5)
        else:
            ccov_effective = self.ccov

        self.C = ((1.0 - ccov_effective) * self.C + 
                  ccov_effective * (rank_one + (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Stagnation-triggered covariance explosions for escaping local optima."""
        # Track best known fitness for stagnation detection
        if not hasattr(self, 'best_known_fitness'):
            self.best_known_fitness = float('inf')

        f_opt = max(abs(self.f_opt), 1e-10)
        if self.f_opt < self.best_known_fitness:
            self.best_known_fitness = self.f_opt
            self.stagnation_depth = 0.0
        else:
            self.stagnation_depth = np.log1p(max(self.best_known_fitness - self.f_opt, 0)) + 1.0

        stagnation_factor = np.clip(self.stagnation_depth / 10.0, 0.0, 1.0)

        ccov_base = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        ccov = ccov_base * (1.0 + 5.0 * stagnation_factor)
        ccov = np.clip(ccov, 1e-8, 0.5)

        cc_adaptive = self.cc * (1.0 + 2.0 * stagnation_factor)
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.5)

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

        C_new = ((1.0 - ccov) * self.C + 
                 ccov * rank_one + 
                 (1.0 - 1.0 / self.mueff) * ccov * 2.0 * rank_mu)

        # Stagnation escape: inject large anisotropic perturbations
        if stagnation_factor > 0.3:
            escape_strength = stagnation_factor * 0.5
            n_directions = min(self.dim, max(3, int(self.dim * 0.3)))

            eigvals, eigvecs = np.linalg.eigh(self.C)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            for k in range(n_directions):
                direction = eigvecs[:, -(k + 1)]
                perturbation = escape_strength * np.outer(direction, direction)
                C_new += perturbation

            self.sigma *= (1.0 + stagnation_factor * 2.0)

        self.C = self._ensure_positive_definite(C_new)

        # Maintain Cholesky factor L for sampling
        try:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            self.L = np.diag(np.sqrt(diag_C))
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Eigendecomposition-directed covariance with adaptive eigenvalue control."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute population diversity to detect premature convergence
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Compute rank-mu update matrix
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += abs(w)

        if total_weight > 1e-30:
            rank_mu /= total_weight

        # Build target covariance as weighted combination
        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)

        rank_one = np.outer(y_mean, y_mean) / max(np.dot(y_mean, y_mean), 1e-30)

        C_target = (1.0 - ccov_1 - ccov_mu) * self.C
        C_target += ccov_1 * rank_one
        C_target += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])

        # Ensure symmetry and positive definiteness
        C_target = 0.5 * (C_target + C_target.T)
        min_eig = np.min(np.linalg.eigvalsh(C_target))
        if min_eig < 1e-12:
            C_target += (1e-11 - min_eig) * np.eye(self.dim)

        # Perform eigendecomposition for direct eigenvalue control
        try:
            eigvals, eigvecs = np.linalg.eigh(C_target)
        except np.linalg.LinAlgError:
            C_target = np.eye(self.dim) * np.mean(np.diag(self.C))
            eigvals, eigvecs = np.linalg.eigh(C_target)

        # Stability check
        eigvals = np.maximum(eigvals, 1e-15)

        # Adaptive eigenvalue modification based on convergence state
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-15)

        if cond > 1e6 or diversity < 0.05:
            # Escape mode: boost smallest eigenvalues significantly
            eigvals = eigvals ** 0.5
            min_eig_val = np.min(eigvals)
            boost_factor = 2.0 + 3.0 * (1.0 - diversity)
            eigvals = eigvals + boost_factor * min_eig_val
            eigvals = np.clip(eigvals, 1e-14, None)
        elif cond < 1e2 and diversity > 0.2:
            # Refinement mode: shrink largest eigenvalues for precision
            max_eig_val = np.max(eigvals)
            shrink_factor = 0.7
            eigvals = np.where(eigvals > 0.5 * max_eig_val, eigvals * shrink_factor, eigvals)
            eigvals = np.maximum(eigvals, 1e-15)
        else:
            # Balanced mode: moderate regularization
            eigvals = eigvals * 0.95 + 0.05 * np.mean(eigvals)

        # Reconstruct covariance from modified eigenvalues
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)

        # Inject orthogonal random exploration when diversity is critically low
        if diversity < 0.02:
            for _ in range(3):
                z = np.random.randn(self.dim)
                z = z / max(np.linalg.norm(z), 1e-15)
                random_dir = np.outer(z, z)
                self.C += 0.1 * np.mean(eigvals) * random_dir

        # Final safety check
        self.C = self._ensure_positive_definite(self.C)

        # Update Cholesky factor for sampling
        try:
            self.L = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            self.L = np.diag(np.sqrt(diag_C))
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Eigenvalue spread control with adaptive learning rates."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute eigenvalue spread of covariance matrix
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(eigvals.min(), 1e-15)
        eig_max = np.maximum(eigvals.max(), 1e-15)
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-15)

        # Adaptive learning rates based on condition number
        log_cond = np.log10(max(cond, 1.0))
        spread_factor = 1.0 + max(0.0, log_cond - 2.0) * 0.5

        # Base learning rates from CMA-ES theory
        ccov_base = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        # Scale up learning rates when condition number is large
        ccov_adaptive = min(ccov_base * spread_factor, 0.5)
        cc_adaptive = min(self.cc * spread_factor, 0.3)

        # Evolution path update
        self.pc = (1.0 - cc_adaptive) * self.pc + \
                  np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combined update with adaptive rate
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * (rank_one + (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu))

        # Inject exploration when condition number is problematic
        if log_cond > 3.0:
            # Add small random perturbation to prevent eigenvalue collapse
            noise_scale = 0.01 * ccov_adaptive * np.mean(eigvals)
            self.C += noise_scale * np.eye(self.dim)

        # Hard reset if covariance becomes pathological
        if cond > 1e8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc = np.zeros(self.dim)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_10_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Condition-number triggered restart with adaptive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute condition number as diversity proxy
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

        # Adaptive learning rate based on condition number
        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
            cc_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.5
            cc_scale = 0.5
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        # Fitness variance for additional adaptation signal
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if rel_var < 1e-4:
            ccov_scale *= 0.5
            cc_scale *= 0.5

        # Trigger covariance restart on extreme condition number
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

        # Adaptive learning rates
        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path with adaptive rate
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
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
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
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
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
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
        """Cholesky factor update - maintains C = LL^T directly."""
        if self.L is None:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        z_1 = y_mean / np.maximum(np.linalg.norm(y_mean), 1e-10)
        
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += w
        
        if total_weight > 0:
            rank_mu /= total_weight
        
        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        
        C_new = (1.0 - ccov_1 - ccov_mu) * self.C
        C_new += ccov_1 * np.outer(z_1, z_1)
        C_new += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])
        
        self.C = self._ensure_positive_definite(C_new)
        
        try:
            v = np.linalg.solve(self.L.T, z_1)
            alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
            self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
    
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
        """Dual active evolution paths with exponential history weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        alpha = 0.05
        if not hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        self.pc_weighted = (1.0 - alpha) * self.pc_weighted + alpha * y_mean
        
        beta = 0.02
        if not hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            cross_contribution = np.sqrt(beta) * (y_mean - self.prev_y_mean)
            self.p_cross = (1.0 - beta) * self.p_cross + cross_contribution
        self.prev_y_mean = y_mean.copy()
        
        rank_one_primary = np.outer(self.pc, self.pc)
        rank_one_weighted = np.outer(self.pc_weighted, self.pc_weighted)
        rank_one_cross = np.outer(self.p_cross, self.p_cross)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_primary = self.ccov
        ccov_weighted = 0.3 * self.ccov
        ccov_cross = 0.1 * self.ccov
        
        self.C = ((1.0 - ccov_primary) * self.C + 
                  ccov_primary * rank_one_primary +
                  ccov_weighted * rank_one_weighted +
                  ccov_cross * rank_one_cross +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
        
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1e7:
            self.C *= 0.5
    
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
        """Compute population diversity as mean per-dimension std."""
        return np.mean(np.std(self.population, axis=0))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
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