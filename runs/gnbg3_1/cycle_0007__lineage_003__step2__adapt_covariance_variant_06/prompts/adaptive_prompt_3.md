Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_06` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.314496e-02            5.214621e-02            5.352354e-02            5.151604e-02            5.200102e-02            5.299018e-02            5.214538e-02            5.267580e-02            -inf                    5.358856e-02            
1      4.447001e-02            4.410901e-02            4.091350e-02            4.351635e-02            4.038533e-02            4.328113e-02            4.443803e-02            4.103992e-02            -inf                    4.302266e-02            
2      2.877613e-01            1.631121e-01            2.819369e-01            3.284314e+00            1.687314e-01            2.362928e-01            4.565337e-01            1.526904e-01            -inf                    1.920584e-01            
3      1.347502e-01            1.349136e-01            2.558322e-01            1.909647e+00            4.755272e-01            4.714490e-01            1.565329e-01            1.327860e-01            -inf                    1.645042e-01            
4      5.231861e-01            5.181813e-01            5.229636e-01            5.197531e-01            5.252674e-01            5.057800e-01            5.149735e-01            5.145598e-01            -inf                    5.141057e-01            
5      9.226954e-05            9.067314e-05            1.417305e-04            1.647184e-04            1.061072e-04            1.359884e-04            3.227146e-04            8.059445e-05            -inf                    9.944339e-05            
6      4.082087e-02            1.607060e-02            1.641077e-02            3.204723e-02            1.662916e+01            7.945815e-01            1.539463e-02            1.585452e-02            -inf                    2.089781e+01            
7      8.003612e-02            7.729452e-02            1.529641e-01            2.687450e-01            1.503586e-01            2.151646e-01            1.002530e-01            7.555326e-02            -inf                    1.074607e-01            
8      2.510861e-01            2.536976e-01            3.393747e-01            3.211349e-01            2.600942e-01            2.523616e-01            2.583369e-01            2.580552e-01            -inf                    3.190725e-01            
9      7.031523e-02            1.080236e-01            9.540115e-02            5.971527e-01            3.223093e+00            1.922503e-01            7.148082e-02            1.367821e-01            -inf                    7.218848e+00            
10     1.880487e+01            2.811036e+01            2.127870e+01            2.531310e+01            1.579647e+01            2.097589e+01            2.896481e+01            2.288840e+01            -inf                    2.561593e+01            
11     1.598485e+01            3.258797e+01            1.018220e+01            1.229073e+01            3.441612e+01            3.498162e+01            2.085027e+01            3.121113e+01            -inf                    1.940348e+01            
12     1.843518e+01            3.852786e+01            1.793698e+01            1.808133e+01            1.859355e+01            1.676153e+01            2.361568e+01            2.996173e+01            -inf                    2.384409e+01            
13     2.988347e+00            3.467626e+00            4.184724e+00            2.690485e+00            3.825424e+00            2.434386e+00            3.611408e+00            3.880886e+00            -inf                    4.391302e+00            
14     2.851132e+00            2.772594e+00            2.833002e+00            2.879144e+00            2.835766e+00            2.990719e+00            2.856217e+00            3.019892e+00            -inf                    3.157694e+00            
15     2.814486e+00            2.905423e+00            2.742295e+00            2.786894e+00            2.771052e+00            2.679600e+00            2.938377e+00            2.941645e+00            -inf                    2.776075e+00            
16     1.072903e+02            1.430655e+02            1.001806e+02            1.410790e+02            8.261067e+01            7.506825e+01            1.445023e+02            1.195523e+02            -inf                    8.804059e+01            
17     4.180814e+01            1.493007e+02            4.461494e+01            5.370868e+02            4.212295e+01            4.117781e+01            7.199203e+01            5.211794e+01            -inf                    4.374382e+01            
18     1.701105e+01            2.463565e+01            2.053514e+01            1.846530e+01            1.890239e+01            1.622052e+01            2.164571e+01            2.041085e+01            -inf                    2.283735e+01            
19     1.652616e+01            2.082271e+01            2.173609e+01            1.856627e+01            3.342419e+01            3.436466e+01            2.253518e+01            2.361245e+01            -inf                    2.026201e+01            
20     2.055197e+01            2.361303e+01            1.970467e+01            2.206281e+01            2.168719e+01            1.998964e+01            1.980982e+01            2.334671e+01            -inf                    2.150191e+01            
21     4.509719e+00            4.513746e+00            4.580903e+00            4.563240e+00            4.553138e+00            4.574589e+00            4.636465e+00            4.574643e+00            -inf                    4.597714e+00            
22     8.070405e+00            9.050553e+00            6.866943e+00            9.345097e+00            9.291045e+00            7.707025e+00            7.911789e+00            9.770380e+00            -inf                    8.360111e+00            
23     1.977124e+01            2.187217e+01            2.182718e+01            1.796590e+01            1.964283e+01            2.056304e+01            1.904582e+01            2.270928e+01            -inf                    2.306528e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_idea_0.py  (error=5.151604e-02)
Task  1: variant_04_idea_0.py  (error=4.038533e-02)
Task  2: variant_07_idea_0.py  (error=1.526904e-01)
Task  3: variant_07_idea_0.py  (error=1.327860e-01)
Task  4: variant_05_idea_0.py  (error=5.057800e-01)
Task  5: variant_07_idea_0.py  (error=8.059445e-05)
Task  6: variant_06_idea_0.py  (error=1.539463e-02)
Task  7: variant_07_idea_0.py  (error=7.555326e-02)
Task  8: original.py  (error=2.510861e-01)
Task  9: original.py  (error=7.031523e-02)
Task 10: variant_04_idea_0.py  (error=1.579647e+01)
Task 11: variant_02_idea_0.py  (error=1.018220e+01)
Task 12: variant_05_idea_0.py  (error=1.676153e+01)
Task 13: variant_05_idea_0.py  (error=2.434386e+00)
Task 14: variant_01_idea_0.py  (error=2.772594e+00)
Task 15: variant_05_idea_0.py  (error=2.679600e+00)
Task 16: variant_05_idea_0.py  (error=7.506825e+01)
Task 17: variant_05_idea_0.py  (error=4.117781e+01)
Task 18: variant_05_idea_0.py  (error=1.622052e+01)
Task 19: original.py  (error=1.652616e+01)
Task 20: variant_02_idea_0.py  (error=1.970467e+01)
Task 21: original.py  (error=4.509719e+00)
Task 22: variant_02_idea_0.py  (error=6.866943e+00)
Task 23: variant_03_idea_0.py  (error=1.796590e+01)

WIN COUNTS:
  variant_05_idea_0.py: 7 wins
  variant_07_idea_0.py: 4 wins
  original.py: 4 wins
  variant_02_idea_0.py: 3 wins
  variant_03_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins
  variant_01_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Active CMA-ES with negative weights for diversity-promoting rank-μ update."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Standard evolution path update
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Active CMA-ES: compute both positive and negative weights
        n_pos = self.mu // 2
        n_neg = self.mu - n_pos

        # Sort population by fitness (ascending - best first)
        sorted_indices = np.argsort(self.fitness)
        sorted_pop = self.population[sorted_indices]

        # Positive weights for top mu individuals (standard)
        pos_weights = np.log(n_pos + 0.5) - np.log(np.arange(1, n_pos + 1))
        pos_weights /= np.sum(pos_weights)
        pos_weights = np.maximum(pos_weights, 0.0)

        # Negative weights for bottom mu individuals (inverted order)
        neg_weights = np.log(n_neg + 0.5) - np.log(np.arange(1, n_neg + 1))
        neg_weights /= np.sum(neg_weights)
        neg_weights = np.maximum(neg_weights, 0.0)

        # Build rank-mu update with both positive and negative contributions
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(n_pos):
            diff = (sorted_pop[i] - self.old_mean) / self.sigma
            rank_mu += pos_weights[i] * np.outer(diff, diff)

        for i in range(n_neg):
            idx = self.mu - 1 - i
            diff = (sorted_pop[idx] - self.old_mean) / self.sigma
            rank_mu -= 0.25 * neg_weights[i] * np.outer(diff, diff)

        # Combine updates with active covariance learning rate
        c_c = 2.0 / ((self.dim + 1.41) ** 2)
        c_1 = 1.0 / (self.dim + 2.0)
        c_mu = self.mueff / (self.dim + 2.0)

        alpha_c = 1.0 + c_1 + c_mu
        c_1_scaled = c_1 / alpha_c
        c_mu_scaled = c_mu / alpha_c
        c_c_scaled = c_c / alpha_c

        self.C = ((1.0 - c_c_scaled) * self.C + 
                  c_c_scaled * rank_one +
                  c_mu_scaled * rank_mu)

        # Ensure numerical stability
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_02_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Forced diversity injection with eigenvalue floor to prevent catastrophic collapse."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Compute condition number and spread
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        eig_spread = eig_min / (eig_max + 1e-15)

        # Fitness variance for stagnation detection
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        # Detect collapse: extreme condition or vanishing spread
        is_collapsed = (cond > 1e6) or (eig_spread < 1e-6) or (rel_var < 1e-6)

        # Determine base learning rates
        if is_collapsed:
            # Force diversity: use small ccov, inject fresh variance
            ccov_eff = self.ccov * 0.2
            cc_eff = self.cc * 0.3

            # Enforce eigenvalue floor to prevent re-collapse
            target_min = 0.01 * np.mean(eigvals)
            if eig_min < target_min:
                self.C += (target_min - eig_min) * np.eye(self.dim)

            # Add isotropic perturbation for exploration burst
            noise_scale = 0.1 * np.mean(eigvals)
            self.C += noise_scale * np.eye(self.dim)
        else:
            # Normal operation with moderate damping based on condition
            damp_factor = 1.0 + min(cond / 1e4, 2.0)
            ccov_eff = self.ccov / damp_factor
            cc_eff = self.cc / max(damp_factor * 0.5, 0.5)

        # Recompute pc with adapted rate if collapsed
        if is_collapsed:
            self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
            rank_one = np.outer(self.pc, self.pc)

        # Standard rank-mu update
        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        # Final safety: enforce minimum eigenvalue after update
        final_eigvals = np.linalg.eigvalsh(self.C)
        final_min = np.min(final_eigvals)
        if final_min < 1e-12:
            self.C += (1e-10 - final_min) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_03_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Catastrophic diversification with eigenvalue floor for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Detect pathological stagnation: no improvement + near-zero diversity
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = pop_variance / (expected_var + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        # Pathological convergence detection: fitness stalled AND population collapsed
        pathological = (rel_var < 1e-6) and (diversity_ratio < 1e-5) and (eig_spread < 1e-6)

        if pathological:
            # Force covariance explosion: add massive random perturbation
            rand_dir = np.random.randn(self.dim)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)

            # Scale perturbation by current max eigenvalue and sigma
            explosion_magnitude = np.max(eigvals) * 50.0 * max(self.sigma, 1.0)
            explosion_matrix = explosion_magnitude * np.outer(rand_dir, rand_dir)

            # Blend explosion into covariance
            self.C = 0.5 * self.C + 0.5 * explosion_matrix

            # Enforce eigenvalue floor to prevent immediate re-collapse
            min_eigenvalue = (self.ub[0] - self.lb[0]) ** 2 * 1e-6
            for i in range(self.dim):
                if self.C[i, i] < min_eigenvalue:
                    self.C[i, i] = min_eigenvalue

            # Reset evolution path to remove directional memory
            self.pc = np.zeros(self.dim)

            # Boost step size to capitalize on new exploration direction
            self.sigma = min(self.sigma * 3.0, 10.0)

            self.C = self._ensure_positive_definite(self.C)
            return

        # Standard CMA-ES adaptation when not pathological
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adaptive learning rate based on condition number
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
        if cond > 1e6:
            ccov_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.5
        else:
            ccov_scale = 1.0

        ccov_adapt = self.ccov * ccov_scale

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Multi-hypothesis covariance pooling with performance-weighted fusion."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Initialize covariance pool on first call
        if not hasattr(self, 'C_pool'):
            self.C_pool = [self.C.copy() for _ in range(3)]
            self.C_scores = [1.0, 1.0, 1.0]
            self.C_pool_updates = [0, 0, 0]

        # --- Hypothesis 1: Exploitation-focused (tight covariance) ---
        C_exploit = ((1.0 - self.ccov) * self.C_pool[0] + 
                     self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # --- Hypothesis 2: History-preserving (slow adaptation) ---
        C_history = 0.95 * self.C_pool[1] + 0.05 * self.C
        C_history = self._ensure_positive_definite(C_history)

        # --- Hypothesis 3: Exploration-focused (eigenvalue inflation) ---
        eigvals_C, eigvecs = np.linalg.eigh(self.C_pool[2])
        eigvals_C = np.maximum(eigvals_C, 1e-10)
        log_eigen_sum = np.sum(np.log(eigvals_C))
        C_explore = ((1.0 - self.ccov * 0.5) * self.C_pool[2] + 
                     self.ccov * 0.5 * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        eigvals_explore, _ = np.linalg.eigh(C_explore)
        if len(eigvals_explore) > 0 and np.all(eigvals_explore > 0):
            log_eigen_explore = np.sum(np.log(eigvals_explore))
            if log_eigen_explore < log_eigen_sum - 0.1 * self.dim:
                C_explore = 1.1 * C_explore
        C_explore = self._ensure_positive_definite(C_explore)

        # --- Update pool with new hypotheses ---
        self.C_pool[0] = self._ensure_positive_definite(C_exploit)
        self.C_pool[1] = self._ensure_positive_definite(C_history)
        self.C_pool[2] = self._ensure_positive_definite(C_explore)

        # --- Update scores based on fitness improvement ---
        if hasattr(self, 'prev_fitness_sum'):
            fitness_change = self.prev_fitness_sum - np.sum(self.fitness)
            for i in range(3):
                # Score increases with positive fitness change
                self.C_scores[i] = max(0.1, self.C_scores[i] * 0.9 + 0.1 * max(fitness_change, 0.0))
        self.prev_fitness_sum = np.sum(self.fitness)

        # --- Weighted fusion based on scores ---
        total_score = sum(self.C_scores) + 1e-10
        weights = [s / total_score for s in self.C_scores]

        self.C = weights[0] * self.C_pool[0] + weights[1] * self.C_pool[1] + weights[2] * self.C_pool[2]
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_05_idea_0.py (7 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Diversity-triggered anisotropic restart with elite preservation."""
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

        # Anisotropic restart if still stuck after adaptation
        fit_var_after = np.var(self.fitness)
        rel_var_after = fit_var_after / (fit_scale ** 2 + 1e-10)
        eigvals_after = np.linalg.eigvalsh(self.C)
        eig_spread_after = np.min(eigvals_after) / (np.max(eigvals_after) + 1e-10)

        needs_restart = (rel_var_after < 0.005) and (eig_spread_after < 1e-5)

        if needs_restart:
            self.pc = np.zeros(self.dim)
            self.C = np.eye(self.dim) * (self.sigma ** 2)

            sorted_idx = np.argsort(self.fitness)
            elites = [self.population[i].copy() for i in sorted_idx[:min(5, self.NP)]]

            spread = 0.3 * self.sigma
            new_pop = self.x_opt + np.random.randn(self.NP, self.dim) * spread
            new_pop = self._clip_to_bounds(new_pop)

            n_elites = min(5, self.NP)
            new_pop[:n_elites] = elites[:n_elites]

            self.population = new_pop
            self.fitness = self.func(self.population)

            self.old_mean = self.mean.copy()
            self.mean = np.mean(self.population, axis=0)

            self.stagnation_counter = 0
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Stagnation-depth-driven covariance explosion with directional kick."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Stagnation depth drives exploration intensity
        stagnation_ratio = min(self.stagnation_counter / max(self.max_stagnation, 1), 1.0)

        # Exponential scaling: stagnation_ratio=0.5 → ~2.7x, stagnation_ratio=0.8 → ~10x
        explosion_mult = 1.0 + 10.0 * np.tanh(3.0 * stagnation_ratio)

        # Decay explosion strength when improving
        if not hasattr(self, 'explosion_strength'):
            self.explosion_strength = 1.0
        if stagnation_ratio < 0.1:
            self.explosion_strength *= 0.95
            self.explosion_strength = max(self.explosion_strength, 1.0)
        else:
            self.explosion_strength = explosion_mult

        # Base adaptive ccov
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        ccov_base = min(self.ccov * (1.0 + 5.0 * min(rel_var, 0.1)), 0.5)
        ccov_scaled = ccov_base * self.explosion_strength
        ccov_scaled = min(ccov_scaled, 0.8)

        # Directional kick: push covariance along the current search path
        eigvals, eigvecs = np.linalg.eigh(self.C)
        max_eig = np.max(eigvals)

        if np.linalg.norm(self.pc) > 1e-12:
            kick_dir = self.pc / np.linalg.norm(self.pc)
        else:
            kick_dir = eigvecs[:, -1]
            kick_dir /= (np.linalg.norm(kick_dir) + 1e-12)

        kick_magnitude = 0.5 * max_eig * self.explosion_strength
        kick_matrix = kick_magnitude * np.outer(kick_dir, kick_dir)

        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu) +
                  0.1 * ccov_scaled * kick_matrix)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_07_idea_0.py (4 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Stagnation-triggered escape with large perturbation and covariance reset."""

        # Detect stagnation earlier (shorter threshold than default)
        stagnation_threshold = max(15, self.dim)
        is_stagnant = self.stagnation_counter > stagnation_threshold

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

        if is_stagnant:
            # Generate random escape direction
            escape_dir = np.random.randn(self.dim)
            escape_dir /= (np.linalg.norm(escape_dir) + 1e-10)

            # Compute perturbation magnitude based on step size and covariance scale
            eigvals = np.linalg.eigvalsh(self.C)
            cov_scale = np.sqrt(np.max(eigvals) + 1e-10)
            jump_magnitude = 15.0 * self.sigma * cov_scale

            # Apply large perturbation to mean
            self.mean = self.mean + jump_magnitude * escape_dir

            # Reset covariance to identity (wider exploration)
            self.C = np.eye(self.dim)

            # Reset evolution path
            self.pc = np.zeros(self.dim)

            # Increase step size for wider search
            self.sigma *= 2.5

            # Reset stagnation counter
            self.stagnation_counter = 0

            # Ensure mean stays within bounds
            self.mean = self._clip_to_bounds(self.mean)
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)
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