Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_original` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.581447e-02            5.846175e-02            5.863098e-02            5.634829e-02            -inf                    5.601890e-02            5.609951e-02            5.668996e-02            5.742661e-02            5.834620e-02            
1      4.225025e-02            4.436173e-02            4.361137e-02            4.357799e-02            -inf                    4.442802e-02            4.244507e-02            4.285232e-02            4.307051e-02            4.540912e-02            
2      1.550907e-01            1.587501e-01            1.492978e-01            1.510194e-01            -inf                    1.567387e-01            1.536278e-01            1.576864e-01            1.561298e-01            1.495295e-01            
3      1.731658e-01            1.815577e-01            1.526925e-01            1.474846e-01            -inf                    1.376338e-01            1.339758e-01            1.346956e-01            2.403666e-01            3.978631e-01            
4      5.133402e-01            5.157695e-01            5.154360e-01            5.087134e-01            -inf                    5.232360e-01            5.076901e-01            5.146248e-01            5.145004e-01            5.182099e-01            
5      1.328620e-04            8.854777e-05            9.933211e-05            9.194950e-05            -inf                    9.606260e-05            9.520711e-05            1.033396e-04            9.377370e-05            9.280901e-05            
6      1.577348e-02            1.481415e-02            1.561941e-02            1.623565e-02            -inf                    1.565731e-02            2.743433e-02            1.577767e-02            1.519600e-02            1.465328e-02            
7      8.061778e-02            8.072789e-02            7.459518e-02            7.863377e-02            -inf                    8.186423e-02            8.280924e-02            7.948502e-02            7.903932e-02            8.105726e-02            
8      2.568841e-01            2.557443e-01            2.522915e-01            2.526107e-01            -inf                    2.494481e-01            2.500990e-01            2.560862e-01            2.710674e-01            2.507659e-01            
9      9.359551e-02            2.452385e-01            9.300284e-02            1.040138e-01            -inf                    9.100165e-02            1.126775e-01            1.960519e-01            7.234376e-02            1.280102e-01            
10     1.004256e+01            1.315189e+01            1.121908e+01            1.214423e+01            -inf                    1.001884e+01            1.540009e+01            1.296601e+01            9.720441e+00            1.779489e+01            
11     1.611530e+01            1.416510e+01            3.782366e+01            3.408174e+01            -inf                    2.568994e+01            3.769455e+01            1.407465e+01            2.243462e+01            1.886508e+01            
12     1.431520e+01            1.815113e+01            1.827344e+01            1.043057e+01            -inf                    1.803386e+01            2.351820e+01            1.446547e+01            1.243264e+01            1.477366e+01            
13     3.185759e+00            2.923934e+00            2.463517e+00            2.848489e+00            -inf                    3.121440e+00            3.546400e+00            2.491856e+00            2.759965e+00            2.614946e+00            
14     2.765349e+00            2.655373e+00            2.739140e+00            2.790758e+00            -inf                    2.781898e+00            2.742426e+00            2.737109e+00            2.655484e+00            2.667316e+00            
15     2.708644e+00            2.640691e+00            2.755389e+00            2.321204e+00            -inf                    2.780949e+00            2.690521e+00            2.759609e+00            2.730808e+00            2.862484e+00            
16     7.913191e+01            7.112965e+01            7.755933e+01            1.064000e+02            -inf                    8.226159e+01            8.760124e+01            9.702706e+01            9.418401e+01            6.518425e+01            
17     4.045684e+01            4.892107e+01            2.720603e+02            3.155724e+02            -inf                    3.131828e+01            5.358056e+02            3.613511e+01            4.172359e+01            6.601403e+01            
18     1.283604e+01            1.880427e+01            1.705865e+01            1.483771e+01            -inf                    1.734587e+01            1.908525e+01            1.548568e+01            1.754643e+01            2.320372e+01            
19     1.339200e+01            1.429952e+01            1.644674e+01            2.163034e+01            -inf                    1.470252e+01            1.698856e+01            1.660657e+01            1.308205e+01            1.836679e+01            
20     1.828716e+01            2.142304e+01            2.051524e+01            1.990750e+01            -inf                    1.952237e+01            2.053500e+01            2.062714e+01            2.151700e+01            2.019202e+01            
21     4.361961e+00            4.360999e+00            4.304912e+00            4.361023e+00            -inf                    4.384306e+00            4.523220e+00            4.480823e+00            4.462162e+00            4.537062e+00            
22     8.522054e+00            8.973710e+00            8.908651e+00            8.121411e+00            -inf                    9.485117e+00            7.915573e+00            8.963366e+00            8.251879e+00            8.296579e+00            
23     1.942306e+01            1.860935e+01            1.843315e+01            2.108157e+01            -inf                    2.237062e+01            1.812837e+01            1.718736e+01            1.772304e+01            2.285009e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=5.581447e-02)
Task  1: original.py  (error=4.225025e-02)
Task  2: variant_02_idea_0.py  (error=1.492978e-01)
Task  3: variant_06_idea_0.py  (error=1.339758e-01)
Task  4: variant_06_idea_0.py  (error=5.076901e-01)
Task  5: variant_01_idea_0.py  (error=8.854777e-05)
Task  6: variant_09_idea_0.py  (error=1.465328e-02)
Task  7: variant_02_idea_0.py  (error=7.459518e-02)
Task  8: variant_05_idea_0.py  (error=2.494481e-01)
Task  9: variant_08_idea_0.py  (error=7.234376e-02)
Task 10: variant_08_idea_0.py  (error=9.720441e+00)
Task 11: variant_07_idea_0.py  (error=1.407465e+01)
Task 12: variant_03_idea_0.py  (error=1.043057e+01)
Task 13: variant_02_idea_0.py  (error=2.463517e+00)
Task 14: variant_01_idea_0.py  (error=2.655373e+00)
Task 15: variant_03_idea_0.py  (error=2.321204e+00)
Task 16: variant_09_idea_0.py  (error=6.518425e+01)
Task 17: variant_05_idea_0.py  (error=3.131828e+01)
Task 18: original.py  (error=1.283604e+01)
Task 19: variant_08_idea_0.py  (error=1.308205e+01)
Task 20: original.py  (error=1.828716e+01)
Task 21: variant_02_idea_0.py  (error=4.304912e+00)
Task 22: variant_06_idea_0.py  (error=7.915573e+00)
Task 23: variant_07_idea_0.py  (error=1.718736e+01)

WIN COUNTS:
  original.py: 4 wins
  variant_02_idea_0.py: 4 wins
  variant_06_idea_0.py: 3 wins
  variant_08_idea_0.py: 3 wins
  variant_01_idea_0.py: 2 wins
  variant_09_idea_0.py: 2 wins
  variant_05_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_03_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
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
```

# --- From variant_02_idea_0.py (4 wins) ---
```python
def _adapt_covariance_original(self):
        """Directional memory with eigenspace injection for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Track successful search directions with momentum
        if not hasattr(self, 'step_memory'):
            self.step_memory = []
            self.step_memory_fitness = []
            self.step_memory_size = 5

        # Store recent successful steps
        for i in range(min(3, self.NP)):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    step = (self.trials[i] - self.population[i]) / self.sigma
                    step_norm = np.linalg.norm(step)
                    if step_norm > 1e-10:
                        step_normalized = step / step_norm
                        self.step_memory.append(step_normalized)
                        self.step_memory_fitness.append(self.fitness[i] - self.trial_fitness[i])

        # Maintain bounded memory of directions
        while len(self.step_memory) > self.step_memory_size * 3:
            if self.step_memory_fitness:
                min_idx = np.argmin(self.step_memory_fitness)
                self.step_memory.pop(min_idx)
                self.step_memory_fitness.pop(min_idx)
            else:
                self.step_memory.pop(0)

        # Compute directional covariance from step memory
        C_dir = np.zeros((self.dim, self.dim))
        if len(self.step_memory) >= 3:
            steps = np.array(self.step_memory)
            weights = np.array(self.step_memory_fitness) if self.step_memory_fitness else np.ones(len(self.step_memory))
            weights = np.maximum(weights, 1e-10)
            weights /= np.sum(weights)
            centered = steps - np.mean(steps, axis=0)
            C_dir = np.dot(centered.T * weights, centered) + 0.1 * np.eye(self.dim)

        # Detect stagnation
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improvement_threshold = 1e-8 * max(1.0, abs(self.f_opt))
        is_stagnant = improvement < improvement_threshold

        # Adaptive learning rates
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        ccov_base = min(0.5, 0.01 / (self.dim + 1.0))
        ccov_adapt = ccov_base * max(0.1, min(10.0, rel_var * 100.0 + 1.0))
        ccov_dir_weight = min(0.3, ccov_adapt * 2.0) if is_stagnant else ccov_adapt * 0.5

        # Evolution path update
        cc_adapt = min(self.cc * 2.0, (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim))
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with directional memory
        self.C = ((1.0 - ccov_adapt - ccov_dir_weight) * self.C +
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        if len(self.step_memory) >= 3:
            self.C = (1.0 - ccov_dir_weight) * self.C + ccov_dir_weight * C_dir

        # Inject exploration along eigendirections when stagnant
        if is_stagnant:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            perturbation = np.zeros((self.dim, self.dim))
            for k in range(min(3, self.dim)):
                if k < len(eigvals):
                    v = eigvecs[:, k]
                    perturbation += 0.05 * eigvals[k] * np.outer(v, v)
            self.C = self.C + perturbation

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_03_idea_0.py (2 wins) ---
```python
def _adapt_covariance_original(self):
        """Stagnation-triggered covariance expansion with condition-based damping."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Monitor recent fitness improvement to detect stagnation
        if not hasattr(self, 'recent_improvements'):
            self.recent_improvements = []
        self.recent_improvements.append(self.f_opt)
        if len(self.recent_improvements) > 10:
            self.recent_improvements.pop(0)

        # Detect stagnation: little improvement over recent generations
        if len(self.recent_improvements) >= 5:
            recent_change = abs(self.recent_improvements[-1] - self.recent_improvements[0])
            is_stagnant = recent_change < 1e-8 * max(abs(self.f_opt), 1.0)
        else:
            is_stagnant = False

        # Compute condition number of C to detect covariance collapse
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = max(np.min(eigvals), 1e-10)
        eig_max = np.max(eigvals)
        cond_C = eig_max / eig_min

        # Condition-based damping: reduce learning when C is ill-conditioned
        # to prevent runaway elongation
        cond_damping = np.sqrt(np.log1p(cond_C)) / np.sqrt(np.log1p(self.dim))
        cond_damping = np.clip(cond_damping, 0.3, 3.0)

        # Aggressive stagnation detection for high-error tasks
        if cond_C > 1e6:
            is_stagnant = True

        # Stagnation-triggered expansion: boost learning rates and inject variance
        if is_stagnant:
            # Strong boost to escape local optima
            stagnation_boost = 5.0
            # Inject diagonal variance proportional to sigma
            diag_injection = 0.2 * self.sigma * np.eye(self.dim)
        else:
            stagnation_boost = 1.0
            diag_injection = np.zeros((self.dim, self.dim))

        # Effective learning rates with stagnation boost and condition damping
        cc_eff = min(self.cc * stagnation_boost, 0.5)
        ccov_eff = min(self.ccov * stagnation_boost / cond_damping, 0.5)
        ccov_eff = max(ccov_eff, 1e-8)

        # Evolution path update with effective rate
        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update (unchanged from original)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Covariance update with stagnation-triggered expansion
        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        # Inject diagonal variance when stagnant to force re-exploration
        self.C = self.C + diag_injection

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_05_idea_0.py (2 wins) ---
```python
def _adapt_covariance_original(self):
        """Original covariance adaptation with restart on stagnation."""
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

        # Check for stagnation and trigger covariance restart
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / (max(eig_min, 1e-10))

        # Restart conditions: extreme ill-conditioning OR stagnation
        should_restart = False
        restart_reason = ""

        # Condition-based restart: matrix too elongated
        if cond > 1e6:
            should_restart = True
            restart_reason = "ill_conditioned"

        # Stagnation-based restart: best not improving
        if hasattr(self, 'stagnation_counter') and self.stagnation_counter > self.max_stagnation:
            should_restart = True
            restart_reason = "stagnation"

        # Spectrum collapse: all eigenvalues similar but tiny
        if eig_max > 1e-10 and eig_min / eig_max < 1e-8:
            should_restart = True
            restart_reason = "spectrum_collapse"

        if should_restart:
            # Reinitialize covariance to identity (spherical exploration)
            self.C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
            # Increase step size for wider exploration after restart
            self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.1)
            self.sigma = max(self.sigma, 1e-6)
```

# --- From variant_06_idea_0.py (3 wins) ---
```python
def _adapt_covariance_original(self):
        """Natural Evolution Strategy (NES) gradient-based covariance adaptation."""
        # Compute normalized fitness weights (fitness-shaping) for gradient estimation
        fitness_array = np.array(self.fitness, dtype=np.float64)
        f_min = np.min(fitness_array)
        f_max = np.max(fitness_array)
        fit_range = max(f_max - f_min, 1e-10)

        # Rank-based fitness shaping: u_i = max(0, log(mu/2+1) - log(rank_i))
        sorted_indices = np.argsort(fitness_array)
        u_weights = np.zeros(self.NP)
        for rank, idx in enumerate(sorted_indices[:self.mu]):
            u_weights[idx] = max(0.0, np.log(self.mu / 2.0 + 1.0) - np.log(rank + 1.0))

        u_sum = np.sum(u_weights)
        if u_sum > 1e-10:
            u_weights /= u_sum
        else:
            u_weights = self.weights[:self.NP]
            u_weights /= np.sum(u_weights)

        # Compute NES gradient: sum_i u_i * grad_log_p(x_i) where p is the sampling distribution
        # For Gaussian distribution N(mean, sigma^2 * C):
        # grad_log_p(x) = -0.5 * inv(C) @ (x - mean) @ (x - mean).T + 0.5 * inv(C)
        # Weighted sum gives the natural gradient direction for covariance

        # Ensure C is invertible
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        C_reg = self.C.copy()
        if min_eig < 1e-8:
            C_reg += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            C_inv = np.linalg.inv(C_reg)
        except np.linalg.LinAlgError:
            diag_C = np.diag(C_reg)
            diag_C = np.maximum(diag_C, 1e-10)
            C_inv = np.diag(1.0 / diag_C)

        # Compute mean-centered deviations
        centered = self.population[:self.NP] - self.mean  # (NP, dim)

        # Compute covariance gradient: G = sum_i u_i * (y_i @ y_i^T - C) where y_i = C^{-1/2} @ (x_i - mean)
        # Use eigen decomposition for stable C^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(C_reg)
        eigvals = np.maximum(eigvals, 1e-10)
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Compute whitened coordinates
        y_coords = centered @ C_inv_sqrt.T  # (NP, dim)

        # Natural gradient for mean
        grad_mean = np.zeros(self.dim)
        for i in range(self.NP):
            grad_mean += u_weights[i] * y_coords[i]
        grad_mean = C_inv_sqrt @ grad_mean

        # Natural gradient for covariance (Friston's precision update)
        grad_cov = np.zeros((self.dim, self.dim))
        for i in range(self.NP):
            outer_y = np.outer(y_coords[i], y_coords[i])
            grad_cov += u_weights[i] * (outer_y - np.eye(self.dim))

        # Learning rates for NES
        eta_mean = 1.0
        eta_cov = 0.5 / (self.dim + 2.0)

        # Update mean using natural gradient
        self.mean = self.mean + eta_mean * self.sigma * (C_inv_sqrt @ grad_mean)

        # Update covariance using natural gradient with momentum
        delta_cov = eta_cov * grad_cov

        # Add momentum to prevent erratic updates
        if not hasattr(self, 'cov_momentum'):
            self.cov_momentum = np.zeros((self.dim, self.dim))
        self.cov_momentum = 0.7 * self.cov_momentum + 0.3 * delta_cov

        # Adaptive learning rate based on fitness gradient magnitude
        fitness_improvement = abs(f_min - f_max) / max(abs(f_min), 1.0)
        lr_scale = np.clip(1.0 + np.log1p(fitness_improvement * 10.0), 0.1, 5.0)

        # Apply update with adaptive scaling
        self.C = self.C + lr_scale * self.cov_momentum

        # Ensure positive definiteness
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _adapt_covariance_original(self):
        """Eigenvalue floor with stagnation detection for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-12)

        # Detect stagnation: mean movement in eigenspace
        y_mean_scaled = y_mean @ eigvecs
        progress_per_dir = np.abs(y_mean_scaled)

        # Eigenvalue floor to prevent collapse
        min_eig = 1e-6 * np.max(eigvals)
        eigvals_clamped = np.maximum(eigvals, min_eig)

        # Boost stagnated eigendirections
        progress_threshold = 1e-4 * np.linalg.norm(y_mean_scaled) + 1e-10
        stagnation_mask = progress_per_dir < progress_threshold

        if np.any(stagnation_mask):
            boost_factor = 5.0
            eigvals_boosted = eigvals_clamped.copy()
            eigvals_boosted[stagnation_mask] *= boost_factor
            eigvals_clamped = eigvals_boosted

        # Reconstruct covariance from modified eigenvalues
        self.C = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)

        # Standard evolution path update
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Blend with standard rank-mu update
        blend_weight = 0.7
        self.C = ((1.0 - blend_weight * self.ccov) * self.C + 
                  blend_weight * self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_08_idea_0.py (3 wins) ---
```python
def _adapt_covariance_original(self):
        """Success-ratio covariance adaptation with dynamic rank weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        # Compute fitness median for success ratio baseline
        median_fit = np.median(self.fitness)

        # Compute success-ratio weights dynamically (key difference from static weights)
        success_weights = np.zeros(self.mu)
        for i in range(self.mu):
            fitness_i = self.fitness[i]
            # Ratio of improvement over median (capped for stability)
            improvement_ratio = (median_fit - fitness_i) / (abs(median_fit) + 1e-10)
            # Only positive improvements contribute, scaled logarithmically
            if improvement_ratio > 0:
                success_weights[i] = np.log1p(improvement_ratio * 10.0)

        # Normalize success weights
        sum_sw = np.sum(success_weights)
        if sum_sw > 1e-15:
            success_weights /= sum_sw
            # Blend with original log-based weights for stability
            final_weights = 0.6 * success_weights + 0.4 * self.weights[:self.mu]
            final_weights /= np.sum(final_weights)
        else:
            final_weights = self.weights[:self.mu]

        # Rank-mu update with success-based weighting
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += final_weights[i] * np.outer(diff, diff)

        # Adaptive learning rate based on condition number
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min

        # Increase learning rate when poorly conditioned (harder problems)
        if cond > 1e4:
            ccov_eff = np.clip(self.ccov * 2.0, 1e-8, 0.5)
            cc_eff = np.clip(self.cc * 1.5, 0.001, 0.3)
        elif cond > 1e2:
            ccov_eff = np.clip(self.ccov * 1.2, 1e-8, 0.5)
            cc_eff = self.cc
        else:
            ccov_eff = self.ccov
            cc_eff = self.cc

        # Re-compute pc with adaptive rate
        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Combine updates
        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _adapt_covariance_original(self):
        """Progressive restart with normalized intensity for escaping severe stagnation."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Standard evolution path update (unchanged)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)

        # Detect stagnation via fitness variance
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        threshold = max(fit_scale * 1e-6, 1e-8)
        is_stagnant = fit_var < threshold

        if is_stagnant:
            # Normalize restart intensity by generational progress to prevent excessive restarts
            gen_norm = max(self.generation, 1)
            restart_intensity = min(1.0 + np.log1p(gen_norm) / gen_norm, 5.0)

            # Preserve best solution
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]

            # Reinitialize population around best solution
            spread = restart_intensity * 0.5 * (self.ub[0] - self.lb[0])
            samples = np.random.randn(self.NP, self.dim)
            for d in range(self.dim):
                bins = np.linspace(elite[d] - spread, elite[d] + spread, self.NP + 1)
                bins = np.clip(bins, self.lb[d], self.ub[d])
                perm = np.random.permutation(self.NP)
                samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)

            self.population = self._clip_to_bounds(samples)
            self.fitness = self.func(self.population)
            self.mean = np.mean(self.population, axis=0)
            self.old_mean = self.mean.copy()

            # Reset and scale step size
            self.sigma = min(self.sigma * (1.0 + restart_intensity * 0.5), 0.3 * (self.ub[0] - self.lb[0]))

            # Reset and perturb covariance matrix
            eigvals = np.linalg.eigvalsh(self.C)
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.C += restart_intensity * 0.2 * np.random.randn(self.dim, self.dim)
            self.C = 0.5 * (self.C + self.C.T)
            self.C = self._ensure_positive_definite(self.C)

            # Reset evolution paths
            self.pc = np.zeros(self.dim)

            # Scale up learning rates for faster adaptation after restart
            self.cc = min(self.cc * (1.0 + restart_intensity * 0.3), 0.3)
            self.ccov = min(self.ccov * (1.0 + restart_intensity * 0.5), 0.5)

            # Update best if elite is better
            if elite_fit < self.f_opt:
                self.f_opt = elite_fit
                self.x_opt = elite.copy()
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