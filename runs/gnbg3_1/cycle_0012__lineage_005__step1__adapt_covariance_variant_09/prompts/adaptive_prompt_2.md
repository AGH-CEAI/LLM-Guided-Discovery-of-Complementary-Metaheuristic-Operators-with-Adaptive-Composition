Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_09` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.463715e-02            5.864673e-02            5.518186e-02            -inf                    5.450900e-02            5.482874e-02            5.775035e-02            5.679021e-02            5.826360e-02            5.853001e-02            5.535223e-02            
1      4.197947e-02            5.446572e-02            4.285112e-02            -inf                    4.422320e-02            4.394275e-02            4.641493e-02            4.546945e-02            4.518451e-02            4.237665e-02            4.402135e-02            
2      1.465092e-01            1.565134e-01            1.414206e-01            -inf                    1.598773e-01            1.659511e-01            1.758802e-01            2.008199e-01            3.789740e-01            1.351018e-01            1.641430e-01            
3      1.270906e-01            2.165449e+00            2.003083e-01            -inf                    1.556996e-01            3.818408e-01            1.597710e-01            4.838591e-01            2.219085e-01            1.624217e-01            1.577479e-01            
4      5.140848e-01            6.185958e-01            5.152479e-01            -inf                    5.126535e-01            5.087928e-01            5.179805e-01            5.122345e-01            5.235706e-01            5.118927e-01            5.167145e-01            
5      6.900990e-05            5.630481e-04            7.558967e-05            -inf                    7.080878e-05            9.654794e-05            9.772120e-05            6.851669e-05            1.059739e-04            9.518598e-05            7.749278e-05            
6      1.485090e-02            3.045946e+01            2.297347e+01            -inf                    3.707055e+01            2.406420e+01            2.267352e+01            3.079984e+01            2.994683e+01            2.445263e+01            2.917905e+01            
7      7.593510e-02            5.957874e-01            8.989103e-02            -inf                    8.763726e-02            7.776266e-02            8.060679e-02            7.552654e-02            9.029802e-02            7.789648e-02            9.693693e-02            
8      2.441312e-01            9.954370e-01            2.764835e-01            -inf                    3.086750e-01            2.909705e-01            2.671686e-01            3.887635e-01            2.514563e-01            2.530702e-01            2.555029e-01            
9      8.639168e-02            7.616213e+00            7.284599e+00            -inf                    7.415064e+00            7.321951e+00            7.431069e+00            7.457616e+00            6.547545e+00            7.434385e+00            7.487059e+00            
10     1.885743e+01            1.020885e+01            1.734294e+01            -inf                    1.988832e+01            2.478280e+01            2.030335e+01            8.328740e+00            1.615154e+01            8.554474e+00            1.538420e+01            
11     2.123507e+01            4.700589e+01            2.079368e+01            -inf                    3.159096e+01            5.078043e+01            4.116839e+01            8.673445e+00            3.489549e+01            1.368350e+01            2.062575e+01            
12     2.263761e+01            5.522686e+00            1.975458e+01            -inf                    1.568145e+01            1.933047e+01            2.316023e+01            1.345105e+01            2.557261e+01            1.397371e+01            1.712485e+01            
13     2.448977e+00            3.542119e+00            2.844601e+00            -inf                    2.798904e+00            3.420076e+00            2.311230e+00            2.248071e+00            2.841204e+00            2.135466e+00            2.174529e+00            
14     2.676600e+00            2.854034e+00            2.793208e+00            -inf                    2.781312e+00            2.845740e+00            2.735657e+00            2.750813e+00            2.798484e+00            2.738783e+00            2.744695e+00            
15     2.827153e+00            2.773340e+00            2.807912e+00            -inf                    2.742781e+00            2.891001e+00            2.823410e+00            2.810770e+00            2.831039e+00            2.757955e+00            2.821213e+00            
16     6.619597e+01            6.576840e+01            6.170414e+01            -inf                    6.366467e+01            8.268048e+01            6.762005e+01            6.588254e+01            7.156429e+01            7.685461e+01            6.890983e+01            
17     1.156140e+02            5.150365e+01            3.407308e+02            -inf                    2.816757e+01            1.427621e+02            3.840023e+01            2.304587e+01            1.726653e+02            2.617464e+01            4.175194e+01            
18     2.024885e+01            1.649981e+01            1.703931e+01            -inf                    1.171477e+01            2.438841e+01            2.167926e+01            1.255813e+01            1.746464e+01            1.512258e+01            2.081114e+01            
19     1.641992e+01            2.669970e+01            1.777649e+01            -inf                    1.732943e+01            1.759412e+01            2.067155e+01            1.637659e+01            1.508041e+01            1.395897e+01            2.371862e+01            
20     1.960875e+01            2.003709e+01            1.937518e+01            -inf                    1.651538e+01            1.943989e+01            1.908507e+01            1.887494e+01            2.130425e+01            1.996536e+01            1.810639e+01            
21     4.466429e+00            4.530367e+00            4.564601e+00            -inf                    4.448944e+00            4.554600e+00            4.455583e+00            4.384007e+00            4.630907e+00            4.350500e+00            4.581017e+00            
22     1.163650e+01            1.221896e+01            1.169114e+01            -inf                    1.004665e+01            1.223857e+01            1.078766e+01            1.067777e+01            1.139783e+01            9.892614e+00            1.147036e+01            
23     2.406307e+01            2.309864e+01            2.423068e+01            -inf                    2.239629e+01            2.184115e+01            2.112907e+01            1.952683e+01            1.944601e+01            2.220763e+01            2.288334e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_04_idea_0.py  (error=5.450900e-02)
Task  1: original.py  (error=4.197947e-02)
Task  2: variant_09_idea_0.py  (error=1.351018e-01)
Task  3: original.py  (error=1.270906e-01)
Task  4: variant_05_idea_0.py  (error=5.087928e-01)
Task  5: variant_07_idea_0.py  (error=6.851669e-05)
Task  6: original.py  (error=1.485090e-02)
Task  7: variant_07_idea_0.py  (error=7.552654e-02)
Task  8: original.py  (error=2.441312e-01)
Task  9: original.py  (error=8.639168e-02)
Task 10: variant_07_idea_0.py  (error=8.328740e+00)
Task 11: variant_07_idea_0.py  (error=8.673445e+00)
Task 12: variant_01_idea_0.py  (error=5.522686e+00)
Task 13: variant_09_idea_0.py  (error=2.135466e+00)
Task 14: original.py  (error=2.676600e+00)
Task 15: variant_04_idea_0.py  (error=2.742781e+00)
Task 16: variant_02_idea_0.py  (error=6.170414e+01)
Task 17: variant_07_idea_0.py  (error=2.304587e+01)
Task 18: variant_04_idea_0.py  (error=1.171477e+01)
Task 19: variant_09_idea_0.py  (error=1.395897e+01)
Task 20: variant_04_idea_0.py  (error=1.651538e+01)
Task 21: variant_09_idea_0.py  (error=4.350500e+00)
Task 22: variant_09_idea_0.py  (error=9.892614e+00)
Task 23: variant_08_idea_0.py  (error=1.944601e+01)

WIN COUNTS:
  original.py: 6 wins
  variant_09_idea_0.py: 5 wins
  variant_07_idea_0.py: 5 wins
  variant_04_idea_0.py: 4 wins
  variant_05_idea_0.py: 1 wins
  variant_01_idea_0.py: 1 wins
  variant_02_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """sep-CMA-ES: Separable covariance adaptation for robust dimension-wise updates."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Adaptive learning rates based on recent improvement
        if not hasattr(self, 'recent_improvements'):
            self.recent_improvements = []

        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        self.recent_improvements.append(improvement)
        if len(self.recent_improvements) > 10:
            self.recent_improvements.pop(0)

        avg_improvement = np.mean(self.recent_improvements) if self.recent_improvements else 0.0
        improv_scale = np.clip(1.0 + np.log1p(avg_improvement * 1e6), 0.5, 3.0)

        cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
        ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)

        # Update evolution path
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Separable rank-one update: per-dimension contribution from evolution path
        pc_sq = self.pc ** 2

        # Separable rank-mu update: per-dimension weighted population variance
        rank_mu_diag = np.zeros(self.dim)
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu_diag += self.weights[i] * (diff ** 2)

        # Update diagonal of covariance matrix (separable adaptation)
        for d in range(self.dim):
            self.C[d, d] = ((1.0 - ccov_adapt) * self.C[d, d] +
                            ccov_adapt * pc_sq[d] +
                            (1.0 - 1.0 / self.mueff) * ccov_adapt * rank_mu_diag[d])
            self.C[d, d] = max(self.C[d, d], 1e-10)

        # Keep only diagonal (separable approach)
        np.fill_diagonal(self.C, np.diag(self.C))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_02_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Hierarchical stagnation detection with targeted eigenspace injection."""
        # Compute population fitness statistics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        # Compute covariance condition and spectrum
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-10)
        eig_spread = eig_min / (eig_max + 1e-10)

        # Compute expected variance and diversity
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        pop_variance = np.mean(np.var(self.population, axis=0))
        diversity_ratio = pop_variance / (expected_var + 1e-10)

        # Initialize stagnation counters if needed
        if not hasattr(self, 'stagnation_levels'):
            self.stagnation_levels = {'fitness': 0, 'cov': 0, 'path': 0, 'diversity': 0}

        # Level 1: Fitness stagnation
        if rel_var < 1e-4:
            self.stagnation_levels['fitness'] += 1
        else:
            self.stagnation_levels['fitness'] = max(0, self.stagnation_levels['fitness'] - 2)

        # Level 2: Covariance collapse
        if cond > 1e6 or eig_spread < 1e-6:
            self.stagnation_levels['cov'] += 1
        else:
            self.stagnation_levels['cov'] = max(0, self.stagnation_levels['cov'] - 2)

        # Level 3: Diversity loss
        if diversity_ratio < 1e-3:
            self.stagnation_levels['diversity'] += 1
        else:
            self.stagnation_levels['diversity'] = max(0, self.stagnation_levels['diversity'] - 2)

        # Check if any stagnation level exceeds threshold
        threshold = max(5, self.dim // 2)
        trigger_level = max(self.stagnation_levels.values())
        is_stagnant = trigger_level >= threshold

        if is_stagnant:
            # Compute eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(self.C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[idx], 1e-10)
            eigvecs = eigvecs[:, idx]

            # Inject along top k eigendirections with diminishing strength
            k = min(self.dim, max(3, self.dim // 3))
            injection_strength = 0.5 * min(diversity_ratio + 0.1, 1.0)

            for i in range(k):
                # Stronger injection for directions with smaller eigenvalues (collapsed axes)
                rel_strength = injection_strength * (k - i) / k
                self.C += rel_strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Also add isotropic component for uniform exploration boost
            self.C += 0.1 * injection_strength * np.eye(self.dim)

            # Reset evolution path to encourage new direction
            self.pc *= 0.2

            # Reset stagnation counters
            for key in self.stagnation_levels:
                self.stagnation_levels[key] = 0

            self.C = self._ensure_positive_definite(self.C)
        else:
            # Standard covariance update with adaptive learning rates
            y_mean = (self.mean - self.old_mean) / self.sigma

            # Adaptive cc based on condition
            cc_adapt = self.cc * np.clip(1.0 + np.log10(max(cond, 1.0)) * 0.1, 0.5, 2.0)
            cc_adapt = np.clip(cc_adapt, 0.01, 0.3)

            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            # Adaptive ccov based on diversity
            ccov_scale = np.clip(1.0 + 2.0 * np.log1p(diversity_ratio * 10.0), 0.1, 3.0)
            ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one + 
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            # Prevent extreme conditioning during normal operation
            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / np.min(eigvals_check)
            if cond_check > 1e7:
                self.C *= 0.5

            self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_04_idea_0.py (4 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Immediate stagnation-triggered covariance reset with rank-2 updates."""
        stagnation_threshold = max(20, self.dim)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            # Aggressive multi-axis perturbation: inject fresh variance along eigendirections
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            sorted_idx = np.argsort(eigvals)

            # Perturb along top 60% of eigenvectors with decreasing strength
            k = max(1, int(0.6 * self.dim))
            for i in range(k):
                idx = sorted_idx[self.dim - 1 - i]
                strength = 3.0 * (k - i) / k
                self.C += strength * np.outer(eigvecs[:, idx], eigvecs[:, idx])

            # Add uniform axis-aligned perturbation
            self.C += 1.5 * np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.01
            self.stagnation_counter = 0
        else:
            # Standard rank-1 update with progress-weighted scaling
            y_mean = (self.mean - self.old_mean) / self.sigma

            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            improv_scale = np.clip(1.0 + 5.0 * np.log1p(improvement * 1e4), 0.5, 3.0)

            cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            # Rank-mu update (normalized)
            rank_mu = np.zeros((self.dim, self.dim))
            total_w = 0.0
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                w = self.weights[i]
                rank_mu += w * np.outer(diff, diff)
                total_w += abs(w)

            if total_w > 0:
                rank_mu /= total_w

            ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one + 
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            # Condition number check to prevent ill-conditioning
            eigvals_check = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals_check) / np.min(eigvals_check)
            if cond > 1e6:
                self.C *= 0.5

            self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Eigenspace diversity-driven covariance reshaping for escaping deceptive local optima."""
        if not hasattr(self, 'eigenspace_history'):
            self.eigenspace_history = []
            self.es_history_size = 5

        y_mean = (self.mean - self.old_mean) / self.sigma

        # Eigendecomposition for eigenspace analysis
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)
        eigvecs = np.maximum(eigvecs, 0.0)

        # Project mean displacement onto eigenspace to track directional usage
        y_proj = eigvecs.T @ y_mean
        proj_squared = y_proj ** 2

        # Track eigenspace utilization history
        self.eigenspace_history.append(proj_squared)
        if len(self.eigenspace_history) > self.es_history_size:
            self.eigenspace_history.pop(0)

        # Compute average utilization per direction
        if len(self.eigenspace_history) >= 2:
            history_array = np.array(self.eigenspace_history)
            avg_utilization = np.mean(history_array, axis=0)
            min_util = np.min(avg_utilization)
            max_util = np.max(avg_utilization)
            util_range = max(max_util - min_util, 1e-15)

            # Identify underutilized directions (potential escape routes)
            underutil_threshold = min_util + 0.2 * util_range
            underutil_mask = avg_utilization < underutil_threshold

            # Stagnation detection
            if not hasattr(self, 'stagnation_counter'):
                self.stagnation_counter = 0
            if not hasattr(self, 'prev_f_opt_es'):
                self.prev_f_opt_es = self.f_opt

            delta_f = self.prev_f_opt_es - self.f_opt
            self.prev_f_opt_es = self.f_opt

            if abs(delta_f) < 1e-10 * max(abs(self.f_opt), 1.0):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            # Adaptive reshaping strength based on stagnation and utilization imbalance
            stagnation_level = min(self.stagnation_counter / max(20, self.dim), 1.0)
            imbalance = np.sum(underutil_mask) / self.dim

            reshape_strength = stagnation_level * (0.5 + 0.5 * imbalance)
            reshape_strength = np.clip(reshape_strength, 0.0, 0.8)
        else:
            reshape_strength = 0.0
            underutil_mask = np.zeros(self.dim, dtype=bool)

        # Compute base learning rates
        ccov_base = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        # Adapt ccov based on stagnation
        if self.stagnation_counter > max(30, self.dim):
            ccov_adaptive = np.clip(ccov_base * 3.0, 1e-8, 0.5)
            cc_adaptive = np.clip(self.cc * 1.5, 0.01, 0.4)
        else:
            ccov_adaptive = np.clip(ccov_base, 1e-10, 0.5)
            cc_adaptive = np.clip(self.cc, 0.01, 0.3)

        # Update evolution path
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with normalized weights
        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        # Standard covariance update
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        # Apply eigenspace diversity reshaping when needed
        if reshape_strength > 0.01:
            eigvals_new, eigvecs_new = np.linalg.eigh(self.C)
            eigvals_new = np.maximum(eigvals_new, 1e-15)

            # Boost underutilized eigendirections
            boost_factors = np.ones(self.dim)
            if np.any(underutil_mask):
                boost_factors[underutil_mask] = 1.5 + 0.5 * stagnation_level

            # Also add small isotropic exploration
            boost_factors = boost_factors * (1.0 + 0.1 * reshape_strength)

            eigvals_reshaped = eigvals_new * boost_factors
            eigvals_reshaped = np.maximum(eigvals_reshaped, 1e-15)

            # Reconstruct covariance matrix in reshaped eigenspace
            self.C = eigvecs_new @ np.diag(eigvals_reshaped) @ eigvecs_new.T

        # Ensure positive definiteness
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_07_idea_0.py (5 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Eigenvalue-aware anisotropic stretching for escaping ill-conditioned traps."""
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

        # Eigenvalue-aware anisotropic stretching
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)
        eig_max = np.max(eigvals)
        eig_min = np.min(eigvals)
        cond = eig_max / eig_min

        # Detect ill-conditioned covariance (collapsed along some directions)
        stretch_threshold = max(1e5, self.dim * 100)
        needs_stretch = cond > stretch_threshold

        # Also check for eigenvalue starvation (smallest eigenvalue too small)
        starvation_threshold = 1e-10 * eig_max
        needs_stretch = needs_stretch or (eig_min < starvation_threshold and cond > 1e3)

        if needs_stretch:
            # Compute stretch factors inversely proportional to eigenvalue magnitude
            # Small eigenvalues get large stretch, large eigenvalues get small stretch
            stretch_factors = np.ones(self.dim)
            for j in range(self.dim):
                stretch_factors[j] = max(1.0, 5.0 * (eig_max / (eigvals[j] + 1e-15)) ** 0.5)

            # Apply anisotropic perturbation along each eigenvector
            stretch_perturb = np.zeros((self.dim, self.dim))
            for j in range(self.dim):
                v_j = eigvecs[:, j]
                stretch_perturb += (stretch_factors[j] - 1.0) * np.outer(v_j, v_j)

            self.C = self.C + 0.3 * stretch_perturb

            # Renormalize to prevent runaway growth
            new_eigvals = np.linalg.eigvalsh(self.C)
            if np.max(new_eigvals) > 10.0 * eig_max:
                self.C *= 0.5

            # Reset evolution path to prevent interference
            self.pc *= 0.5

            self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Restart-triggered covariance shattering for escaping deep local optima."""
        stagnation_threshold = max(50, self.dim * 3)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        # Compute condition and diversity metrics
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-15)
        eig_max = np.maximum(np.max(eigvals), 1e-15)
        cond = eig_max / eig_min

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = pop_variance / (expected_var + 1e-10)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        # Trigger conditions: stagnation + (ill-conditioned OR low diversity OR low variance)
        trigger_condition = is_stagnant and (cond > 1e6 or diversity < 1e-4 or rel_var < 1e-4)

        if trigger_condition:
            # SHATTER: completely rebuild covariance along novel orthogonal axes
            dim = self.dim

            # Compute anisotropic scaling from fitness landscape
            if not hasattr(self, 'prev_fitness_improvements'):
                self.prev_fitness_improvements = []

            recent_improvement = 0.0
            if hasattr(self, 'prev_fitness_improvements') and len(self.prev_fitness_improvements) > 0:
                recent_improvement = np.mean(self.prev_fitness_improvements[-5:])

            # Adaptive exploration radius: larger when stuck with large error
            error_magnitude = max(abs(self.f_opt), 1.0)
            exploration_radius = np.clip(np.log1p(error_magnitude) / dim, 0.1, 2.0)

            # Build novel covariance: diagonal with varying scales along rotated axes
            # Use Hadamard-like pattern for low-correlation axes
            novel_C = np.zeros((dim, dim))

            # Primary exploration axis: toward best solution
            if self.generation > 0:
                primary_dir = self.x_opt - self.mean
                norm_dir = np.linalg.norm(primary_dir)
                if norm_dir > 1e-10:
                    primary_dir /= norm_dir
                    # Large eigenvalue along best direction
                    novel_C += exploration_radius * 2.0 * np.outer(primary_dir, primary_dir)

            # Secondary axes: orthogonal to primary, with varying scales
            # Create dim-1 orthogonal directions using Gram-Schmidt on random vectors
            basis_vectors = []
            if len(basis_vectors) < dim:
                # Start with axis-aligned vectors, orthogonalize
                for i in range(dim):
                    v = np.zeros(dim)
                    v[i] = 1.0
                    # Orthogonalize against existing basis
                    for u in basis_vectors:
                        v = v - np.dot(u, v) * u
                    norm_v = np.linalg.norm(v)
                    if norm_v > 1e-10:
                        v /= norm_v
                        basis_vectors.append(v)

            # Assign decreasing eigenvalues to orthogonal directions
            for i, v in enumerate(basis_vectors):
                scale = exploration_radius * np.exp(-2.0 * i / dim)
                novel_C += scale * np.outer(v, v)

            # Fallback: if basis_vectors insufficient, use diagonal
            if len(basis_vectors) < dim:
                for i in range(dim):
                    novel_C[i, i] += exploration_radius * np.exp(-i / dim)

            self.C = self._ensure_positive_definite(novel_C)
            self.pc = np.zeros(dim)  # reset evolution path
            self.stagnation_counter = 0

            # Adaptive restart: keep some good individuals
            n_elite = max(2, self.NP // 10)
            elite_indices = np.argsort(self.fitness)[:n_elite]
            elite_mean = np.mean(self.population[elite_indices], axis=0)

            # Blend current best with elite mean for reinitialization
            blend_weight = 0.3
            self.mean = blend_weight * self.x_opt + (1.0 - blend_weight) * elite_mean
        else:
            # Standard adaptive update with momentum-based learning rate
            y_mean = (self.mean - self.old_mean) / self.sigma

            # Momentum-based improvement tracking
            if not hasattr(self, 'prev_fitness_improvements'):
                self.prev_fitness_improvements = []

            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            self.prev_fitness_improvements.append(improvement)
            if len(self.prev_fitness_improvements) > 20:
                self.prev_fitness_improvements.pop(0)

            # Adaptive learning rates based on improvement momentum
            if len(self.prev_fitness_improvements) >= 5:
                avg_improvement = np.mean(self.prev_fitness_improvements[-5:])
                momentum = np.clip(1.0 + np.log1p(avg_improvement * 1e4), 0.1, 3.0)
            else:
                momentum = 1.0

            cc_adaptive = np.clip(self.cc * momentum, 0.001, 0.3)
            self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adaptive = np.clip(self.ccov * momentum, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adaptive) * self.C +
                      ccov_adaptive * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_09_idea_0.py (5 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Eigenvalue-weighted covariance adaptation with diversity-driven exploration."""
        # Compute covariance eigendecomposition for geometric analysis
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)

        # Geometric analysis metrics
        cond = np.max(eigvals) / np.min(eigvals)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-15)
        eig_entropy = -np.sum((eigvals / np.sum(eigvals)) * np.log(eigvals / np.sum(eigvals) + 1e-15))
        max_eig_ratio = np.max(eigvals) / (np.median(eigvals) + 1e-15)

        # Population diversity via geometric span
        pop_diffs = self.population[:self.mu] - self.old_mean
        pop_projections = pop_diffs @ eigvecs
        proj_var = np.var(pop_projections, axis=0)
        proj_coverage = np.sum(proj_var > 1e-10 * np.mean(proj_var))
        diversity_ratio = proj_coverage / max(self.dim, 1)

        # Detect problematic states
        is_ill_conditioned = cond > 1e5 or eig_spread < 1e-6
        is_low_diversity = diversity_ratio < 0.3
        is_stagnant = self.stagnation_counter > max(20, self.dim * 2)
        needs_exploration = is_ill_conditioned or is_low_diversity or is_stagnant

        # Condition-number-adaptive ccov scaling
        cond_scaling = np.clip(np.log10(cond + 1) / 3.0, 0.1, 10.0)

        # Eigenvalue-weighted base ccov (emphasize under-represented directions)
        inv_eigvals = 1.0 / (eigvals + 1e-10)
        weights_normalized = inv_eigvals / np.sum(inv_eigvals)
        eig_weight = np.sum(weights_normalized * np.log(eigvals + 1e-10))

        base_ccov_1 = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2)
        base_ccov_mu = (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        # Scale ccov based on condition and diversity
        if needs_exploration:
            ccov_scale = 2.0 * cond_scaling
            cc_scale = 0.5
        else:
            ccov_scale = 1.0 / (1.0 + eig_weight * 0.5)
            cc_scale = 1.0

        ccov_1 = np.clip(base_ccov_1 * ccov_scale, 1e-10, 0.3)
        ccov_mu = np.clip(base_ccov_mu * ccov_scale, 1e-10, 0.3)
        cc_adapt = np.clip(self.cc * cc_scale, 0.001, 0.3)

        # Evolution path update
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update (eigenvalue-weighted)
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with eigenvalue-based reweighting
        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            # Reweight by inverse eigenvalue of that direction
            proj = diff @ eigvecs
            eig_reweight = np.sum((proj ** 2) / (eigvals + 1e-10))
            w_eff = w * np.clip(eig_reweight / (np.sum(proj ** 2) + 1e-10), 0.1, 10.0)
            rank_mu += w_eff * np.outer(diff, diff)
            total_w += abs(w_eff)

        if total_w > 0:
            rank_mu /= total_w

        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Diversity-driven exploration boost
        if needs_exploration:
            # Add exploration along top and bottom eigenvectors
            k_perturb = min(self.dim, max(2, self.dim // 3))
            for i in range(k_perturb):
                # Top eigenvectors (strong perturbation to break narrow basins)
                strength_top = 0.5 * (k_perturb - i) / k_perturb
                self.C += strength_top * np.outer(eigvecs[:, -(i + 1)], eigvecs[:, -(i + 1)])
                # Bottom eigenvectors (gentle boost for dormant directions)
                strength_bottom = 0.2 * (i + 1) / k_perturb
                self.C += strength_bottom * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Axis-aligned perturbation for uniform coverage
            self.C += 0.3 * np.eye(self.dim)

            # Dampen pc to reset biased search direction
            self.pc *= 0.2

            # Reset stagnation counter after exploration
            self.stagnation_counter = 0

        # Ensure positive definiteness
        self.C = self._ensure_positive_definite(self.C)

        # Emergency restart on extreme ill-conditioning
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.min(eigvals_check)
        if cond_check > 1e8:
            self.C = np.eye(self.dim) * np.mean(eigvals_check)
            self.pc = np.zeros(self.dim)
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
        """Explosive eigendirection perturbation on stagnation for multimodal escape."""
        stagnation_threshold = max(30, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            # Explosive exploration: perturb along all principal eigendirections
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)

            # Perturb along top k eigenvectors with decreasing strength
            k = min(self.dim, max(3, self.dim // 4))
            for i in range(k):
                strength = 2.0 * (k - i) / k  # stronger for top eigenvectors
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Also add axis-aligned perturbation for uniform coverage
            self.C += 0.5 * np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.1  # reset evolution path
            self.stagnation_counter = 0
        else:
            # Standard covariance update with improvement-weighted scaling
            y_mean = (self.mean - self.old_mean) / self.sigma

            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            improv_scale = np.clip(1.0 + 10.0 * np.log1p(improvement * 1e6), 0.1, 5.0)

            cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals_check) / np.min(eigvals_check)
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