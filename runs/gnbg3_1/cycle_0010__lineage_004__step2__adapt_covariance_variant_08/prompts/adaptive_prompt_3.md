Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_08` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.693508e-02            5.956891e-02            5.645121e-02            -inf                    5.694165e-02            -inf                    5.588045e-02            6.058819e-02            5.630607e-02            5.622726e-02            5.608249e-02            
1      4.314960e-02            4.394420e-02            4.738787e-02            -inf                    4.265579e-02            -inf                    4.384655e-02            4.539975e-02            4.435133e-02            4.180192e-02            4.240407e-02            
2      1.952052e-01            1.505612e-01            1.527230e-01            -inf                    1.436237e-01            -inf                    2.396392e-01            1.504673e-01            1.562634e-01            1.513903e-01            1.440424e-01            
3      1.302377e-01            2.492586e-01            1.311335e-01            -inf                    1.275716e-01            -inf                    1.331010e-01            1.337727e-01            1.339808e-01            2.404144e-01            1.307111e-01            
4      5.126635e-01            5.583763e-01            5.117033e-01            -inf                    5.180809e-01            -inf                    5.198890e-01            5.099087e-01            5.180774e-01            5.135700e-01            5.162655e-01            
5      1.091499e-04            1.021981e-04            1.068328e-04            -inf                    1.026140e-04            -inf                    8.794352e-05            1.146559e-04            9.430640e-05            9.202483e-05            8.840722e-05            
6      1.580902e-02            1.527492e-02            1.491334e-02            -inf                    1.525625e-02            -inf                    1.555504e-02            1.566782e-02            1.488211e-02            1.503686e-02            1.564329e-02            
7      8.315822e-02            8.246468e-02            8.292402e-02            -inf                    8.079540e-02            -inf                    7.895844e-02            1.522883e-01            7.714989e-02            7.841770e-02            7.442408e-02            
8      2.550828e-01            2.538429e-01            1.848378e+00            -inf                    2.600928e-01            -inf                    4.509200e-01            2.576270e-01            4.440958e-01            3.177474e-01            2.460600e-01            
9      1.912892e-01            1.917685e-01            2.093801e+00            -inf                    1.371433e-01            -inf                    1.390571e-01            6.986602e-02            2.658272e-01            1.046276e-01            7.226689e-02            
10     1.949887e+01            1.248161e+01            2.341410e+01            -inf                    1.087830e+01            -inf                    1.520162e+01            3.106237e+01            1.534842e+01            1.985450e+01            1.348303e+01            
11     3.868308e+01            2.180727e+01            2.661666e+01            -inf                    3.181183e+01            -inf                    3.126429e+01            5.753443e+01            3.557992e+01            2.313211e+01            4.292417e+01            
12     1.502848e+01            1.570116e+01            2.003584e+01            -inf                    2.878084e+01            -inf                    2.080764e+01            1.764462e+01            1.884118e+01            1.679376e+01            1.745455e+01            
13     3.127491e+00            2.693489e+00            2.936084e+00            -inf                    2.557403e+00            -inf                    3.387609e+00            6.392882e+00            2.872053e+00            2.518344e+00            1.698113e+00            
14     2.774192e+00            2.900283e+00            2.803610e+00            -inf                    2.657729e+00            -inf                    2.725748e+00            2.897645e+00            2.666614e+00            2.941881e+00            2.691049e+00            
15     2.795656e+00            2.743396e+00            2.777347e+00            -inf                    2.806353e+00            -inf                    2.742165e+00            2.727438e+00            2.758361e+00            2.740972e+00            2.841946e+00            
16     1.096026e+02            3.604968e+02            2.296587e+02            -inf                    1.731971e+02            -inf                    1.658305e+02            3.225438e+02            1.787791e+02            2.560307e+02            6.333563e+01            
17     4.492466e+01            4.941449e+01            5.939084e+02            -inf                    9.988462e+02            -inf                    1.663473e+02            1.888397e+03            2.804339e+02            1.611454e+02            3.462647e+01            
18     1.818309e+01            1.383094e+01            2.091016e+01            -inf                    1.753236e+01            -inf                    1.970656e+01            2.087773e+01            1.815182e+01            1.930697e+01            1.927086e+01            
19     1.645640e+01            1.801356e+01            1.824462e+01            -inf                    2.275834e+01            -inf                    2.891319e+01            2.326788e+01            1.603803e+01            1.898371e+01            1.772518e+01            
20     2.119021e+01            2.152421e+01            2.337066e+01            -inf                    2.153880e+01            -inf                    2.524864e+01            2.077596e+01            2.349856e+01            2.273780e+01            2.018037e+01            
21     4.333023e+00            4.500156e+00            4.537922e+00            -inf                    4.448804e+00            -inf                    4.361632e+00            4.538897e+00            4.376698e+00            4.402833e+00            4.555958e+00            
22     9.177052e+00            9.191219e+00            9.615094e+00            -inf                    7.628223e+00            -inf                    9.041792e+00            9.011631e+00            7.633360e+00            9.803217e+00            1.080803e+01            
23     1.862057e+01            2.276188e+01            2.396680e+01            -inf                    1.831699e+01            -inf                    2.428861e+01            2.182645e+01            2.554543e+01            2.147385e+01            2.018838e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_idea_0.py  (error=5.588045e-02)
Task  1: variant_09_idea_0.py  (error=4.180192e-02)
Task  2: variant_04_idea_0.py  (error=1.436237e-01)
Task  3: variant_04_idea_0.py  (error=1.275716e-01)
Task  4: variant_07_idea_0.py  (error=5.099087e-01)
Task  5: variant_06_idea_0.py  (error=8.794352e-05)
Task  6: variant_08_idea_0.py  (error=1.488211e-02)
Task  7: variant_10_idea_0.py  (error=7.442408e-02)
Task  8: variant_10_idea_0.py  (error=2.460600e-01)
Task  9: variant_07_idea_0.py  (error=6.986602e-02)
Task 10: variant_04_idea_0.py  (error=1.087830e+01)
Task 11: variant_01_idea_0.py  (error=2.180727e+01)
Task 12: original.py  (error=1.502848e+01)
Task 13: variant_10_idea_0.py  (error=1.698113e+00)
Task 14: variant_04_idea_0.py  (error=2.657729e+00)
Task 15: variant_07_idea_0.py  (error=2.727438e+00)
Task 16: variant_10_idea_0.py  (error=6.333563e+01)
Task 17: variant_10_idea_0.py  (error=3.462647e+01)
Task 18: variant_01_idea_0.py  (error=1.383094e+01)
Task 19: variant_08_idea_0.py  (error=1.603803e+01)
Task 20: variant_10_idea_0.py  (error=2.018037e+01)
Task 21: original.py  (error=4.333023e+00)
Task 22: variant_04_idea_0.py  (error=7.628223e+00)
Task 23: variant_04_idea_0.py  (error=1.831699e+01)

WIN COUNTS:
  variant_04_idea_0.py: 6 wins
  variant_10_idea_0.py: 6 wins
  variant_07_idea_0.py: 3 wins
  variant_06_idea_0.py: 2 wins
  variant_08_idea_0.py: 2 wins
  variant_01_idea_0.py: 2 wins
  original.py: 2 wins
  variant_09_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Cholesky-direct active covariance adaptation with diagonal perturbation."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Update evolution path with adaptive learning rate
        cc_adapt = min(self.cc * 1.5, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update from evolution path
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update from weighted population differences
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adaptive learning rates
        ccov_1 = min(2.0 / (self.dim + 2.0), self.ccov * 2.0)
        ccov_mu = min(self.mueff / (self.dim + 2.0) / (self.dim + 4.0), ccov_1 * 0.5)

        # Combine covariance update
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Add small diagonal noise for exploration (diversification)
        diag_noise = 1e-6 * (self.ub[0] - self.lb[0])
        self.C += diag_noise * np.eye(self.dim)

        # Ensure positive definiteness
        self.C = self._ensure_positive_definite(self.C)

        # Check condition number and apply damping if needed
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / max(np.min(eigvals), 1e-15)

        if cond > 1e7:
            # Scale down and add regularization
            self.C *= 0.9
            self.C += 1e-4 * np.eye(self.dim)
        elif cond < 1e3 and self.generation > 20:
            # Increase adaptation if well-conditioned
            self.C += 0.01 * np.eye(self.dim)
```

# --- From variant_04_idea_0.py (6 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Restart-triggered aggressive covariance injection for escaping catastrophic stagnation."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Check stagnation severity and covariance health
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.maximum(np.min(eigvals), 1e-15)

        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)

        # Determine boost level based on how stuck we are
        if stagnation_ratio > 0.7 or cond > 1e5:
            # Catastrophic stagnation: inject fresh random direction + maximum boost
            inject_dir = np.random.randn(self.dim)
            inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
            random_rank_one = np.outer(inject_dir, inject_dir)
            inject_strength = 0.5
            ccov_boost = 10.0
            cc_boost = 5.0
        elif stagnation_ratio > 0.4 or cond > 1e4:
            # Moderate stagnation: partial injection + strong boost
            inject_dir = np.random.randn(self.dim)
            inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
            random_rank_one = np.outer(inject_dir, inject_dir)
            inject_strength = 0.2
            ccov_boost = 4.0
            cc_boost = 2.5
        elif stagnation_ratio > 0.2:
            # Mild stagnation: small injection + moderate boost
            inject_dir = np.random.randn(self.dim)
            inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
            random_rank_one = np.outer(inject_dir, inject_dir)
            inject_strength = 0.1
            ccov_boost = 2.0
            cc_boost = 1.5
        else:
            # Normal mode: no injection
            random_rank_one = np.zeros((self.dim, self.dim))
            inject_strength = 0.0
            ccov_boost = 1.0
            cc_boost = 1.0

        # Apply boosted learning rates
        cc_adapted = min(self.cc * cc_boost, 0.5)
        self.pc = (1.0 - cc_adapted) * self.pc + np.sqrt(cc_adapted * (2.0 - cc_adapted)) * y_mean

        # Rank-one update (primary evolution path)
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Apply boosted covariance update with injection
        ccov_adapted = min(self.ccov * ccov_boost, 0.6)
        self.C = ((1.0 - ccov_adapted) * self.C +
                  ccov_adapted * rank_one +
                  inject_strength * random_rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapted * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        # Full reset on extreme ill-conditioning
        eigvals_new = np.linalg.eigvalsh(self.C)
        cond_new = np.max(eigvals_new) / np.maximum(np.min(eigvals_new), 1e-15)
        if cond_new > 1e8:
            self.C = np.eye(self.dim) * np.mean(eigvals_new)
            self.pc = np.zeros(self.dim)
            self.L = None
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Coordinate system rotation for non-separable and multi-modal landscapes."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Stagnation detection for rotation trigger
        if not hasattr(self, 'rotation_stagnation'):
            self.rotation_stagnation = 0

        is_improving = self.f_opt < getattr(self, 'prev_f_opt_rot', self.f_opt) - 1e-10
        self.prev_f_opt_rot = self.f_opt

        if is_improving:
            self.rotation_stagnation = 0
        else:
            self.rotation_stagnation += 1

        # Rotate covariance matrix when stuck (handles non-separability)
        if self.rotation_stagnation >= 3 and np.linalg.norm(self.pc) > 1e-10:
            # Compute rotation from evolution path direction
            pc_norm = np.linalg.norm(self.pc)
            u = self.pc / pc_norm

            # Householder reflection for orthogonal rotation
            v = u / (1.0 + abs(np.dot(u, np.ones(self.dim) / np.sqrt(self.dim)) + 1e-10))
            v = v - (np.dot(v, np.ones(self.dim)) / self.dim) * np.ones(self.dim)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v = v / v_norm
                # Apply rotation: C <- Q @ C @ Q.T where Q = I - 2vv^T
                self.C = self.C - 2.0 * np.outer(v, np.dot(v, self.C))
                self.C = self.C - 2.0 * np.outer(np.dot(self.C, v), v) + 4.0 * np.outer(np.dot(v, self.C), v)

            # Boost exploration along rotation axes
            self.C += 0.1 * np.eye(self.dim)
            self.rotation_stagnation = 0

        # Standard rank-one and rank-mu updates
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        # Inject diagonal exploration on stagnation to escape local optima
        if self.rotation_stagnation >= 2:
            self.C += 0.05 * np.eye(self.dim)

        # Ensure positive definiteness and condition number bound
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)

        if eig_min < 1e-10:
            self.C += (1e-9 - eig_min) * np.eye(self.dim)

        cond = eig_max / max(eig_min, 1e-10)
        if cond > 1e7:
            self.C *= 0.5
```

# --- From variant_07_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Restart-triggered covariance reset for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Monitor fitness history for stagnation detection
        if not hasattr(self, 'restart_fitness_history'):
            self.restart_fitness_history = []
        self.restart_fitness_history.append(self.f_opt)
        max_history = max(20, self.dim * 2)
        if len(self.restart_fitness_history) > max_history:
            self.restart_fitness_history.pop(0)

        should_restart = False
        if len(self.restart_fitness_history) >= max_history:
            fit_range = max(self.restart_fitness_history) - min(self.restart_fitness_history)
            fit_scale = max(abs(self.f_opt), 1.0, np.median(self.restart_fitness_history))
            if fit_range < 1e-6 * fit_scale:
                should_restart = True

        # Also restart on extreme stagnation counter
        if self.stagnation_counter > self.max_stagnation // 2:
            should_restart = True

        if should_restart:
            # Full covariance reset to escape local optima
            self.C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)

            # Aggressive step size expansion
            bound_range = max(self.ub[0] - self.lb[0], 1.0)
            self.sigma = min(self.sigma * 3.0, 0.1 * bound_range)
            self.sigma = max(self.sigma, 1e-8)

            # Clear history after restart
            self.restart_fitness_history = []
            should_restart = False

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Eigenvalue-balanced restart escape for severe multimodality."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

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

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        eig_spread = eig_min / (eig_max + 1e-15)

        # Detect stagnation: best hasn't improved
        is_stagnant = hasattr(self, 'stagnation_counter') and self.stagnation_counter > self.dim * 2

        # Detect severe ill-conditioning or collapsed spectrum
        is_ill_conditioned = cond > 1e6 or eig_spread < 1e-6

        # Detect convergence trap: low diversity + high condition
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        is_converged_trap = rel_var < 1e-3 and cond > 1e4

        needs_escape = is_stagnant or is_ill_conditioned or is_converged_trap

        if needs_escape:
            # Aggressively rebalance eigenvalues: shrink large, boost small
            target_cond = min(cond, 1e4)
            eigvals_balanced = np.exp(np.linspace(np.log(eig_min), np.log(eig_min * target_cond), self.dim))
            eigvals_balanced = np.clip(eigvals_balanced, 1e-10, None)

            # Diagonalize and rescale
            Q = np.linalg.eig(self.C)[1]
            self.C = Q @ np.diag(eigvals_balanced) @ Q.T
            self.C = 0.5 * (self.C + self.C.T)

            # Inject exploration along principal axes
            self.C += 0.2 * np.mean(eigvals) * np.eye(self.dim)

            # Boost learning rates temporarily
            ccov_escape = min(self.ccov * 5.0, 0.4)
            cc_escape = min(self.cc * 3.0, 0.25)
        else:
            ccov_escape = self.ccov
            cc_escape = self.cc

        # Recompute pc with escaped learning rate if needed
        if needs_escape:
            self.pc = (1.0 - cc_escape) * self.pc + np.sqrt(cc_escape * (2.0 - cc_escape)) * y_mean
            rank_one = np.outer(self.pc, self.pc)

        self.C = ((1.0 - ccov_escape) * self.C + 
                  ccov_escape * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_escape * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        # Hard reset on extreme conditions
        eigvals_final = np.linalg.eigvalsh(self.C)
        cond_final = np.max(eigvals_final) / np.min(eigvals_final)
        if cond_final > 1e8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * np.mean(eigvals_final)
            self.pc = np.zeros(self.dim)
            self.L = None
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Condition-aware covariance adaptation with stagnation-triggered exploration bursts."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Track improvement velocity
        if not hasattr(self, 'improvement_history'):
            self.improvement_history = []
        current_improvement = max(0.0, self.f_opt_prev - self.f_opt)
        self.improvement_history.append(current_improvement)
        if len(self.improvement_history) > 5:
            self.improvement_history.pop(0)
        avg_improvement = np.mean(self.improvement_history) if self.improvement_history else 1e-15

        # Detect stagnation: no meaningful improvement in recent generations
        is_stagnant = (avg_improvement < 1e-8 and len(self.improvement_history) >= 5)

        # Compute condition number of current covariance
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min

        # Condition-aware learning rate: reduce update speed for ill-conditioned problems
        cond_factor = np.clip(np.log10(cond + 1.0) / 10.0, 0.01, 1.0)

        # Stagnation trigger: aggressive exploration burst
        if is_stagnant:
            # Reset improvement history after triggering
            self.improvement_history = []

            # Add exploration burst to covariance
            burst_strength = np.clip(0.5 * np.log10(cond + 1.0), 0.1, 2.0)
            self.C = (1.0 - burst_strength) * self.C + burst_strength * np.eye(self.dim)

            # Also reset evolution path to allow new direction
            self.pc = np.zeros(self.dim)

        # Adaptive learning rates based on condition number
        cc_adapt = np.clip(self.cc * cond_factor, 0.001, 0.3)
        ccov_adapt = np.clip(self.ccov * cond_factor, 1e-10, 0.5)

        # Evolution path update with condition-aware rate
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with condition-aware weighting
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        # Additional diversity injection for ill-conditioned problems
        if cond > 1e6:
            self.C += 0.01 * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)

        # Condition-based covariance reduction to prevent numerical issues
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.maximum(np.min(eigvals_check), 1e-15)
        if cond_check > 1e7:
            self.C *= 0.5
```

# --- From variant_10_idea_0.py (6 wins) ---
```python
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