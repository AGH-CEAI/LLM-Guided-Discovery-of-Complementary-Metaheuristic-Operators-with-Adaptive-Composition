Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_01` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.198624e-02            5.611165e-02            -inf                    5.317593e-02            5.498550e-02            -inf                    5.693508e-02            5.747791e-02            -inf                    5.723476e-02            5.630775e-02            
1      4.459700e-02            4.244527e-02            -inf                    4.113104e-02            4.423742e-02            -inf                    4.314960e-02            4.263842e-02            -inf                    4.450479e-02            4.539120e-02            
2      4.555690e-01            1.526739e-01            -inf                    3.338407e-01            1.479386e-01            -inf                    1.952052e-01            4.046211e+00            -inf                    2.037289e-01            1.865108e+00            
3      1.776511e-01            1.315487e-01            -inf                    2.313673e-01            2.557958e-01            -inf                    1.302377e-01            1.380589e-01            -inf                    1.376964e-01            4.251093e-01            
4      5.123177e-01            5.195793e-01            -inf                    5.159771e-01            5.587777e-01            -inf                    5.126635e-01            5.334196e-01            -inf                    5.140326e-01            5.179925e-01            
5      9.922140e-05            1.020803e-04            -inf                    8.617705e-05            2.772166e-04            -inf                    1.091499e-04            1.305159e-04            -inf                    1.230749e-04            1.942641e-04            
6      1.494947e-02            1.508154e-02            -inf                    5.633913e-02            1.539367e-02            -inf                    1.580902e-02            1.467647e-02            -inf                    1.558820e-02            1.564490e-02            
7      2.811058e-01            7.666448e-02            -inf                    9.083758e-02            8.582964e-02            -inf                    8.315822e-02            1.320130e-01            -inf                    7.969663e-02            7.679089e-02            
8      2.633893e-01            2.488070e-01            -inf                    2.569579e-01            2.563611e-01            -inf                    2.550828e-01            4.359541e-01            -inf                    2.567986e-01            2.516120e-01            
9      1.255197e-01            7.015379e-02            -inf                    2.433067e-01            6.828789e-01            -inf                    1.912892e-01            1.267931e-01            -inf                    6.998076e-02            3.085635e-01            
10     1.884560e+01            2.179017e+01            -inf                    1.084702e+01            1.612150e+01            -inf                    1.949887e+01            2.395105e+01            -inf                    2.168406e+01            2.476930e+01            
11     1.598662e+01            6.524017e+01            -inf                    3.709559e+01            2.587938e+01            -inf                    3.868308e+01            5.326628e+01            -inf                    2.824617e+01            3.445766e+01            
12     1.854798e+01            2.601059e+01            -inf                    1.662823e+01            1.822671e+01            -inf                    1.502848e+01            2.443100e+01            -inf                    2.643198e+01            2.979926e+01            
13     2.836848e+00            3.357190e+00            -inf                    2.422489e+00            4.388693e+00            -inf                    3.127491e+00            3.237847e+00            -inf                    2.762074e+00            2.708990e+00            
14     2.747525e+00            2.758997e+00            -inf                    2.690090e+00            2.907299e+00            -inf                    2.774192e+00            2.814057e+00            -inf                    2.896076e+00            2.922415e+00            
15     2.784337e+00            2.654100e+00            -inf                    2.793259e+00            2.773188e+00            -inf                    2.795656e+00            2.776947e+00            -inf                    2.830483e+00            2.827963e+00            
16     1.213889e+02            8.247239e+01            -inf                    1.331511e+02            1.092784e+02            -inf                    1.096026e+02            1.265706e+02            -inf                    8.740148e+01            1.507870e+02            
17     3.869824e+01            5.020314e+02            -inf                    5.418067e+01            1.353562e+03            -inf                    4.492466e+01            9.353186e+02            -inf                    5.634219e+02            3.471220e+02            
18     2.186745e+01            2.284899e+01            -inf                    1.760332e+01            2.130994e+01            -inf                    1.818309e+01            2.120615e+01            -inf                    1.468251e+01            1.813274e+01            
19     1.969005e+01            2.251108e+01            -inf                    2.274799e+01            2.820934e+01            -inf                    1.645640e+01            1.789106e+01            -inf                    2.948039e+01            1.522709e+01            
20     2.133874e+01            1.873657e+01            -inf                    2.143669e+01            2.041338e+01            -inf                    2.119021e+01            2.114891e+01            -inf                    2.245666e+01            2.143002e+01            
21     4.463087e+00            4.295017e+00            -inf                    4.406315e+00            4.488381e+00            -inf                    4.333023e+00            4.440060e+00            -inf                    4.397216e+00            4.455895e+00            
22     8.599603e+00            6.773821e+00            -inf                    8.017963e+00            9.000348e+00            -inf                    9.177052e+00            7.566422e+00            -inf                    7.625631e+00            9.533320e+00            
23     1.721509e+01            2.090722e+01            -inf                    2.091910e+01            1.889575e+01            -inf                    1.862057e+01            1.538347e+01            -inf                    1.996717e+01            2.151420e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=5.198624e-02)
Task  1: variant_03_idea_0.py  (error=4.113104e-02)
Task  2: variant_04_idea_0.py  (error=1.479386e-01)
Task  3: variant_06_idea_0.py  (error=1.302377e-01)
Task  4: original.py  (error=5.123177e-01)
Task  5: variant_03_idea_0.py  (error=8.617705e-05)
Task  6: variant_07_idea_0.py  (error=1.467647e-02)
Task  7: variant_01_idea_0.py  (error=7.666448e-02)
Task  8: variant_01_idea_0.py  (error=2.488070e-01)
Task  9: variant_09_idea_0.py  (error=6.998076e-02)
Task 10: variant_03_idea_0.py  (error=1.084702e+01)
Task 11: original.py  (error=1.598662e+01)
Task 12: variant_06_idea_0.py  (error=1.502848e+01)
Task 13: variant_03_idea_0.py  (error=2.422489e+00)
Task 14: variant_03_idea_0.py  (error=2.690090e+00)
Task 15: variant_01_idea_0.py  (error=2.654100e+00)
Task 16: variant_01_idea_0.py  (error=8.247239e+01)
Task 17: original.py  (error=3.869824e+01)
Task 18: variant_09_idea_0.py  (error=1.468251e+01)
Task 19: variant_10_idea_0.py  (error=1.522709e+01)
Task 20: variant_01_idea_0.py  (error=1.873657e+01)
Task 21: variant_01_idea_0.py  (error=4.295017e+00)
Task 22: variant_01_idea_0.py  (error=6.773821e+00)
Task 23: variant_07_idea_0.py  (error=1.538347e+01)

WIN COUNTS:
  variant_01_idea_0.py: 7 wins
  variant_03_idea_0.py: 5 wins
  original.py: 4 wins
  variant_06_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_09_idea_0.py: 2 wins
  variant_04_idea_0.py: 1 wins
  variant_10_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (7 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Momentum-enhanced covariance adaptation with adaptive learning rates."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Momentum term: EMA of mean shift direction for stable adaptation
        beta = 0.5
        if not hasattr(self, 'y_mean_ema'):
            self.y_mean_ema = np.zeros(self.dim)
        self.y_mean_ema = beta * self.y_mean_ema + (1.0 - beta) * y_mean

        # Adaptive learning rate based on eigenvalue spread
        eigvals = np.linalg.eigvalsh(self.C)
        eig_ratio = np.max(eigvals) / (np.min(eigvals) + 1e-10)
        cond_factor = min(1.0, 1.0 / np.log1p(eig_ratio + 1.0))

        # Momentum-enhanced evolution path
        cc_momentum = self.cc * (0.5 + 0.5 * cond_factor)
        cc_momentum = np.clip(cc_momentum, 0.01, 0.2)

        self.pc = (1.0 - cc_momentum) * self.pc + np.sqrt(cc_momentum * (2.0 - cc_momentum)) * self.y_mean_ema

        # Rank-one update with momentum
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adaptive covariance learning rate
        ccov_adaptive = self.ccov * (0.5 + 0.5 * cond_factor)
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        # Combine with momentum-weighted original covariance
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_03_idea_0.py (5 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Rank-distribution triggered exploration bursts for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute rank-based diversity: how spread out are solutions in fitness space?
        ranks = np.argsort(np.argsort(self.fitness))  # rank of each individual
        rank_spread = np.max(ranks) - np.min(ranks)
        rank_diversity = rank_spread / max(len(ranks) - 1, 1)

        # Normalize by expected spread for uniform distribution
        expected_spread = 1.0
        rank_ratio = rank_diversity / max(expected_spread, 1e-10)
        rank_ratio = np.clip(rank_ratio, 0.01, 10.0)

        # Base learning rates
        ccov_1_base = 1.0 / (self.dim + 2.0)
        ccov_mu_base = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt_base = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Exploration boost when ranks are clustered (low diversity)
        # This directly measures whether top solutions are close in fitness
        if rank_ratio < 0.3:  # Clustered ranks = need exploration
            exploration_boost = 3.0
            diag_noise = 0.1
        elif rank_ratio < 0.5:
            exploration_boost = 2.0
            diag_noise = 0.05
        elif rank_ratio < 0.7:
            exploration_boost = 1.5
            diag_noise = 0.02
        else:
            exploration_boost = 1.0
            diag_noise = 0.0

        # Apply exploration boost to rank-one update (drives principal axis movement)
        ccov_1 = min(ccov_1_base * exploration_boost, 0.3)
        ccov_mu = ccov_mu_base * min(exploration_boost, 2.0)
        cc_adapt = min(cc_adapt_base * exploration_boost, 0.3)

        # Evolution path update
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

        # Inject diagonal noise for exploration when needed
        if diag_noise > 0:
            eigvals = np.linalg.eigvalsh(self.C)
            noise = np.random.randn(self.dim) * diag_noise * np.mean(eigvals)
            self.C += np.diag(noise)

        # Restart when severely converged (extreme case)
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        if eig_spread < 1e-8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Eigenvalue history smoothing with adaptive regularization for ill-conditioned landscapes."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute current eigenvalues and condition number
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        # Track eigenvalue history for drift detection
        if not hasattr(self, 'eigval_history'):
            self.eigval_history = []
            self.eigval_ema = eigvals.copy()

        self.eigval_history.append(eigvals.copy())
        if len(self.eigval_history) > 10:
            self.eigval_history.pop(0)

        # Exponential smoothing of eigenvalues (spectrum adaptation)
        alpha_smooth = 0.1
        self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * eigvals

        # Detect eigenvalue drift (ratio of current to EMA)
        if len(self.eigval_history) >= 3:
            drift_ratio = np.median(eigvals) / (np.median(self.eigval_ema) + 1e-10)
            drift_ratio = np.clip(drift_ratio, 0.1, 10.0)
        else:
            drift_ratio = 1.0

        # Adaptive regularization based on eigenvalue health
        reg_strength = 0.0
        if cond > 1e6 or eig_spread < 1e-6:
            reg_strength = 0.1
        elif cond > 1e4 or eig_spread < 1e-4:
            reg_strength = 0.05
        elif drift_ratio < 0.5 or drift_ratio > 2.0:
            reg_strength = 0.02

        # Force eigenvalues toward geometric mean for badly conditioned cases
        if cond > 1e7 or eig_spread < 1e-8:
            target_eigvals = np.full(self.dim, np.exp(np.mean(np.log(eigvals + 1e-10))))
            self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * target_eigvals

        # Adaptive learning rates with eigenvalue-health scaling
        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
            cc_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.6
            cc_scale = 0.6
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        # Scale down when eigenvalues are drifting
        drift_penalty = 0.5 if drift_ratio < 0.5 or drift_ratio > 2.0 else 1.0
        ccov_scale *= drift_penalty
        cc_scale *= drift_penalty

        # Fitness-based adaptation signal
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
            self.eigval_ema = np.full(self.dim, np.mean(eigvals))
            return

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

        # Apply eigenvalue regularization
        if reg_strength > 0:
            current_trace = np.trace(self.C)
            target_trace = np.sum(self.eigval_ema)
            self.C = (1.0 - reg_strength) * self.C + reg_strength * (target_trace / self.dim) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
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
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Isotropic reset with aggressive exploration scaling for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Detect stagnation: no improvement in recent generations
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        # Also trigger reset if covariance is too elongated
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-10)

        # Fitness-based trigger: if best fitness hasn't improved and condition is high
        fit_var = np.var(self.fitness)
        rel_var = fit_var / (max(abs(self.f_opt), 1.0) ** 2 + 1e-10)
        is_converged = rel_var < 1e-3 and cond > 1e3

        # Force isotropic reset on stagnation or extreme conditioning
        if is_stagnant or is_converged or cond > 1e7:
            # Reset to isotropic covariance with larger spread
            C_mean = np.mean(eigvals)
            self.C = np.eye(self.dim) * max(C_mean, self.sigma ** 2)
            self.pc = np.zeros(self.dim)
            self.L = None

            # Increase step size to encourage exploration
            if self.sigma < 0.5 * (self.ub[0] - self.lb[0]) / 3.0:
                self.sigma *= 2.0
                self.sigma = np.clip(self.sigma, 1e-10, 10.0)

        # Adaptive learning rates based on condition number
        if cond > 1e6:
            ccov_scale = 0.2
            cc_scale = 0.2
        elif cond > 1e4:
            ccov_scale = 0.4
            cc_scale = 0.4
        elif cond > 1e2:
            ccov_scale = 0.7
            cc_scale = 0.7
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        # Additional scaling for stagnant/poorly-conditioned cases
        if is_stagnant or is_converged:
            ccov_scale *= 2.0
            cc_scale *= 1.5

        ccov_scale = min(ccov_scale, 2.0)
        cc_scale = min(cc_scale, 1.0)

        # Adaptive learning rates
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

        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Orthogonal-escape evolution path for multimodal function escape."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute perpendicular direction to pc for escape
        if not hasattr(self, 'p_escape'):
            self.p_escape = None

        # Trigger escape mechanism at 50% of stagnation threshold
        if self.stagnation_counter >= self.max_stagnation // 2:
            if self.p_escape is None and np.dot(self.pc, self.pc) > 1e-20:
                # Compute perpendicular direction using QR decomposition
                v = self.pc / (np.linalg.norm(self.pc) + 1e-20)
                random_vec = np.random.randn(self.dim)
                random_vec -= np.dot(random_vec, v) * v
                norm_r = np.linalg.norm(random_vec)
                if norm_r > 1e-10:
                    self.p_escape = random_vec / norm_r

        # Apply main evolution path update
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Rank-one and rank-mu updates
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine main and escape updates
        if self.p_escape is not None:
            rank_escape = np.outer(self.p_escape, self.p_escape)
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one + 
                      0.5 * self.ccov * rank_escape +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
            # Decay escape path
            self.p_escape *= 0.95
        else:
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one + 
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_01(self):
        """Aggressive covariance reset with step-size reduction for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute condition number and eigenvalue spread
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

        # Check for stagnation indicators
        recent_improved = 0
        for i in range(min(len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
        success_rate = recent_improved / max(len(self.trial_fitness), 1)

        # Trigger aggressive reset when:
        # 1. Extreme condition number (ill-conditioned landscape)
        # 2. Very small eigenvalue spread (collapsed covariance)
        # 3. Stagnation with low success rate (trapped in local optimum)
        should_reset = (cond > 1e6) or (eig_spread < 1e-6) or \
                       (success_rate < 0.05 and self.stagnation_counter > 20)

        if should_reset:
            # Reset to spherical covariance (isotropic search)
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

            # Shrink step-size to refine search after reset
            self.sigma = max(self.sigma * 0.3, 1e-10)

            # Use conservative learning rates after reset
            ccov_1 = 0.5 / (self.dim + 2.0)
            ccov_mu = 0.5 * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
            cc_adapt = 0.3 * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        else:
            # Normal adaptation when not in trap
            ccov_1 = self.ccov
            ccov_mu = self.ccov
            cc_adapt = self.cc

        # Evolution path update
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