Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_covariance_matrix` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.420195e-07            1.667109e-05            1.074926e-07            4.209885e+01            5.706037e+00            7.903272e-07            2.861478e-06            1.087140e-07            1.114895e-07            1.814654e-06            2.835273e-07            
1      2.940412e-07            3.568752e-05            2.895937e-07            1.005032e+02            7.199510e+00            1.352938e-06            5.350006e-06            2.988239e-07            2.574046e-07            4.173478e-06            4.756851e-07            
2      2.424825e-05            2.789521e-03            2.560808e-05            3.875686e+02            6.218130e+00            2.557667e-05            4.232675e-04            2.693619e-05            2.729753e-05            3.933822e-04            3.518922e-05            
3      4.153943e-04            1.567293e-02            4.202992e-04            1.958632e+02            2.336067e+01            4.119941e-04            3.155508e-03            4.157775e-04            4.445991e-04            3.155440e-03            4.200994e-04            
4      2.489961e-02            9.937588e-02            2.484200e-02            4.581607e+00            1.779403e+00            3.188838e-02            5.720951e-02            2.548185e-02            2.570711e-02            5.495956e-02            2.774660e-02            
5      1.000000e-08            1.000000e-08            1.000000e-08            3.984279e+08            3.882089e+02            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
6      5.372434e-07            1.258793e-03            5.370227e-07            2.182217e+03            2.861450e+02            5.612610e-07            1.137122e-05            5.796778e-07            1.529568e+01            1.996921e-05            5.154168e-07            
7      1.753360e-05            1.483449e-03            1.761106e-05            2.415337e+02            8.736803e+00            2.348716e-05            2.369101e-04            1.802359e-05            1.848941e-05            2.164225e-04            2.029990e-05            
8      1.997055e-03            2.974523e-02            2.020407e-03            3.851366e+01            2.494481e+00            1.978457e-03            9.476586e-03            2.040996e-03            2.110190e-03            9.239473e-03            2.004354e-03            
9      5.873080e-04            5.648659e-02            5.491247e-04            9.717821e+01            7.192545e+00            3.687014e+00            2.267203e-03            4.066137e+00            6.638211e+00            1.086262e+00            5.840267e-04            
10     3.413339e+00            4.164460e+00            5.731155e+00            8.881273e+01            6.173363e+01            5.770136e+00            4.834605e+00            6.032105e+00            6.005547e+00            6.277203e+00            5.084738e+00            
11     1.745365e+00            8.944050e-01            4.185109e+00            2.776167e+02            1.444706e+02            2.991330e+00            2.803939e+00            2.288184e+00            4.357229e+00            2.396140e+00            2.428650e+00            
12     1.381109e+00            2.189497e+00            2.460709e+00            6.924716e+01            5.046098e+01            1.350429e+00            3.740459e+00            2.195412e+00            2.363404e+00            1.707133e+00            1.998974e+00            
13     1.343336e+00            9.005530e-01            1.641989e+00            2.723960e+01            1.154494e+01            1.554708e+00            1.848421e+00            1.846065e+00            1.352559e+00            1.590958e+00            1.726690e+00            
14     3.904585e+00            3.984978e+00            3.586635e+00            6.806328e+00            2.792274e+00            3.954651e+00            2.658228e+00            2.721627e+00            3.622887e+00            3.165793e+00            3.460247e+00            
15     1.084512e+00            1.147843e+00            1.279996e+00            3.453105e+00            3.217631e+00            1.020905e+00            1.464382e+00            1.272178e+00            1.078384e+00            1.077169e+00            1.162053e+00            
16     5.913093e+02            4.572545e+02            3.926171e+02            2.286509e+03            5.285932e+02            5.164957e+02            3.099381e+02            3.031440e+02            6.423688e+02            7.720416e+02            5.888306e+02            
17     3.120103e+00            3.121941e+00            3.120103e+00            3.620222e+04            1.047689e+04            3.120110e+00            3.120519e+00            3.120104e+00            3.120103e+00            3.120114e+00            3.120105e+00            
18     5.403979e+00            2.743817e+00            1.432803e+01            3.901673e+01            2.796961e+01            1.358029e+01            5.892330e+00            2.674329e+00            1.408913e+01            1.072411e+01            1.609863e+01            
19     3.151107e+01            3.113016e+01            3.222608e+01            1.547200e+02            3.143054e+01            2.901529e+01            1.731980e+01            2.201878e+01            3.112234e+01            3.115740e+01            3.080416e+01            
20     2.342560e+01            2.029017e+01            2.224940e+01            3.098743e+01            2.561086e+01            2.363355e+01            1.820764e+01            2.019687e+01            2.410329e+01            2.477015e+01            2.294410e+01            
21     4.505597e+00            4.552099e+00            4.465616e+00            5.179742e+00            4.624588e+00            4.474316e+00            4.484039e+00            4.473616e+00            4.521144e+00            4.429609e+00            4.536014e+00            
22     2.319560e+00            2.317815e+00            2.454104e+00            1.273270e+01            2.908084e+00            2.912438e+00            2.240277e+00            2.483329e+00            2.511712e+00            2.576554e+00            2.733021e+00            
23     1.906595e+01            1.777700e+01            1.915735e+01            5.159740e+01            1.943753e+01            2.069091e+01            6.675497e+00            6.296887e+00            2.296871e+01            1.920888e+01            1.847875e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_02_idea_0.py  (error=1.074926e-07)
Task  1: variant_08_idea_0.py  (error=2.574046e-07)
Task  2: original.py  (error=2.424825e-05)
Task  3: variant_05_idea_0.py  (error=4.119941e-04)
Task  4: variant_02_idea_0.py  (error=2.484200e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_10_idea_0.py  (error=5.154168e-07)
Task  7: original.py  (error=1.753360e-05)
Task  8: variant_05_idea_0.py  (error=1.978457e-03)
Task  9: variant_02_idea_0.py  (error=5.491247e-04)
Task 10: original.py  (error=3.413339e+00)
Task 11: variant_01_idea_0.py  (error=8.944050e-01)
Task 12: variant_05_idea_0.py  (error=1.350429e+00)
Task 13: variant_01_idea_0.py  (error=9.005530e-01)
Task 14: variant_06_idea_0.py  (error=2.658228e+00)
Task 15: variant_05_idea_0.py  (error=1.020905e+00)
Task 16: variant_07_idea_0.py  (error=3.031440e+02)
Task 17: original.py  (error=3.120103e+00)
Task 18: variant_07_idea_0.py  (error=2.674329e+00)
Task 19: variant_06_idea_0.py  (error=1.731980e+01)
Task 20: variant_06_idea_0.py  (error=1.820764e+01)
Task 21: variant_09_idea_0.py  (error=4.429609e+00)
Task 22: variant_06_idea_0.py  (error=2.240277e+00)
Task 23: variant_07_idea_0.py  (error=6.296887e+00)

WIN COUNTS:
  original.py: 4 wins
  variant_05_idea_0.py: 4 wins
  variant_06_idea_0.py: 4 wins
  variant_02_idea_0.py: 3 wins
  variant_07_idea_0.py: 3 wins
  variant_01_idea_0.py: 2 wins
  variant_08_idea_0.py: 1 wins
  variant_10_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Update covariance matrix using active CMA-ES strategy.
            Uses both best (positive) and worst (negative) weighted individuals
            for more robust covariance adaptation.
            """
            dim = self.dim

            # Rank-1 update component (evolution path)
            rank_one = np.outer(self.p_c, self.p_c)

            # Correction for h_sigma
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Get selected individuals
            selected_best = sorted_population[:self.mu]

            # Positive rank-mu update from best individuals
            diffs_pos = (selected_best - old_mean[np.newaxis, :]) / self.sigma
            weighted_diffs_pos = self.weights[:, np.newaxis] * diffs_pos
            rank_mu_pos = diffs_pos.T @ weighted_diffs_pos

            # Active CMA-ES: negative rank-mu update from worst individuals
            # Use negative weights on worst individuals to decrease variance
            n_neg = min(self.mu, len(sorted_population) - self.mu)
            selected_worst = sorted_population[-n_neg:]

            # Compute negative weights (reverse order of worst individuals)
            neg_raw_weights = np.log(n_neg + 0.5) - np.log(np.arange(1, n_neg + 1))
            neg_weights = -neg_raw_weights / np.sum(neg_raw_weights) * 0.5  # Scaled negative weights

            # Differences for worst individuals
            diffs_neg = (selected_worst - old_mean[np.newaxis, :]) / self.sigma
            weighted_diffs_neg = neg_weights[:, np.newaxis] * diffs_neg
            rank_mu_neg = diffs_neg.T @ weighted_diffs_neg

            # Combined rank-mu (positive - negative)
            rank_mu = rank_mu_pos - rank_mu_neg

            # Learning rate adjustment for active CMA-ES
            c_1_eff = self.c_1
            c_mu_eff = self.c_mu_cov

            # Combined update
            self.C = (1.0 - c_1_eff - c_mu_eff + delta_h * c_1_eff) * self.C + \
                     c_1_eff * rank_one + \
                     c_mu_eff * rank_mu

            # Ensure symmetry (numerical robustness)
            self.C = np.triu(self.C) + np.triu(self.C, 1).T

            # Eigenvalue clamping for condition number control
            eigenvals = np.linalg.eigvalsh(self.C)
            eigenvals = np.maximum(eigenvals, 1e-20)

            # Clamp condition number to prevent extreme elongation
            max_eigen = np.max(eigenvals)
            min_eigen = np.min(eigenvals)
            cond_limit = 1e12

            if max_eigen / min_eigen > cond_limit:
                scale_factor = np.sqrt(max_eigen / (min_eigen * cond_limit))
                self.C *= scale_factor

            # Ensure positive definiteness
            min_eigen_check = np.min(np.linalg.eigvalsh(self.C))
            if min_eigen_check < 1e-15:
                self.C += np.eye(dim) * (1e-14 - min_eigen_check)
```

# --- From variant_02_idea_0.py (3 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Update covariance matrix using fitness-proportional weighting for rank-mu
            and an aggressive rank-1 component to escape local optima.
            """
            # Rank-1 update component - use larger weight for exploration
            rank_one = np.outer(self.p_c, self.p_c)

            # Correction for h_sigma
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Rank-mu update with FITNESS-PROPORTIONAL weights
            selected = sorted_population[:self.mu]
            diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)

            # Compute fitness-based weights: use normalized rank improvement
            # Best individual gets highest weight, exponential decay
            fitness_weights = np.exp(-0.5 * np.arange(self.mu))  # (mu,)
            fitness_weights = fitness_weights / np.sum(fitness_weights)  # normalize

            weighted_diffs = fitness_weights[:, np.newaxis] * diffs  # (mu, dim)
            rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

            # Adaptive learning rates: increase c_1 for exploration on hard tasks
            # Use larger rank-1 weight when sigma is large (exploration phase)
            adaptive_c1 = self.c_1 * (1.0 + 0.5 * np.log1p(self.sigma / 30.0))
            adaptive_c1 = np.clip(adaptive_c1, self.c_1, self.c_1 * 3.0)

            # Combined update with adaptive rates
            self.C = (1.0 - adaptive_c1 - self.c_mu_cov + delta_h * adaptive_c1) * self.C + \
                     adaptive_c1 * rank_one + \
                     self.c_mu_cov * rank_mu

            # Ensure positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
```

# --- From variant_05_idea_0.py (4 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Update covariance matrix using adaptive rank-1 and rank-mu updates.
            Uses current mean for rank-mu computation and momentum-based adaptation.
            """
            # Compute mean shift quality for adaptive learning
            mean_shift = self.mean - old_mean
            mean_shift_norm = np.linalg.norm(mean_shift)

            # Adaptive momentum: scale by how much the mean moved (relative to sigma)
            # Larger shifts → more aggressive adaptation
            shift_ratio = mean_shift_norm / (self.sigma + 1e-20)
            adaptive_momentum = np.clip(0.9 ** (1.0 / (1.0 + shift_ratio)), 0.85, 0.99)

            # Rank-1 update component
            rank_one = np.outer(self.p_c, self.p_c)

            # Correction for h_sigma
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Adaptive learning rates based on mean shift quality
            # When making good progress, increase learning rates
            progress_factor = np.clip(shift_ratio / 10.0, 0.5, 2.0)
            adaptive_c_1 = self.c_1 * progress_factor
            adaptive_c_mu = self.c_mu_cov * progress_factor

            # Rank-mu update: use CURRENT mean (not old_mean) to capture latest direction
            selected = sorted_population[:self.mu]
            diffs = (selected - self.mean[np.newaxis, :]) / self.sigma  # (mu, dim)
            weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
            rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

            # Combined update with adaptive momentum
            self.C = (adaptive_momentum * (1.0 - adaptive_c_1 - adaptive_c_mu) + 
                      (1.0 - adaptive_momentum)) * self.C + \
                     adaptive_c_1 * rank_one + \
                     adaptive_c_mu * rank_mu + \
                     delta_h * self.c_1 * np.eye(self.dim)

            # Ensure symmetry and positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals = np.linalg.eigvalsh(self.C)
            min_eig = np.min(eigvals)
            if min_eig < 1e-12:
                self.C += (1e-11 - min_eig) * np.eye(self.dim)
```

# --- From variant_06_idea_0.py (4 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Best-anchored rank-mu update for diversity preservation.
            Key insight: anchoring to old_mean loses inter-basin diversity.
            Anchoring to current best preserves exploration across regions.
            """
            dim = self.dim

            # Rank-1 update (evolution path - unchanged)
            rank_one = np.outer(self.p_c, self.p_c)
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Use current best as anchor (not old_mean!)
            # This preserves diversity across different basins found
            best_solution = sorted_population[0]

            # Top performers centered on best (not old_mean)
            selected = sorted_population[:self.mu]
            diffs = (selected - best_solution[np.newaxis, :]) / max(self.sigma, 1e-30)
            weighted_diffs = self.weights[:, np.newaxis] * diffs
            rank_mu = diffs.T @ weighted_diffs

            # Adaptive dampening based on condition number
            cond = self.D[-1] / max(self.D[0], 1e-30)
            if cond > 1e7:
                damp = 0.2
            elif cond > 1e5:
                damp = 0.5
            else:
                damp = 1.0

            # Stagnation detection for exploration boost
            stagnation = (self.stagnation_counter > self.dim)

            # Effective rank-mu learning rate
            c_mu_eff = self.c_mu_cov * damp
            if stagnation:
                c_mu_eff *= 2.5  # Boost exploration when stuck

            # Combined update
            self.C = (1.0 - self.c_1 - c_mu_eff + delta_h * self.c_1) * self.C + \
                     self.c_1 * rank_one + \
                     c_mu_eff * rank_mu

            # Ensure symmetry and positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals = np.linalg.eigvalsh(self.C)
            if np.min(eigvals) < 1e-15:
                self.C += np.eye(dim) * (1e-14 - np.min(eigvals))
```

# --- From variant_07_idea_0.py (3 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Diversity-modulated adaptive covariance update.
            Learning rates scale with population spread and evolution path length,
            enabling aggressive adaptation on ill-conditioned tasks.
            """
            # Rank-1 component
            rank_one = np.outer(self.p_c, self.p_c)
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Rank-mu component
            selected = sorted_population[:self.mu]
            diffs = (selected - old_mean[np.newaxis, :]) / self.sigma
            weighted_diffs = self.weights[:, np.newaxis] * diffs
            rank_mu = diffs.T @ weighted_diffs

            # Adaptive diversity-based scaling
            # Measure population spread in the current basis
            pop_spread = np.mean(np.std(selected, axis=0) ** 2)
            spread_ratio = pop_spread / (np.trace(self.C) / max(self.dim, 1) + 1e-20)

            # Evolution path magnitude as condition indicator
            path_norm = np.linalg.norm(self.p_c) + 1e-20
            path_adaptive = min(path_norm / np.sqrt(self.dim), 3.0)

            # Diversity modulation: boost learning when population is diverse
            diversity_boost = 1.0 + 2.0 * np.tanh(spread_ratio - 0.5)

            # Condition-adaptive scaling: amplify on ill-conditioned landscapes
            condition_factor = 1.0 + path_adaptive

            # Compute adaptive learning rates
            c_1_adaptive = self.c_1 * diversity_boost * condition_factor
            c_mu_adaptive = self.c_mu_cov * diversity_boost * condition_factor

            # Cap to maintain positive semidefiniteness
            c_1_adaptive = min(c_1_adaptive, 0.5)
            c_mu_adaptive = min(c_mu_adaptive, 0.9 - c_1_adaptive)

            # Combined update with adaptive rates
            self.C = (1.0 - c_1_adaptive - c_mu_adaptive + delta_h * c_1_adaptive) * self.C + \
                     c_1_adaptive * rank_one + \
                     c_mu_adaptive * rank_mu

            # Ensure symmetry and numerical health
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Update covariance matrix using rank-1, rank-mu with positive weights,
            and active covariance update with negative weights (CMA-ES active update).
            """
            # Rank-1 update component
            rank_one = np.outer(self.p_c, self.p_c)

            # Correction for h_sigma
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Standard rank-mu update with positive weights
            selected = sorted_population[:self.mu]
            diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
            weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
            rank_mu_pos = diffs.T @ weighted_diffs  # (dim, dim)

            # Active covariance update: use worst individuals with negative weights
            n_neg = max(1, self.pop_size // 4)
            neg_start = max(self.mu, self.pop_size - n_neg)
            worst = sorted_population[neg_start:]  # bottom performers
            if len(worst) >= 2:
                neg_diffs = (worst - old_mean[np.newaxis, :]) / self.sigma  # (n_neg, dim)
                # Negative weights: proportional to 1/rank, normalized to sum to -0.5
                neg_raw_weights = np.log(self.pop_size + 0.5) - np.log(np.arange(neg_start + 1, self.pop_size + 1))
                neg_weights = neg_raw_weights / np.sum(np.abs(neg_raw_weights)) * (-0.5)
                neg_weighted_diffs = neg_weights[:, np.newaxis] * neg_diffs  # (n_neg, dim)
                rank_mu_neg = neg_diffs.T @ neg_weighted_diffs  # (dim, dim)
            else:
                rank_mu_neg = np.zeros((self.dim, self.dim))

            # Combined update with active component
            c_pos = self.c_mu_cov
            c_neg = min(0.5, c_pos * 0.5)  # scale negative component
            self.C = (1.0 - self.c_1 - c_pos - c_neg + delta_h * self.c_1) * self.C + \
                     self.c_1 * rank_one + \
                     c_pos * rank_mu_pos + \
                     c_neg * rank_mu_neg

            # Ensure symmetry and positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            # Add small identity for numerical stability
            min_eig = np.min(np.linalg.eigvalsh(self.C))
            if min_eig < 1e-12:
                self.C += (1e-11 - min_eig) * np.eye(self.dim)
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Update covariance matrix using adaptive diversity-preserving rank-μ update.
            """
            # Compute fitness spread to adapt selection pressure
            fitness_values = np.array([p.fitness if hasattr(p, 'fitness') else float('inf') 
                                       for p in sorted_population[:self.mu]])
            fitness_range = np.max(fitness_values) - np.min(fitness_values) + 1e-20

            # Adaptive mu: use more individuals when fitness spread is high (diverse population)
            spread_ratio = fitness_range / (abs(np.min(fitness_values)) + 1.0)
            adaptive_mu = max(2, min(self.mu, int(self.mu * min(1.0, spread_ratio * 0.5) + self.mu * 0.5)))

            # Use exponential weights instead of log weights for smoother selection pressure
            selected = sorted_population[:adaptive_mu]
            n_selected = len(selected)

            # Exponential weights: more gradual than log weights
            ranks = np.arange(1, n_selected + 1)
            exp_weights = np.exp(-ranks / (n_selected / 3.0))  # decay factor based on relative position
            exp_weights = exp_weights / np.sum(exp_weights)

            # Compute differences from old mean (more numerically stable)
            diffs = (selected - old_mean[np.newaxis, :]) / max(self.sigma, 1e-20)  # (mu', dim)

            # Weighted differences with exponential weights
            weighted_diffs = exp_weights[:, np.newaxis] * diffs  # (mu', dim)

            # Rank-μ update component
            rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

            # Rank-1 update component (evolution path)
            rank_one = np.outer(self.p_c, self.p_c)

            # Correction for h_sigma
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Diversity-preserving regularization: add small fraction of identity
            # This prevents eigenvalue collapse when stuck
            condition_number = np.max(self.D) / (np.min(self.D) + 1e-20)
            reg_strength = min(0.01, 0.001 * np.log(max(2, condition_number)))
            diversity_reg = reg_strength * np.eye(self.dim)

            # Effective learning rate scales with adaptive_mu ratio
            mu_ratio = adaptive_mu / max(1, self.mu)
            effective_c_mu = self.c_mu_cov * (0.5 + 0.5 * mu_ratio)
            effective_c_1 = self.c_1 * (1.5 - 0.5 * mu_ratio)

            # Combined update
            self.C = (1.0 - effective_c_1 - effective_c_mu + delta_h * effective_c_1) * self.C + \
                     effective_c_1 * rank_one + \
                     effective_c_mu * rank_mu + \
                     diversity_reg

            # Ensure symmetry and positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals = np.linalg.eigvalsh(self.C)
            if np.any(eigvals <= 0):
                self.C += (abs(np.min(eigvals)) + 1e-12) * np.eye(self.dim)
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """
            Update covariance matrix using rank-1 update plus fitness-weighted
            full-population covariance (instead of only top mu individuals).
            """
            # Rank-1 update component (unchanged)
            rank_one = np.outer(self.p_c, self.p_c)
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Full population differences from CURRENT mean (key change)
            all_diffs = (sorted_population - self.mean[np.newaxis, :]) / self.sigma  # (pop, dim)

            # Compute fitness-based exponential weights for ALL individuals
            # Better individuals get higher weights, spread controlled by temperature
            pop_size = len(sorted_population)
            # Use rank-based weights: weight_i = exp(-rank_i / (pop_size * temperature))
            temperature = 0.3
            ranks = np.arange(1, pop_size + 1)
            raw_weights = np.exp(-ranks / (pop_size * temperature))
            fit_weights = raw_weights / raw_weights.sum()

            # Weighted differences from current mean
            weighted_diffs = fit_weights[:, np.newaxis] * all_diffs  # (pop, dim)
            full_pop_cov = all_diffs.T @ weighted_diffs  # (dim, dim)

            # Adaptive learning rate based on fitness improvement signal
            if self.best_f < self.last_best_f:
                c_full = self.c_mu_cov * 1.5  # Increase adaptation on improvement
            else:
                c_full = self.c_mu_cov * 0.5  # Reduce on stagnation

            # Combined update with full-population covariance
            self.C = (1.0 - self.c_1 - c_full + delta_h * self.c_1) * self.C + \
                     self.c_1 * rank_one + \
                     c_full * full_pop_cov

            # Ensure symmetry and positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
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


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance matrix adaptation using a rank-1 update
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update
    - Periodic restarts when diversity collapses
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = max(4 * dim, 20)
        self.mu = self.pop_size // 2  # number of parents
        self.sigma = 30.0  # initial step size
        self.min_sigma = 1e-12
        self.max_sigma = 100.0

        # Weights for recombination
        self.weights = self._compute_recombination_weights()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # State
        self.mean = None
        self.C = None
        self.p_sigma = None
        self.p_c = None
        self.eigen_decomp_gen = 0
        self.B = None
        self.D = None
        self.invsqrtC = None

        self.best_x = None
        self.best_f = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_f = np.inf

    def _compute_recombination_weights(self):
        """Compute log-based recombination weights for the top mu individuals."""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = raw_weights / np.sum(raw_weights)
        return weights

    def _initialize_state(self):
        """Initialize or reset the distribution parameters (mean, covariance, paths)."""
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=self.dim)
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.sigma = 30.0
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrtC = np.eye(self.dim)
        self.eigen_decomp_gen = 0
        self.stagnation_counter = 0
        self.last_best_f = np.inf

    def _update_eigen_decomposition(self):
        """Perform eigendecomposition of the covariance matrix C for sampling."""
        # Symmetrize C to avoid numerical issues
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        # Clamp eigenvalues to avoid negative values from numerical errors
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        inv_D = 1.0 / self.D
        self.invsqrtC = self.B @ np.diag(inv_D) @ self.B.T
        self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        """
        Sample a batch of candidates from the multivariate normal distribution
        N(mean, sigma^2 * C). Returns array of shape (pop_size, dim).
        """
        # Sample standard normals and transform
        z = np.random.randn(self.pop_size, self.dim)
        # Transform: x = mean + sigma * B * D * z
        scaled = z * self.D[np.newaxis, :]  # (pop_size, dim)
        rotated = scaled @ self.B.T  # (pop_size, dim)
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds [-100, 100]^dim."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        """
        Sort population by fitness (ascending = better) and return sorted arrays.
        Handles NaN by assigning them worst possible fitness.
        """
        safe_fitness = np.where(np.isnan(fitness), np.inf, fitness)
        sorted_indices = np.argsort(safe_fitness)
        return population[sorted_indices], safe_fitness[sorted_indices], z_vectors[sorted_indices]

    def _update_mean(self, sorted_population):
        """
        Update the distribution mean using weighted recombination of the best mu individuals.
        """
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]  # (mu, dim)
        self.mean = self.weights @ selected  # (dim,)
        return old_mean

    def _update_evolution_paths(self, old_mean):
        """
        Update the cumulative step-size path (p_sigma) and the covariance path (p_c).
        """
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Update step-size path
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

        # Heaviside function for p_c update
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # Update covariance path
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using rank-1 and rank-mu updates.
        """
        # Rank-1 update component
        rank_one = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

        # Combined update
        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu

    def _update_step_size(self):
        """Adapt the global step size sigma using cumulative step-size adaptation (CSA)."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)

    def _track_best(self, population, fitness):
        """Update the global best solution found so far, ignoring NaN entries."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_fitness = fitness[valid_mask]
        valid_pop = population[valid_mask]
        local_best_idx = np.argmin(valid_fitness)
        if valid_fitness[local_best_idx] < self.best_f:
            self.best_f = valid_fitness[local_best_idx]
            self.best_x = valid_pop[local_best_idx].copy()

    def _check_stagnation(self, best_gen_fitness):
        """
        Detect if the search has stagnated and trigger a restart if needed.
        Returns True if a restart was performed.
        """
        improvement = self.last_best_f - best_gen_fitness
        if improvement < 1e-12 * (1.0 + abs(self.last_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_f = min(self.last_best_f, best_gen_fitness)

        # Check for various restart conditions
        should_restart = False
        if self.stagnation_counter > 20 + self.dim:
            should_restart = True
        if self.sigma < self.min_sigma * 10:
            should_restart = True
        if np.max(self.D) / np.min(self.D) > 1e7:
            should_restart = True

        if should_restart:
            self._perform_restart()
            return True
        return False

    def _perform_restart(self):
        """
        Restart the search with a new random mean, reset covariance and paths,
        but keep the global best tracked.
        """
        old_best_x = self.best_x.copy() if self.best_x is not None else None
        old_best_f = self.best_f

        self._initialize_state()

        # If we have a known best, bias the new mean slightly towards it
        if old_best_x is not None:
            blend = np.random.uniform(0.0, 0.3)
            self.mean = blend * old_best_x + (1.0 - blend) * self.mean
            self.mean = np.clip(self.mean, self.lb, self.ub)

        self.best_x = old_best_x
        self.best_f = old_best_f

    def _should_update_eigen(self):
        """Determine if eigendecomposition should be recomputed this generation."""
        update_interval = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        return (self.generation - self.eigen_decomp_gen) >= update_interval

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        """
        If func returned fewer values than expected (budget exhaustion),
        truncate population and z_vectors to match.
        """
        n_valid = len(fitness)
        return population[:n_valid], fitness[:n_valid], z_vectors[:n_valid]

    def _inject_best_into_population(self, population, z_vectors):
        """
        Occasionally inject the global best solution into the population to
        preserve elitism across restarts.
        """
        if self.best_x is not None and self.generation % 5 == 0:
            population[-1] = self.best_x.copy()
            # Approximate z_vector (not exact but acceptable)
            z_vectors[-1] = np.zeros(self.dim)
        return population, z_vectors

    def __call__(self, func, stopping_condition):
        """
        Main optimization loop: sample, evaluate, sort, update distribution.
        """
        self._initialize_state()
        self._update_eigen_decomposition()

        # Initial evaluation of a small seed population around mean
        seed_pop = self._clip_to_bounds_batch(
            self.mean[np.newaxis, :] + self.sigma * np.random.randn(self.pop_size, self.dim)
        )
        seed_fit = func(seed_pop)
        if stopping_condition():
            self._track_best(seed_pop, seed_fit[:len(seed_pop)])
            return self.best_f, self.best_x

        self._track_best(seed_pop, seed_fit)

        while not stopping_condition():
            # Eigen decomposition update
            if self._should_update_eigen():
                try:
                    self._update_eigen_decomposition()
                except np.linalg.LinAlgError:
                    self._perform_restart()
                    self._update_eigen_decomposition()

            # Sample new population
            population, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            # Inject best occasionally
            population, z_vectors = self._inject_best_into_population(population, z_vectors)
            population = self._clip_to_bounds_batch(population)

            # Evaluate
            fitness = func(population)
            if stopping_condition():
                if len(fitness) > 0:
                    population, fitness, z_vectors = self._handle_truncated_fitness(
                        population, fitness, z_vectors
                    )
                    self._track_best(population, fitness)
                break

            # Handle truncated returns
            if len(fitness) < len(population):
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._track_best(population, fitness)
                break

            # Track best
            self._track_best(population, fitness)

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(population, fitness, z_vectors)

            # Need at least mu valid individuals
            valid_count = np.sum(np.isfinite(sorted_fit))
            if valid_count < self.mu:
                self.generation += 1
                continue

            # Update mean
            old_mean = self._update_mean(sorted_pop)

            # Update evolution paths
            h_sigma = self._update_evolution_paths(old_mean)

            # Update covariance matrix
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)

            # Update step size
            self._update_step_size()

            # Check stagnation
            best_gen_f = sorted_fit[0]
            if np.isfinite(best_gen_f):
                restarted = self._check_stagnation(best_gen_f)
                if restarted:
                    self._update_eigen_decomposition()

            self.generation += 1

        return self.best_f, self.best_x

```