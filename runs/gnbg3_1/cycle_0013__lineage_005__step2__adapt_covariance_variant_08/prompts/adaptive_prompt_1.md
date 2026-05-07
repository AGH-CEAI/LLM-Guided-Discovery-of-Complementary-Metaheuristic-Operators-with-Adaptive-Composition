Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_08` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.463715e-02            5.812975e-02            5.652616e-02            -inf                    -inf                    5.265602e-02            5.503138e-02            5.677969e-02            5.827112e-02            5.429045e-02            5.653840e-02            
1      4.197947e-02            4.496215e-02            4.289350e-02            -inf                    -inf                    4.081530e-02            4.221913e-02            4.643953e-02            4.663777e-02            4.464926e-02            4.172198e-02            
2      1.465092e-01            1.578556e-01            1.373009e-01            -inf                    -inf                    1.426695e-01            1.475510e-01            1.532958e-01            1.585427e-01            1.400095e-01            1.482687e-01            
3      1.270906e-01            1.291268e-01            4.516244e-01            -inf                    -inf                    1.306833e-01            1.886356e-01            2.248265e+00            1.362566e-01            1.312086e-01            1.308047e-01            
4      5.140848e-01            5.148583e-01            5.145529e-01            -inf                    -inf                    5.067673e-01            5.192349e-01            5.212964e-01            5.174754e-01            5.129761e-01            5.120652e-01            
5      6.900990e-05            1.126927e-04            1.420815e-04            -inf                    -inf                    7.122171e-05            8.698636e-05            8.829762e-05            9.407862e-05            7.725065e-05            8.122965e-05            
6      1.485090e-02            1.546407e-02            1.524205e-02            -inf                    -inf                    1.494098e-02            1.465741e-02            1.502716e-02            1.558549e-02            1.489332e-02            1.499144e-02            
7      7.593510e-02            7.948120e-02            7.835545e-02            -inf                    -inf                    7.542333e-02            7.706659e-02            3.239940e-01            8.135884e-02            7.410207e-02            7.675424e-02            
8      2.441312e-01            2.617683e-01            2.625169e-01            -inf                    -inf                    2.486560e-01            2.569260e-01            2.544153e-01            4.599967e-01            2.457658e-01            2.416415e-01            
9      8.639168e-02            7.222536e-02            1.075249e-01            -inf                    -inf                    7.224975e-02            1.007810e-01            1.206267e-01            2.543533e+00            1.409080e-01            7.023935e-02            
10     1.885743e+01            1.243494e+01            7.834287e+00            -inf                    -inf                    1.515009e+01            2.079791e+01            1.666236e+01            2.344180e+01            1.901303e+01            1.110135e+01            
11     2.123507e+01            1.626618e+01            2.437678e+01            -inf                    -inf                    9.388758e+01            2.204607e+01            3.465809e+01            3.233279e+01            3.913052e+01            4.588277e+01            
12     2.263761e+01            1.768508e+01            2.283120e+00            -inf                    -inf                    2.391685e+01            1.836142e+01            1.689701e+01            2.196902e+01            2.516637e+01            1.823175e+01            
13     2.448977e+00            2.751837e+00            2.791782e+00            -inf                    -inf                    2.146485e+00            3.220713e+00            2.924804e+00            4.814134e+00            2.218308e+00            2.688688e+00            
14     2.676600e+00            2.703046e+00            2.804990e+00            -inf                    -inf                    2.899815e+00            2.917696e+00            2.784685e+00            2.858818e+00            2.731255e+00            2.711645e+00            
15     2.827153e+00            2.692457e+00            2.822256e+00            -inf                    -inf                    2.793762e+00            2.646308e+00            2.549043e+00            2.769926e+00            2.764613e+00            2.929336e+00            
16     6.619597e+01            8.564369e+01            1.094909e+02            -inf                    -inf                    6.315044e+01            7.705344e+01            2.552406e+02            1.171852e+02            8.901652e+01            9.817973e+01            
17     1.156140e+02            3.727647e+01            4.463504e+01            -inf                    -inf                    2.191383e+02            2.347577e+02            3.934400e+02            6.147811e+02            3.618554e+01            7.506603e+01            
18     2.024885e+01            1.807794e+01            1.698619e+01            -inf                    -inf                    2.214526e+01            1.951435e+01            1.889925e+01            2.151961e+01            1.435423e+01            1.800373e+01            
19     1.641992e+01            1.386868e+01            3.243287e+01            -inf                    -inf                    1.198275e+01            2.037167e+01            2.385825e+01            1.714357e+01            1.413960e+01            2.667789e+01            
20     1.960875e+01            1.881006e+01            2.192002e+01            -inf                    -inf                    1.896320e+01            2.014493e+01            2.317884e+01            2.598199e+01            1.988706e+01            1.952472e+01            
21     4.466429e+00            4.279396e+00            4.625441e+00            -inf                    -inf                    4.444204e+00            4.323638e+00            4.330795e+00            4.457838e+00            4.284471e+00            4.291245e+00            
22     1.163650e+01            8.372090e+00            1.120581e+01            -inf                    -inf                    1.058443e+01            9.291381e+00            8.788652e+00            1.027531e+01            1.104516e+01            1.178407e+01            
23     2.406307e+01            2.045874e+01            1.999309e+01            -inf                    -inf                    1.915982e+01            2.303422e+01            2.353358e+01            2.420814e+01            2.076518e+01            2.064732e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_05_idea_0.py  (error=5.265602e-02)
Task  1: variant_05_idea_0.py  (error=4.081530e-02)
Task  2: variant_02_idea_0.py  (error=1.373009e-01)
Task  3: original.py  (error=1.270906e-01)
Task  4: variant_05_idea_0.py  (error=5.067673e-01)
Task  5: original.py  (error=6.900990e-05)
Task  6: variant_06_idea_0.py  (error=1.465741e-02)
Task  7: variant_09_idea_0.py  (error=7.410207e-02)
Task  8: variant_10_idea_0.py  (error=2.416415e-01)
Task  9: variant_10_idea_0.py  (error=7.023935e-02)
Task 10: variant_02_idea_0.py  (error=7.834287e+00)
Task 11: variant_01_idea_0.py  (error=1.626618e+01)
Task 12: variant_02_idea_0.py  (error=2.283120e+00)
Task 13: variant_05_idea_0.py  (error=2.146485e+00)
Task 14: original.py  (error=2.676600e+00)
Task 15: variant_07_idea_0.py  (error=2.549043e+00)
Task 16: variant_05_idea_0.py  (error=6.315044e+01)
Task 17: variant_09_idea_0.py  (error=3.618554e+01)
Task 18: variant_09_idea_0.py  (error=1.435423e+01)
Task 19: variant_05_idea_0.py  (error=1.198275e+01)
Task 20: variant_01_idea_0.py  (error=1.881006e+01)
Task 21: variant_01_idea_0.py  (error=4.279396e+00)
Task 22: variant_01_idea_0.py  (error=8.372090e+00)
Task 23: variant_05_idea_0.py  (error=1.915982e+01)

WIN COUNTS:
  variant_05_idea_0.py: 7 wins
  variant_01_idea_0.py: 4 wins
  variant_02_idea_0.py: 3 wins
  original.py: 3 wins
  variant_09_idea_0.py: 3 wins
  variant_10_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (4 wins) ---
```python
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
```

# --- From variant_02_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Multi-trigger adaptive eigenspace rescaling with minor-axis emphasis."""
        # --- Multi-failure detection ---
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-12)
        cond = np.max(eigvals) / np.min(eigvals)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = pop_variance / (expected_var + 1e-10)

        # Trigger conditions: stagnation OR high condition OR fitness collapse OR low diversity
        is_stagnant = self.stagnation_counter > max(20, self.dim)
        is_ill_cond = cond > 1e5
        is_collapsed = rel_var < 1e-3
        is_low_diversity = diversity < 1e-3
        needs_rescue = is_stagnant or is_ill_cond or is_collapsed or is_low_diversity

        if needs_rescue:
            # Compute adaptive rescue strength based on condition number
            log_cond = np.log10(cond + 1.0)
            rescue_strength = np.clip(0.5 + 0.5 * log_cond, 0.5, 5.0)

            # Inverse-strength weighting: amplify minor axes more
            sorted_indices = np.argsort(eigvals)
            for idx, i in enumerate(sorted_indices):
                # Rank-based strength: minor axes get MORE perturbation
                strength = rescue_strength * (idx + 1) / self.dim
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Add uniform isotropic perturbation for baseline exploration
            self.C += 0.3 * np.eye(self.dim)

            # Correlation repair: reduce off-diagonal elements that create deceptive ridges
            diag_C = np.diag(self.C)
            diag_sqrt = np.sqrt(np.maximum(diag_C, 1e-10))
            corr = self.C / np.outer(diag_sqrt, diag_sqrt)
            corr = np.clip(corr, -0.95, 0.95)
            self.C = np.diag(diag_C) + 0.2 * (corr - np.diag(np.diag(corr)))

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.2
            self.stagnation_counter = 0
        else:
            # Standard covariance update with condition-aware scaling
            y_mean = (self.mean - self.old_mean) / self.sigma

            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            improv_scale = np.clip(1.0 + 5.0 * np.log1p(improvement * 1e6), 0.2, 3.0)

            # Reduce learning rate when covariance is already well-conditioned
            cond_factor = np.clip(1.0 / np.log10(cond + 10.0), 0.1, 1.0)

            cc_adapt = np.clip(self.cc * improv_scale * cond_factor, 0.001, 0.3)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale * cond_factor, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C +
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            # Passive condition control
            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / np.min(eigvals_check)
            if cond_check > 1e7:
                self.C *= 0.5
```

# --- From variant_05_idea_0.py (7 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Forced perturbation with mean relocation for escaping deep local optima."""
        stagnation_threshold = max(30, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            # Compute fitness-based weights for selecting a good restart point
            fit_min = np.min(self.fitness)
            fit_max = np.max(self.fitness)
            fit_range = max(fit_max - fit_min, 1e-10)
            weights_reloc = np.exp(-2.0 * (self.fitness - fit_min) / fit_range)
            weights_reloc /= np.sum(weights_reloc)

            # Select a promising but diverse candidate from population
            candidate_idx = np.random.choice(self.NP, p=weights_reloc)
            candidate = self.population[candidate_idx]

            # Compute population spread for perturbation scale
            pop_spread = np.mean(np.std(self.population, axis=0))
            perturb_scale = max(pop_spread * 3.0, (self.ub[0] - self.lb[0]) * 0.1)

            # Relocate mean to diverse region with forced exploration
            random_dir = np.random.randn(self.dim)
            random_dir /= (np.linalg.norm(random_dir) + 1e-10)
            self.mean = candidate + perturb_scale * random_dir

            # Clip to bounds
            self.mean = self._clip_to_bounds(self.mean)

            # Dramatically increase step size for global search
            self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.2)

            # Reset covariance to moderate isotropic for fresh start
            eigvals = np.linalg.eigvalsh(self.C)
            avg_eig = np.mean(eigvals)
            self.C = np.eye(self.dim) * max(avg_eig, 1e-4)

            # Reset evolution path
            self.pc *= 0.1
            self.stagnation_counter = 0
        else:
            # Normal CMA-ES update
            y_mean = (self.mean - self.old_mean) / self.sigma
            self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            self.C = ((1.0 - self.ccov) * self.C +
                      self.ccov * rank_one +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            # Check condition number for numerical stability
            eigvals_check = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals_check) / np.min(eigvals_check)
            if cond > 1e7:
                self.C *= 0.5
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Fitness variance + condition-triggered multi-direction exploration for catastrophic local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Detect convergence using MULTIPLE signals
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        cond = np.max(eigvals) / np.min(eigvals)

        # Trigger exploration on BOTH low variance AND high condition
        is_converging = (rel_var < 1e-3) and (cond > 1e4)

        if is_converging:
            # AGGRESSIVE exploration: perturb along WEAKEST eigendirections
            # (opposite of current approach which perturbs strongest)
            k_perturb = min(self.dim, max(5, self.dim // 2))

            for i in range(k_perturb):
                # Perturb weakest directions most (they need the most help)
                strength = 3.0 * (i + 1) / k_perturb  # stronger for weaker eigenvectors
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Add substantial diagonal noise for uniform exploration
            self.C += 1.0 * np.eye(self.dim)

            # Reset evolution path completely
            self.pc = np.zeros(self.dim)

            self.C = self._ensure_positive_definite(self.C)
        else:
            # Standard update with diversity-scaled learning rates
            ccov_scale = 1.0 + 20.0 * min(rel_var, 0.1)
            ccov_scale = np.clip(ccov_scale, 0.1, 10.0)

            cc_adapt = np.clip(self.cc * ccov_scale, 0.001, 0.5)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            # Prevent extreme conditioning
            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / np.min(eigvals_check)
            if cond_check > 1e7:
                self.C *= 0.5
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Continuous landscape-adaptive dual-path covariance for multimodal escape."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Exploitation path: standard CMA-ES covariance update
        C_exploit = ((1.0 - self.ccov) * self.C + 
                     self.ccov * rank_one + 
                     (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        # Exploration path: adaptive isotropic covariance for uniform coverage
        pop_range = self.ub[0] - self.lb[0]
        explore_radius = 0.05 * pop_range * self.sigma / np.sqrt(self.dim)
        C_explore = np.eye(self.dim) * (explore_radius ** 2 + 1e-10)

        # Landscape sensing: detect multi-modality, ill-conditioning, and stagnation
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        # Detect local optima trapping: no improvement but population has moved
        is_trapped = (self.stagnation_counter > 10 and rel_var > 1e-4)

        # Compute continuous mixing weight from landscape metrics
        multimodality_factor = np.clip(rel_var * 10.0, 0.0, 1.0)
        illcond_factor = np.clip(1.0 - eig_spread, 0.0, 1.0)
        trap_factor = 1.0 if is_trapped else 0.0

        mix = 0.05 + 0.15 * multimodality_factor + 0.30 * illcond_factor + 0.30 * trap_factor
        mix = np.clip(mix, 0.01, 0.7)

        # Blend exploitation and exploration paths
        self.C = (1.0 - mix) * C_exploit + mix * C_explore

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_09_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Adaptive diversity injection via convergence pressure and eigendirection restart."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute convergence pressure metrics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = max(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-15)
        cond = eig_max / eig_min

        # Combined convergence pressure (0 = converged, 1 = diverse)
        pressure = min(1.0, rel_var * 10.0 + eig_spread * 5.0)
        pressure = max(0.0, pressure)

        # Adaptive threshold based on dimension and condition number
        adaptive_threshold = max(15, self.dim * 1.5) * (1.0 + 0.1 * np.log1p(cond))
        is_stagnant = self.stagnation_counter > adaptive_threshold
        is_converged = pressure < 0.05 or cond > 1e6

        # Graduated diversity injection
        if is_stagnant or is_converged:
            # Strong perturbation: restart along eigendirections
            eigvecs = np.linalg.eigh(self.C)[1]

            k = min(self.dim, max(2, self.dim // 3))
            for i in range(k):
                strength = 3.0 * (k - i) / k  # stronger for top eigenvectors
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            self.C += 0.8 * np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.05
            self.stagnation_counter = 0
        else:
            # Dynamic learning rate scaling based on convergence pressure
            improv_scale = 1.0 + 5.0 * (1.0 - pressure)
            improv_scale = np.clip(improv_scale, 0.2, 10.0)

            cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.5)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.6)
            self.C = ((1.0 - ccov_adapt) * self.C +
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            # Prevent extreme conditioning
            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / max(np.min(eigvals_check), 1e-15)
            if cond_check > 1e6:
                self.C *= 0.5
                self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_08(self):
        """Anticipatory diversity-driven restart with exponential eigendirection perturbation."""
        # Compute convergence indicators
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-10)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-10)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Proactive restart probability based on multiple convergence indicators
        # Using log(1+|f_opt|) to handle tasks at different scales
        restart_prob = 0.0
        restart_prob += 0.3 * max(0.0, 1.0 - rel_var / 1e-4) if rel_var < 1e-4 else 0.0
        restart_prob += 0.4 * max(0.0, 1.0 - np.log10(cond + 1) / 7.0) if cond > 1e3 else 0.0
        restart_prob += 0.3 * max(0.0, 1.0 - diversity / 1e-3) if diversity < 1e-3 else 0.0
        restart_prob *= np.clip(1.0 + np.log1p(abs(self.f_opt)), 0.1, 10.0)

        # Trigger restart probabilistically
        if np.random.random() < restart_prob:
            # Exponential eigendirection perturbation
            eigvecs = np.linalg.eigh(self.C)[1]

            # Strong perturbation along bottom eigenvectors (smallest variance directions)
            # These are the directions most likely to be trapped
            k = min(self.dim, max(2, self.dim // 3))
            for i in range(k):
                strength = 3.0 * (i + 1) / k  # stronger for bottom eigenvectors
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Add global exploration
            self.C += np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.05  # nearly reset evolution path
            self.stagnation_counter = 0

        # Standard covariance update with diversity-sensitive scaling
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Adaptive learning rates based on diversity
        diversity_factor = np.clip(1.0 / (diversity + 1e-4), 0.1, 10.0)
        ccov_scale = np.clip(0.5 + 0.5 * diversity_factor, 0.3, 3.0)
        cc_scale = np.clip(0.5 + 0.5 * diversity_factor, 0.5, 2.0)

        cc_adapt = np.clip(self.cc * cc_scale, 0.01, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        # Prevent extreme conditioning
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.maximum(np.min(eigvals_check), 1e-10)
        if cond_check > 1e7:
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