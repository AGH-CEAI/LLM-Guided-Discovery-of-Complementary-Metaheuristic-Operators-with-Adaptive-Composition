Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.237632e+01            3.840817e+01            6.349140e+00            4.761177e+01            5.593885e+00            9.705480e+01            1.355797e+01            -inf                    7.963484e+01            5.718331e+00            7.391785e+01            
1      4.477634e-01            7.089631e+01            5.325645e+00            1.009361e+02            5.155145e+00            1.937029e+02            7.579225e+00            -inf                    1.507177e+02            1.384336e+01            1.396705e+02            
2      1.334752e+02            4.170799e+02            5.429154e+00            4.406180e+02            7.110640e+01            8.357195e+02            1.618381e+02            -inf                    6.815735e+02            1.918093e+01            6.178407e+02            
3      3.320172e+00            1.784917e+02            2.088624e+01            2.196940e+02            2.733353e+01            3.518871e+02            3.903971e+01            -inf                    2.998940e+02            3.724215e+01            2.709880e+02            
4      2.472958e+00            4.544784e+00            1.472686e+00            4.900685e+00            2.151145e+00            5.882228e+00            2.927181e+00            -inf                    5.657795e+00            1.914946e+00            5.452307e+00            
5      6.346784e+01            3.921937e+08            4.874213e+04            3.456579e+07            6.598907e+04            6.288887e+09            1.125984e+06            -inf                    2.927180e+09            3.299176e+05            2.098641e+09            
6      9.003834e+02            2.162674e+03            8.860051e+02            5.059308e+03            9.765720e+02            1.107013e+04            1.456177e+03            -inf                    6.817081e+03            7.398393e+02            7.559198e+03            
7      7.717793e+00            2.169858e+02            2.458221e+01            2.572457e+02            1.995347e+01            4.729733e+02            2.888585e+01            -inf                    3.962665e+02            3.319447e+01            3.579274e+02            
8      6.225298e+00            3.762095e+01            5.142242e+00            3.983123e+01            8.353622e+00            5.679243e+01            1.334071e+01            -inf                    5.066555e+01            7.177583e+00            4.799927e+01            
9      1.291122e+02            2.151441e+02            8.345866e+01            2.975355e+02            1.581059e+02            4.291149e+02            1.615674e+02            -inf                    3.381503e+02            7.395860e+01            2.820288e+02            
10     6.475976e+01            2.813906e+02            7.052328e+01            1.680038e+02            2.133154e+01            5.236847e+02            6.650453e+01            -inf                    4.212685e+02            1.592059e+01            2.715077e+02            
11     2.822261e+02            5.840122e+02            4.951196e+02            4.312234e+02            1.973556e+02            1.857235e+03            8.614814e+01            -inf                    1.721643e+03            3.217417e+02            1.182935e+03            
12     6.787851e+01            1.130312e+02            1.885254e+02            1.207028e+02            3.199278e+01            3.294293e+02            5.915090e+01            -inf                    2.733314e+02            4.342706e+01            2.146255e+02            
13     2.559391e+01            3.866584e+01            2.931712e+01            3.930432e+01            2.397749e+01            5.948366e+01            1.663522e+01            -inf                    5.904496e+01            2.991482e+01            4.486270e+01            
14     1.114015e+01            1.313877e+01            8.155101e+00            1.175400e+01            8.994276e+00            1.756091e+01            1.240947e+01            -inf                    2.085049e+01            7.212517e+00            1.768615e+01            
15     3.621330e+00            3.596624e+00            4.117945e+00            3.541928e+00            3.182933e+00            4.469738e+00            3.315983e+00            -inf                    4.230619e+00            3.656681e+00            4.110100e+00            
16     5.034389e+03            9.389805e+03            3.369277e+03            3.136172e+03            1.928435e+03            3.118164e+04            5.148707e+03            -inf                    2.223696e+04            4.398077e+03            1.480318e+04            
17     1.856005e+05            8.704218e+04            2.764220e+05            8.141052e+04            4.567762e+04            3.046030e+05            5.509861e+04            -inf                    3.779535e+05            2.643240e+05            1.742224e+05            
18     4.842410e+01            6.737104e+01            9.372420e+01            6.350741e+01            4.621246e+01            9.033009e+01            3.913050e+01            -inf                    9.999115e+01            5.272514e+01            7.834093e+01            
19     2.154754e+02            2.478216e+02            1.612632e+02            2.355309e+02            1.688979e+02            4.751978e+02            2.341306e+02            -inf                    5.393420e+02            1.536943e+02            3.986708e+02            
20     3.389117e+01            4.428890e+01            5.259571e+01            4.501736e+01            2.362516e+01            7.224760e+01            3.496134e+01            -inf                    6.556767e+01            2.722969e+01            5.601237e+01            
21     5.733614e+00            5.786986e+00            5.975167e+00            5.587649e+00            5.267047e+00            5.933771e+00            5.686132e+00            -inf                    6.076485e+00            5.858899e+00            5.814837e+00            
22     1.540552e+01            1.793037e+01            1.879876e+01            1.702604e+01            1.382281e+01            2.246838e+01            1.427621e+01            -inf                    2.268021e+01            1.447493e+01            1.924800e+01            
23     6.113997e+01            1.081962e+02            5.819484e+01            1.050127e+02            8.253619e+01            1.388372e+02            8.542341e+01            -inf                    1.375568e+02            4.579004e+01            1.187770e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_04_catD_idea_0.py  (error=5.593885e+00)
Task  1: original.py  (error=4.477634e-01)
Task  2: variant_02_catB_idea_0.py  (error=5.429154e+00)
Task  3: original.py  (error=3.320172e+00)
Task  4: variant_02_catB_idea_0.py  (error=1.472686e+00)
Task  5: original.py  (error=6.346784e+01)
Task  6: variant_09_catA_idea_0.py  (error=7.398393e+02)
Task  7: original.py  (error=7.717793e+00)
Task  8: variant_02_catB_idea_0.py  (error=5.142242e+00)
Task  9: variant_09_catA_idea_0.py  (error=7.395860e+01)
Task 10: variant_09_catA_idea_0.py  (error=1.592059e+01)
Task 11: variant_06_catF_idea_0.py  (error=8.614814e+01)
Task 12: variant_04_catD_idea_0.py  (error=3.199278e+01)
Task 13: variant_06_catF_idea_0.py  (error=1.663522e+01)
Task 14: variant_09_catA_idea_0.py  (error=7.212517e+00)
Task 15: variant_04_catD_idea_0.py  (error=3.182933e+00)
Task 16: variant_04_catD_idea_0.py  (error=1.928435e+03)
Task 17: variant_04_catD_idea_0.py  (error=4.567762e+04)
Task 18: variant_06_catF_idea_0.py  (error=3.913050e+01)
Task 19: variant_09_catA_idea_0.py  (error=1.536943e+02)
Task 20: variant_04_catD_idea_0.py  (error=2.362516e+01)
Task 21: variant_04_catD_idea_0.py  (error=5.267047e+00)
Task 22: variant_04_catD_idea_0.py  (error=1.382281e+01)
Task 23: variant_09_catA_idea_0.py  (error=4.579004e+01)

WIN COUNTS:
  variant_04_catD_idea_0.py: 8 wins
  variant_09_catA_idea_0.py: 6 wins
  original.py: 4 wins
  variant_02_catB_idea_0.py: 3 wins
  variant_06_catF_idea_0.py: 3 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_04_catD_idea_0.py       5.5939e+00       variant_09_catA_idea_0.py        5.7183e+00         1.0×    6.9×
     1   original.py                     4.4776e-01       variant_04_catD_idea_0.py        5.1551e+00        11.5×  158.3×
     2   variant_02_catB_idea_0.py       5.4292e+00       variant_09_catA_idea_0.py        1.9181e+01         3.5×   76.8×
     3   original.py                     3.3202e+00       variant_02_catB_idea_0.py        2.0886e+01         6.3×   53.8×
     4   variant_02_catB_idea_0.py       1.4727e+00       variant_09_catA_idea_0.py        1.9149e+00         1.3×    3.1×
     5   original.py                     6.3468e+01       variant_02_catB_idea_0.py        4.8742e+04       768.0×  544619×
     6   variant_09_catA_idea_0.py       7.3984e+02       variant_02_catB_idea_0.py        8.8601e+02         1.2×    2.9×
     7   original.py                     7.7178e+00       variant_04_catD_idea_0.py        1.9953e+01         2.6×   28.1×
     8   variant_02_catB_idea_0.py       5.1422e+00       original.py                      6.2253e+00         1.2×    7.3×
     9   variant_09_catA_idea_0.py       7.3959e+01       variant_02_catB_idea_0.py        8.3459e+01         1.1×    2.9×
    10   variant_09_catA_idea_0.py       1.5921e+01       variant_04_catD_idea_0.py        2.1332e+01         1.3×   10.6×
    11   variant_06_catF_idea_0.py       8.6148e+01       variant_04_catD_idea_0.py        1.9736e+02         2.3×    5.7×
    12   variant_04_catD_idea_0.py       3.1993e+01       variant_09_catA_idea_0.py        4.3427e+01         1.4×    3.8×
    13   variant_06_catF_idea_0.py       1.6635e+01       variant_04_catD_idea_0.py        2.3977e+01         1.4×    2.3×
    14   variant_09_catA_idea_0.py       7.2125e+00       variant_02_catB_idea_0.py        8.1551e+00         1.1×    1.7×
    15   variant_04_catD_idea_0.py       3.1829e+00       variant_06_catF_idea_0.py        3.3160e+00         1.0×    1.1×
    16   variant_04_catD_idea_0.py       1.9284e+03       variant_03_catC_idea_0.py        3.1362e+03         1.6×    2.7×
    17   variant_04_catD_idea_0.py       4.5678e+04       variant_06_catF_idea_0.py        5.5099e+04         1.2×    4.1×
    18   variant_06_catF_idea_0.py       3.9130e+01       variant_04_catD_idea_0.py        4.6212e+01         1.2×    1.7×
    19   variant_09_catA_idea_0.py       1.5369e+02       variant_02_catB_idea_0.py        1.6126e+02         1.0×    1.5×
    20   variant_04_catD_idea_0.py       2.3625e+01       variant_09_catA_idea_0.py        2.7230e+01         1.2×    1.9×
    21   variant_04_catD_idea_0.py       5.2670e+00       variant_03_catC_idea_0.py        5.5876e+00         1.1×    1.1×
    22   variant_04_catD_idea_0.py       1.3823e+01       variant_06_catF_idea_0.py        1.4276e+01         1.0×    1.3×
    23   variant_09_catA_idea_0.py       4.5790e+01       variant_02_catB_idea_0.py        5.8195e+01         1.3×    2.3×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 5 (original.py, variant_02_catB_idea_0.py, variant_04_catD_idea_0.py, variant_06_catF_idea_0.py, variant_09_catA_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 4  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 1  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — original.py (6.35e+01) vs variant_02_catB_idea_0.py (4.87e+04) = 768.0× gap to runner-up, 544618.9× gap to median competitor. A 50/50 blend with the runner-up would yield ~2.44e+04, which is ~384.5× WORSE than the winner alone.

STRONGLY RECOMMENDED MECHANISM: per-task fingerprint-and-commit (PROBE-AND-COMMIT pattern below) or mechanism #4 (contextual bandit). Mechanism #2 (ensemble/blending) is DISCOURAGED: there are 4 tasks where the winner beats the runner-up by >=10× and ensemble blending will degrade those wins.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_catB_idea_0.py (3 wins) ---
```python
def _position_update_batch(self):
        """Update positions via SVD-based population whitening with anisotropic damping."""
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            # Guard against degenerate population
            singular_values = np.clip(singular_values, 1e-10, None)

            # Compute condition number and total variance
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10

            # Normalize singular values to [0, 1] for per-component scaling
            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Compute spectral modulation: directions with LARGE spread get DAMPENED,
            # directions with SMALL spread get AMPLIFIED (counteract anisotropy)
            # Using inverse-square-law for stronger effect on collapsed directions
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)

            # Normalize to prevent explosion
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            # Blend with uniform scaling based on condition number
            # High condition number -> more aggressive per-component modulation
            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            # Project velocity onto principal axes
            vel_proj = self.velocity @ Vt.T

            # Apply component-wise spectral scaling
            vel_scaled = vel_proj * per_component_scale

            # Reconstruct and add to population
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_04_catD_idea_0.py (8 wins) ---
```python
def _position_update_batch(self):
        """Update positions using fitness-landscape rank signals only (no distances/covariance)."""
        # Compute fitness rank percentiles (0=worst, 1=best)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Compute Spearman-like correlation between fitness rank and distance to global best
        if self.global_best is not None:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc_corr = 0.0
        else:
            fdc_corr = 0.0

        # Detect landscape ruggedness: low correlation = rugged = more exploration
        exploration_factor = np.clip(1.0 - fdc_corr, 0.3, 2.0)

        # Track success-history: count recent improvements per particle
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9  # Decay
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # Scale velocity by exploration factor and success history
        vel_scale = exploration_factor * (1.0 + 0.3 * success_norm)
        vel_scale = np.clip(vel_scale, 0.5, 2.0)

        # Fitness-percentile perturbation: push lower-ranked particles more
        # High percentile (close to 1) = good fitness = small perturbation
        # Low percentile (close to 0) = bad fitness = large perturbation
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        exploration_perturb = exploration_factor * fitness_perturb
        random_perturb = exploration_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        # Apply rank-based velocity modulation
        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + random_perturb

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_06_catF_idea_0.py (3 wins) ---
```python
def _position_update_batch(self):
        """Update positions with temporal-drift-modulated velocity scaling."""
        # Track temporal dynamics of population centroid
        centroid = np.mean(self.population, axis=0)

        # Initialize EMA state on first call
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)

        # Update EMA of centroid position
        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid

        # Compute current centroid velocity and update its EMA
        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

        # Detect convergence/oscillation via autocorrelation of centroid velocity history
        if hasattr(self, '_centroid_vel_history') and len(self._centroid_vel_history) >= 5:
            recent = np.array(self._centroid_vel_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            autocorr = 0.0

        if not hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)

        # Stagnation detection via EMA of global best improvement rate
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        # Compute temporal modulation factor
        convergence_signal = np.clip(autocorr, -1.0, 1.0)

        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7

        # Modulate based on convergence signal: high autocorr = converging = higher scaling
        temporal_scale = base_scale * (1.0 + 0.4 * convergence_signal)

        # Velocity magnitude factor: prevent overshooting when centroid moves fast
        vel_magnitude = np.linalg.norm(self._ema_centroid_velocity) + 1e-10
        vel_factor = np.clip(1.0 / (1.0 + vel_magnitude * 0.005), 0.5, 1.5)
        temporal_scale *= vel_factor

        # Apply spectral conditioning for anisotropic swarms
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
            temporal_scale *= spectral_factor
        except np.linalg.LinAlgError:
            pass

        # Apply temporally-modulated velocity to positions
        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_09_catA_idea_0.py (6 wins) ---
```python
def _position_update_batch(self):
        """Update positions using centroid gravity and k-NN density modulation."""
        centroid = np.mean(self.population, axis=0)

        # Compute distance to centroid for each particle
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

        # Compute k-NN local density (average distance to k nearest neighbors)
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10

        # Global spread: average pairwise distance as normalization factor
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

        # Centroid attraction strength: stronger for distant particles
        # Use inverse distance weighting, clipped to avoid explosion
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

        # Direction toward centroid
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dir

        # k-NN density modulation: repulse from dense regions
        # Particles in sparse regions get more centroid pull
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)

        # Combined geometric correction
        correction = centroid_attraction * to_centroid_dir * density_modulation

        # Base position update + geometric correction, scaled by inertia
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction

        self.population = self._clip_to_bounds(new_population)
```

────────────────────────────────────────────────────────────────────────
ADAPTATION MECHANISM MENU — pick ONE primary mechanism and motivate it.
Different mechanisms are appropriate to different operator types. A bandit
over six near-equivalent metrics is wasted; an ensemble that VOTES gives
a useful new signal even when the candidates are similar.

  1) DISCRETE ARM SELECTION (multi-armed bandit family)
     - Thompson Sampling, UCB1, EXP3, ε-greedy
     - Best when: operators are mutually exclusive, easy to credit-assign,
       reward is on a stable scale.
     - REQUIRED if you use this: rank-normalize rewards within a sliding
       window so magnitudes don't dominate. NEVER hard-code multipliers
       like *1e5 or *1e10 — those are scale-dependent magic constants.

  2) ENSEMBLE / VOTING / BLENDING
     - Run all candidates every generation and combine their outputs:
       weighted average, median, soft-vote, rank-aggregation.
     - Weights can be learned online (e.g. exponentially weighted majority).
     - Best when: candidates are correlated but each captures a different
       failure mode. Robust on its own — does not need credit assignment.

  3) CONDITIONAL / RULE-BASED DISPATCH
     - Hand-coded rules that pick the operator from a feature of the
       current population state: stagnation count, condition number of C,
       fitness-rank entropy, generation index, restart count, etc.
     - Best when: the failure modes are diagnosable. Fast convergence
       because there is no exploration cost.

  4) CONTEXTUAL BANDIT / FEATURE-CONDITIONED SELECTION
     - Like (1) but the arm-distribution depends on a feature vector
       describing the current state of the search (population shape,
       improvement rate, diversity, generation, ...).
     - Best when: different operators are best in different regimes and
       you can measure the regime cheaply.

  5) HYPER-HEURISTIC / SCHEDULE LEARNING
     - Maintain a *sequence* of operators rather than a single per-step
       choice (e.g. operator A for k_1 generations, then B for k_2).
     - Schedule itself is adapted (e.g. via genetic encoding of the
       sequence, or adapting k_i with success-history).
     - Best when: operators benefit from being applied in bursts.

  6) SELF-ADAPTIVE PARAMETERS (continuous, not discrete)
     - Instead of picking among discrete operators, evolve continuous
       parameters that morph the operator (e.g. CR, F, σ, mutation
       radius). Parameters can be encoded per-individual and inherited.
     - Best when: the operator family is a continuous family.

  7) CO-EVOLUTION
     - Run TWO populations: one of solutions, one of operators / operator
       configurations. Operators evolve based on their measured fitness.
     - Best when: the operator design space is large and you want
       genuine novelty.

  8) RESTART-AWARE / EPOCHED ADAPTATION
     - Different operators on different restart epochs. Memory of which
       operators worked in past epochs informs the current epoch's
       distribution.
     - Best when: the algorithm restarts often (multimodal landscapes).

  9) META-CONTROLLED ADAPTATION
     - A small online controller (e.g. a learned linear policy or a
       hand-tuned PID-like loop) maps measured signals to operator
       weights. The controller's gains are themselves adapted slowly.
     - Best when: the dynamics are reasonably smooth.

────────────────────────────────────────────────────────────────────────
PER-TASK FINGERPRINTING / PROBE-AND-COMMIT (a specialisation of mechanism #4)

When the per-task winner table shows that DIFFERENT TASKS have DIFFERENT
winners with LARGE GAPS (>=10×), no single operator wins everywhere and
ensemble blending mathematically cannot preserve the per-task advantages
(a 50/50 blend of 1 and 100 is 50.5, not 1). The right pattern is:

  PHASE A (PROBE, ~3-8% of total budget):
    - Run a SHORT exploratory burst with each candidate operator in turn,
      OR with a deterministic round-robin across operators each generation,
      using a small sub-population.
    - From the probe burst, extract a CHEAP FINGERPRINT of the landscape:
        * convergence rate of the best-so-far (slope of log-best vs gen)
        * fitness rank-correlation across consecutive generations (proxy
          for landscape ruggedness / smoothness)
        * effective dimensionality from the early covariance (number of
          eigenvalues that explain >95% of variance)
        * separability proxy: ratio of axis-aligned vs diagonal step
          improvement counts
        * basin-of-attraction estimate: variance of best-so-far across
          parallel mini-restarts
        * stagnation index: fraction of probe gens with no improvement

  PHASE B (COMMIT, ~92-97% of budget):
    - Compute a fingerprint VECTOR (concatenate the cheap signals above).
    - For each candidate operator, compute its EMPIRICAL win rate on the
      probe burst (rank-based, not raw fitness).
    - COMMIT to the operator that won the probe burst, and run it for the
      remaining budget. Optionally keep a small probability ε of revisiting
      a runner-up if the committed operator stagnates.

This pattern is the correct response when the per-task gap table is
blend-hostile. It is a CONTEXTUAL BANDIT (mechanism #4) where the context
is computed once at the start of the run rather than every generation.

IMPORTANT: the fingerprint is computed FROM THE LANDSCAPE THE OPTIMIZER IS
ON, not from `func` identity (the optimizer never knows which GNBG task it
is solving). The cheap signals listed above are observable from any
black-box `func` and require no oracle knowledge.

────────────────────────────────────────────────────────────────────────
CALIBRATION RULES (mandatory — these are the most common failure modes):

a) NEVER use scale-dependent magic constants in reward shaping.
   FORBIDDEN: `np.log1p(diversity_improvement * 1e5)`,
              `np.log1p(func_improvement * 1e10)`,
              any hard-coded multiplier > 100.
   ALLOWED:   rank-based credit (was this generation in the top tercile of
              the last K generations?), z-score normalization, relative
              improvement (delta / |incumbent|), success/failure binary.

b) If you use a sliding window of size K, declare K as a parameter and
   make K >= 3 * num_arms so each arm is sampled enough times to be
   estimable. If num_arms is large, prefer ensemble (mechanism #2) over
   discrete selection.

c) Mixed reward signals (e.g. 0.6*diversity + 0.4*fitness) MUST be on
   the same scale. Either both rank-based, both z-scored, or both
   relative-improvement. Mixing log1p(x*1e5) with log1p(y*1e10) is wrong.

d) The mechanism must be VISIBLY DIFFERENT from a plain bandit-over-
   operators if the variants are highly correlated. If you can't credit-
   assign reliably, switch mechanism — don't try to fix a broken
   reward signal with more clipping/decay.

e) Initial state matters. Don't start with `alpha = beta = 1` (uniform
   prior) if you have no reason to think the prior is uninformative —
   the original method may already be a strong choice.


ADDITIONAL CALIBRATION RULES (apply when the per-task gap table is
blend-hostile, i.e. ≥1 task with gap ≥ 10×):

f) HARD-CODED REGIME→WEIGHT TABLES ARE FORBIDDEN. Lines like
       weights = {'converging': [0.50, 0.20, 0.15, 0.15], ...}
   are a known failure mode: they cap the dominant variant at the
   chosen constant (e.g. 0.50) regardless of how much it actually wins
   by. If task X has variant_07 winning by 100×, a 0.50 weight on
   variant_07 still loses ~50× of its advantage. Either:
     - DISPATCH (mechanism #3): pick ONE variant from a rule and use
       only that one (weight 1.0). No blending.
     - LEARN WEIGHTS ONLINE: initialize weights uniformly and update
       them from rank-based fitness improvement. The hand-coded prior
       table is the bug.

g) IF YOU IMPLEMENT A REGIME DETECTOR: every regime label it can return
   MUST appear in the dispatch / weighting logic. Dead branches like
   `regime_to_variant = {'exploring': 3, ...}` where `'exploring'` is
   never returned from the detector are a silent bug — the
   credit-assignment update never reaches that arm.

h) FOR EACH ARM/VARIANT YOU INCLUDE: there must exist at least one
   reachable code path that COMMITS TO THAT ARM ALONE (weight 1.0 or
   exclusive selection). If every branch of your dispatcher mixes the
   arm with at least one other arm, the arm's per-task advantage cannot
   be recovered. Test mentally: "if variant_X is 100× better on task T,
   does my dispatcher have a state in which it picks variant_X with
   weight ≥ 0.95?" If no, redesign.
────────────────────────────────────────────────────────────────────────

DECISION GUIDE — choose your mechanism by looking at the data above:

- If the per-task gap table shows ANY task with gap >=100× and at least
  two distinct winners → MUST use PROBE-AND-COMMIT or mechanism #3
  (rule-based dispatch). Ensemble blending is mathematically excluded.
- If the per-task gap table shows >=3 tasks with gap >=10× and at least
  two distinct winners → strongly prefer PROBE-AND-COMMIT or mechanism
  #4 (contextual bandit). Ensemble blending will cap your achievable
  per-task accuracy.
- If ONE variant clearly dominates (>50% of wins AND no other variant
  has a gap >=10× win on its own tasks) → use mechanism #3 with that
  variant as default.
- If wins are spread roughly evenly across many variants AND all gaps
  are small (<5×) → mechanism #1 (bandit) or #4 (contextual bandit).
- If variants compute correlated quantities and ALL gaps are < 3× →
  mechanism #2 (ensemble/voting) is acceptable.
- If there is a clear regime split (variant X wins early, variant Y
  wins late) → mechanism #5 (schedule) or #4 (contextual).
- If the operator is a continuous family → mechanism #6 (self-adaptive
  parameters) is more powerful than picking among discrete snapshots.

REQUIREMENTS:
- Respond with the COMPLETE class code inside a single ```python``` block.
- Include `import numpy as np` at the top.
- At the START of the class docstring, write a SHORT comment naming the
  adaptation mechanism you chose (from the menu above) and 1–2 lines
  explaining why it fits the per-task winner-gap table — not just the
  win-count distribution. If the gap table is blend-hostile, the
  docstring must explicitly state this and explain how your dispatcher
  preserves the per-task winner's advantage.
- You may add new attributes in __init__ and new private helper methods.
- Do NOT change signatures of existing public methods (__call__, etc.).
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt).
- Handle ALL edge cases: no -inf, no NaN, no crashes, clip to bounds.
- If you use any of mechanisms #1, #4, #5, #8: rewards MUST be rank-based
  or z-scored within a sliding window. NO scale-dependent constants.
- If the per-task gap table contains any task with gap ≥ 10×, you MUST
  use a dispatch-style mechanism (PROBE-AND-COMMIT, #3, or #4) — NOT
  ensemble blending. Linear blending mathematically cannot preserve
  large per-task gaps.

Original algorithm:
```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE: Task 5 has the winner (variant_10)
    beating the median competitor by 13917×. A 50/50 ensemble blend of
    31.17 and 60.30 yields ~45.7, which is 1.5× WORSE than the winner
    alone. Linear blending cannot preserve per-task advantages.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): short round-robin burst with each operator,
        tracking rank-based improvement scores on the ACTUAL landscape.
      - PHASE B (commit): run ONLY the winning operator for the
        remaining budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if the committed operator stagnates, allow a
        small probability of switching to the runner-up.
    
    Win-count distribution (10/7/2/2/2/1 across 6 winners) confirms
    that different operators win on different tasks, making a single
    default operator suboptimal without probing.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # --- PROBE-AND-COMMIT STATE ---
        # Map operator ID -> (score_sum, count) for rank-based UCB scoring
        self.operator_scores = {}   # {op_id: [score1, score2, ...]}
        self.operator_total_score = {}
        self.operator_sample_count = {}
        self._committed_operator = None  # Set after probe
        
        # Operators available for selection
        self._operators = [
            'original',
            'variant_01_catA',
            'variant_03_catC',
            'variant_04_catD',
            'variant_05_catE',
            'variant_10_catB',
        ]
        for op in self._operators:
            self.operator_scores[op] = []
            self.operator_total_score[op] = 0.0
            self.operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator, min 20 total
        self._probe_gens_per_op = 3
        self._probe_total_budget = max(len(self._operators) * self._probe_gens_per_op, 20)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Track if we're in probe phase
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        # Running rank of best fitness in current probe window (for scoring)
        self._probe_fitness_history = []  # best fitness per gen in probe
        self._probe_op_history = []       # which operator per gen
        
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        range_width = self.upper_bound - self.lower_bound
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
        
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        # Handle budget exhaustion gracefully
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            # Ring neighborhood indices
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            # Include self in neighborhood comparison
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on axis-aligned geometric spread ratio."""
        # Compute axis-aligned bounding box
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10

        # Geometric mean vs arithmetic mean ratio of spreads
        # Low ratio -> anisotropic (elongated) swarm -> increase exploration (higher inertia)
        # High ratio -> isotropic (spherical) swarm -> increase exploitation (lower inertia)
        geo_mean = np.exp(np.mean(np.log(spread)))
        arith_mean = np.mean(spread)
        spread_ratio = geo_mean / (arith_mean + 1e-10)

        # Normalize ratio: for dim=10, random uniform spread_ratio ~0.9
        # Tight clustering gives spread_ratio ~0.01
        spread_ratio_normalized = np.clip(spread_ratio / 0.9, 0.0, 1.0)

        # Invert: low ratio -> high inertia (exploration)
        # Generational decay pushes toward exploitation over time
        target_inertia = (1.0 - spread_ratio_normalized) * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

        # Smooth convergence toward target
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Increase neighborhood size for more diversity
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            # Decrease neighborhood size for faster convergence
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        # Adapt cognitive coefficient based on personal best improvement
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        
        # Adapt social coefficient inversely to cognitive
        social = self.social_base * (2.0 - improvement_ratio)
        
        return cognitive, social
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE VARIANTS (from benchmark winners)
    # -------------------------------------------------------------------------
    
    def _velocity_update_original(self):
        """Original velocity update with DE/rand/1 mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_01_catA(self):
        """Update velocities using geometric layout: centroid, k-NN repulsion, spread."""
        centroid = np.mean(self.population, axis=0)
        
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10
        
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.cognitive_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        social = self.social_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        social_component = social * r2 * (self.local_best - self.population)
        
        centroid_direction = centroid - self.population
        centroid_strength = np.exp(-dist_to_centroid / 100.0)
        centroid_component = 0.3 * centroid_strength * centroid_direction
        
        k = min(5, self.np - 1)
        knn_repulsion = np.zeros((self.np, self.dim))
        for i in range(self.np):
            diffs = self.population - self.population[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nn_indices = np.argpartition(dists, k)[:k]
            nn_dists = dists[nn_indices]
            for j, nn_idx in enumerate(nn_indices):
                if nn_dists[j] > 1e-10:
                    diff = self.population[i] - self.population[nn_idx]
                    knn_repulsion[i] += diff / (nn_dists[j] ** 2 + 1e-10)
        
        knn_component = 0.1 * knn_repulsion
        
        spread_modulation = np.mean(spread) / (spread + 1e-10)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            centroid_component +
            knn_component
        )
        new_velocity = new_velocity * spread_modulation
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_03_catC(self):
        """Update velocities with entropy-modulated distribution-guided exploration."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mean_pop = np.mean(self.population, axis=0)
        centered = self.population - mean_pop
        
        cov_pop = np.cov(centered.T)
        cov_pop += np.eye(self.dim) * 1e-8
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_pop)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            
            log_det = np.sum(np.log(eigenvalues))
            entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + log_det)
            
            max_entropy = self.dim * np.log(self.upper_bound - self.lower_bound + 1e-10)
            min_entropy = self.dim * np.log(1e-6)
            entropy_normalized = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0, 1)
            
            dispersive_strength = 0.5 * (1.0 - entropy_normalized)
            
            inv_sqrt_eigen = 1.0 / np.sqrt(eigenvalues + 1e-10)
            scaled_diff = centered * inv_sqrt_eigen
            mahal_dist = np.linalg.norm(scaled_diff, axis=1, keepdims=True)
            
            max_mahal = np.max(mahal_dist) + 1e-10
            mahal_normalized = mahal_dist / max_mahal
            
            to_mean = mean_pop - self.population
            to_mean_norm = np.linalg.norm(to_mean, axis=1, keepdims=True) + 1e-10
            to_mean_dir = to_mean / to_mean_norm
            
            dispersive_component = (
                dispersive_strength *
                mahal_normalized *
                to_mean_dir *
                np.random.uniform(0, 1, (self.np, self.dim))
            )
            
            random_explore = 0.3 * (1.0 - entropy_normalized) * np.random.uniform(-1, 1, (self.np, self.dim))
            
            entropy_velocity = dispersive_component + random_explore
            
        except np.linalg.LinAlgError:
            entropy_velocity = 0.1 * np.random.uniform(-1, 1, (self.np, self.dim))
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            entropy_velocity
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_04_catD(self):
        """Update velocities with fitness-rank-adaptive components."""
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            
            if np.std(distances) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                corr = np.corrcoef(
                    np.argsort(np.argsort(self.current_fitness)),
                    np.argsort(np.argsort(distances))
                )[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0
        
        social = self.social_base * np.clip(1.0 + corr, 0.2, 1.5)
        cognitive = self.cognitive_base
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_strength = 0.3 * np.clip(1.0 - corr, 0.1, 1.0)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            mutation_strength * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_05_catE(self):
        """Update velocities using k-NN graph topology and betweenness-based social routing."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        k = min(max(3, self.neighborhood_size), self.np - 1)
        
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        adjacency = np.zeros((self.np, self.np), dtype=np.float64)
        for i in range(self.np):
            adjacency[i, knn_indices[i]] = 1.0
            adjacency[knn_indices[i], i] = 1.0
        
        degree = np.sum(adjacency, axis=1)
        degree_norm = degree / (degree.max() + 1e-10)
        
        d_sqrt_inv = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
        laplacian = np.eye(self.np) - d_sqrt_inv @ adjacency @ d_sqrt_inv
        
        np.random.seed(42)
        v = np.random.randn(self.np)
        v = v / (np.linalg.norm(v) + 1e-10)
        for _ in range(20):
            v = laplacian @ v
            v = v / (np.linalg.norm(v) + 1e-10)
        
        fiedler = np.abs(v @ laplacian @ v) / (np.dot(v, v) + 1e-10)
        connectivity_strength = np.clip(fiedler * 5.0, 0.1, 2.0)
        
        betweenness = np.zeros(self.np)
        n_samples = min(50, self.np)
        sample_nodes = np.random.choice(self.np, n_samples, replace=False)
        
        for src in sample_nodes:
            dist = np.full(self.np, np.inf)
            pred = [[] for _ in range(self.np)]
            dist[src] = 0
            queue = [src]
            
            while queue:
                curr = queue.pop(0)
                for nb in knn_indices[curr]:
                    if dist[nb] == np.inf:
                        dist[nb] = dist[curr] + 1
                        queue.append(nb)
                    if dist[nb] == dist[curr] + 1:
                        pred[nb].append(curr)
            
            sigma = np.zeros(self.np)
            sigma[src] = 1
            for d in range(int(dist.max()) + 1) if dist.max() < np.inf else []:
                for node in np.where(dist == d)[0]:
                    for p in pred[node]:
                        sigma[node] += sigma[p]
            
            for node in range(self.np):
                if node != src and dist[node] < np.inf:
                    for p in pred[node]:
                        betweenness[node] += sigma[p] / (sigma[node] + 1e-10)
        
        betweenness_norm = betweenness / (betweenness.max() + 1e-10)
        
        social_component = np.zeros((self.np, self.dim))
        
        for i in range(self.np):
            neighbor_best_idx = knn_indices[i][np.argmin(self.personal_best_fitness[knn_indices[i]])]
            neighbor_best_pos = self.personal_best[neighbor_best_idx]
            
            influence_weight = 0.5 * (1.0 + degree_norm[i]) * (1.0 + betweenness_norm[i])
            social_scaled = social * connectivity_strength
            
            social_component[i] = social_scaled * r2[i] * influence_weight * (neighbor_best_pos - self.population[i])
        
        if fiedler < 0.2:
            rescue_strength = np.clip((0.2 - fiedler) * 3.0, 0.0, 0.8)
            if self.global_best is not None:
                rescue = rescue_strength * (self.global_best - self.population)
            else:
                best_third = np.argsort(self.personal_best_fitness)[:max(1, self.np // 3)]
                centroid = np.mean(self.population[best_third], axis=0)
                rescue = rescue_strength * (centroid - self.population)
            social_component += rescue
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.05 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(mutation_active, 0.2 * (mutation_vectors - self.population), 0.0)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_10_catB(self):
        """Update velocities with spectral-condition-guided anisotropic mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        centered = self.population - np.mean(self.population, axis=0)
        
        try:
            _, singular_values, right_sv = np.linalg.svd(centered, full_matrices=False)
            
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            mutation_strength = np.clip(0.5 / (1.0 + 0.1 * np.log1p(cond)), 0.1, 0.5)
            
            total_variance = np.sum(singular_values ** 2) + 1e-10
            cumvar = np.cumsum(singular_values ** 2) / total_variance
            effective_dim = np.searchsorted(cumvar, 0.95) + 1
            
            if effective_dim < self.dim * 0.5:
                principal_axes = right_sv[:effective_dim].T
                parallel_component = self.population @ principal_axes @ principal_axes.T
                orthogonal_residual = self.population - parallel_component
                orthogonal_strength = 0.3 * (1.0 - effective_dim / self.dim)
                mutation_component = orthogonal_strength * orthogonal_residual
            else:
                mutation_component = np.zeros((self.np, self.dim))
            
            mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
            mutation_threshold = mutation_strength * (1.0 - self.generation / 5000)
            mutation_active = mutation_mask < mutation_threshold
            
            mutation_vectors = np.zeros((self.np, self.dim))
            for i in range(self.np):
                indices = np.random.choice(
                    [j for j in range(self.np) if j != i], 3, replace=False
                )
                mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                    self.population[indices[1]] - self.population[indices[2]]
                )
            
            de_component = np.where(
                mutation_active,
                mutation_strength * (mutation_vectors - self.population),
                0.0
            )
            
            total_mutation = mutation_component + de_component
            
        except np.linalg.LinAlgError:
            mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
            mutation_threshold = 0.3 * (1.0 - self.generation / 5000)
            mutation_active = mutation_mask < mutation_threshold
            
            mutation_vectors = np.zeros((self.np, self.dim))
            for i in range(self.np):
                indices = np.random.choice(
                    [j for j in range(self.np) if j != i], 3, replace=False
                )
                mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                    self.population[indices[1]] - self.population[indices[2]]
                )
            
            total_mutation = np.where(
                mutation_active,
                0.3 * (mutation_vectors - self.population),
                0.0
            )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            total_mutation
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> velocity update method
    # -------------------------------------------------------------------------
    
    def _dispatch_velocity_update(self, operator):
        """Route to the appropriate velocity update implementation."""
        if operator == 'original':
            self._velocity_update_original()
        elif operator == 'variant_01_catA':
            self._velocity_update_variant_01_catA()
        elif operator == 'variant_03_catC':
            self._velocity_update_variant_03_catC()
        elif operator == 'variant_04_catD':
            self._velocity_update_variant_04_catD()
        elif operator == 'variant_05_catE':
            self._velocity_update_variant_05_catE()
        elif operator == 'variant_10_catB':
            self._velocity_update_variant_10_catB()
        else:
            # Fallback to original
            self._velocity_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score for rank-based evaluation.
        
        Uses rank-based scoring: a generation's score is its percentile
        rank among all probe generations seen so far (lower fitness = better
        = higher score). This is scale-independent.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        # Compute rank-based score: fraction of probe gens with WORSE fitness
        if len(self._probe_fitness_history) < 2:
            score = 1.0  # First sample gets best score
        else:
            current_best = min(self._probe_fitness_history)
            current_worst = max(self._probe_fitness_history)
            if current_worst > current_best:
                # Normalize: higher score = better (lower fitness)
                normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
                score = np.clip(normalized, 0.0, 1.0)
            else:
                score = 1.0
        
        self.operator_scores[operator].append(score)
        self.operator_total_score[operator] += score
        self.operator_sample_count[operator] += 1
    
    def _compute_ucb_score(self, operator):
        """Compute UCB1-style score for operator selection."""
        n = self.operator_sample_count[operator]
        if n == 0:
            return float('inf')  # Unexplored operators get priority
        
        total_n = sum(self.operator_sample_count.values())
        avg_score = self.operator_total_score[operator] / n
        
        # UCB1 exploration bonus
        if total_n > 0:
            exploration = np.sqrt(2.0 * np.log(total_n) / n)
        else:
            exploration = float('inf')
        
        return avg_score + exploration
    
    def _select_best_operator(self):
        """Select operator with highest UCB score from probe data."""
        best_op = self._operators[0]
        best_score = -float('inf')
        
        for op in self._operators:
            ucb = self._compute_ucb_score(op)
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
    def _select_runner_up_operator(self):
        """Select the second-best operator by UCB score."""
        scores = [(op, self._compute_ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2:
            return scores[1][0]
        return scores[0][0]
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            if self.global_best is not None:
                perturbation = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + perturbation)
                self.personal_best[0] = self.population[0]
                self.personal_best_fitness[0] = np.inf
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT operator selection.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Reset probe state
        for op in self._operators:
            self.operator_scores[op] = []
            self.operator_total_score[op] = 0.0
            self.operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation phase
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Use the current probe operator
                self._dispatch_velocity_update(current_op)
                self._position_update_batch()
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < self.np:
                    self.np = actual_n
                    self.population = self.population[:actual_n]
                    self.velocity = self.velocity[:actual_n]
                    self.current_fitness = self.current_fitness[:actual_n]
                    break
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                
                # Record probe score
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_score(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= len(self._operators):
                    self._in_probe_phase = False
                    self._committed_operator = self._select_best_operator()
                    self._runner_up_operator = self._select_runner_up_operator()
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._dispatch_velocity_update(self._runner_up_operator)
                else:
                    self._dispatch_velocity_update(self._committed_operator)
                
                self._position_update_batch()
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < self.np:
                    self.np = actual_n
                    self.population = self.population[:actual_n]
                    self.velocity = self.velocity[:actual_n]
                    self.current_fitness = self.current_fitness[:actual_n]
                    break
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```