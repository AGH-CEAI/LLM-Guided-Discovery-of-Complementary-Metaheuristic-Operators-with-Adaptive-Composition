Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_temporal_drift` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.746969e-01            1.558076e-01            1.661515e-01            1.531156e-01            1.697580e-01            -inf                    1.660694e-01            -inf                    1.756874e-01            1.748903e-01            -inf                    
1      3.517951e+00            4.564696e+00            3.870592e+00            3.148386e+00            3.162463e+00            -inf                    3.357051e+00            -inf                    3.389974e+00            4.133576e+00            -inf                    
2      4.116799e-01            4.867982e-01            3.847184e-01            4.556218e-01            4.214071e-01            -inf                    4.416822e-01            -inf                    4.166384e-01            3.928957e-01            -inf                    
3      6.439517e+00            7.214199e+00            6.383018e+00            6.712256e+00            6.674890e+00            -inf                    6.680014e+00            -inf                    6.506683e+00            6.532007e+00            -inf                    
4      1.164510e+00            1.215758e+00            1.286179e+00            1.192330e+00            1.225665e+00            -inf                    1.237997e+00            -inf                    1.162384e+00            1.161161e+00            -inf                    
5      9.701416e+00            1.177836e+02            2.320836e+01            9.405765e+00            4.890018e+01            -inf                    7.000477e+01            -inf                    9.877183e+01            1.620522e+02            -inf                    
6      7.318297e+02            6.187939e+02            6.920484e+02            7.251801e+02            7.009466e+02            -inf                    6.243778e+02            -inf                    7.477845e+02            7.995832e+02            -inf                    
7      1.172967e+01            1.745032e+01            1.111533e+01            1.521417e+01            1.257293e+01            -inf                    1.354847e+01            -inf                    1.232921e+01            1.137018e+01            -inf                    
8      2.896967e+00            2.835020e+00            2.954491e+00            3.026623e+00            2.977006e+00            -inf                    3.044974e+00            -inf                    3.036355e+00            3.029141e+00            -inf                    
9      1.131047e+02            9.577211e+01            1.058123e+02            9.694174e+01            1.030941e+02            -inf                    1.042220e+02            -inf                    1.102830e+02            1.060068e+02            -inf                    
10     2.563100e+01            1.713414e+01            1.441460e+01            1.277662e+01            2.047441e+01            -inf                    1.538310e+01            -inf                    1.656555e+01            2.116687e+01            -inf                    
11     2.327755e+02            2.081846e+02            1.772015e+02            2.161945e+02            1.693468e+02            -inf                    2.017942e+02            -inf                    2.165880e+02            2.325110e+02            -inf                    
12     6.750187e+01            6.018796e+01            5.533376e+01            6.831272e+01            6.687179e+01            -inf                    5.307079e+01            -inf                    7.555342e+01            6.265958e+01            -inf                    
13     2.267144e+01            2.283927e+01            1.706566e+01            1.706948e+01            1.972677e+01            -inf                    2.141325e+01            -inf                    2.277343e+01            1.858572e+01            -inf                    
14     1.082220e+01            1.035435e+01            8.626900e+00            9.925819e+00            1.029812e+01            -inf                    1.043091e+01            -inf                    1.065929e+01            1.038090e+01            -inf                    
15     3.626828e+00            3.682288e+00            3.651577e+00            3.757009e+00            3.525660e+00            -inf                    3.606058e+00            -inf                    3.560200e+00            3.707731e+00            -inf                    
16     2.536155e+03            2.030115e+03            1.615383e+03            2.154949e+03            2.396293e+03            -inf                    2.460200e+03            -inf                    2.754414e+03            2.532544e+03            -inf                    
17     4.777815e+04            5.226830e+04            4.529916e+04            5.262577e+04            6.390788e+04            -inf                    5.457380e+04            -inf                    5.345054e+04            5.833383e+04            -inf                    
18     4.876153e+01            4.917630e+01            4.851987e+01            4.720826e+01            5.233083e+01            -inf                    4.927689e+01            -inf                    4.811160e+01            4.992306e+01            -inf                    
19     1.733395e+02            1.593706e+02            1.544668e+02            1.652803e+02            1.620819e+02            -inf                    1.618500e+02            -inf                    1.695064e+02            1.677589e+02            -inf                    
20     2.731258e+01            2.765563e+01            2.546812e+01            2.566590e+01            2.640386e+01            -inf                    2.597494e+01            -inf                    2.673898e+01            2.637599e+01            -inf                    
21     5.618457e+00            5.439139e+00            5.407335e+00            5.657757e+00            5.475270e+00            -inf                    5.517170e+00            -inf                    5.597348e+00            5.606888e+00            -inf                    
22     1.356750e+01            1.329225e+01            1.369064e+01            1.385695e+01            1.404021e+01            -inf                    1.356095e+01            -inf                    1.411631e+01            1.366842e+01            -inf                    
23     7.253302e+01            6.920657e+01            7.018665e+01            6.631414e+01            7.195251e+01            -inf                    6.915479e+01            -inf                    7.107855e+01            7.176506e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_catC_idea_0.py  (error=1.531156e-01)
Task  1: variant_03_catC_idea_0.py  (error=3.148386e+00)
Task  2: variant_02_catB_idea_0.py  (error=3.847184e-01)
Task  3: variant_02_catB_idea_0.py  (error=6.383018e+00)
Task  4: variant_09_catA_idea_0.py  (error=1.161161e+00)
Task  5: variant_03_catC_idea_0.py  (error=9.405765e+00)
Task  6: variant_01_catA_idea_0.py  (error=6.187939e+02)
Task  7: variant_02_catB_idea_0.py  (error=1.111533e+01)
Task  8: variant_01_catA_idea_0.py  (error=2.835020e+00)
Task  9: variant_01_catA_idea_0.py  (error=9.577211e+01)
Task 10: variant_03_catC_idea_0.py  (error=1.277662e+01)
Task 11: variant_04_catD_idea_0.py  (error=1.693468e+02)
Task 12: variant_06_catF_idea_0.py  (error=5.307079e+01)
Task 13: variant_02_catB_idea_0.py  (error=1.706566e+01)
Task 14: variant_02_catB_idea_0.py  (error=8.626900e+00)
Task 15: variant_04_catD_idea_0.py  (error=3.525660e+00)
Task 16: variant_02_catB_idea_0.py  (error=1.615383e+03)
Task 17: variant_02_catB_idea_0.py  (error=4.529916e+04)
Task 18: variant_03_catC_idea_0.py  (error=4.720826e+01)
Task 19: variant_02_catB_idea_0.py  (error=1.544668e+02)
Task 20: variant_02_catB_idea_0.py  (error=2.546812e+01)
Task 21: variant_02_catB_idea_0.py  (error=5.407335e+00)
Task 22: variant_01_catA_idea_0.py  (error=1.329225e+01)
Task 23: variant_03_catC_idea_0.py  (error=6.631414e+01)

WIN COUNTS:
  variant_02_catB_idea_0.py: 10 wins
  variant_03_catC_idea_0.py: 6 wins
  variant_01_catA_idea_0.py: 4 wins
  variant_04_catD_idea_0.py: 2 wins
  variant_09_catA_idea_0.py: 1 wins
  variant_06_catF_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_03_catC_idea_0.py       1.5312e-01       variant_01_catA_idea_0.py        1.5581e-01         1.0×    1.1×
     1   variant_03_catC_idea_0.py       3.1484e+00       variant_04_catD_idea_0.py        3.1625e+00         1.0×    1.1×
     2   variant_02_catB_idea_0.py       3.8472e-01       variant_09_catA_idea_0.py        3.9290e-01         1.0×    1.1×
     3   variant_02_catB_idea_0.py       6.3830e+00       original.py                      6.4395e+00         1.0×    1.0×
     4   variant_09_catA_idea_0.py       1.1612e+00       variant_08_catH_idea_0.py        1.1624e+00         1.0×    1.0×
     5   variant_03_catC_idea_0.py       9.4058e+00       original.py                      9.7014e+00         1.0×    7.4×
     6   variant_01_catA_idea_0.py       6.1879e+02       variant_06_catF_idea_0.py        6.2438e+02         1.0×    1.2×
     7   variant_02_catB_idea_0.py       1.1115e+01       variant_09_catA_idea_0.py        1.1370e+01         1.0×    1.1×
     8   variant_01_catA_idea_0.py       2.8350e+00       original.py                      2.8970e+00         1.0×    1.1×
     9   variant_01_catA_idea_0.py       9.5772e+01       variant_03_catC_idea_0.py        9.6942e+01         1.0×    1.1×
    10   variant_03_catC_idea_0.py       1.2777e+01       variant_02_catB_idea_0.py        1.4415e+01         1.1×    1.3×
    11   variant_04_catD_idea_0.py       1.6935e+02       variant_02_catB_idea_0.py        1.7720e+02         1.0×    1.3×
    12   variant_06_catF_idea_0.py       5.3071e+01       variant_02_catB_idea_0.py        5.5334e+01         1.0×    1.3×
    13   variant_02_catB_idea_0.py       1.7066e+01       variant_03_catC_idea_0.py        1.7069e+01         1.0×    1.3×
    14   variant_02_catB_idea_0.py       8.6269e+00       variant_03_catC_idea_0.py        9.9258e+00         1.2×    1.2×
    15   variant_04_catD_idea_0.py       3.5257e+00       variant_08_catH_idea_0.py        3.5602e+00         1.0×    1.0×
    16   variant_02_catB_idea_0.py       1.6154e+03       variant_01_catA_idea_0.py        2.0301e+03         1.3×    1.5×
    17   variant_02_catB_idea_0.py       4.5299e+04       original.py                      4.7778e+04         1.1×    1.2×
    18   variant_03_catC_idea_0.py       4.7208e+01       variant_08_catH_idea_0.py        4.8112e+01         1.0×    1.0×
    19   variant_02_catB_idea_0.py       1.5447e+02       variant_01_catA_idea_0.py        1.5937e+02         1.0×    1.1×
    20   variant_02_catB_idea_0.py       2.5468e+01       variant_03_catC_idea_0.py        2.5666e+01         1.0×    1.0×
    21   variant_02_catB_idea_0.py       5.4073e+00       variant_01_catA_idea_0.py        5.4391e+00         1.0×    1.0×
    22   variant_01_catA_idea_0.py       1.3292e+01       variant_06_catF_idea_0.py        1.3561e+01         1.0×    1.0×
    23   variant_03_catC_idea_0.py       6.6314e+01       variant_06_catF_idea_0.py        6.9155e+01         1.0×    1.1×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 6 (variant_01_catA_idea_0.py, variant_02_catB_idea_0.py, variant_03_catC_idea_0.py, variant_04_catD_idea_0.py, variant_06_catF_idea_0.py, variant_09_catA_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 0  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (4 wins) ---
```python
def _position_update_temporal_drift(self):
        """Geometry-modulated velocity scaling (Category A: Geometry / spatial).

        Uses literal geometric properties of the swarm layout:
        - Convex hull volume (detects convergence/divergence)
        - Centroid-relative distance distribution (spread signal)
        - k-NN radial density (local clustering)
        - Axis-aligned bounding box elongation (shape anisotropy)
        - Temporal tracking of centroid drift (oscillation detection)
        """
        centroid = np.mean(self.population, axis=0)

        # --- GEOMETRIC PROPERTY 1: Convex Hull ---
        # Use hull volume as convergence signal (shrinking hull = converging)
        try:
            from scipy.spatial import ConvexHull
            if self.dim <= 3 and self.np >= self.dim + 2:
                hull = ConvexHull(self.population)
                hull_volume = hull.volume
            else:
                # Fallback for high-dim: use centroid-radius sphere volume proxy
                dists = np.linalg.norm(self.population - centroid, axis=1)
                max_dist = np.max(dists) + 1e-10
                hull_volume = max_dist ** self.dim
        except:
            dists = np.linalg.norm(self.population - centroid, axis=1)
            max_dist = np.max(dists) + 1e-10
            hull_volume = max_dist ** self.dim

        # Initialize hull volume history for temporal tracking
        if not hasattr(self, '_hull_volume_history'):
            self._hull_volume_history = []
        self._hull_volume_history.append(hull_volume)
        if len(self._hull_volume_history) > 20:
            self._hull_volume_history.pop(0)

        # Hull shrinkage rate (negative = converging, positive = diverging)
        if len(self._hull_volume_history) >= 3:
            recent_vol = np.array(self._hull_volume_history[-3:])
            hull_shrink_rate = (recent_vol[-1] - recent_vol[0]) / (recent_vol[0] + 1e-10)
        else:
            hull_shrink_rate = 0.0

        # --- GEOMETRIC PROPERTY 2: Distance to Centroid Distribution ---
        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)

        # Interquartile range as robust spread measure
        q25, q50, q75 = np.percentile(dists_to_centroid, [25, 50, 75])
        iqr = q75 - q25 + 1e-10
        median_dist = q50 + 1e-10

        # Normalized spread: compare IQR to median distance
        spread_normalized = iqr / median_dist
        spread_signal = np.clip(spread_normalized / 10.0, 0.0, 2.0)

        # --- GEOMETRIC PROPERTY 3: k-NN Radial Density ---
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_sorted = np.sort(sq_dists, axis=1)
        knn_dists = np.sqrt(knn_sorted[:, :k])
        knn_mean_dist = np.mean(knn_dists, axis=1) + 1e-10  # Local density proxy

        # Density ratio: high = sparse region, low = dense region
        global_knn_mean = np.mean(knn_mean_dist) + 1e-10
        density_ratio = knn_mean_dist / global_knn_mean

        # --- GEOMETRIC PROPERTY 4: Axis-Aligned Bounding Box Elongation ---
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        spans = pop_max - pop_min + 1e-10

        max_span = np.max(spans)
        min_span = np.min(spans)
        elongation = max_span / min_span
        elongation = np.clip(elongation / 10.0, 0.5, 2.0)

        # --- COMBINE GEOMETRIC SIGNALS INTO VELOCITY SCALE ---
        # Base scale from spread: low spread (converged) → reduce velocity
        if median_dist < 5.0:
            base_scale = 0.7
        elif median_dist < 20.0:
            base_scale = 1.0
        else:
            base_scale = 1.3

        # Modulate by hull shrinkage: converging → reduce further
        hull_factor = 1.0 - np.clip(hull_shrink_rate * 2.0, -0.5, 0.5)

        # Modulate by elongation: elongated → anisotropic-like correction
        elongation_factor = 2.0 - elongation  # High elongation → lower factor
        elongation_factor = np.clip(elongation_factor, 0.5, 1.5)

        # Combine: geometric_scale = base * hull_factor * elongation_factor
        geometric_scale = base_scale * hull_factor * elongation_factor

        # Per-particle adjustment from k-NN density
        # Sparse regions (high density_ratio) → more velocity, dense → less
        density_modulation = np.clip(density_ratio, 0.5, 2.0)

        # --- TEMPORAL TRACKING: Centroid Drift for Oscillation Detection ---
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)
            self._centroid_vel_history = []

        # EMA of centroid position
        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid

        # Centroid velocity (rate of centroid movement)
        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

        # Track velocity history for autocorrelation
        self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)

        # Oscillation detection via autocorrelation
        if len(self._centroid_vel_history) >= 5:
            recent = np.array(self._centroid_vel_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            autocorr = 0.0

        # Oscillation factor: high positive autocorr = oscillating → reduce velocity
        oscillation_factor = 1.0 + 0.4 * np.clip(autocorr, -1.0, 1.0)

        # --- FINAL VELOCITY UPDATE ---
        # Apply geometric scale with temporal oscillation modulation
        final_scale = geometric_scale * oscillation_factor
        final_scale = np.clip(final_scale, 0.5, 2.5)

        # Per-particle velocity adjustment from density
        vel_adjustment = density_modulation[:, np.newaxis]

        new_population = self.population + final_scale * self.velocity * vel_adjustment
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_02_catB_idea_0.py (10 wins) ---
```python
def _position_update_temporal_drift(self):
        """Spectral entropy-based velocity modulation (Category B).

        Replaces temporal autocorrelation with eigenvalue distribution analysis.
        Spectral entropy detects convergence from covariance shape; condition
        number drives anisotropic scaling. Targets worst tasks with large
        condition numbers (17, 16, 6, 11) where population collapses to subspaces.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_spectral_ema_improvement'):
                self._spectral_ema_improvement = 0.0
            self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement

            if self._spectral_ema_improvement > 1e-6:
                base_scale = 1.2
            elif self._spectral_ema_improvement > 1e-10:
                base_scale = 1.0
            else:
                base_scale = 0.7

            temporal_scale = base_scale * (1.0 + 0.4 * spectral_signal)

            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                temporal_scale *= (1.0 + 0.3 * log_damp)

            U = np.linalg.eigvalsh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = np.linalg.eigh(cov)[1][:, idx]

            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            per_comp_scale = np.clip(sv_norm, 0.1, 2.0)
            vel_scaled = vel_proj * per_comp_scale
            new_population = self.population + temporal_scale * (vel_scaled @ V.T)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_03_catC_idea_0.py (6 wins) ---
```python
def _position_update_temporal_drift(self):
        """Gaussian entropy / KL-divergence velocity modulation (Category C).

        Key insight: Treat population as probability distribution. Use Gaussian
        entropy and KL divergence between consecutive generations to detect
        convergence (low entropy, small KL) and modulate velocity accordingly.
        This is orthogonal to centroid-dynamics (F) and spectral (B) approaches.
        """
        # --- Fit Gaussian to population and compute entropy ---
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)

            # Gaussian entropy: H = 0.5 * log(det(2*pi*e*Sigma))
            # Equivalent to: H = 0.5 * sum(log(2*pi*e * lambda_i))
            log_dets = np.log(2 * np.pi * np.e * eigenvalues)
            current_entropy = 0.5 * np.sum(log_dets)
            current_entropy = max(current_entropy, 1e-10)
        except np.linalg.LinAlgError:
            current_entropy = 1.0

        # --- Track entropy history ---
        if not hasattr(self, '_entropy_history'):
            self._entropy_history = []
        self._entropy_history.append(current_entropy)
        if len(self._entropy_history) > 20:
            self._entropy_history.pop(0)

        # --- Entropy-based convergence signal ---
        if len(self._entropy_history) >= 5:
            recent_entropies = np.array(self._entropy_history[-5:])
            entropy_trend = (recent_entropies[-1] - recent_entropies[0]) / (len(recent_entropies) + 1e-10)
            entropy_std = np.std(recent_entropies) + 1e-10
            entropy_signal = entropy_trend / entropy_std
        else:
            entropy_signal = 0.0

        # --- KL divergence between consecutive distributions ---
        if not hasattr(self, '_prev_mean') or not hasattr(self, '_prev_cov_diag'):
            kl_div = 0.5
        else:
            current_mean = np.mean(self.population, axis=0)
            try:
                cov_diag = np.var(self.population, axis=0) + 1e-10
                prev_var = self._prev_cov_diag + 1e-10
                current_var = cov_diag + 1e-10

                # Simplified KL(N1 || N2) for diagonal Gaussians
                mean_diff_sq = np.sum((current_mean - self._prev_mean) ** 2)
                var_ratio = np.sum(prev_var / current_var)
                trace_term = np.sum(current_var / prev_var)
                det_ratio = np.sum(np.log(current_var / prev_var))

                kl_div = 0.5 * (mean_diff_sq / (np.sum(prev_var) + 1e-10) + 
                               trace_term - self.dim + det_ratio)
                kl_div = abs(kl_div)
            except:
                kl_div = 0.5

        self._prev_mean = np.mean(self.population, axis=0)
        self._prev_cov_diag = np.var(self.population, axis=0) + 1e-10

        if not hasattr(self, '_kl_history'):
            self._kl_history = []
        self._kl_history.append(kl_div)
        if len(self._kl_history) > 20:
            self._kl_history.pop(0)

        # --- Fitness entropy (additional signal) ---
        try:
            fitness_normalized = (self.current_fitness - np.min(self.current_fitness) + 1e-10)
            fitness_normalized = fitness_normalized / (np.max(fitness_normalized) + 1e-10)
            fitness_normalized = np.clip(fitness_normalized, 1e-10, 1.0)
            fitness_entropy = -np.sum(fitness_normalized * np.log(fitness_normalized + 1e-10))
            fitness_entropy = fitness_entropy / (np.log(self.np) + 1e-10)
        except:
            fitness_entropy = 0.5

        # --- Improvement rate ---
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        # --- Compute velocity scale from information-theoretic signals ---
        # Low entropy trend = converging = need more exploration (higher scale)
        # High KL divergence = active exploration = can be more aggressive
        # Low fitness entropy = fitness clustered = harder landscape

        exploration_factor = 1.0 - np.clip(entropy_signal, -2.0, 2.0)
        exploration_factor = np.clip(exploration_factor, 0.3, 2.0)

        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7

        # KL-driven boost: high KL = more change = scale up exploration
        kl_boost = 1.0 + 0.3 * np.clip(kl_div, 0.0, 2.0)

        # Fitness entropy: low entropy (clustered fitness) = harder = more exploration
        fitness_explore = 1.0 + 0.4 * (1.0 - fitness_entropy)

        temporal_scale = base_scale * exploration_factor * kl_boost * fitness_explore
        temporal_scale = np.clip(temporal_scale, 0.3, 2.5)

        # --- Spectral conditioning for numerical robustness ---
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
            temporal_scale *= spectral_factor
        except np.linalg.LinAlgError:
            pass

        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_04_catD_idea_0.py (2 wins) ---
```python
def _position_update_temporal_drift(self):
        """Fitness-rank-modulated velocity scaling (Category D).

        Key insight: Use ONLY fitness signals (ranks, success history) to modulate
        velocity. Per-particle success tracking identifies exploit/explore regimes.
        Fitness-distance correlation detects funnel vs deceptive landscapes.
        """
        # Per-particle success history (fitness signal, not distance)
        if not hasattr(self, '_particle_success_count'):
            self._particle_success_count = np.zeros(self.np)

        improved = self.personal_best_fitness >= self.current_fitness
        self._particle_success_count[improved] += 1
        self._particle_success_count[~improved] *= 0.9

        # Per-particle success rate (0 to 1)
        max_success = np.max(self._particle_success_count) + 1e-10
        success_rate = self._particle_success_count / max_success

        # Fitness percentile rank (Category D: purely fitness-based)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Fitness-distance correlation: Spearman-like (Category D)
        if self.global_best is not None and self.np > 2:
            # Use squared distance rank (proxy, but computed from positions)
            # Then correlate with fitness rank
            dists_sq = np.sum((self.population - self.global_best) ** 2, axis=1)
            dist_ranks = np.argsort(np.argsort(dists_sq)) / max(1, self.np - 1)

            # Spearman correlation between fitness rank and distance rank
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc_corr = 0.0
        else:
            fdc_corr = 0.0

        # Track global improvement for regime detection
        if self.global_best is not None and hasattr(self, '_prev_global_fitness'):
            improvement = max(0.0, self._prev_global_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        # Regime detection from fitness signals only (Category D)
        if self._ema_improvement > 1e-6:
            regime = 'improving'
        elif self._ema_improvement > 1e-10:
            regime = 'slow'
        else:
            regime = 'stagnant'

        # Fitness-distance correlation informs exploration need
        # fdc_corr > 0: funnel landscape (good), fdc_corr < 0: deceptive
        if fdc_corr > 0.3:
            landscape_quality = 1.2  # funnel - exploit more
        elif fdc_corr < -0.1:
            landscape_quality = 0.7  # deceptive - explore more
        else:
            landscape_quality = 1.0

        # Per-particle velocity scaling (Category D: fitness-based only)
        # High success + high fitness rank = exploit (lower scale)
        # Low success + low fitness rank = explore (higher scale)
        fitness_exploit = 1.0 - 0.5 * fitness_ranks  # 0.5 to 1.0
        success_explore = 0.8 + 0.4 * (1.0 - success_rate)  # 0.8 to 1.2

        vel_scale = fitness_exploit * success_explore

        # Regime-dependent global modifier
        if regime == 'improving':
            regime_mod = 1.1 * landscape_quality
        elif regime == 'slow':
            regime_mod = 1.0
        else:  # stagnant
            regime_mod = 0.8 * landscape_quality

        vel_scale = vel_scale * regime_mod
        vel_scale = np.clip(vel_scale, 0.3, 2.0)

        # Apply velocity with per-particle scaling
        new_population = self.population + self.velocity * vel_scale[:, np.newaxis]
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_06_catF_idea_0.py (1 wins) ---
```python
def _position_update_temporal_drift(self):
        """Fitness-variance temporal tracking (variant_06_catF).

        Category F: Track fitness variance ACROSS GENERATIONS via EMA of
        variance, its rate-of-change, and stagnation windows to detect
        premature convergence. Different from centroid-drift variant_06
        which tracked geometric state; this tracks distributional dynamics.
        """
        # Compute population fitness variance (distributional, not geometric)
        current_variance = np.var(self.current_fitness)

        # Initialize EMA tracking across generations
        if not hasattr(self, '_ema_fitness_variance'):
            self._ema_fitness_variance = current_variance if np.isfinite(current_variance) else 1.0
        if not hasattr(self, '_ema_variance_velocity'):
            self._ema_variance_velocity = 0.0
        if not hasattr(self, '_fitness_variance_history'):
            self._fitness_variance_history = []

        # Update EMA of fitness variance
        alpha_var = 0.15
        self._ema_fitness_variance = alpha_var * current_variance + (1 - alpha_var) * self._ema_fitness_variance
        self._ema_fitness_variance = max(self._ema_fitness_variance, 1e-15)

        # Track rate-of-change of variance (convergence signal)
        if len(self._fitness_variance_history) > 0:
            prev_var = self._fitness_variance_history[-1]
            variance_delta = current_variance - prev_var
            alpha_dv = 0.2
            self._ema_variance_velocity = alpha_dv * variance_delta + (1 - alpha_dv) * self._ema_variance_velocity
        else:
            self._ema_variance_velocity = 0.0

        # Record history for autocorrelation
        self._fitness_variance_history.append(current_variance)
        if len(self._fitness_variance_history) > 25:
            self._fitness_variance_history.pop(0)

        # Autocorrelation of variance (temporal pattern detection)
        if len(self._fitness_variance_history) >= 5:
            recent = np.array(self._fitness_variance_history[-5:])
            norm_a = np.linalg.norm(recent[:-1]) + 1e-10
            norm_b = np.linalg.norm(recent[1:]) + 1e-10
            var_autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
        else:
            var_autocorr = 0.0

        # Stagnation detection: variance not changing over time window
        if len(self._fitness_variance_history) >= 10:
            recent_window = np.array(self._fitness_variance_history[-10:])
            var_relative_change = np.std(recent_window) / (np.mean(recent_window) + 1e-10)
        else:
            var_relative_change = 1.0

        # Global best improvement EMA (temporal fitness signal)
        if self.global_best is not None and hasattr(self, '_prev_best_fitness'):
            improvement = max(0.0, self._prev_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.25 * improvement + 0.75 * self._ema_improvement

        # Compute temporal convergence signal from variance dynamics
        # Negative EMA_variance_velocity = variance shrinking = convergence
        # var_autocorr > 0 = stable pattern = stagnation risk
        convergence_pressure = -np.sign(self._ema_variance_velocity) * np.abs(self._ema_variance_velocity)
        convergence_pressure = np.clip(convergence_pressure, -1.0, 1.0)
        pattern_signal = np.clip(var_autocorr, -1.0, 1.0)

        # Stagnation flag
        stagnation_detected = (var_relative_change < 0.05) and (self._ema_improvement < 1e-8)

        # Base velocity scale from improvement regime
        if stagnation_detected:
            base_scale = 1.6  # Boost exploration when stagnant
        elif self._ema_improvement > 1e-6:
            base_scale = 0.9  # Fine exploitation when improving fast
        elif self._ema_improvement > 1e-12:
            base_scale = 1.1  # Moderate when improving slowly
        else:
            base_scale = 1.3  # Increase exploration when plateaued

        # Modulate by variance dynamics
        temporal_scale = base_scale * (1.0 + 0.3 * convergence_pressure + 0.2 * pattern_signal)

        # Variance magnitude factor: low variance = clustered = needs more velocity
        var_magnitude = np.log1p(self._ema_fitness_variance)
        var_factor = np.clip(1.5 / (1.0 + var_magnitude * 0.1), 0.6, 1.8)
        temporal_scale *= var_factor

        # Apply temporal scaling to velocity
        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_09_catA_idea_0.py (1 wins) ---
```python
def _position_update_temporal_drift(self):
        """Convex-hull volume + principal-axis expansion (Category A: Geometry).

        Key insight: Detect population collapse via convex hull volume in PC-space;
        use principal axis projections to direct expansion away from centroid.
        Targets tasks where swarm is geometrically trapped (e.g., Task 17: 4.5e+04 error).
        """
        # --- Geometric Signal 1: Convex Hull Volume ---
        try:
            from scipy.spatial import ConvexHull
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            pc_population = self.population @ eigenvectors[:, :min(3, self.dim)]
            hull = ConvexHull(pc_population)
            hull_volume = hull.volume if hasattr(hull, 'volume') else hull.area
            hull_volume = max(hull_volume, 1e-30)
        except:
            hull_volume = 1.0

        search_space_vol = (self.upper_bound - self.lower_bound) ** min(3, self.dim)
        volume_ratio = hull_volume / (search_space_vol + 1e-30)
        volume_ratio = np.clip(volume_ratio, 1e-10, 1.0)

        # Small volume_ratio = collapsed population → strong expansion
        if volume_ratio < 1e-6:
            expansion_strength = 2.5
        elif volume_ratio < 1e-3:
            expansion_strength = 1.8
        elif volume_ratio < 0.01:
            expansion_strength = 1.3
        else:
            expansion_strength = 1.0

        # --- Geometric Signal 2: Principal Axis Projection Analysis ---
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            total_var = np.sum(eigenvalues)
            var_ratios = eigenvalues / (total_var + 1e-30)
            effective_dims = np.sum(var_ratios > 0.01)
            elongation_factor = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        except:
            effective_dims = self.dim
            elongation_factor = 1.0

        elongation_modulation = np.clip(elongation_factor / 50.0, 0.5, 2.0)

        # --- Geometric Signal 3: Axis-Aligned Spread Ratio ---
        pop_range = np.ptp(self.population, axis=0)
        search_range = self.upper_bound - self.lower_bound
        spread_ratios = pop_range / (search_range + 1e-30)
        min_spread = np.min(spread_ratios)
        max_spread = np.max(spread_ratios)
        spread_aspect = max_spread / (min_spread + 1e-30)
        spread_aspect = np.clip(spread_aspect, 1.0, 100.0)

        if spread_aspect > 50.0:
            spread_modulation = 1.8
        elif spread_aspect > 20.0:
            spread_modulation = 1.4
        else:
            spread_modulation = 1.0

        # --- Geometric Signal 4: k-NN Density Gradient ---
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_mean = np.mean(np.sqrt(knn_dists), axis=1) + 1e-10
        global_knn_mean = np.mean(knn_mean)
        density_signal = knn_mean / (global_knn_mean + 1e-10)
        density_signal = np.clip(density_signal, 0.5, 2.0)

        # --- Combine Geometric Signals into Spatial Scale ---
        spatial_scale = expansion_strength * elongation_modulation * spread_modulation
        spatial_scale = spatial_scale * (0.7 + 0.3 * np.mean(density_signal))
        spatial_scale = np.clip(spatial_scale, 0.4, 2.5)

        # --- Apply Position Update ---
        new_population = self.population + spatial_scale * self.velocity
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
    
    The gap table is BLEND-HOSTILE:
      - Task 5: original wins with 768× gap to runner-up, 544618× to median.
        A 50/50 blend of 63.5 and 48742 yields ~24403, which is ~384× WORSE
        than the winner alone. Linear blending cannot preserve this advantage.
      - Task 1: original wins with 11.5× gap.
      - Task 3: original wins with 6.3× gap.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin burst with each of the 5 winning
        operators, tracking rank-based improvement scores on the ACTUAL
        landscape. Each operator gets 3 generations in a sub-population.
      - PHASE B (commit): run ONLY the winning operator for the remaining
        budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if committed operator stagnates (>20 gens, p=0.05),
        allow a small probability of switching to the runner-up.
    
    Win-count distribution (8/6/4/3/3 across 5 winners) confirms that
    different operators win on different tasks. The probe extracts a
    cheap FINGERPRINT from the landscape (convergence slope, rank-
    correlation, effective dimensionality) to route to the right operator.
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
        self.current_fitness = None
        
        # --- PROBE-AND-COMMIT STATE ---
        # Map operator ID -> list of rank-based scores
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        
        # THE 5 ACTUAL WINNING POSITION UPDATE STRATEGIES
        self._operators = [
            'original',      # variant_00: eigenvalue-anisotropic velocity modulation
            'svd_whitening', # variant_02: SVD-based population whitening (3 wins)
            'fitness_rank',  # variant_04: fitness-landscape rank signals only (8 wins)
            'temporal_drift',# variant_06: temporal-drift-modulated velocity scaling (3 wins)
            'centroid_knn',  # variant_09: centroid gravity + k-NN density (6 wins)
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator, min 15 total
        self._probe_gens_per_op = 3
        self._probe_total_budget = max(len(self._operators) * self._probe_gens_per_op, 15)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        # Running history for rank-based scoring
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Fingerprint signals (computed during probe)
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
        
        # Commit phase stagnation tracking
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
    
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
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
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
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
    
    def _compute_fingerprint(self):
        """Compute cheap landscape fingerprint from current population state.
        
        Returns a dict of signals observable from any black-box func:
          - convergence_slope: slope of log(best_fitness) vs generation
          - rank_corr: Spearman-like correlation between fitness rank and
                       distance to global best (proxy for landscape smoothness)
          - eff_dim: effective dimensionality from early covariance
          - diversity: current population diversity
        """
        signals = {}
        
        # 1. Convergence slope (from probe history)
        if len(self._probe_fitness_history) >= 3:
            gens = np.arange(len(self._probe_fitness_history))
            bests = np.array(self._probe_fitness_history)
            # Use log-slope for positive fitness; fallback to linear for negative/zero
            valid = bests > 0
            if np.sum(valid) >= 2:
                log_bests = np.log1p(bests[valid])
                signals['convergence_slope'] = np.polyfit(gens[valid], log_bests, 1)[0]
            else:
                signals['convergence_slope'] = np.polyfit(gens, bests, 1)[0]
        else:
            signals['convergence_slope'] = 0.0
        
        # 2. Rank correlation (fitness vs distance to global best)
        if self.global_best is not None and self.np > 2:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                signals['rank_corr'] = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                signals['rank_corr'] = 0.0
        else:
            signals['rank_corr'] = 0.0
        
        # 3. Effective dimensionality from covariance
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cumvar = np.cumsum(sorted(eigenvalues, reverse=True)) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        # 4. Diversity
        signals['diversity'] = self._compute_diversity()
        
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        
        return signals
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight using condition number of population covariance."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            # Spectral signal: map condition number to inertia adjustment
            # High cond = anisotropic/ill-conditioned → more exploration (higher inertia)
            # Low cond = well-conditioned → more exploitation (lower inertia)
            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            # Map spectral signal to inertia range [0.4, 0.95]
            target_inertia = 0.4 + 0.55 * spectral_signal

            # Blend with time-based baseline
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (shared across all position update strategies)
    # -------------------------------------------------------------------------
    
    def _velocity_update_base(self):
        """Base velocity update: cognitive + social + DE mutation."""
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
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (the 5 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation."""
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
    
    def _position_update_svd_whitening(self):
        """SVD-based population whitening with anisotropic damping (variant_02).
        
        3 wins on tasks 2, 4, 8. Key insight: counteracts anisotropy by
        dampening high-spread directions and amplifying collapsed directions.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            singular_values = np.clip(singular_values, 1e-10, None)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Inverse-square-law: stronger effect on collapsed directions
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_fitness_rank(self):
        """Graph-Laplacian eigenvalue-based velocity modulation (Category E).

        Key insight: Build k-NN graph over population; use Laplacian eigenvalues
        to detect clustering/convergence. Small spectral gap = clustered population
        → increase exploration. Small component particles → stronger nudges toward
        larger components. This topology-aware approach is orthogonal to fitness-
        landscape (D) and spectral-matrix (B) approaches used previously.
        """
        # Build k-NN graph over population
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Connected component analysis using graph traversal
        visited = np.zeros(self.np, dtype=bool)
        components = []
        for start in range(self.np):
            if visited[start]:
                continue
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for neighbor in knn_indices[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            components.append(component)

        component_sizes = np.array([len(c) for c in components])
        particle_component_size = np.array([component_sizes[np.argmax([i in c for c in components])] for i in range(self.np)])

        # Graph Laplacian spectral analysis: second-smallest eigenvalue of normalized Laplacian
        try:
            # Compute adjacency (k-NN symmetric)
            row_indices = np.repeat(np.arange(self.np), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            adj_size = self.np * self.np
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
            adj = adj + adj.T
            adj.data[:] = 1.0

            # Normalized Laplacian: I - D^{-1/2} * A * D^{-1/2}
            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
            lap = dspy.identity(self.np) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)

            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)

            # Modularity-like signal from top eigenvalues
            modularity_signal = 1.0 - eigenvalues[0] if len(eigenvalues) > 0 else 0.0
        except:
            spectral_gap = 1.0
            modularity_signal = 0.0

        # Fitness rank signal (from Category D)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Graph-based exploration factor
        # Small spectral_gap = clustered → more exploration needed
        clustering_explore = 1.0 + 0.5 * (1.0 - spectral_gap)
        # Small component → particles may be isolated → stronger exploration
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.3, 1.0)
        graph_explore = clustering_explore * (2.0 - size_factor)
        graph_explore = np.clip(graph_explore, 0.5, 2.0)

        # Success history (from Category D)
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # Velocity scaling combines graph signals with fitness rank
        vel_scale = graph_explore * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.5, 2.5)

        # Fitness-based directional perturbation
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        random_perturb = graph_explore[:, np.newaxis] * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_temporal_drift(self):
        """Temporal-drift-modulated velocity scaling (variant_06).
        
        3 wins on tasks 11, 13, 18. Key insight: EMA of centroid dynamics
        detects convergence/oscillation; modulates scaling accordingly.
        """
        centroid = np.mean(self.population, axis=0)

        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)

        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid

        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

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

        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        convergence_signal = np.clip(autocorr, -1.0, 1.0)

        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7

        temporal_scale = base_scale * (1.0 + 0.4 * convergence_signal)

        vel_magnitude = np.linalg.norm(self._ema_centroid_velocity) + 1e-10
        vel_factor = np.clip(1.0 / (1.0 + vel_magnitude * 0.005), 0.5, 1.5)
        temporal_scale *= vel_factor

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
            temporal_scale *= spectral_factor
        except np.linalg.LinAlgError:
            pass

        new_population = self.population + temporal_scale * self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_centroid_knn(self):
        """Centroid gravity + k-NN density modulation (variant_09).
        
        6 wins on tasks 6, 9, 10, 14, 19, 23. Key insight: particles in
        sparse regions get more centroid pull; dense regions get repulsion.
        """
        centroid = np.mean(self.population, axis=0)

        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10

        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)

        correction = centroid_attraction * to_centroid_dir * density_modulation

        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'svd_whitening':
            self._position_update_svd_whitening()
        elif operator == 'fitness_rank':
            self._position_update_fitness_rank()
        elif operator == 'temporal_drift':
            self._position_update_temporal_drift()
        elif operator == 'centroid_knn':
            self._position_update_centroid_knn()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score using rank-based (scale-independent) scoring.
        
        Score = fraction of probe generations with WORSE (higher) fitness.
        This is rank-based, not raw fitness — no scale-dependent constants.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        if len(self._probe_fitness_history) < 2:
            score = 1.0
        else:
            current_best = min(self._probe_fitness_history)
            current_worst = max(self._probe_fitness_history)
            if current_worst > current_best:
                # Higher score = better (lower fitness)
                normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
                score = np.clip(normalized, 0.0, 1.0)
            else:
                score = 1.0
        
        self._operator_scores[operator].append(score)
        self._operator_total_score[operator] += score
        self._operator_sample_count[operator] += 1
    
    def _compute_ucb_score(self, operator):
        """Compute UCB1-style score for operator selection.
        
        Uses rank-based average (already in [0,1]) so exploration bonus
        is on the same scale — no magic constants.
        """
        n = self._operator_sample_count[operator]
        if n == 0:
            return float('inf')
        
        total_n = sum(self._operator_sample_count.values())
        avg_score = self._operator_total_score[operator] / n
        
        # UCB1 exploration bonus — scale is [0,1], so this is well-calibrated
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        
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
    # RESTART & LOCAL BEST
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT position-update selection.
        
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
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset EMA states for temporal_drift
        if hasattr(self, '_ema_centroid'):
            del self._ema_centroid
        if hasattr(self, '_ema_centroid_velocity'):
            del self._ema_centroid_velocity
        if hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        if hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        if hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        
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
            if hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Velocity update (shared base), then position update (PROBED)
                self._velocity_update_base()
                self._dispatch_position_update(current_op)
                
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
                
                # Record probe score (rank-based, scale-independent)
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
                    # Compute fingerprint from probe data
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._velocity_update_base()
                    self._dispatch_position_update(self._runner_up_operator)
                else:
                    self._velocity_update_base()
                    self._dispatch_position_update(self._committed_operator)
                
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
                
                # Track commit-phase improvement for potential review
                if self.global_best_fitness < self._commit_best_fitness:
                    self._commit_best_fitness = self.global_best_fitness
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```