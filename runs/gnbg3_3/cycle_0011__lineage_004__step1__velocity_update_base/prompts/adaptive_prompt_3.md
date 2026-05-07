Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_velocity_update_base` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.631587e-01            1.472258e+01            1.031819e+01            1.229195e-01            -inf                    1.010851e+00            1.913582e-01            1.049908e+00            1.765568e-01            1.035643e+00            1.104579e+01            
1      2.701172e+00            2.546445e+01            4.244655e+01            9.087629e+00            -inf                    4.543035e+00            3.063340e+00            4.571940e+00            6.517138e+00            6.871428e+00            6.056601e+01            
2      4.252344e-01            7.725277e+00            5.924421e+00            7.852600e-01            -inf                    1.123650e+01            4.513176e-01            2.775816e+01            6.707172e+00            1.413996e+01            1.134216e+01            
3      6.780165e+00            1.200162e+02            1.332772e+02            3.654038e+01            -inf                    2.360496e+01            6.667789e+00            1.705652e+01            5.477179e+01            3.612241e+01            1.752760e+02            
4      1.188679e+00            2.596524e+00            2.509746e+00            1.269261e+00            -inf                    1.932818e+00            1.304550e+00            1.820856e+00            1.313731e+00            1.846765e+00            3.471801e+00            
5      2.633829e+01            1.275817e+05            1.708896e+02            5.563827e+00            -inf                    2.650723e+04            1.021639e+06            2.387361e+06            7.386891e+03            6.576989e+05            2.070590e+02            
6      6.861531e+02            7.110552e+02            6.149003e+02            6.440781e+02            -inf                    3.439512e+02            4.222499e+02            7.738942e+02            7.493027e+02            7.131527e+02            6.174143e+02            
7      1.071404e+01            1.523189e+02            1.670372e+02            3.631914e+01            -inf                    1.751696e+01            9.266946e+00            1.066941e+01            2.373189e+01            3.222088e+01            2.122654e+02            
8      2.950122e+00            2.150219e+01            3.023229e+01            6.560539e+00            -inf                    7.192821e+00            3.139830e+00            5.609404e+00            3.844313e+00            8.127546e+00            3.618276e+01            
9      1.048843e+02            7.813860e+01            5.842466e+01            8.105431e+01            -inf                    9.076680e+01            5.698562e+01            1.056391e+02            8.518082e+01            5.791352e+01            6.022288e+01            
10     1.416184e+01            2.107345e+02            1.123374e+02            2.112800e+01            -inf                    3.219529e+00            1.461286e+01            2.362748e+01            1.920019e+01            1.023427e+02            1.703519e+02            
11     1.965966e+02            6.960991e+02            5.389889e+02            3.322009e+02            -inf                    1.086216e+02            1.803609e+02            2.049618e+02            2.648178e+02            3.904003e+02            4.577579e+02            
12     7.250700e+01            1.478295e+02            1.243050e+02            9.309058e+01            -inf                    9.720769e+00            5.184972e+01            7.660832e+01            8.367714e+01            1.015695e+02            1.801510e+02            
13     1.684541e+01            3.538266e+01            2.816919e+01            2.553182e+01            -inf                    2.110556e+01            1.362991e+01            2.094536e+01            2.233580e+01            3.068796e+01            3.263444e+01            
14     8.846425e+00            6.041757e+00            7.590241e+00            6.701097e+00            -inf                    8.976748e+00            7.790337e+00            1.013142e+01            7.182501e+00            6.089103e+00            7.529951e+00            
15     3.680040e+00            4.035434e+00            4.068630e+00            3.985184e+00            -inf                    3.373267e+00            3.551066e+00            3.653243e+00            4.010678e+00            3.836246e+00            3.958522e+00            
16     2.122926e+03            7.550685e+03            1.881243e+03            1.386536e+03            -inf                    2.000097e+03            1.092066e+03            2.013258e+03            1.509743e+03            4.062172e+03            3.442060e+03            
17     5.020825e+04            9.768787e+04            1.326513e+05            5.699801e+04            -inf                    4.068943e+04            4.206989e+04            5.052142e+04            9.304268e+04            3.744873e+04            1.502630e+05            
18     5.119816e+01            6.013815e+01            6.455827e+01            5.882612e+01            -inf                    4.424094e+01            4.556705e+01            4.906365e+01            5.020808e+01            4.724206e+01            6.569217e+01            
19     1.401594e+02            1.754175e+02            1.140159e+02            1.216149e+02            -inf                    1.213527e+02            1.010909e+02            1.531490e+02            1.496974e+02            1.283515e+02            1.335394e+02            
20     2.550005e+01            3.644146e+01            3.249693e+01            2.487680e+01            -inf                    2.365881e+01            2.337658e+01            2.651031e+01            2.455478e+01            3.063085e+01            3.777444e+01            
21     5.474189e+00            5.657495e+00            5.822648e+00            5.689950e+00            -inf                    5.661022e+00            5.324856e+00            5.456701e+00            5.818613e+00            5.338097e+00            5.703590e+00            
22     1.376412e+01            1.617235e+01            1.647980e+01            1.570874e+01            -inf                    1.304271e+01            1.283383e+01            1.292038e+01            1.485595e+01            1.385383e+01            1.683807e+01            
23     6.506394e+01            6.038886e+01            5.674129e+01            6.184945e+01            -inf                    6.874191e+01            7.383028e+01            6.999957e+01            6.379494e+01            6.068630e+01            6.004857e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_catC_idea_0.py  (error=1.229195e-01)
Task  1: original.py  (error=2.701172e+00)
Task  2: original.py  (error=4.252344e-01)
Task  3: variant_06_catF_idea_0.py  (error=6.667789e+00)
Task  4: original.py  (error=1.188679e+00)
Task  5: variant_03_catC_idea_0.py  (error=5.563827e+00)
Task  6: variant_05_catE_idea_0.py  (error=3.439512e+02)
Task  7: variant_06_catF_idea_0.py  (error=9.266946e+00)
Task  8: original.py  (error=2.950122e+00)
Task  9: variant_06_catF_idea_0.py  (error=5.698562e+01)
Task 10: variant_05_catE_idea_0.py  (error=3.219529e+00)
Task 11: variant_05_catE_idea_0.py  (error=1.086216e+02)
Task 12: variant_05_catE_idea_0.py  (error=9.720769e+00)
Task 13: variant_06_catF_idea_0.py  (error=1.362991e+01)
Task 14: variant_01_catA_idea_0.py  (error=6.041757e+00)
Task 15: variant_05_catE_idea_0.py  (error=3.373267e+00)
Task 16: variant_06_catF_idea_0.py  (error=1.092066e+03)
Task 17: variant_09_catA_idea_0.py  (error=3.744873e+04)
Task 18: variant_05_catE_idea_0.py  (error=4.424094e+01)
Task 19: variant_06_catF_idea_0.py  (error=1.010909e+02)
Task 20: variant_06_catF_idea_0.py  (error=2.337658e+01)
Task 21: variant_06_catF_idea_0.py  (error=5.324856e+00)
Task 22: variant_06_catF_idea_0.py  (error=1.283383e+01)
Task 23: variant_02_catB_idea_0.py  (error=5.674129e+01)

WIN COUNTS:
  variant_06_catF_idea_0.py: 9 wins
  variant_05_catE_idea_0.py: 6 wins
  original.py: 4 wins
  variant_03_catC_idea_0.py: 2 wins
  variant_01_catA_idea_0.py: 1 wins
  variant_09_catA_idea_0.py: 1 wins
  variant_02_catB_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_03_catC_idea_0.py       1.2292e-01       original.py                      1.6316e-01         1.3×    8.4×
     1   original.py                     2.7012e+00       variant_06_catF_idea_0.py        3.0633e+00         1.1×    2.5×
     2   original.py                     4.2523e-01       variant_06_catF_idea_0.py        4.5132e-01         1.1×   18.2×
     3   variant_06_catF_idea_0.py       6.6678e+00       original.py                      6.7802e+00         1.0×    5.5×
     4   original.py                     1.1887e+00       variant_03_catC_idea_0.py        1.2693e+00         1.1×    1.6×
     5   variant_03_catC_idea_0.py       5.5638e+00       original.py                      2.6338e+01         4.7×   4764×
     6   variant_05_catE_idea_0.py       3.4395e+02       variant_06_catF_idea_0.py        4.2225e+02         1.2×    2.0×
     7   variant_06_catF_idea_0.py       9.2669e+00       variant_07_catG_idea_0.py        1.0669e+01         1.2×    3.5×
     8   original.py                     2.9501e+00       variant_06_catF_idea_0.py        3.1398e+00         1.1×    2.4×
     9   variant_06_catF_idea_0.py       5.6986e+01       variant_09_catA_idea_0.py        5.7914e+01         1.0×    1.4×
    10   variant_05_catE_idea_0.py       3.2195e+00       original.py                      1.4162e+01         4.4×    7.3×
    11   variant_05_catE_idea_0.py       1.0862e+02       variant_06_catF_idea_0.py        1.8036e+02         1.7×    3.1×
    12   variant_05_catE_idea_0.py       9.7208e+00       variant_06_catF_idea_0.py        5.1850e+01         5.3×    9.6×
    13   variant_06_catF_idea_0.py       1.3630e+01       original.py                      1.6845e+01         1.2×    1.9×
    14   variant_01_catA_idea_0.py       6.0418e+00       variant_09_catA_idea_0.py        6.0891e+00         1.0×    1.3×
    15   variant_05_catE_idea_0.py       3.3733e+00       variant_06_catF_idea_0.py        3.5511e+00         1.1×    1.2×
    16   variant_06_catF_idea_0.py       1.0921e+03       variant_03_catC_idea_0.py        1.3865e+03         1.3×    1.8×
    17   variant_09_catA_idea_0.py       3.7449e+04       variant_05_catE_idea_0.py        4.0689e+04         1.1×    1.5×
    18   variant_05_catE_idea_0.py       4.4241e+01       variant_06_catF_idea_0.py        4.5567e+01         1.0×    1.2×
    19   variant_06_catF_idea_0.py       1.0109e+02       variant_02_catB_idea_0.py        1.1402e+02         1.1×    1.3×
    20   variant_06_catF_idea_0.py       2.3377e+01       variant_05_catE_idea_0.py        2.3659e+01         1.0×    1.1×
    21   variant_06_catF_idea_0.py       5.3249e+00       variant_09_catA_idea_0.py        5.3381e+00         1.0×    1.1×
    22   variant_06_catF_idea_0.py       1.2834e+01       variant_07_catG_idea_0.py        1.2920e+01         1.0×    1.2×
    23   variant_02_catB_idea_0.py       5.6741e+01       variant_10_catB_idea_0.py        6.0049e+01         1.1×    1.1×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_idea_0.py, variant_02_catB_idea_0.py, variant_03_catC_idea_0.py, variant_05_catE_idea_0.py, variant_06_catF_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 1  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 1  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — variant_03_catC_idea_0.py (5.56e+00) vs original.py (2.63e+01) = 4.7× gap to runner-up, 4764.2× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.60e+01, which is ~2.9× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (1 wins) ---
```python
def _velocity_update_base(self):
        """Geometry-aware velocity update using axis-aligned spread and centroid geometry."""
        cognitive, social = self._adaptive_coefficients()

        # === GEOMETRIC COMPUTATIONS (Category A) ===
        centroid = np.mean(self.population, axis=0)

        # Axis-aligned spread (range) per dimension
        dim_min = np.min(self.population, axis=0)
        dim_max = np.max(self.population, axis=0)
        dim_range = dim_max - dim_min + 1e-10

        # Normalized spread per dimension: 0 = collapsed, 1 = well-spread
        max_range = np.max(dim_range)
        normalized_spread = np.clip(dim_range / (max_range + 1e-10), 0.01, 1.0)

        # Inverse spread: amplify velocity in collapsed dimensions
        spread_modulation = 1.0 / normalized_spread
        spread_modulation = np.clip(spread_modulation, 0.3, 3.0)
        spread_modulation = spread_modulation / (np.max(spread_modulation) + 1e-10)

        # Centroid-relative geometry: distance of each particle to centroid
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(dist_to_centroid) + 1e-10

        # Particles far from centroid get stronger pull toward center
        centroid_attraction_strength = np.clip(dist_to_centroid / (2.0 * global_spread), 0.0, 1.5)

        # Direction toward centroid for each particle
        to_centroid = centroid - self.population
        to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_norm

        # === RANDOM COEFFICIENTS ===
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # === MODULATED COGNITIVE COMPONENT ===
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # === MODULATED SOCIAL COMPONENT ===
        social_component = social * r2 * (self.local_best - self.population)

        # Add centroid correction: particles far from center get pulled inward
        centroid_correction = 0.2 * centroid_attraction_strength * to_centroid_dir

        # === GEOMETRY-MODULATED MUTATION COMPONENT ===
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

        # Apply spread modulation to mutation: more mutation in collapsed dimensions
        mutation_component = np.where(
            mutation_active,
            0.3 * spread_modulation * (mutation_vectors - self.population),
            0.0
        )

        # === COMBINE ALL COMPONENTS ===
        # Apply spread modulation to cognitive/social (dampen high-spread dims, amplify collapsed)
        new_velocity = (
            self.inertia_weight * self.velocity * spread_modulation +
            cognitive_component * spread_modulation +
            social_component * spread_modulation +
            mutation_component +
            centroid_correction
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_02_catB_idea_0.py (1 wins) ---
```python
def _velocity_update_base(self):
        """Spectral velocity modulation: dampen high-variance directions, amplify low-variance."""
        cognitive, social = self._adaptive_coefficients()

        # --- SPECTRAL ANALYSIS ---
        centered = self.population - np.mean(self.population, axis=0)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            cond = eigenvalues[0] / eigenvalues[-1]

            # Per-direction modulation: inverse-variance weighting
            # High variance direction → population already covers it → dampen
            # Low variance direction → potential unexplored region → amplify
            total_var = np.sum(eigenvalues) + 1e-10
            var_ratio = eigenvalues / total_var

            # Modulation: sqrt inverse prevents over-amplification of near-zero eigenvalues
            spectral_scale = np.sqrt(var_ratio[-1] / (var_ratio + 1e-10))
            spectral_scale = np.clip(spectral_scale, 0.3, 3.0)

            # Condition-based blending: mild modulation for well-conditioned, stronger for ill-conditioned
            blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            effective_scale = 1.0 * (1.0 - blend) + spectral_scale * blend

            # Project velocity onto eigenbasis, apply modulation, project back
            vel_proj = self.velocity @ eigenvectors
            vel_modulated = vel_proj * effective_scale
            modulated_velocity = vel_modulated @ eigenvectors.T
        except np.linalg.LinAlgError:
            modulated_velocity = self.velocity.copy()

        # --- COGNITIVE + SOCIAL ---
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)

        # --- DE MUTATION ---
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

        # --- COMBINE WITH SPECTRAL MODULATION ---
        new_velocity = (
            self.inertia_weight * modulated_velocity +
            cognitive_component +
            social_component +
            mutation_component
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_03_catC_idea_0.py (2 wins) ---
```python
def _velocity_update_base(self):
        """Information-theoretic velocity update using population entropy and distribution matching.

        Category C: Treats population as probability distribution, computes entropy from
        covariance, and uses entropy signal to modulate velocity magnitude.

        Key insight: Low population entropy = concentrated/bunched = premature convergence
        → boost exploration. High entropy = diverse spread → can exploit more.

        Also uses KL-like distribution matching toward a well-conditioned reference.
        """
        cognitive, social = self._adaptive_coefficients()

        # === ENTROPY COMPUTATION FROM POPULATION DISTRIBUTION ===
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)

            # Gaussian entropy: H = 0.5 * log((2*pi*e)^d * det(cov))
            # Higher det(cov) = more spread = higher entropy
            dim = self.dim
            var_range = (self.upper_bound - self.lower_bound) ** 2
            det_cov = np.prod(eigenvalues) + 1e-10

            entropy = 0.5 * (dim * np.log(2 * np.pi * np.e * var_range / dim) + np.log(det_cov))

            # Normalize entropy to [0, 1] using theoretical bounds
            max_entropy = 0.5 * dim * np.log(2 * np.pi * np.e * var_range / dim)
            min_entropy = 0.5 * dim * np.log(2 * np.pi * np.e * 1e-4)
            normalized_entropy = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0.0, 1.0)

            # Also compute condition number as anisotropy signal
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            anisotropy = np.clip(np.log1p(cond) / np.log1p(1e4), 0.0, 1.0)

        except:
            normalized_entropy = 0.5
            anisotropy = 0.5

        # === ENTROPY-BASED EXPLORATION/EXPLOITATION BALANCE ===
        # Low entropy -> concentrated population -> boost exploration
        # High entropy -> diverse population -> can exploit more
        exploration_boost = 1.0 + 0.6 * (1.0 - normalized_entropy)
        exploitation_damp = 1.0 - 0.3 * normalized_entropy

        # === FITNESS ENTROPY (distribution of fitness values) ===
        try:
            # Compute entropy of fitness distribution (histogram-based)
            hist, _ = np.histogram(self.current_fitness, bins=min(10, self.np), density=True)
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_fitness_entropy = np.log(min(10, self.np))
            normalized_fitness_entropy = np.clip(fitness_entropy / (max_fitness_entropy + 1e-10), 0.0, 1.0)
        except:
            normalized_fitness_entropy = 0.5

        # === COMBINED INFORMATION-THEORETIC SIGNAL ===
        combined_entropy = 0.6 * normalized_entropy + 0.4 * normalized_fitness_entropy

        # === KL-LIKE DISTRIBUTION MATCHING ===
        # Compute deviation from well-conditioned reference (identity-scaled covariance)
        # This provides a "pull" toward better-conditioned configurations
        try:
            eigenvalues_normalized = eigenvalues / (eigenvalues[0] + 1e-10)
            kl_like = np.mean(np.log(eigenvalues_normalized + 1e-10))
            kl_modulation = 1.0 + 0.2 * kl_like  # Positive kl_like = elongated -> reduce
            kl_modulation = np.clip(kl_modulation, 0.5, 1.5)
        except:
            kl_modulation = 1.0

        # === RANDOM COEFFICIENTS ===
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # === VELOCITY COMPONENTS ===
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)

        # === DE MUTATION COMPONENT ===
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

        # === COMBINE WITH INFORMATION-THEORETIC MODULATION ===
        # Scale cognitive/social by exploration boost and KL modulation
        # Scale mutation by entropy (more entropy = less need for mutation exploration)
        mutation_scale = 1.0 - 0.3 * combined_entropy

        new_velocity = (
            self.inertia_weight * self.velocity +
            exploration_boost * cognitive_component +
            exploitation_damp * social_component +
            kl_modulation * mutation_scale * mutation_component
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_05_catE_idea_0.py (6 wins) ---
```python
def _velocity_update_base(self):
        """Topology-aware velocity update using k-NN graph centrality."""
        cognitive, social = self._adaptive_coefficients()

        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)

        # === CATEGORY E: k-NN Graph Centrality ===
        k = min(5, self.np - 1)

        # Build k-NN distance matrix
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Compute average distance to k nearest neighbors (inverse centrality)
        knn_dists = np.sqrt(np.sort(sq_dists, axis=1)[:, :k])
        avg_knn_dist = np.mean(knn_dists, axis=1) + 1e-10

        # Graph centrality: inverse of average neighbor distance
        # Higher centrality = well-connected = less correction needed
        global_avg_dist = np.mean(avg_knn_dist) + 1e-10
        centrality = global_avg_dist / avg_knn_dist

        # Normalize centrality to [0.5, 2.0] range
        centrality = np.clip(centrality / (np.max(centrality) + 1e-10) * 1.5, 0.5, 2.0)

        # Compute connected components to detect fragmentation
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

        n_components = len(components)
        component_sizes = np.array([len(c) for c in components])

        # Per-particle component size factor
        particle_component_size = np.array([
            component_sizes[next((i for i, c in enumerate(components) if p in c), 0)]
            for p in range(self.np)
        ])

        # Fragmentation correction: particles in small components get stronger boost
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.1, 1.0)

        # Combined topology correction: low centrality + small component = strong pull
        topology_correction_strength = (2.0 - centrality) * (2.0 - size_factor)
        topology_correction_strength = np.clip(topology_correction_strength, 0.0, 2.0)

        # Apply topology-corrected velocity toward global best
        if self.global_best is not None:
            to_global = self.global_best - self.population
            to_global_dist = np.linalg.norm(to_global, axis=1, keepdims=True) + 1e-10
            to_global_dir = to_global / to_global_dist
            topology_correction = topology_correction_strength[:, np.newaxis] * to_global_dir
        else:
            # Fallback: random perturbation scaled by isolation
            topology_correction = topology_correction_strength[:, np.newaxis] * np.random.uniform(-1, 1, (self.np, self.dim))

        # === DE Mutation Component ===
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

        # === Combine all components ===
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            0.5 * topology_correction +
            mutation_component
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_06_catF_idea_0.py (9 wins) ---
```python
def _velocity_update_base(self):
        """Velocity update with temporal dynamics tracking (Category F).

        Tracks across generations:
        - Centroid drift rate (EMA of centroid displacement)
        - Population spread change rate (EMA of std deviation delta)
        - Velocity autocorrelation (momentum persistence signal)

        Uses these temporal signals to modulate inertia, cognitive/social
        coefficients, and mutation intensity based on detected swarm dynamics.
        """
        # === TEMPORAL STATE TRACKING ===
        # Initialize EMA state on first call
        if not hasattr(self, '_ema_centroid_drift'):
            self._ema_centroid_drift = 0.0
            self._ema_spread_change = 0.0
            self._ema_improvement = 0.0
            self._prev_centroid = np.mean(self.population, axis=0)
            self._prev_spread = np.std(self.population)
            self._prev_velocity_mag = np.mean(np.linalg.norm(self.velocity, axis=1))
            self._velocity_history = []

        # Compute current centroid and track drift
        current_centroid = np.mean(self.population, axis=0)
        centroid_drift = np.linalg.norm(current_centroid - self._prev_centroid)
        self._prev_centroid = current_centroid.copy()

        # Update EMA of drift rate
        self._ema_centroid_drift = 0.2 * centroid_drift + 0.8 * self._ema_centroid_drift

        # Track population spread change rate
        current_spread = np.std(self.population)
        spread_delta = abs(current_spread - self._prev_spread)
        self._prev_spread = current_spread
        self._ema_spread_change = 0.2 * spread_delta + 0.8 * self._ema_spread_change

        # Velocity autocorrelation for stagnation detection
        current_velocity_mag = np.mean(np.linalg.norm(self.velocity, axis=1))
        if self._prev_velocity_mag > 1e-10:
            autocorr = current_velocity_mag / (self._prev_velocity_mag + 1e-10)
        else:
            autocorr = 1.0
        self._prev_velocity_mag = current_velocity_mag

        # Store history for autocorrelation
        self._velocity_history.append(autocorr)
        if len(self._velocity_history) > 10:
            self._velocity_history.pop(0)

        # === TEMPORAL-BASED PARAMETER MODULATION ===
        # Normalize temporal signals
        normalized_drift = self._ema_centroid_drift / (current_spread + 1e-10)
        normalized_spread_change = self._ema_spread_change / (current_spread + 1e-10)

        # Stagnation detection: low drift + low spread change over time
        stagnation_signal = 1.0 - min(1.0, np.mean(self._velocity_history[-5:]) if len(self._velocity_history) >= 5 else 1.0)

        # Adaptive inertia based on temporal signals
        if stagnation_signal > 0.7:
            # Stagnating: boost inertia for exploration
            temporal_inertia = min(0.95, self.inertia_weight + 0.1)
        elif normalized_drift < 0.02:
            # Low drift (converging): reduce inertia
            temporal_inertia = max(0.4, self.inertia_weight - 0.05)
        else:
            # Normal exploration: time-based decay
            temporal_inertia = max(0.4, 0.729 - 0.15 * (self.generation / 3000))

        # Adaptive cognitive/social based on temporal dynamics
        cognitive, social = self._adaptive_coefficients()

        if stagnation_signal > 0.5:
            # Stagnating: boost cognitive, reduce social (personal search)
            temporal_cog_mult = 1.3
            temporal_social_mult = 0.7
        elif normalized_drift > 0.1 and normalized_spread_change > 0.05:
            # High movement (exploring): balanced
            temporal_cog_mult = 1.0
            temporal_social_mult = 1.0
        elif normalized_drift < 0.02:
            # Converging: boost social for faster convergence
            temporal_cog_mult = 0.8
            temporal_social_mult = 1.2
        else:
            temporal_cog_mult = 1.0
            temporal_social_mult = 1.0

        cognitive = cognitive * temporal_cog_mult
        social = social * temporal_social_mult

        # === BASE VELOCITY COMPONENTS ===
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)

        # === MUTATION COMPONENT (temporal-adaptive) ===
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        # Increase mutation when stagnating or spread is collapsing
        mutation_base = 0.1 * (1.0 - self.generation / 5000)
        mutation_boost = 0.05 * stagnation_signal + 0.03 * (1.0 - min(1.0, normalized_spread_change))
        mutation_threshold = max(0.01, mutation_base - mutation_boost)
        mutation_active = mutation_mask < mutation_threshold

        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )

        # Boost mutation magnitude when stagnating
        mutation_scale = 0.3 * (1.0 + 0.5 * stagnation_signal)
        mutation_component = np.where(
            mutation_active,
            mutation_scale * (mutation_vectors - self.population),
            0.0
        )

        # === COMBINE WITH TEMPORAL INERTIA ===
        new_velocity = (
            temporal_inertia * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_09_catA_idea_0.py (1 wins) ---
```python
def _velocity_update_base(self):
        """Base velocity update with k-NN density + axis-aligned spread modulation (Category A).

        Geometric mechanisms:
          1. k-NN density: Particles in sparse regions get AMPLIFIED cognitive/social pulls;
             particles in dense clusters get DAMPENED velocity to prevent over-crowding.
          2. Axis-aligned spread: Per-dimension std modulates velocity inversely —
             collapsed dimensions (low std) get amplified exploration, spread-out
             dimensions (high std) get dampened to avoid redundant exploration.

        This is purely geometric (distances, k-NN structure, axis-aligned spread)
        — no fitness values, eigenvalues, or information-theoretic quantities.
        """
        # === MECHANISM 1: k-NN DENSITY ESTIMATION ===
        k = min(5, self.np - 1)

        # Compute squared pairwise distances (N x N matrix)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)

        # k-NN distances: average distance to k nearest neighbors
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        mean_knn_dist = np.mean(knn_dists, axis=1) + 1e-10

        # Density = inverse of k-NN distance (high distance = sparse = high density factor)
        # Add small epsilon to avoid division issues
        density_factor = mean_knn_dist / (np.mean(mean_knn_dist) + 1e-10)
        density_factor = np.clip(density_factor, 0.3, 3.0)

        # === MECHANISM 2: AXIS-ALIGNED SPREAD MODULATION ===
        # Compute per-dimension standard deviation (geometric spread)
        per_dim_std = np.std(self.population, axis=0) + 1e-10
        mean_spread = np.mean(per_dim_std) + 1e-10

        # Spread ratio: how spread out is each dimension relative to average?
        spread_ratio = per_dim_std / mean_spread

        # Inverse spread modulation: collapsed dimensions (low spread_ratio) get
        # AMPLIFIED velocity, spread-out dimensions get dampened
        spread_modulation = 1.0 / (spread_ratio + 0.5)
        spread_modulation = np.clip(spread_modulation, 0.5, 2.5)

        # === COMBINE GEOMETRIC SIGNALS ===
        # Density-based scaling: sparse particles move more, dense particles move less
        density_scale = density_factor[:, np.newaxis]  # (np, 1) broadcast to (np, dim)

        # Spread-based scaling: per-dimension, broadcast across particles
        spread_scale = spread_modulation[np.newaxis, :]  # (1, dim) broadcast to (np, dim)

        # Combined geometric modulation
        geometric_scale = density_scale * spread_scale
        geometric_scale = np.clip(geometric_scale, 0.2, 4.0)

        # === STANDARD COGNITIVE + SOCIAL COMPONENTS ===
        cognitive, social = self._adaptive_coefficients()

        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)

        # Apply geometric modulation to cognitive and social components
        cognitive_component = cognitive_component * geometric_scale
        social_component = social_component * geometric_scale

        # === DE MUTATION COMPONENT (also modulated geometrically) ===
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

        # Apply geometric modulation to mutation component as well
        mutation_component = mutation_component * geometric_scale

        # === VELOCITY UPDATE ===
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
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
        """Hybrid: Temporal drift tracking + topology-based fragmentation detection (Category H).

        Combines TWO distinct mechanisms:
          1. Temporal: Centroid drift rate across generations to detect convergence phases
          2. Topology: k-NN connected component analysis to detect population fragmentation

        Switching logic is data-driven:
          - Low drift + fragmented population → strong exploration boost
          - High drift (exploring) → trust spectral signals
          - Fragmented + improving → moderate exploration
          - Well-connected + converging → exploit via spectral damping

        Targets worst tasks (17, 16, 6, 11) where premature fragmentation causes
        the population to get trapped in disconnected local optima.
        """
        try:
            # === MECHANISM 1: TEMPORAL DRIFT TRACKING ===
            centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_prev_centroid'):
                self._prev_centroid = centroid.copy()
                self._centroid_drift_history = []

            drift = np.linalg.norm(centroid - self._prev_centroid)
            self._prev_centroid = centroid.copy()

            if not hasattr(self, '_centroid_drift_history'):
                self._centroid_drift_history = []
            self._centroid_drift_history.append(drift)
            if len(self._centroid_drift_history) > 20:
                self._centroid_drift_history.pop(0)

            # EMA of drift rate
            if not hasattr(self, '_ema_drift'):
                self._ema_drift = 0.0
            self._ema_drift = 0.3 * drift + 0.7 * self._ema_drift

            # Normalize drift by population spread
            population_spread = np.std(self.population) + 1e-10
            normalized_drift = self._ema_drift / population_spread

            # === MECHANISM 2: TOPOLOGY FRAGMENTATION DETECTION ===
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

            # Connected component analysis via BFS
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

            n_components = len(components)
            component_sizes = np.array([len(c) for c in components])

            # Fragmentation ratio: how uneven is the distribution?
            max_component_size = np.max(component_sizes)
            fragmentation_ratio = 1.0 - (max_component_size / (self.np + 1e-10))

            # Per-particle component membership
            particle_to_component = np.zeros(self.np, dtype=int)
            for idx, comp in enumerate(components):
                for particle_idx in comp:
                    particle_to_component[particle_idx] = idx

            # === PRINCIPLED SWITCHING LOGIC ===
            # State classification based on temporal and topological signals
            is_converging = normalized_drift < 0.05  # Low drift = converging
            is_fragmented = n_components > 1 and fragmentation_ratio > 0.3

            # === SPECTRAL ANALYSIS (for when not fragmented) ===
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

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

            # === TEMPORAL IMPROVEMENT SIGNAL ===
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_spectral_ema_improvement'):
                self._spectral_ema_improvement = 0.0
            self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement

            # === DECISION TREE WITH PRINCIPLED SWITCHING ===
            if is_fragmented and is_converging:
                # CRITICAL: Population is fragmented AND stuck → aggressive exploration
                # Weight heavily toward topology-based correction
                base_scale = 1.5
                topology_weight = 0.8
                spectral_weight = 0.2
            elif is_fragmented and not is_converging:
                # Fragmented but still moving → moderate exploration
                base_scale = 1.3
                topology_weight = 0.6
                spectral_weight = 0.4
            elif not is_fragmented and is_converging:
                # Well-connected but converging → use spectral exploitation
                base_scale = 0.9
                topology_weight = 0.2
                spectral_weight = 0.8
            else:
                # Well-connected and exploring → balanced
                base_scale = 1.1
                topology_weight = 0.4
                spectral_weight = 0.6

            # === COMBINE MECHANISMS ===
            # Spectral contribution
            if self._spectral_ema_improvement > 1e-6:
                spectral_base = 1.2
            elif self._spectral_ema_improvement > 1e-10:
                spectral_base = 1.0
            else:
                spectral_base = 0.7

            spectral_scale = spectral_base * (1.0 + 0.4 * spectral_signal)
            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                spectral_scale *= (1.0 + 0.3 * log_damp)

            # Topology contribution: inter-component repulsion
            # Particles in smaller components get pushed toward larger ones
            component_centroids = np.array([np.mean(self.population[c], axis=0) for c in components])
            largest_comp_idx = np.argmax(component_sizes)
            target_centroid = component_centroids[largest_comp_idx]

            # Per-particle correction toward largest component centroid
            topology_correction = np.zeros((self.np, self.dim))
            for i in range(self.np):
                comp_idx = particle_to_component[i]
                comp_size = component_sizes[comp_idx]
                if comp_idx != largest_comp_idx:
                    # Scale correction by inverse component size (smaller = stronger push)
                    correction_strength = 0.5 * (1.0 - comp_size / self.np)
                    direction = target_centroid - self.population[i]
                    direction_norm = np.linalg.norm(direction) + 1e-10
                    topology_correction[i] = correction_strength * (direction / direction_norm)

            # Combine spectral and topology contributions
            combined_scale = spectral_weight * spectral_scale + topology_weight * (1.0 + 0.5 * fragmentation_ratio)

            # Apply combined scale to velocity
            U = np.linalg.eigvalsh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = np.linalg.eigh(cov)[1][:, idx]

            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            per_comp_scale = np.clip(sv_norm, 0.1, 2.0)
            vel_scaled = vel_proj * per_comp_scale

            # Final update: spectral velocity + topology correction
            new_population = self.population + combined_scale * (vel_scaled @ V.T) + topology_correction

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

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