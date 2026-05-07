Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_velocity_update_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      9.639481e+00            7.674367e+00            nan                     6.999778e+00            5.740866e+00            1.209525e+01            1.017377e+01            nan                     9.229758e+00            1.129295e+01            1.125414e+01            
1      2.917541e+00            8.797120e+00            nan                     1.922629e+01            9.011537e+00            2.896616e+01            4.043242e+00            nan                     2.519382e+01            3.208687e+00            2.996917e+00            
2      1.426079e+02            4.862173e+01            nan                     9.277420e+01            1.315078e+02            1.378718e+02            1.425208e+02            nan                     1.495205e+02            1.518694e+02            1.227617e+02            
3      6.843786e+00            4.233810e+01            nan                     6.824103e+01            4.164186e+01            9.979646e+01            6.944551e+00            nan                     7.692576e+01            1.878091e+01            4.496952e+00            
4      2.804566e+00            2.547035e+00            nan                     2.939824e+00            2.511436e+00            3.384386e+00            2.788286e+00            nan                     3.327975e+00            2.926228e+00            2.153815e+00            
5      9.844950e+01            3.364981e+05            nan                     2.143861e+06            5.311697e+05            4.403772e+07            6.030204e+01            nan                     6.935412e+06            1.964807e+02            3.117167e+01            
6      1.317773e+03            7.869097e+02            nan                     9.512466e+02            1.350128e+03            2.854084e+03            1.459368e+03            nan                     1.802792e+03            1.130151e+03            9.866575e+02            
7      8.312018e+00            3.320651e+01            nan                     5.256194e+01            3.486420e+01            8.553474e+01            9.172504e+00            nan                     8.710346e+01            9.218095e+00            7.238755e+00            
8      1.176810e+01            1.149830e+01            nan                     1.628423e+01            1.141652e+01            2.098432e+01            1.195136e+01            nan                     2.026190e+01            1.245215e+01            6.868494e+00            
9      1.567469e+02            1.033462e+02            nan                     9.874748e+01            1.433847e+02            2.678483e+02            1.572153e+02            nan                     1.382838e+02            1.364335e+02            1.260747e+02            
10     7.889535e+01            3.221648e+02            1.756537e+02            1.587612e+02            9.963374e+01            6.976026e+01            7.138014e+01            nan                     1.147448e+02            9.914247e+01            7.487764e+01            
11     2.596490e+02            1.333932e+03            7.588731e+02            1.055874e+03            4.558696e+02            3.925721e+02            3.149561e+02            nan                     6.371750e+02            7.329751e+02            6.959390e+02            
12     1.095714e+02            2.476603e+02            1.721187e+02            1.709117e+02            1.617429e+02            6.523172e+01            9.881509e+01            nan                     1.430458e+02            1.616641e+02            1.175789e+02            
13     2.382198e+01            5.109743e+01            3.334228e+01            3.052095e+01            2.319044e+01            2.671748e+01            2.377934e+01            nan                     2.755768e+01            2.745176e+01            3.162564e+01            
14     1.273579e+01            1.354897e+01            1.387450e+01            1.197852e+01            1.161213e+01            1.569116e+01            1.193776e+01            nan                     1.207413e+01            1.139315e+01            1.006989e+01            
15     3.860743e+00            4.172722e+00            4.079794e+00            4.085274e+00            4.104740e+00            3.442630e+00            3.930066e+00            nan                     4.082271e+00            4.088291e+00            4.137339e+00            
16     5.401070e+03            1.165102e+04            nan                     5.726123e+03            4.517383e+03            1.167600e+04            5.749440e+03            nan                     6.423717e+03            4.391058e+03            4.364766e+03            
17     6.764446e+04            3.476876e+05            1.955249e+05            2.079961e+05            1.003506e+05            6.134690e+04            6.682094e+04            nan                     9.522522e+04            2.611209e+05            2.803601e+05            
18     5.351889e+01            9.852704e+01            7.990063e+01            8.209375e+01            7.949371e+01            4.343204e+01            5.684676e+01            nan                     6.710180e+01            8.949144e+01            9.266892e+01            
19     2.309333e+02            3.146111e+02            2.776909e+02            2.468791e+02            2.088112e+02            2.929373e+02            2.285209e+02            nan                     2.348974e+02            2.023878e+02            2.012136e+02            
20     3.724459e+01            6.087782e+01            nan                     4.135440e+01            3.747247e+01            3.659457e+01            3.620370e+01            nan                     3.996582e+01            3.750896e+01            3.454556e+01            
21     5.667701e+00            6.049254e+00            5.747906e+00            5.747796e+00            5.768467e+00            5.562990e+00            5.702474e+00            nan                     5.681063e+00            5.887338e+00            5.898251e+00            
22     1.536783e+01            2.036490e+01            1.753271e+01            1.835786e+01            1.566875e+01            1.442617e+01            1.520494e+01            nan                     1.689087e+01            1.859648e+01            1.833363e+01            
23     7.888192e+01            9.947328e+01            9.377858e+01            8.335871e+01            7.813504e+01            1.039846e+02            7.721050e+01            nan                     8.061223e+01            7.448324e+01            5.508135e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_04_catD_idea_0.py  (error=5.740866e+00)
Task  1: original.py  (error=2.917541e+00)
Task  2: variant_01_catA_idea_0.py  (error=4.862173e+01)
Task  3: variant_10_catB_idea_0.py  (error=4.496952e+00)
Task  4: variant_10_catB_idea_0.py  (error=2.153815e+00)
Task  5: variant_10_catB_idea_0.py  (error=3.117167e+01)
Task  6: variant_01_catA_idea_0.py  (error=7.869097e+02)
Task  7: variant_10_catB_idea_0.py  (error=7.238755e+00)
Task  8: variant_10_catB_idea_0.py  (error=6.868494e+00)
Task  9: variant_03_catC_idea_0.py  (error=9.874748e+01)
Task 10: variant_05_catE_idea_0.py  (error=6.976026e+01)
Task 11: original.py  (error=2.596490e+02)
Task 12: variant_05_catE_idea_0.py  (error=6.523172e+01)
Task 13: variant_04_catD_idea_0.py  (error=2.319044e+01)
Task 14: variant_10_catB_idea_0.py  (error=1.006989e+01)
Task 15: variant_05_catE_idea_0.py  (error=3.442630e+00)
Task 16: variant_10_catB_idea_0.py  (error=4.364766e+03)
Task 17: variant_05_catE_idea_0.py  (error=6.134690e+04)
Task 18: variant_05_catE_idea_0.py  (error=4.343204e+01)
Task 19: variant_10_catB_idea_0.py  (error=2.012136e+02)
Task 20: variant_10_catB_idea_0.py  (error=3.454556e+01)
Task 21: variant_05_catE_idea_0.py  (error=5.562990e+00)
Task 22: variant_05_catE_idea_0.py  (error=1.442617e+01)
Task 23: variant_10_catB_idea_0.py  (error=5.508135e+01)

WIN COUNTS:
  variant_10_catB_idea_0.py: 10 wins
  variant_05_catE_idea_0.py: 7 wins
  variant_04_catD_idea_0.py: 2 wins
  original.py: 2 wins
  variant_01_catA_idea_0.py: 2 wins
  variant_03_catC_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_04_catD_idea_0.py       5.7409e+00       variant_03_catC_idea_0.py        6.9998e+00         1.2×    1.7×
     1   original.py                     2.9175e+00       variant_10_catB_idea_0.py        2.9969e+00         1.0×    3.1×
     2   variant_01_catA_idea_0.py       4.8622e+01       variant_03_catC_idea_0.py        9.2774e+01         1.9×    2.9×
     3   variant_10_catB_idea_0.py       4.4970e+00       original.py                      6.8438e+00         1.5×    9.3×
     4   variant_10_catB_idea_0.py       2.1538e+00       variant_04_catD_idea_0.py        2.5114e+00         1.2×    1.3×
     5   variant_10_catB_idea_0.py       3.1172e+01       variant_06_catF_idea_0.py        6.0302e+01         1.9×  13918×
     6   variant_01_catA_idea_0.py       7.8691e+02       variant_03_catC_idea_0.py        9.5125e+02         1.2×    1.7×
     7   variant_10_catB_idea_0.py       7.2388e+00       original.py                      8.3120e+00         1.1×    4.7×
     8   variant_10_catB_idea_0.py       6.8685e+00       variant_04_catD_idea_0.py        1.1417e+01         1.7×    1.8×
     9   variant_03_catC_idea_0.py       9.8747e+01       variant_01_catA_idea_0.py        1.0335e+02         1.0×    1.4×
    10   variant_05_catE_idea_0.py       6.9760e+01       variant_06_catF_idea_0.py        7.1380e+01         1.0×    1.4×
    11   original.py                     2.5965e+02       variant_06_catF_idea_0.py        3.1496e+02         1.2×    2.7×
    12   variant_05_catE_idea_0.py       6.5232e+01       variant_06_catF_idea_0.py        9.8815e+01         1.5×    2.5×
    13   variant_04_catD_idea_0.py       2.3190e+01       variant_06_catF_idea_0.py        2.3779e+01         1.0×    1.2×
    14   variant_10_catB_idea_0.py       1.0070e+01       variant_09_catA_idea_0.py        1.1393e+01         1.1×    1.2×
    15   variant_05_catE_idea_0.py       3.4426e+00       original.py                      3.8607e+00         1.1×    1.2×
    16   variant_10_catB_idea_0.py       4.3648e+03       variant_09_catA_idea_0.py        4.3911e+03         1.0×    1.3×
    17   variant_05_catE_idea_0.py       6.1347e+04       variant_06_catF_idea_0.py        6.6821e+04         1.1×    3.2×
    18   variant_05_catE_idea_0.py       4.3432e+01       original.py                      5.3519e+01         1.2×    1.8×
    19   variant_10_catB_idea_0.py       2.0121e+02       variant_09_catA_idea_0.py        2.0239e+02         1.0×    1.2×
    20   variant_10_catB_idea_0.py       3.4546e+01       variant_06_catF_idea_0.py        3.6204e+01         1.0×    1.1×
    21   variant_05_catE_idea_0.py       5.5630e+00       original.py                      5.6677e+00         1.0×    1.0×
    22   variant_05_catE_idea_0.py       1.4426e+01       variant_06_catF_idea_0.py        1.5205e+01         1.1×    1.2×
    23   variant_10_catB_idea_0.py       5.5081e+01       variant_09_catA_idea_0.py        7.4483e+01         1.4×    1.5×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 6 (original.py, variant_01_catA_idea_0.py, variant_03_catC_idea_0.py, variant_04_catD_idea_0.py, variant_05_catE_idea_0.py, variant_10_catB_idea_0.py)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 1  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 1  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — variant_10_catB_idea_0.py (3.12e+01) vs variant_06_catF_idea_0.py (6.03e+01) = 1.9× gap to runner-up, 13917.6× gap to median competitor. A 50/50 blend with the runner-up would yield ~4.57e+01, which is ~1.5× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (2 wins) ---
```python
def _velocity_update_batch(self):
        """Update velocities using geometric layout: centroid, k-NN repulsion, spread."""
        # Compute centroid of population
        centroid = np.mean(self.population, axis=0)

        # Compute axis-aligned spread (range) per dimension
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10

        # Compute distance from centroid for each particle
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

        # Generate random matrices
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # Cognitive component: attraction to personal best
        cognitive = self.cognitive_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # Social component: attraction to local best
        social = self.social_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        social_component = social * r2 * (self.local_best - self.population)

        # Geometric centroid component: attract toward swarm centroid
        centroid_direction = centroid - self.population
        centroid_strength = np.exp(-dist_to_centroid / 100.0)
        centroid_component = 0.3 * centroid_strength * centroid_direction

        # k-NN repulsion from spatial neighbors (purely geometric)
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

        # Spread-based velocity modulation: boost under-explored dimensions
        spread_modulation = np.mean(spread) / (spread + 1e-10)

        # Velocity update
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            centroid_component +
            knn_component
        )
        new_velocity = new_velocity * spread_modulation

        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_03_catC_idea_0.py (1 wins) ---
```python
def _velocity_update_batch(self):
        """Update velocities with entropy-modulated distribution-guided exploration."""
        cognitive, social = self._adaptive_coefficients()

        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)

        # --- INFORMATION-THEORETIC COMPONENT (Category C) ---
        # Fit a multivariate Gaussian to the current population
        mean_pop = np.mean(self.population, axis=0)
        centered = self.population - mean_pop

        # Compute population covariance with regularization for numerical stability
        cov_pop = np.cov(centered.T)
        cov_pop += np.eye(self.dim) * 1e-8  # Regularization

        try:
            # Eigendecomposition of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov_pop)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)  # Ensure positive

            # Compute differential entropy of the Gaussian: H = 0.5 * log((2πe)^d * det(cov))
            # Entropy increases with spread/determinant of covariance
            log_det = np.sum(np.log(eigenvalues))
            entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + log_det)

            # Normalize entropy to [0, 1] relative to expected range for this dimension
            # Max entropy for uniform distribution in range [-100, 100] ≈ dim * log(range)
            max_entropy = self.dim * np.log(self.upper_bound - self.lower_bound + 1e-10)
            min_entropy = self.dim * np.log(1e-6)  # Highly concentrated
            entropy_normalized = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0, 1)

            # Entropy-based dispersive force: when entropy is LOW, push particles toward mean
            # This combats premature convergence detected via information-theoretic measure
            dispersive_strength = 0.5 * (1.0 - entropy_normalized)

            # Compute Mahalanobis-like distance for each particle from distribution center
            # Use inverse square root of eigenvalues for scaling
            inv_sqrt_eigen = 1.0 / np.sqrt(eigenvalues + 1e-10)
            scaled_diff = centered * inv_sqrt_eigen  # Broadcast: (np, dim)
            mahal_dist = np.linalg.norm(scaled_diff, axis=1, keepdims=True)  # (np, 1)

            # Normalize distances to [0, 1] per generation
            max_mahal = np.max(mahal_dist) + 1e-10
            mahal_normalized = mahal_dist / max_mahal

            # Direction toward mean (normalized)
            to_mean = mean_pop - self.population  # (np, dim)
            to_mean_norm = np.linalg.norm(to_mean, axis=1, keepdims=True) + 1e-10
            to_mean_dir = to_mean / to_mean_norm

            # Dispersive velocity component: push toward mean, modulated by entropy and distance
            # Particles far from center get pushed more; particles near center get less push
            dispersive_component = (
                dispersive_strength *
                mahal_normalized *
                to_mean_dir *
                np.random.uniform(0, 1, (self.np, self.dim))
            )

            # Also add a small random perturbation scaled by inverse entropy
            # (more random exploration when population is collapsed)
            random_explore = 0.3 * (1.0 - entropy_normalized) * np.random.uniform(-1, 1, (self.np, self.dim))

            entropy_velocity = dispersive_component + random_explore

        except np.linalg.LinAlgError:
            # Fallback: small random exploration on decomposition failure
            entropy_velocity = 0.1 * np.random.uniform(-1, 1, (self.np, self.dim))

        # Velocity update with inertia, cognitive, social, mutation, and entropy-based components
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            entropy_velocity
        )

        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_04_catD_idea_0.py (2 wins) ---
```python
def _velocity_update_batch(self):
        """Update velocities with fitness-rank-adaptive components."""
        # Compute Spearman rank correlation between fitness and distance to global best
        if self.global_best is not None:
            # Compute distances from each particle to global best
            distances = np.linalg.norm(self.population - self.global_best, axis=1)

            # Compute Spearman rank correlation between fitness and distance
            # (negative correlation = better fitness is closer to global best)
            if np.std(distances) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                corr = np.corrcoef(
                    np.argsort(np.argsort(self.current_fitness)),
                    np.argsort(np.argsort(distances))
                )[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0

        # Adaptive social coefficient based on rank correlation
        # High correlation -> trust social component more; low -> rely on exploration
        social = self.social_base * np.clip(1.0 + corr, 0.2, 1.5)
        cognitive = self.cognitive_base

        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)

        # Mutation injection strength inversely proportional to correlation
        # Low correlation = rugged landscape = more exploration needed
        mutation_strength = 0.3 * np.clip(1.0 - corr, 0.1, 1.0)

        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold

        # Generate mutation vectors from random distinct indices
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )

        # Apply mutation where active
        mutation_component = np.where(
            mutation_active,
            mutation_strength * (mutation_vectors - self.population),
            0.0
        )

        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )

        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_05_catE_idea_0.py (7 wins) ---
```python
def _velocity_update_batch(self):
        """Update velocities using k-NN graph topology and betweenness-based social routing."""
        cognitive, social = self._adaptive_coefficients()

        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # Build k-NN graph based on Euclidean distance in position space
        k = min(max(3, self.neighborhood_size), self.np - 1)

        # Compute pairwise squared distances (vectorized)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)  # Exclude self

        # Find k-nearest neighbors for each particle
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Build adjacency matrix
        adjacency = np.zeros((self.np, self.np), dtype=np.float64)
        for i in range(self.np):
            adjacency[i, knn_indices[i]] = 1.0
            adjacency[knn_indices[i], i] = 1.0

        # Compute degree centrality
        degree = np.sum(adjacency, axis=1)
        degree_norm = degree / (degree.max() + 1e-10)

        # Compute Fiedler eigenvalue (algebraic connectivity) via power iteration approximation
        # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        d_sqrt_inv = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
        laplacian = np.eye(self.np) - d_sqrt_inv @ adjacency @ d_sqrt_inv

        # Power iteration to find smallest non-zero eigenvalue (Fiedler)
        np.random.seed(42)  # Reproducible but still varying per call
        v = np.random.randn(self.np)
        v = v / (np.linalg.norm(v) + 1e-10)
        for _ in range(20):
            v = laplacian @ v
            v = v / (np.linalg.norm(v) + 1e-10)

        # Rayleigh quotient for Fiedler value estimate
        fiedler = np.abs(v @ laplacian @ v) / (np.dot(v, v) + 1e-10)
        connectivity_strength = np.clip(fiedler * 5.0, 0.1, 2.0)  # Low = fragmented

        # Compute betweenness centrality via approximate method (sample paths)
        betweenness = np.zeros(self.np)
        n_samples = min(50, self.np)
        sample_nodes = np.random.choice(self.np, n_samples, replace=False)

        for src in sample_nodes:
            # BFS from source
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

            # Count shortest paths through each node
            sigma = np.zeros(self.np)
            sigma[src] = 1
            for d in range(int(dist.max()) + 1) if dist.max() < np.inf else []:
                for node in np.where(dist == d)[0]:
                    for p in pred[node]:
                        sigma[node] += sigma[p]

            # Accumulate betweenness
            for node in range(self.np):
                if node != src and dist[node] < np.inf:
                    for p in pred[node]:
                        betweenness[node] += sigma[p] / (sigma[node] + 1e-10)

        betweenness_norm = betweenness / (betweenness.max() + 1e-10)

        # Graph-based social component: route influence through topology
        social_component = np.zeros((self.np, self.dim))

        for i in range(self.np):
            # Get neighborhood best from k-NN graph
            neighbor_best_idx = knn_indices[i][np.argmin(self.personal_best_fitness[knn_indices[i]])]
            neighbor_best_pos = self.personal_best[neighbor_best_idx]

            # Weight by degree and betweenness (bridge particles have more influence)
            influence_weight = 0.5 * (1.0 + degree_norm[i]) * (1.0 + betweenness_norm[i])

            # Scale social coefficient by connectivity
            social_scaled = social * connectivity_strength

            social_component[i] = social_scaled * r2[i] * influence_weight * (neighbor_best_pos - self.population[i])

        # Fragmentation rescue: if Fiedler is very low, pull toward global best
        if fiedler < 0.2:
            rescue_strength = np.clip((0.2 - fiedler) * 3.0, 0.0, 0.8)
            if self.global_best is not None:
                rescue = rescue_strength * (self.global_best - self.population)
            else:
                # Use centroid of best particles
                best_third = np.argsort(self.personal_best_fitness)[:max(1, self.np // 3)]
                centroid = np.mean(self.population[best_third], axis=0)
                rescue = rescue_strength * (centroid - self.population)
            social_component += rescue

        # DE/rand/1 mutation (reduced to avoid overwhelming graph-based dynamics)
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

        # Velocity update
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )

        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```

# --- From variant_10_catB_idea_0.py (10 wins) ---
```python
def _velocity_update_batch(self):
        """Update velocities with spectral-condition-guided anisotropic mutation."""
        cognitive, social = self._adaptive_coefficients()

        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)

        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)

        # Compute spectral condition number from centered population
        centered = self.population - np.mean(self.population, axis=0)

        # Use SVD for numerical stability (avoids covariance matrix issues)
        try:
            _, singular_values, right_sv = np.linalg.svd(centered, full_matrices=False)

            # Condition number as ratio of max to min singular value
            # High cond => population in low-rank subspace => needs more exploration
            # Low cond => well-distributed across dimensions
            cond = singular_values[0] / (singular_values[-1] + 1e-10)

            # Modulate mutation strength inversely with condition number
            # When cond is high (population collapsed), reduce mutation to avoid over-exploitation
            # When cond is low (good spread), can use stronger mutation
            mutation_strength = np.clip(0.5 / (1.0 + 0.1 * np.log1p(cond)), 0.1, 0.5)

            # Determine effective dimensionality from singular values
            total_variance = np.sum(singular_values ** 2) + 1e-10
            cumvar = np.cumsum(singular_values ** 2) / total_variance
            effective_dim = np.searchsorted(cumvar, 0.95) + 1

            # When population is in low effective dimension, inject velocity
            # orthogonal to principal axes to break anisotropy
            if effective_dim < self.dim * 0.5:
                # Project population onto principal subspace and reconstruct residual
                principal_axes = right_sv[:effective_dim].T  # (dim, effective_dim)

                # Compute component orthogonal to principal subspace
                parallel_component = self.population @ principal_axes @ principal_axes.T
                orthogonal_residual = self.population - parallel_component

                # Add scaled orthogonal component as "spectral mutation"
                orthogonal_strength = 0.3 * (1.0 - effective_dim / self.dim)
                mutation_component = orthogonal_strength * orthogonal_residual
            else:
                mutation_component = np.zeros((self.np, self.dim))

            # DE mutation with spectral-modulated strength
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

            # Combine spectral mutation and DE mutation
            total_mutation = mutation_component + de_component

        except np.linalg.LinAlgError:
            # Fallback to standard DE mutation on decomposition failure
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

        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            total_mutation
        )

        # Clip velocity to bounds
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
    Adaptive Topology Particle Swarm Optimizer with DE Mutation (ATPSO-DM).
    
    Combines:
    - Ring topology PSO with adaptive neighborhood size
    - Adaptive inertia weight based on swarm diversity
    - DE/rand/1 mutation injection for exploration
    - Adaptive cognitive/social coefficients
    - Diversity-guided restart mechanism
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
        """Adapt inertia weight based on swarm diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Low diversity: increase exploration
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            # High diversity: increase exploitation
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            # Normal range: gradual convergence
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
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
    
    def _velocity_update_batch(self):
        """Update velocities with adaptive components and DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        # Generate random matrices once for efficiency
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        # Cognitive component: attraction to personal best
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        # Social component: attraction to local best (ring topology)
        social_component = social * r2 * (self.local_best - self.population)
        
        # DE/rand/1 mutation injection for enhanced exploration
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)  # Decreases over time
        mutation_active = mutation_mask < mutation_threshold
        
        # Generate mutation vectors from random distinct indices
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        # Apply mutation where active
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        # Velocity update with inertia, cognitive, social, and mutation
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        # Clip velocity to bounds
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
        # Compute population covariance for spectral analysis
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)

        # Eigendecomposition for spectral scaling
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute condition-number-based scaling (stuck populations have high condition number)
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)

            # Project velocity onto principal axes and scale
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale

            # Transform back to original space
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            # Fallback to standard update on decomposition failure
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            # Reinitialize worst 30% of particles
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            # Generate new random particles
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            # Reset fitness for replaced particles
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            # Perturb global best slightly
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
        Run the optimizer.
        
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
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        # Handle truncated initial evaluation
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        # Check initial stopping condition
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Main optimization loop
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            # Check for budget exhaustion
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Update bests
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            
            # Restart check
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()

```