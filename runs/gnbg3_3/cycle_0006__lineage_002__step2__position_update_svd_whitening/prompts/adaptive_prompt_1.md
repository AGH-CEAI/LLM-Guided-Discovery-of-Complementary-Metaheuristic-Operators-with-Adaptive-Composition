Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_svd_whitening` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.746969e-01            1.797634e-01            1.675346e-01            1.636542e-01            1.645219e-01            -inf                    1.688591e-01            1.815090e-01            1.732047e-01            1.795780e-01            1.732383e-01            
1      3.517951e+00            3.577583e+00            3.463111e+00            2.758538e+00            3.807898e+00            -inf                    3.149738e+00            3.425451e+00            3.396337e+00            3.140744e+00            2.675947e+00            
2      4.116799e-01            4.430807e-01            4.968402e-01            4.458920e-01            4.249657e-01            -inf                    3.861921e-01            4.046704e-01            4.087250e-01            3.898759e-01            3.890708e-01            
3      6.439517e+00            6.459453e+00            7.356818e+00            7.010274e+00            6.635629e+00            -inf                    6.802055e+00            6.691717e+00            6.651507e+00            6.428078e+00            6.818434e+00            
4      1.164510e+00            1.265259e+00            1.183754e+00            1.199043e+00            1.195021e+00            -inf                    1.215238e+00            1.242913e+00            1.230513e+00            1.308289e+00            1.206109e+00            
5      9.701416e+00            2.261902e+01            5.119520e+01            4.491760e+01            7.006945e+02            -inf                    8.633644e+01            2.574367e+01            3.688652e+01            2.203675e+02            3.295515e+01            
6      7.318297e+02            7.707790e+02            7.094679e+02            7.386713e+02            7.509895e+02            -inf                    7.829020e+02            7.605530e+02            7.124512e+02            7.790707e+02            7.656686e+02            
7      1.172967e+01            1.183766e+01            1.501526e+01            1.109817e+01            1.223720e+01            -inf                    1.236106e+01            1.232687e+01            1.137910e+01            1.143547e+01            1.168138e+01            
8      2.896967e+00            2.813357e+00            2.908574e+00            2.875991e+00            3.006811e+00            -inf                    2.952211e+00            2.976371e+00            3.065146e+00            3.106957e+00            3.007209e+00            
9      1.131047e+02            1.127778e+02            1.080074e+02            1.188914e+02            1.194988e+02            -inf                    1.134058e+02            1.147646e+02            1.076267e+02            1.121227e+02            1.101219e+02            
10     2.563100e+01            2.171411e+01            2.237406e+01            2.140570e+01            2.173385e+01            -inf                    2.264796e+01            2.050667e+01            2.174415e+01            2.089822e+01            2.362240e+01            
11     2.327755e+02            2.570171e+02            2.359480e+02            2.160898e+02            2.038083e+02            -inf                    2.359684e+02            2.190825e+02            2.383025e+02            2.282351e+02            2.230432e+02            
12     6.750187e+01            5.380691e+01            7.747383e+01            5.630568e+01            5.116564e+01            -inf                    6.339211e+01            7.869516e+01            7.109120e+01            5.449645e+01            6.393525e+01            
13     2.267144e+01            2.031238e+01            2.102290e+01            2.105261e+01            2.018372e+01            -inf                    2.172393e+01            2.157562e+01            2.143370e+01            1.989462e+01            2.264578e+01            
14     1.082220e+01            1.085025e+01            1.051728e+01            1.083749e+01            1.047330e+01            -inf                    1.078214e+01            1.021718e+01            1.057669e+01            1.071188e+01            1.049553e+01            
15     3.626828e+00            3.444477e+00            3.604435e+00            3.508839e+00            3.532417e+00            -inf                    3.520797e+00            3.645116e+00            3.730687e+00            3.422100e+00            3.647763e+00            
16     2.536155e+03            2.721647e+03            2.548652e+03            2.616996e+03            2.447028e+03            -inf                    2.635728e+03            2.930314e+03            2.694482e+03            2.883846e+03            2.655375e+03            
17     4.777815e+04            4.597229e+04            4.680497e+04            4.061381e+04            4.345166e+04            -inf                    5.582453e+04            5.945768e+04            5.914600e+04            3.512935e+04            4.988296e+04            
18     4.876153e+01            5.132833e+01            4.754128e+01            4.580819e+01            4.651831e+01            -inf                    4.634927e+01            4.819925e+01            4.835026e+01            4.561252e+01            4.781222e+01            
19     1.733395e+02            1.664953e+02            1.614558e+02            1.689649e+02            1.704494e+02            -inf                    1.726651e+02            1.642066e+02            1.855225e+02            1.720322e+02            1.774424e+02            
20     2.731258e+01            2.781739e+01            2.547614e+01            2.696854e+01            2.638036e+01            -inf                    2.601020e+01            2.677892e+01            2.611902e+01            2.628758e+01            2.766878e+01            
21     5.618457e+00            5.436455e+00            5.534474e+00            5.522967e+00            5.587729e+00            -inf                    5.616800e+00            5.631511e+00            5.509893e+00            5.663324e+00            5.491397e+00            
22     1.356750e+01            1.401637e+01            1.422414e+01            1.378445e+01            1.382441e+01            -inf                    1.356569e+01            1.399183e+01            1.358174e+01            1.407280e+01            1.452772e+01            
23     7.253302e+01            6.965317e+01            6.648304e+01            7.320508e+01            7.251372e+01            -inf                    7.037553e+01            7.196432e+01            7.127502e+01            6.982306e+01            7.383003e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_catC_idea_0.py  (error=1.636542e-01)
Task  1: variant_10_catB_idea_0.py  (error=2.675947e+00)
Task  2: variant_06_catF_idea_0.py  (error=3.861921e-01)
Task  3: variant_09_catA_idea_0.py  (error=6.428078e+00)
Task  4: original.py  (error=1.164510e+00)
Task  5: original.py  (error=9.701416e+00)
Task  6: variant_02_catB_idea_0.py  (error=7.094679e+02)
Task  7: variant_03_catC_idea_0.py  (error=1.109817e+01)
Task  8: variant_01_catA_idea_0.py  (error=2.813357e+00)
Task  9: variant_08_catH_idea_0.py  (error=1.076267e+02)
Task 10: variant_07_catG_idea_0.py  (error=2.050667e+01)
Task 11: variant_04_catD_idea_0.py  (error=2.038083e+02)
Task 12: variant_04_catD_idea_0.py  (error=5.116564e+01)
Task 13: variant_09_catA_idea_0.py  (error=1.989462e+01)
Task 14: variant_07_catG_idea_0.py  (error=1.021718e+01)
Task 15: variant_09_catA_idea_0.py  (error=3.422100e+00)
Task 16: variant_04_catD_idea_0.py  (error=2.447028e+03)
Task 17: variant_09_catA_idea_0.py  (error=3.512935e+04)
Task 18: variant_09_catA_idea_0.py  (error=4.561252e+01)
Task 19: variant_02_catB_idea_0.py  (error=1.614558e+02)
Task 20: variant_02_catB_idea_0.py  (error=2.547614e+01)
Task 21: variant_01_catA_idea_0.py  (error=5.436455e+00)
Task 22: variant_06_catF_idea_0.py  (error=1.356569e+01)
Task 23: variant_02_catB_idea_0.py  (error=6.648304e+01)

WIN COUNTS:
  variant_09_catA_idea_0.py: 5 wins
  variant_02_catB_idea_0.py: 4 wins
  variant_04_catD_idea_0.py: 3 wins
  variant_03_catC_idea_0.py: 2 wins
  variant_06_catF_idea_0.py: 2 wins
  original.py: 2 wins
  variant_01_catA_idea_0.py: 2 wins
  variant_07_catG_idea_0.py: 2 wins
  variant_10_catB_idea_0.py: 1 wins
  variant_08_catH_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_03_catC_idea_0.py       1.6365e-01       variant_04_catD_idea_0.py        1.6452e-01         1.0×    1.1×
     1   variant_10_catB_idea_0.py       2.6759e+00       variant_03_catC_idea_0.py        2.7585e+00         1.0×    1.3×
     2   variant_06_catF_idea_0.py       3.8619e-01       variant_10_catB_idea_0.py        3.8907e-01         1.0×    1.1×
     3   variant_09_catA_idea_0.py       6.4281e+00       original.py                      6.4395e+00         1.0×    1.0×
     4   original.py                     1.1645e+00       variant_02_catB_idea_0.py        1.1838e+00         1.0×    1.0×
     5   original.py                     9.7014e+00       variant_01_catA_idea_0.py        2.2619e+01         2.3×    4.6×
     6   variant_02_catB_idea_0.py       7.0947e+02       variant_08_catH_idea_0.py        7.1245e+02         1.0×    1.1×
     7   variant_03_catC_idea_0.py       1.1098e+01       variant_08_catH_idea_0.py        1.1379e+01         1.0×    1.1×
     8   variant_01_catA_idea_0.py       2.8134e+00       variant_03_catC_idea_0.py        2.8760e+00         1.0×    1.1×
     9   variant_08_catH_idea_0.py       1.0763e+02       variant_02_catB_idea_0.py        1.0801e+02         1.0×    1.1×
    10   variant_07_catG_idea_0.py       2.0507e+01       variant_09_catA_idea_0.py        2.0898e+01         1.0×    1.1×
    11   variant_04_catD_idea_0.py       2.0381e+02       variant_03_catC_idea_0.py        2.1609e+02         1.1×    1.1×
    12   variant_04_catD_idea_0.py       5.1166e+01       variant_01_catA_idea_0.py        5.3807e+01         1.1×    1.2×
    13   variant_09_catA_idea_0.py       1.9895e+01       variant_04_catD_idea_0.py        2.0184e+01         1.0×    1.1×
    14   variant_07_catG_idea_0.py       1.0217e+01       variant_04_catD_idea_0.py        1.0473e+01         1.0×    1.0×
    15   variant_09_catA_idea_0.py       3.4221e+00       variant_01_catA_idea_0.py        3.4445e+00         1.0×    1.1×
    16   variant_04_catD_idea_0.py       2.4470e+03       original.py                      2.5362e+03         1.0×    1.1×
    17   variant_09_catA_idea_0.py       3.5129e+04       variant_03_catC_idea_0.py        4.0614e+04         1.2×    1.4×
    18   variant_09_catA_idea_0.py       4.5613e+01       variant_03_catC_idea_0.py        4.5808e+01         1.0×    1.0×
    19   variant_02_catB_idea_0.py       1.6146e+02       variant_07_catG_idea_0.py        1.6421e+02         1.0×    1.1×
    20   variant_02_catB_idea_0.py       2.5476e+01       variant_06_catF_idea_0.py        2.6010e+01         1.0×    1.1×
    21   variant_01_catA_idea_0.py       5.4365e+00       variant_10_catB_idea_0.py        5.4914e+00         1.0×    1.0×
    22   variant_06_catF_idea_0.py       1.3566e+01       original.py                      1.3568e+01         1.0×    1.0×
    23   variant_02_catB_idea_0.py       6.6483e+01       variant_01_catA_idea_0.py        6.9653e+01         1.0×    1.1×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 10 (original.py, variant_01_catA_idea_0.py, variant_02_catB_idea_0.py, variant_03_catC_idea_0.py, variant_04_catD_idea_0.py, variant_06_catF_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 0  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (2 wins) ---
```python
def _position_update_svd_whitening(self):
        """Geometric pairwise distance + axis-aligned spread modulation (Category A).

        Operates on literal geometric layout: pairwise distance distribution,
        axis-aligned bounding box spread, and centroid distances. No spectral
        decomposition - purely geometric statistics of point configuration.
        """
        # Axis-aligned bounding box per dimension
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        axis_spread = pop_max - pop_min + 1e-10

        # Relative spread per dimension (normalized by search bounds)
        bounds_range = self.upper_bound - self.lower_bound + 1e-10
        rel_spread = axis_spread / bounds_range

        # Pairwise distance distribution statistics (geometric feature)
        # Sample pairs for efficiency on large populations
        n = self.np
        if n > 2:
            max_samples = min(500, n * (n - 1) // 2)
            sample_size = min(max_samples, 200)

            # Sample random pairs
            all_pairs = []
            for _ in range(sample_size):
                i, j = np.random.choice(n, 2, replace=False)
                all_pairs.append((i, j))

            distances = np.array([
                np.linalg.norm(self.population[i] - self.population[j])
                for i, j in all_pairs
            ])
            mean_pair_dist = np.mean(distances)
            std_pair_dist = np.std(distances)
        else:
            mean_pair_dist = bounds_range[0]
            std_pair_dist = 0.0

        # Per-particle centroid distance (geometric position)
        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10

        # Geometric modulation: axis-aligned spread per dimension
        # Low spread dimension = potentially collapsed = amplify velocity
        # High spread dimension = already explored = dampen velocity
        spread_modulation = np.clip(1.0 / (rel_spread + 0.05), 0.3, 3.0)

        # Geometric modulation: per-particle centroid proximity
        # Particles far from centroid in sparse regions get exploration boost
        # Particles near centroid in dense regions get dampening
        centroid_scale = np.clip(dist_to_centroid / mean_pair_dist, 0.2, 2.5)

        # Detect collapsed population (all points very close together)
        collapse_threshold = 0.1 * np.mean(bounds_range)
        is_collapsed = mean_pair_dist < collapse_threshold

        # Combine geometric signals
        per_component_scale = spread_modulation * centroid_scale

        if is_collapsed:
            # Isotropic expansion for collapsed population
            per_component_scale = np.clip(per_component_scale, 1.5, 3.0)
        else:
            per_component_scale = np.clip(per_component_scale, 0.3, 2.0)

        # Apply geometric modulation to velocity
        vel_scaled = self.velocity * per_component_scale
        new_population = self.population + vel_scaled

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_02_catB_idea_0.py (4 wins) ---
```python
def _position_update_svd_whitening(self):
        """Eigendecomposition-based whitened coordinate momentum (Category B).

        Key insight: transform velocity to eigenspace, apply momentum-based
        updates, and use condition number to aggressively modulate exploration.
        Targets extreme anisotropy in worst unsolved tasks (17, 16, 6).
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            eigenvalues = np.clip(eigenvalues, 1e-12, None)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)

            total_var = np.sum(eigenvalues) + 1e-10

            if not hasattr(self, '_prev_whitened_vel'):
                self._prev_whitened_vel = np.zeros((self.np, self.dim))

            vel_proj = self.velocity @ eigenvectors

            whiten_factors = np.sqrt(1.0 / eigenvalues + 1e-10)
            whiten_factors = whiten_factors / (np.max(whiten_factors) + 1e-10)

            vel_whitened = vel_proj * whiten_factors

            momentum_coeff = np.clip(0.3 + 0.5 * np.log1p(cond) / np.log1p(1e6), 0.3, 0.8)

            vel_with_momentum = (1.0 - momentum_coeff) * vel_whitened + momentum_coeff * self._prev_whitened_vel

            self._prev_whitened_vel = vel_with_momentum.copy()

            vel_unwhitened = vel_with_momentum / (whiten_factors + 1e-10)

            vel_scaled = vel_unwhitened * np.clip(cond / 100.0, 0.5, 2.5)

            new_population = self.population + vel_scaled

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_03_catC_idea_0.py (2 wins) ---
```python
def _position_update_svd_whitening(self):
        """Entropy-modulated velocity scaling (Category C: Information-theoretic).

        Uses non-parametric entropy estimation from k-NN distances to characterize
        population distribution. Low entropy = clustered = more exploration needed.
        High entropy = dispersed = exploitation-focused movement.
        """
        if self.np < 3:
            self.population = self._clip_to_bounds(self.population + self.velocity)
            return

        # Compute pairwise squared distances
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        distances = np.sqrt(sq_dists)

        k = min(5, self.np - 1)
        sorted_dists = np.sort(distances, axis=1)
        knn_dists = sorted_dists[:, :k]  # k-nearest neighbor distances

        # Entropy estimation via distance ratios (non-parametric)
        # Using ratio of successive NN distances as probability proxy
        entropy_signal = np.zeros(self.np)
        for i in range(self.np):
            d = knn_dists[i]
            if d[0] > 1e-15 and d[-1] > d[0]:
                ratios = d[1:k] / (d[0] + 1e-15)
                p = ratios / (np.sum(ratios) + 1e-15)
                p = np.clip(p, 1e-15, 1.0)
                entropy_signal[i] = -np.sum(p * np.log(p))

        # Normalize entropy to [0, 1] range
        max_entropy = np.log(k)
        entropy_norm = entropy_signal / (max_entropy + 1e-15)
        entropy_norm = np.clip(entropy_norm, 0.0, 1.0)

        # Velocity modulation: inverse relationship with entropy
        # Low entropy (clustered) -> high exploration scaling
        # High entropy (dispersed) -> low exploration scaling
        explore_scale = 1.0 + 1.5 * (1.0 - entropy_norm)
        explore_scale = np.clip(explore_scale, 0.5, 2.5)

        # Apply scaled velocity update
        scaled_velocity = self.velocity * explore_scale[:, np.newaxis]
        new_population = self.population + scaled_velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_04_catD_idea_0.py (3 wins) ---
```python
def _position_update_svd_whitening(self):
        """Fitness-landscape rank-based velocity modulation (Category D).

        Uses Spearman fitness-distance correlation to detect landscape smoothness,
        fitness percentiles for per-particle scaling, and success history for
        momentum. Target: tasks 17, 16, 6, 11, 19, 9 (worst unsolved).
        """
        # Compute fitness ranks (0=best, 1=worst)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Spearman-like fitness-distance correlation via Kendall tau proxy
        if self.global_best is not None and self.np > 3:
            # Distance to global best (only for ranking, not magnitude)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)

            # Kendall tau: count concordant vs discordant pairs
            n = self.np
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    fit_diff = (fitness_ranks[i] - fitness_ranks[j]) * (dist_ranks[i] - dist_ranks[j])
                    if fit_diff > 0:
                        concordant += 1
                    elif fit_diff < 0:
                        discordant += 1
            total_pairs = n * (n - 1) / 2
            kendall_tau = (concordant - discordant) / (total_pairs + 1e-10)
            fdc_signal = np.clip(kendall_tau, -1.0, 1.0)
        else:
            fdc_signal = 0.0

        # Fitness-distance correlation guides exploration/exploitation balance
        # High positive FDC = smooth landscape → exploit (lower scaling for top performers)
        # Low/negative FDC = rugged/deceptive → explore (boost poorly-ranked)
        exploit_weight = 0.5 + 0.5 * fdc_signal  # [0, 1]

        # Success history tracking
        if not hasattr(self, '_fitness_success_count'):
            self._fitness_success_count = np.zeros(self.np)

        improved = self.personal_best_fitness > self.current_fitness
        self._fitness_success_count[improved] += 1
        self._fitness_success_count[~improved] *= 0.9
        success_norm = self._fitness_success_count / (np.max(self._fitness_success_count) + 1.0)

        # Per-particle velocity scale based on fitness percentile
        # Top performers (low rank): moderate scale to refine
        # Middle: higher scale for active search
        # Bottom (high rank): highest scale to escape local optima
        vel_scale = np.ones(self.np)

        top_mask = fitness_ranks < 0.25
        mid_mask = (fitness_ranks >= 0.25) & (fitness_ranks < 0.75)
        bot_mask = fitness_ranks >= 0.75

        # Base scales differ by FDC-driven exploit_weight
        vel_scale[top_mask] = 0.6 + 0.4 * exploit_weight
        vel_scale[mid_mask] = 1.0 + 0.3 * (1.0 - exploit_weight)
        vel_scale[bot_mask] = 1.5 + 0.5 * (1.0 - exploit_weight)

        # Success history boosts: particles with recent wins get momentum
        vel_scale = vel_scale * (1.0 + 0.4 * success_norm)
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        # Fitness-guided directional perturbation (no raw distances)
        # Poor fitness → stronger pull toward better regions
        pull_strength = 0.3 * (fitness_ranks ** 1.5)  # Non-linear: worse particles get much stronger pull
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = pull_strength[:, np.newaxis] * to_best_dir
        else:
            # Fallback: random perturbation weighted by fitness rank
            directional = pull_strength[:, np.newaxis] * np.random.uniform(-1, 1, (self.np, self.dim))

        # Random perturbation scaled by fitness uncertainty (high rank = more noise)
        random_perturb = (0.2 + 0.4 * fitness_ranks[:, np.newaxis]) * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        # Combine: velocity scaling + directional pull + noise
        new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_06_catF_idea_0.py (2 wins) ---
```python
def _position_update_svd_whitening(self):
        """SVD-whitening with temporal collapse detection via EMA of condition number.

        Category F: Tracks condition number and singular value ratios ACROSS
        GENERATIONS using exponential moving averages. Detects when population
        is collapsing (rising EMA condition) vs converging well (falling EMA).
        Uses rate-of-change of EMA to modulate exploration vs exploitation.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            singular_values = np.clip(singular_values, 1e-10, None)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10

            # --- TEMPORAL: EMA of condition number across generations ---
            if not hasattr(self, '_ema_cond'):
                self._ema_cond = cond
            alpha_cond = 0.1
            self._ema_cond = alpha_cond * cond + (1 - alpha_cond) * self._ema_cond

            # Detect COLLAPSE: EMA condition is rising → population becoming anisotropic
            cond_ema_diff = cond - self._ema_cond
            is_collapsing = cond_ema_diff > 0.0

            # --- TEMPORAL: EMA of singular value ratios for degeneracy detection ---
            if len(singular_values) >= 2:
                sv_ratio = singular_values[-1] / (singular_values[0] + 1e-10)
                if not hasattr(self, '_ema_sv_ratio'):
                    self._ema_sv_ratio = sv_ratio
                alpha_sv = 0.05
                self._ema_sv_ratio = alpha_sv * sv_ratio + (1 - alpha_sv) * self._ema_sv_ratio
                # Degeneracy: ratio approaching 1 means collapsed distribution
                is_degenerate = self._ema_sv_ratio > 0.1
            else:
                is_degenerate = False

            # --- TEMPORAL: Rate of change of EMA (momentum of convergence) ---
            if not hasattr(self, '_prev_ema_cond'):
                self._prev_ema_cond = self._ema_cond
            cond_velocity = self._ema_cond - self._prev_ema_cond
            self._prev_ema_cond = self._ema_cond

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Inverse-square-law: stronger effect on collapsed directions
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            # --- TEMPORAL: Modulate based on collapse detection ---
            if is_collapsing:
                # Population collapsing: increase ALL directions for exploration
                collapse_boost = 1.0 + 0.5 * np.clip(cond_ema_diff / (self._ema_cond + 1e-10), 0.0, 2.0)
                per_component_scale *= collapse_boost
            elif is_degenerate:
                # Degenerate distribution: add isotropic exploration
                per_component_scale += 0.3

            # --- TEMPORAL: Momentum-based damping ---
            # If condition number is accelerating in either direction, dampen
            momentum_factor = 1.0 - 0.2 * np.sign(cond_velocity) * np.clip(abs(cond_velocity) / (self._ema_cond + 1e-10), 0.0, 1.0)
            per_component_scale *= np.clip(momentum_factor, 0.5, 1.5)

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_07_catG_idea_0.py (2 wins) ---
```python
def _position_update_svd_whitening(self):
        """Monte Carlo random subspace SVD for robust population whitening.

        Category G: Stochastic / sampling-based
        Uses multiple random orthogonal projections to compute SVD transformations
        in different subspaces, then averages the resulting velocity modulations.
        This stochastic approach provides robustness against population collapse
        and local optima traps on deceptive landscapes.
        """
        centered = self.population - np.mean(self.population, axis=0)
        n_mc_samples = 15  # Monte Carlo sampling count

        try:
            accumulated_velocity = np.zeros_like(self.velocity)

            for _ in range(n_mc_samples):
                # Generate random orthogonal projection matrix (Monte Carlo draw)
                R = np.random.randn(self.dim, self.dim)
                Q, _ = np.linalg.qr(R)

                # Project to random subspace
                proj_pop = centered @ Q
                proj_vel = self.velocity @ Q

                # SVD in projected subspace
                U, singular_values, Vt = np.linalg.svd(proj_pop, full_matrices=False)
                singular_values = np.clip(singular_values, 1e-10, None)

                if singular_values[0] < 1e-10:
                    continue

                sv_norm = singular_values / (singular_values[0] + 1e-10)

                # Inverse-square-law modulation in projected space
                spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
                spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

                cond = singular_values[0] / (singular_values[-1] + 1e-10)
                blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
                uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
                per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

                # Scale velocity in projected space
                vel_scaled = proj_vel * per_component_scale

                # Transform back to original space and accumulate
                accumulated_velocity += vel_scaled @ Q.T

            # Average across Monte Carlo samples
            avg_velocity = accumulated_velocity / n_mc_samples
            new_population = self.population + avg_velocity

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_08_catH_idea_0.py (1 wins) ---
```python
def _position_update_svd_whitening(self):
        """Hybrid: SVD spectral conditioning + graph-based clustering detection.

        Combines two distinct mechanisms:
          1) SVD whitening: counteracts population anisotropy via per-component scaling
          2) Graph clustering: k-NN connected components detect fragmentation/stagnation

        Weighting is principled:
          - Stagnation counter (data-driven) → switches between exploration and exploitation
          - k-NN component count (data-driven) → modulates spectral strength based on fragmentation

        Targets worst tasks (17, 16, 6, 11) which likely have both anisotropy and
        deceptive local optima that existing single-mechanism operators miss.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            # --- MECHANISM 1: SVD spectral conditioning ---
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Inverse-square-law spectral modulation
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            # --- MECHANISM 2: Graph-based clustering detection ---
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

            # Connected component analysis via BFS
            visited = np.zeros(self.np, dtype=bool)
            num_components = 0
            for start in range(self.np):
                if visited[start]:
                    continue
                stack = [start]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    for neighbor in knn_indices[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
                num_components += 1

            # Fragmentation signal: more components = more fragmented population
            fragmentation = num_components / max(1, self.np)
            fragmentation = np.clip(fragmentation, 0.0, 1.0)

            # --- PRINCIPLED WEIGHTING: stagnation-driven switching ---
            # Stagnation counter (data-driven) detects convergence stalls
            stagnation_signal = np.clip(np.log1p(self.stagnation_counter) / np.log1p(100), 0.0, 1.0)

            # When stagnant: boost exploration (higher spectral modulation, more uniform scaling)
            # When not stagnant: rely more on SVD's natural spectral scaling
            exploration_weight = stagnation_signal * 0.5 + 0.25

            # Fragmentation amplifies spectral modulation when population is fragmented
            cluster_scale = 1.0 + fragmentation * stagnation_signal

            # --- COMBINED SCALING ---
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = (
                uniform_scale * (1.0 - exploration_weight) + 
                spectral_modulation * exploration_weight * cluster_scale
            )

            # Stagnation-driven velocity magnitude boost for escaping local optima
            vel_magnitude_scale = 1.0 + stagnation_signal * 0.5
            vel_magnitude_scale = np.clip(vel_magnitude_scale, 1.0, 1.5)

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale * vel_magnitude_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_09_catA_idea_0.py (5 wins) ---
```python
def _position_update_svd_whitening(self):
        """Axis-aligned bounding box normalization + centroid-distance-weighted velocity.

        Category A geometry: uses literal spatial layout — per-dimension bounding box
        to detect anisotropy, then centroid distance to weight exploration. Different
        from SVD-based whitening which reasons about covariance/principal components.
        """
        # Compute axis-aligned bounding box per dimension
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        bb_range = pop_max - pop_min
        bb_range = np.clip(bb_range, 1e-10, None)

        # Condition from bounding box: ratio of max to min range
        bb_cond = np.max(bb_range) / (np.min(bb_range) + 1e-10)
        bb_cond = np.clip(bb_cond, 1.0, 1000.0)

        # Normalize each dimension by its bounding box range
        centered = self.population - np.mean(self.population, axis=0)
        bb_normalized = centered / bb_range

        # Compute centroid distances for each particle
        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        max_dist = np.max(dist_to_centroid) + 1e-10
        norm_dist = dist_to_centroid / max_dist

        # Particles far from centroid: amplify velocity (more exploration needed)
        # Particles near centroid: dampen velocity (exploit current region)
        # Use inverse of normalized distance so far particles get higher weights
        centroid_weights = 1.0 / (norm_dist + 0.1)
        centroid_weights = centroid_weights / (np.max(centroid_weights) + 1e-10)
        centroid_weights = np.clip(centroid_weights, 0.3, 2.5)

        # Compute axis-aligned spread factor per particle
        # High spread in a dimension → dampen velocity in that direction
        # Low spread → amplify
        per_dim_spread = bb_normalized / (np.std(bb_normalized, axis=0) + 1e-10)
        per_dim_spread = np.clip(np.abs(per_dim_spread), 0.0, 3.0)

        # Blend: per-particle centroid weights with per-dimension spread adjustment
        blend_weight = np.clip((bb_cond - 10.0) / 100.0, 0.0, 0.5)

        # Velocity scaling: combine centroid-distance weights with axis-aligned spread
        vel_scale = centroid_weights[:, np.newaxis] * (1.0 - blend_weight * per_dim_spread)
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        # Apply scaling to velocity
        new_population = self.population + vel_scale * self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_10_catB_idea_0.py (1 wins) ---
```python
def _position_update_svd_whitening(self):
        """Condition-adaptive rank-truncated whitening (variant_10, Category B).

        Targets worst unsolved tasks (17: 3.5e4, 16: 2.4e3, 6: 7e2) where
        severe anisotropy/ill-conditioning blocks convergence. Key differences:
        - Tikhonov regularization on eigendecomposition (stable pseudo-inverse)
        - Logarithmic spectral modulation (gentler than inverse-square)
        - Rank-adaptive damping based on effective dimensionality
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-14, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            if eigenvalues[-1] <= 0:
                eigenvalues[-1] = 1e-14

            cond = eigenvalues[0] / eigenvalues[-1]
            total_var = np.sum(eigenvalues) + 1e-10

            # Effective dimensionality: count components explaining 95% variance
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim = np.clip(eff_dim, 1, len(eigenvalues))

            # Rank-adaptive Tikhonov regularization
            # More regularization for lower-rank (more degenerate) populations
            rank_ratio = eff_dim / len(eigenvalues)
            reg_strength = 0.01 * (1.0 - rank_ratio) + 0.001
            reg_strength = np.clip(reg_strength, 0.001, 0.1)

            # Logarithmic spectral modulation (gentler than inverse-square)
            # Maps eigenvalue ratio to [0.5, 2.0] range with log scaling
            log_eig_ratio = np.log1p(eigenvalues / (eigenvalues[-1] + 1e-14))
            log_cond = np.log1p(cond)

            # Base modulation: inverse of normalized eigenvalue (gentle)
            base_modulation = 1.0 / (log_eig_ratio + 1.0)
            base_modulation = base_modulation / (np.max(base_modulation) + 1e-10)

            # Condition-adaptive blend: high cond → more aggressive modulation
            blend_weight = np.clip(log_cond / 10.0, 0.0, 0.8)

            # Uniform scale: dampen all directions when ill-conditioned
            uniform_scale = np.clip(1.0 / np.sqrt(log_cond + 1.0), 0.3, 1.5)

            # Per-component scale with log modulation
            per_component_scale = uniform_scale * (1.0 - blend_weight) + base_modulation * blend_weight

            # Additional damping for high-spread directions (prevent explosion)
            high_spread_mask = eigenvalues / eigenvalues[0] > 0.5
            per_component_scale[high_spread_mask] *= 0.7

            # Velocity modulation in eigenspace
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ eigenvectors.T)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

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