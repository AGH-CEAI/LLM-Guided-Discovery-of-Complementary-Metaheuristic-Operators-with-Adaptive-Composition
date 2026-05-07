Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_temporal_drift` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.505145e-01            1.752734e-01            1.778750e-01            1.654981e-01            1.624475e-01            1.731966e-01            1.576429e-01            1.606246e-01            1.678867e-01            1.678821e-01            
1      3.536913e+00            2.701947e+00            3.489526e+00            3.286417e+00            3.322968e+00            4.866522e+00            4.231380e+00            3.964122e+00            2.513947e+00            3.741425e+00            
2      4.333476e-01            4.295295e-01            4.635850e-01            5.196284e-01            4.968021e-01            4.924369e-01            4.880118e-01            4.328523e-01            4.185369e-01            4.410849e-01            
3      6.542038e+00            6.582139e+00            7.049976e+00            7.344076e+00            1.073227e+01            7.485550e+00            6.752729e+00            7.216222e+00            6.664202e+00            6.472042e+00            
4      1.247580e+00            1.263876e+00            1.223060e+00            1.173089e+00            1.194369e+00            1.210147e+00            1.155502e+00            1.239728e+00            1.269124e+00            1.154683e+00            
5      4.404180e+01            5.809222e+01            1.587539e+02            5.295562e+01            7.889169e+01            1.550843e+02            8.510584e+01            2.101318e+01            1.956844e+01            3.426759e+01            
6      6.992797e+02            7.380886e+02            6.943043e+02            7.228399e+02            6.886570e+02            7.273793e+02            7.109751e+02            7.076340e+02            7.181303e+02            7.115060e+02            
7      1.088133e+01            1.115882e+01            1.225299e+01            1.503289e+01            1.592422e+01            1.501009e+01            1.384936e+01            1.348630e+01            1.075375e+01            1.314576e+01            
8      2.828104e+00            3.053658e+00            2.931531e+00            2.963697e+00            3.108149e+00            3.033065e+00            2.924451e+00            3.065612e+00            2.977160e+00            2.898859e+00            
9      1.077978e+02            9.793100e+01            1.053695e+02            1.037608e+02            1.032803e+02            1.053743e+02            1.068572e+02            1.028551e+02            1.064435e+02            1.083463e+02            
10     1.788038e+01            1.606305e+01            1.775826e+01            1.865667e+01            1.941639e+01            2.456628e+01            1.990771e+01            1.847321e+01            1.711819e+01            1.588236e+01            
11     1.651066e+02            2.248841e+02            2.138644e+02            2.296185e+02            1.983360e+02            2.432968e+02            2.054320e+02            2.007445e+02            2.095083e+02            2.031824e+02            
12     6.510743e+01            9.629016e+01            5.576729e+01            5.689602e+01            6.955195e+01            5.573914e+01            5.486124e+01            6.092770e+01            5.956372e+01            6.164732e+01            
13     1.398979e+01            2.188108e+01            1.882292e+01            2.108226e+01            2.209962e+01            2.274980e+01            2.152976e+01            2.084509e+01            1.403915e+01            2.038677e+01            
14     8.606972e+00            1.030652e+01            1.026651e+01            1.026513e+01            1.017653e+01            1.018538e+01            1.050150e+01            1.020941e+01            8.839393e+00            1.049209e+01            
15     3.521978e+00            3.672619e+00            3.737779e+00            3.690348e+00            3.745741e+00            3.559473e+00            3.590618e+00            3.759453e+00            3.678244e+00            3.639762e+00            
16     1.757107e+03            2.402458e+03            2.263317e+03            2.404259e+03            2.143701e+03            2.283743e+03            2.543989e+03            2.382693e+03            1.139412e+03            2.331897e+03            
17     5.054774e+04            5.570061e+04            6.181312e+04            6.053165e+04            4.270925e+04            5.553997e+04            6.438293e+04            6.206846e+04            5.411050e+04            4.758001e+04            
18     4.589434e+01            4.993423e+01            5.164227e+01            5.032408e+01            4.700035e+01            4.859986e+01            4.857112e+01            5.085838e+01            5.148248e+01            5.026023e+01            
19     1.222706e+02            1.557165e+02            1.673543e+02            1.652347e+02            1.533590e+02            1.557524e+02            1.655199e+02            1.619416e+02            1.579772e+02            1.719295e+02            
20     2.634984e+01            2.533392e+01            2.458444e+01            2.608919e+01            2.478048e+01            2.572064e+01            2.585734e+01            2.529263e+01            2.377158e+01            2.526475e+01            
21     5.704388e+00            5.667760e+00            5.470070e+00            5.600544e+00            5.469269e+00            5.438520e+00            5.567889e+00            5.543234e+00            5.468575e+00            5.502501e+00            
22     1.305462e+01            1.374280e+01            1.387868e+01            1.387682e+01            1.418285e+01            1.372823e+01            1.375784e+01            1.348600e+01            1.393308e+01            1.392889e+01            
23     7.059281e+01            6.908859e+01            7.153962e+01            6.834571e+01            6.741661e+01            6.866903e+01            7.088539e+01            6.971675e+01            6.913598e+01            7.131006e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.505145e-01)
Task  1: variant_08_catH_idea_0.py  (error=2.513947e+00)
Task  2: variant_08_catH_idea_0.py  (error=4.185369e-01)
Task  3: variant_10_catB_idea_0.py  (error=6.472042e+00)
Task  4: variant_10_catB_idea_0.py  (error=1.154683e+00)
Task  5: variant_08_catH_idea_0.py  (error=1.956844e+01)
Task  6: variant_04_catD_idea_0.py  (error=6.886570e+02)
Task  7: variant_08_catH_idea_0.py  (error=1.075375e+01)
Task  8: original.py  (error=2.828104e+00)
Task  9: variant_01_catA_idea_0.py  (error=9.793100e+01)
Task 10: variant_10_catB_idea_0.py  (error=1.588236e+01)
Task 11: original.py  (error=1.651066e+02)
Task 12: variant_06_catF_idea_0.py  (error=5.486124e+01)
Task 13: original.py  (error=1.398979e+01)
Task 14: original.py  (error=8.606972e+00)
Task 15: original.py  (error=3.521978e+00)
Task 16: variant_08_catH_idea_0.py  (error=1.139412e+03)
Task 17: variant_04_catD_idea_0.py  (error=4.270925e+04)
Task 18: original.py  (error=4.589434e+01)
Task 19: original.py  (error=1.222706e+02)
Task 20: variant_08_catH_idea_0.py  (error=2.377158e+01)
Task 21: variant_05_catE_idea_0.py  (error=5.438520e+00)
Task 22: original.py  (error=1.305462e+01)
Task 23: variant_04_catD_idea_0.py  (error=6.741661e+01)

WIN COUNTS:
  original.py: 9 wins
  variant_08_catH_idea_0.py: 6 wins
  variant_10_catB_idea_0.py: 3 wins
  variant_04_catD_idea_0.py: 3 wins
  variant_01_catA_idea_0.py: 1 wins
  variant_06_catF_idea_0.py: 1 wins
  variant_05_catE_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   original.py                     1.5051e-01       variant_06_catF_idea_0.py        1.5764e-01         1.0×    1.1×
     1   variant_08_catH_idea_0.py       2.5139e+00       variant_01_catA_idea_0.py        2.7019e+00         1.1×    1.4×
     2   variant_08_catH_idea_0.py       4.1854e-01       variant_01_catA_idea_0.py        4.2953e-01         1.0×    1.1×
     3   variant_10_catB_idea_0.py       6.4720e+00       original.py                      6.5420e+00         1.0×    1.1×
     4   variant_10_catB_idea_0.py       1.1547e+00       variant_06_catF_idea_0.py        1.1555e+00         1.0×    1.1×
     5   variant_08_catH_idea_0.py       1.9568e+01       variant_07_catG_idea_0.py        2.1013e+01         1.1×    3.0×
     6   variant_04_catD_idea_0.py       6.8866e+02       variant_02_catB_idea_0.py        6.9430e+02         1.0×    1.0×
     7   variant_08_catH_idea_0.py       1.0754e+01       original.py                      1.0881e+01         1.0×    1.3×
     8   original.py                     2.8281e+00       variant_10_catB_idea_0.py        2.8989e+00         1.0×    1.1×
     9   variant_01_catA_idea_0.py       9.7931e+01       variant_07_catG_idea_0.py        1.0286e+02         1.1×    1.1×
    10   variant_10_catB_idea_0.py       1.5882e+01       variant_01_catA_idea_0.py        1.6063e+01         1.0×    1.2×
    11   original.py                     1.6511e+02       variant_04_catD_idea_0.py        1.9834e+02         1.2×    1.3×
    12   variant_06_catF_idea_0.py       5.4861e+01       variant_05_catE_idea_0.py        5.5739e+01         1.0×    1.1×
    13   original.py                     1.3990e+01       variant_08_catH_idea_0.py        1.4039e+01         1.0×    1.5×
    14   original.py                     8.6070e+00       variant_08_catH_idea_0.py        8.8394e+00         1.0×    1.2×
    15   original.py                     3.5220e+00       variant_05_catE_idea_0.py        3.5595e+00         1.0×    1.0×
    16   variant_08_catH_idea_0.py       1.1394e+03       original.py                      1.7571e+03         1.5×    2.0×
    17   variant_04_catD_idea_0.py       4.2709e+04       variant_10_catB_idea_0.py        4.7580e+04         1.1×    1.3×
    18   original.py                     4.5894e+01       variant_04_catD_idea_0.py        4.7000e+01         1.0×    1.1×
    19   original.py                     1.2227e+02       variant_04_catD_idea_0.py        1.5336e+02         1.3×    1.3×
    20   variant_08_catH_idea_0.py       2.3772e+01       variant_02_catB_idea_0.py        2.4584e+01         1.0×    1.1×
    21   variant_05_catE_idea_0.py       5.4385e+00       variant_08_catH_idea_0.py        5.4686e+00         1.0×    1.0×
    22   original.py                     1.3055e+01       variant_07_catG_idea_0.py        1.3486e+01         1.0×    1.1×
    23   variant_04_catD_idea_0.py       6.7417e+01       variant_03_catC_idea_0.py        6.8346e+01         1.0×    1.0×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 7 (original.py, variant_01_catA_idea_0.py, variant_04_catD_idea_0.py, variant_05_catE_idea_0.py, variant_06_catF_idea_0.py, variant_08_catH_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 0  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (1 wins) ---
```python
def _position_update_temporal_drift(self):
        """Geometry-based temporal drift modulation (Category A).

        Uses literal geometric layout: centroid drift magnitude, convex hull
        volume/area, pairwise distance distribution, axis-aligned spread, and
        k-NN connected components. Detects convergence from population geometry
        rather than spectral decomposition.
        """
        # --- Centroid temporal drift ---
        current_centroid = np.mean(self.population, axis=0)
        if not hasattr(self, '_prev_centroid'):
            self._prev_centroid = current_centroid.copy()
            self._centroid_drift_mag = 0.0
            self._centroid_ema = 0.0
        else:
            raw_drift = np.linalg.norm(current_centroid - self._prev_centroid)
            self._centroid_drift_mag = raw_drift
            self._prev_centroid = current_centroid.copy()
            self._centroid_ema = 0.7 * self._centroid_ema + 0.3 * raw_drift

        # --- Pairwise distance statistics ---
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        pairwise_dists = np.sqrt(sq_dists)
        valid_dists = pairwise_dists[pairwise_dists < np.inf]

        if len(valid_dists) > 0:
            mean_dist = np.mean(valid_dists)
            std_dist = np.std(valid_dists) + 1e-10
            dist_cv = std_dist / mean_dist  # coefficient of variation
        else:
            mean_dist, std_dist, dist_cv = 1.0, 1.0, 1.0

        # --- Axis-aligned spread per dimension ---
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)
        axis_spreads = maxs - mins + 1e-10
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads)
        spread_ratio = min_spread / max_spread  # low = anisotropic collapse

        # --- k-NN connected components (geometric clustering) ---
        k = min(5, self.np - 1)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        visited = np.zeros(self.np, dtype=bool)
        n_components = 0
        component_sizes = []
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
            n_components += 1
            component_sizes.append(len(component))

        component_sizes = np.array(component_sizes)
        largest_component_ratio = np.max(component_sizes) / self.np if len(component_sizes) > 0 else 1.0
        fragmentation = 1.0 - largest_component_ratio  # 0 = unified, 1 = fully fragmented

        # --- Convex hull approximation using axis-aligned bounding box ---
        hull_volume = np.prod(axis_spreads)
        hull_volume_log = np.log(hull_volume + 1e-10)
        hull_volume_ema_key = '_hull_volume_ema'
        if not hasattr(self, hull_volume_ema_key):
            setattr(self, hull_volume_ema_key, hull_volume_log)
        else:
            setattr(self, hull_volume_ema_key, 0.9 * getattr(self, hull_volume_ema_key) + 0.1 * hull_volume_log)
        hull_shrink_rate = hull_volume_log - getattr(self, hull_volume_ema_key)

        # --- Geometric convergence signal ---
        # Low centroid drift + high fragmentation + shrinking hull = converged
        drift_signal = np.clip(self._centroid_ema / (mean_dist + 1e-10), 0.0, 2.0)
        spread_signal = np.clip(spread_ratio, 0.0, 1.0)
        component_signal = 1.0 - fragmentation
        hull_signal = np.clip(-hull_shrink_rate / 10.0, 0.0, 1.0)  # shrinking hull = convergence

        # Combined geometric signal: high when population is scattered/exploring, low when converged
        geo_signal = (drift_signal * 0.25 + spread_signal * 0.25 + 
                      component_signal * 0.25 + hull_signal * 0.25)

        # --- Temporal scale from geometric state ---
        if self._centroid_ema > mean_dist * 0.05:
            base_scale = 1.2  # population moving → can explore more
        elif hull_shrink_rate < -0.01:
            base_scale = 0.7  # hull shrinking rapidly → converged, slow down
        else:
            base_scale = 1.0

        temporal_scale = base_scale * (0.8 + 0.4 * geo_signal)

        # --- Per-particle scaling from local density ---
        knn_avg_dist = np.mean(pairwise_dists[np.arange(self.np)[:, None], knn_indices], axis=1, keepdims=True) + 1e-10
        density_factor = np.clip(knn_avg_dist / (mean_dist + 1e-10), 0.5, 2.0)

        # --- Directional correction toward centroid ---
        to_centroid = current_centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_dist
        centroid_pull = 0.2 * to_centroid_dir * density_factor

        # --- Apply position update ---
        new_population = (self.population + 
                          self.inertia_weight * self.velocity * temporal_scale * density_factor + 
                          centroid_pull)

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_04_catD_idea_0.py (3 wins) ---
```python
def _position_update_temporal_drift(self):
        """Fitness-landscape rank-based velocity modulation (Category D).

        Uses fitness ranks, success-history, and fitness-distance correlation
        to modulate velocity. No raw distances or covariance - only fitness signals.
        Targets worst tasks (17, 16, 6, 11) with large deceptive basins where
        population may collapse to local optima.
        """
        # 1. Fitness ranks (0 = best, 1 = worst)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # 2. Success history with exponential decay
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness > self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # 3. Fitness-percentile velocity scaling
        # Worse particles (higher rank) get more exploration velocity
        vel_scale = 0.5 + 1.5 * fitness_ranks
        vel_scale = np.clip(vel_scale, 0.5, 2.0)

        # 4. Fitness-distance correlation (FDC) as landscape ruggedness signal
        # Compute rank-based correlation between fitness and distance to global best
        if self.global_best is not None and self.np > 2:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0

        # 5. FDC-driven exploration factor
        # High FDC (>0) = smooth landscape = more exploitation
        # Low/negative FDC = rugged/deceptive = more exploration
        fdc_explore_factor = np.clip(2.0 - fdc, 0.5, 2.5)

        # 6. Success-based boost for recently improved particles
        success_boost = 1.0 + 0.5 * success_norm
        success_boost = np.clip(success_boost, 1.0, 2.0)

        # 7. Combined scale
        combined_scale = vel_scale * fdc_explore_factor * success_boost
        combined_scale = np.clip(combined_scale, 0.3, 3.0)

        # 8. Compute fitness gradient direction
        # Direction toward centroid of top 20% fittest particles
        top_k = max(1, self.np // 5)
        top_indices = np.argsort(self.current_fitness)[:top_k]
        top_centroid = np.mean(self.population[top_indices], axis=0)

        to_top = top_centroid - self.population
        to_top_dist = np.linalg.norm(to_top, axis=1, keepdims=True) + 1e-10
        to_top_dir = to_top / to_top_dist

        # Gradient strength proportional to fitness rank (worse particles get stronger pull)
        gradient_strength = 0.5 * fitness_ranks[:, np.newaxis]

        # 9. Position update
        scaled_velocity = self.velocity * combined_scale[:, np.newaxis]
        gradient_push = gradient_strength * to_top_dir
        random_perturb = fdc_explore_factor * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + scaled_velocity + gradient_push + random_perturb
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_05_catE_idea_0.py (1 wins) ---
```python
def _position_update_temporal_drift(self):
        """Graph Laplacian escape via topology-aware velocity modulation (Category E).

        Builds k-NN graph over population, analyzes connected components and
        spectral gap of normalized Laplacian to detect convergence/fragmentation,
        identifies bridge particles via centrality approximation, and modulates
        velocities per-particle based on graph topology. Targets worst tasks
        (17, 16, 6, 11) where population gets trapped in disconnected clusters.
        """
        try:
            # --- Build k-NN graph ---
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

            # --- Connected component analysis via BFS ---
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
            particle_component_size = np.array([
                component_sizes[np.argmax([i in c for c in components])]
                for i in range(self.np)
            ])

            # --- Normalized Laplacian spectral gap ---
            try:
                row_idx = np.repeat(np.arange(self.np), k)
                col_idx = knn_indices.ravel()
                data = np.ones(len(row_idx))
                adj = np.zeros((self.np, self.np))
                adj[row_idx, col_idx] = 1.0
                adj = np.clip(adj + adj.T, 0.0, 1.0)
                np.fill_diagonal(adj, 0.0)

                degrees = np.sum(adj, axis=1) + 1e-10
                d_inv_sqrt = 1.0 / np.sqrt(degrees)
                d_mat = np.diag(d_inv_sqrt)
                lap = np.eye(self.np) - d_mat @ adj @ d_mat

                eigenvalues = np.linalg.eigvalsh(lap)
                eigenvalues = np.sort(eigenvalues)
                spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
                spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
            except np.linalg.LinAlgError:
                spectral_gap = 1.0

            # --- Betweenness-centrality approximation ---
            centrality = np.zeros(self.np)
            for i in range(self.np):
                neighbors = knn_indices[i]
                neighbor_degrees = component_sizes[[np.argmax([n in c for c in components]) for n in neighbors]]
                centrality[i] = np.sum(1.0 / (neighbor_degrees + 1e-10))
            centrality = centrality / (np.max(centrality) + 1e-10)

            # --- Modulation factors ---
            # Small spectral gap = fragmented/converged → more exploration
            explore_factor = 1.0 + 0.6 * (1.0 - spectral_gap)

            # Small component = isolated → stronger boost
            size_factor = np.clip(particle_component_size / max(1, self.np), 0.2, 1.0)
            isolation_boost = 1.5 * (1.0 - size_factor) + 1.0

            # High centrality = bridge particles → stabilize
            bridge_damp = 1.0 - 0.3 * centrality

            vel_scale = explore_factor * isolation_boost * bridge_damp
            vel_scale = np.clip(vel_scale, 0.3, 2.5)

            # --- Temporal drift EMA for base scaling ---
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_graph_ema_improvement'):
                self._graph_ema_improvement = 0.0
            self._graph_ema_improvement = 0.3 * improvement + 0.7 * self._graph_ema_improvement

            if self._graph_ema_improvement > 1e-6:
                temporal_scale = 1.2
            elif self._graph_ema_improvement > 1e-10:
                temporal_scale = 1.0
            else:
                temporal_scale = 0.7

            # --- Apply velocity update ---
            new_population = self.population + temporal_scale * self.velocity * vel_scale[:, np.newaxis]

            # --- Targeted perturbation for isolated particles ---
            if spectral_gap < 0.15:
                isolated = particle_component_size <= 2
                if np.any(isolated):
                    random_perturb = np.random.uniform(-1.0, 1.0, (self.np, self.dim))
                    new_population[isolated] += 0.4 * random_perturb[isolated]

        except Exception:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_06_catF_idea_0.py (1 wins) ---
```python
def _position_update_temporal_drift(self):
        """Centroid momentum with velocity autocorrelation stagnation detection (Category F).

        Tracks centroid trajectory across generations; autocorrelation of centroid
        velocity detects oscillation (sign of convergence stall). Cross-correlation
        between particle velocities and centroid drift detects cohesive vs scattered
        motion. Uses motion dynamics, not current population structure, to modulate
        velocity scaling and inject corrective impulses. Targets worst tasks (17, 16,
        6, 11) where premature convergence prevents reaching 1e-08.
        """
        try:
            centroid = np.mean(self.population, axis=0)

            # Initialize temporal state
            if not hasattr(self, '_temporal_centroid'):
                self._temporal_centroid = centroid.copy()
            if not hasattr(self, '_temporal_centroid_history'):
                self._temporal_centroid_history = []
            if not hasattr(self, '_temporal_vel_history'):
                self._temporal_vel_history = []

            # Centroid momentum: EMA of centroid position
            self._temporal_centroid = 0.7 * self._temporal_centroid + 0.3 * centroid

            # Compute centroid velocity (change in EMA position)
            if len(self._temporal_centroid_history) > 0:
                centroid_vel = self._temporal_centroid - self._temporal_centroid_history[-1]
            else:
                centroid_vel = np.zeros(self.dim)

            # Track velocity history for autocorrelation
            self._temporal_vel_history.append(centroid_vel)
            if len(self._temporal_vel_history) > 10:
                self._temporal_vel_history.pop(0)

            # Autocorrelation of centroid velocity: detects oscillation/stagnation
            if len(self._temporal_vel_history) >= 3:
                v1 = np.array(self._temporal_vel_history[-2])
                v2 = np.array(self._temporal_vel_history[-1])
                v1_norm = np.linalg.norm(v1) + 1e-10
                v2_norm = np.linalg.norm(v2) + 1e-10
                velocity_autocorr = np.dot(v1, v2) / (v1_norm * v2_norm)
            else:
                velocity_autocorr = 0.0

            # Cross-correlation: particle velocity alignment with centroid drift
            centroid_drift = centroid - self._temporal_centroid
            drift_norm = np.linalg.norm(centroid_drift) + 1e-10
            drift_dir = centroid_drift / drift_norm

            vel_alignment = np.dot(self.velocity, drift_dir) / (np.linalg.norm(self.velocity, axis=1) + 1e-10)
            cross_corr = np.mean(vel_alignment)

            # Update centroid history
            self._temporal_centroid_history.append(centroid.copy())
            if len(self._temporal_centroid_history) > 10:
                self._temporal_centroid_history.pop(0)

            # Temporal signal combining autocorrelation and cross-correlation
            autocorr_signal = np.clip(velocity_autocorr, -1.0, 1.0)
            cross_signal = np.clip(cross_corr, -1.0, 1.0)

            # Autocorrelation near +1 means oscillating in same direction → stagnation
            # Autocorrelation near -1 means reversing direction → bouncing in valley
            # Combined signal: low |autocorr| + high |cross_corr| = good cohesive convergence
            stagnation_risk = 0.5 * abs(autocorr_signal) - 0.5 * abs(cross_signal)
            stagnation_risk = np.clip(stagnation_risk, -1.0, 1.0)

            # EMA of improvement
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_temporal_ema_improvement'):
                self._temporal_ema_improvement = 0.0
            self._temporal_ema_improvement = 0.3 * improvement + 0.7 * self._temporal_ema_improvement

            # Base velocity scale from improvement EMA
            if self._temporal_ema_improvement > 1e-6:
                base_scale = 1.2
            elif self._temporal_ema_improvement > 1e-10:
                base_scale = 1.0
            else:
                base_scale = 0.7

            # Temporal modulation: stagnation risk reduces scale, high cross-corr boosts
            temporal_modulation = 1.0 + 0.4 * (1.0 - stagnation_risk) + 0.3 * cross_signal
            temporal_scale = base_scale * np.clip(temporal_modulation, 0.5, 2.0)

            # Corrective impulse when stagnation detected
            if stagnation_risk > 0.5:
                drift_rate = np.linalg.norm(centroid_drift)
                if drift_rate < 1e-4:
                    corrective_force = 0.3 * np.random.uniform(-1, 1, self.dim) * drift_norm
                else:
                    corrective_force = -0.2 * centroid_drift
            else:
                corrective_force = np.zeros(self.dim)

            new_population = self.population + temporal_scale * self.velocity + corrective_force
            self.population = self._clip_to_bounds(new_population)

        except Exception:
            new_population = self.population + self.velocity
            self.population = self._clip_to_bounds(new_population)
```

# --- From variant_08_catH_idea_0.py (6 wins) ---
```python
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
```

# --- From variant_10_catB_idea_0.py (3 wins) ---
```python
def _position_update_temporal_drift(self):
        """Directional alignment + subspace expansion (Category B).

        Computes alignment between each eigenvector direction and the global-best
        vector. Velocity components aligned with global-best get boosted; misaligned
        components get dampened. When condition number is high (population collapsed
        to subspace), minor eigenvector directions are amplified to re-expand the
        population. This is orthogonal to SVD whitening (which uses magnitude ratios)
        because it uses DIRECTIONAL alignment with the objective target.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = np.clip(eigenvalues, 1e-10, None)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            sv_scale = np.sqrt(eigenvalues + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)

            if self.global_best is not None:
                global_dir = self.global_best - np.mean(self.population, axis=0)
                global_dir_norm = np.linalg.norm(global_dir) + 1e-10
                global_dir = global_dir / global_dir_norm

                alignment = np.abs(eigenvectors.T @ global_dir)

                alignment_boost = 0.5 + 0.5 * alignment
                alignment_boost = np.clip(alignment_boost, 0.3, 1.5)
            else:
                alignment_boost = np.ones(self.dim)

            spectral_signal = np.clip(cond_log / 5.0, 0.0, 1.0)

            if spectral_signal > 0.3:
                minor_boost = 1.0 + 1.5 * spectral_signal * (1.0 - sv_norm)
                minor_boost = np.clip(minor_boost, 1.0, 3.0)
            else:
                minor_boost = np.ones(self.dim)

            per_comp_scale = alignment_boost * minor_boost
            per_comp_scale = np.clip(per_comp_scale, 0.3, 3.0)

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_comp_scale
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