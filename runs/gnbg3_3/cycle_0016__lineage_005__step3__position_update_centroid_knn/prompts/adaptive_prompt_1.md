Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_centroid_knn` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.406534e-01            -inf                    2.569430e+00            1.149971e+00            3.522544e+01            1.581768e-01            1.535220e-01            9.501697e-01            1.746554e-01            8.518368e-01            6.798755e+01            
1      1.886416e+00            -inf                    4.728557e+00            5.947335e+00            7.271984e+01            3.382530e+00            2.000602e+00            4.292595e+00            1.840311e+00            4.164606e+00            1.322652e+02            
2      2.069233e-01            -inf                    3.028753e+01            1.193744e+01            3.345876e+02            2.173299e-01            1.656482e-01            1.375398e+01            5.141981e+00            8.737109e+00            5.943734e+02            
3      5.508458e+00            -inf                    2.634752e+01            2.664730e+01            1.734111e+02            6.001056e+00            5.800539e+00            2.350749e+01            9.746786e+00            2.198275e+01            2.588374e+02            
4      1.292211e+00            -inf                    2.116658e+00            1.989638e+00            4.507488e+00            1.347274e+00            1.292517e+00            1.895906e+00            1.393448e+00            1.849641e+00            5.319748e+00            
5      3.911267e+02            -inf                    1.016495e+04            1.424485e+03            5.663030e+02            3.833070e+03            1.994687e+03            2.828622e+03            2.370290e+03            1.831056e+02            1.099979e+04            
6      5.371716e+02            -inf                    6.462029e+02            6.412161e+02            3.731926e+03            5.611283e+02            5.080976e+02            4.765597e+02            6.207147e+02            4.350787e+02            6.442467e+03            
7      6.698704e+00            -inf                    1.931434e+01            2.210381e+01            1.950598e+02            7.183653e+00            7.014038e+00            1.750651e+01            8.189778e+00            1.606212e+01            2.962859e+02            
8      2.560489e+00            -inf                    8.263779e+00            8.467618e+00            3.363159e+01            2.834786e+00            2.798732e+00            7.310514e+00            3.837079e+00            6.739219e+00            4.654650e+01            
9      4.203225e+01            -inf                    6.152377e+01            5.448783e+01            2.534412e+02            4.788259e+01            4.756094e+01            3.991590e+01            4.044317e+01            4.925469e+01            3.112457e+02            
10     2.485380e-01            -inf                    1.381308e+02            4.992070e+00            8.999620e+01            3.122069e-01            2.830134e-01            3.382047e+00            6.015025e+00            2.670236e+00            7.360966e+01            
11     8.581919e+01            -inf                    1.874626e+02            1.569246e+02            3.001633e+02            9.787745e+01            1.043041e+02            1.050895e+02            1.178759e+02            1.216908e+02            3.224703e+02            
12     4.561047e-01            -inf                    6.132392e+01            2.877229e+00            6.671094e+01            6.444228e-01            3.841073e-01            5.389684e+00            4.883246e+00            3.783482e+00            5.586228e+01            
13     1.849799e+01            -inf                    3.745777e+01            2.478358e+01            3.500247e+01            2.147453e+01            2.410859e+01            1.965863e+01            2.241722e+01            1.841819e+01            4.074112e+01            
14     8.655709e+00            -inf                    6.403201e+00            6.130926e+00            1.499829e+01            7.896360e+00            7.855698e+00            7.842372e+00            8.165703e+00            8.797823e+00            1.537231e+01            
15     3.200086e+00            -inf                    3.209447e+00            3.155013e+00            3.357932e+00            2.982161e+00            3.155061e+00            3.070703e+00            2.956948e+00            3.237538e+00            3.171472e+00            
16     7.721051e+02            -inf                    6.867461e+03            9.865940e+02            9.843446e+03            7.031982e+02            1.267158e+03            8.670687e+02            1.084769e+03            9.310517e+02            1.051259e+04            
17     1.847027e+04            -inf                    2.033891e+04            2.630466e+04            2.020490e+04            2.524306e+04            3.259472e+04            2.828288e+04            2.756696e+04            1.579464e+04            2.631075e+04            
18     3.492771e+01            -inf                    6.064447e+01            3.967468e+01            4.507235e+01            4.041380e+01            3.853665e+01            3.987450e+01            3.855880e+01            3.935225e+01            4.528099e+01            
19     9.076116e+01            -inf                    1.292559e+02            6.204653e+01            3.103941e+02            8.160115e+01            7.908821e+01            6.339825e+01            7.651468e+01            7.293477e+01            3.249865e+02            
20     2.053786e+01            -inf                    3.935244e+01            2.191111e+01            4.351410e+01            2.063755e+01            2.058252e+01            2.079639e+01            2.204653e+01            2.221331e+01            4.435307e+01            
21     5.405128e+00            -inf                    5.726948e+00            4.955109e+00            5.689530e+00            5.538663e+00            5.401116e+00            5.410430e+00            5.539009e+00            5.710026e+00            5.730428e+00            
22     1.126015e+01            -inf                    1.677235e+01            1.155296e+01            1.698548e+01            1.209722e+01            1.067629e+01            1.196504e+01            1.235958e+01            1.057282e+01            1.666118e+01            
23     5.954521e+01            -inf                    4.533158e+01            5.583305e+01            9.667244e+01            5.729982e+01            6.002103e+01            5.604096e+01            6.062325e+01            6.012973e+01            1.024846e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.406534e-01)
Task  1: variant_08_catH_idea_0.py  (error=1.840311e+00)
Task  2: variant_06_catF_idea_0.py  (error=1.656482e-01)
Task  3: original.py  (error=5.508458e+00)
Task  4: original.py  (error=1.292211e+00)
Task  5: variant_09_catA_idea_0.py  (error=1.831056e+02)
Task  6: variant_09_catA_idea_0.py  (error=4.350787e+02)
Task  7: original.py  (error=6.698704e+00)
Task  8: original.py  (error=2.560489e+00)
Task  9: variant_07_catG_idea_0.py  (error=3.991590e+01)
Task 10: original.py  (error=2.485380e-01)
Task 11: original.py  (error=8.581919e+01)
Task 12: variant_06_catF_idea_0.py  (error=3.841073e-01)
Task 13: variant_09_catA_idea_0.py  (error=1.841819e+01)
Task 14: variant_03_catC_idea_0.py  (error=6.130926e+00)
Task 15: variant_08_catH_idea_0.py  (error=2.956948e+00)
Task 16: variant_05_catE_idea_0.py  (error=7.031982e+02)
Task 17: variant_09_catA_idea_0.py  (error=1.579464e+04)
Task 18: original.py  (error=3.492771e+01)
Task 19: variant_03_catC_idea_0.py  (error=6.204653e+01)
Task 20: original.py  (error=2.053786e+01)
Task 21: variant_03_catC_idea_0.py  (error=4.955109e+00)
Task 22: variant_09_catA_idea_0.py  (error=1.057282e+01)
Task 23: variant_02_catB_idea_0.py  (error=4.533158e+01)

WIN COUNTS:
  original.py: 9 wins
  variant_09_catA_idea_0.py: 5 wins
  variant_03_catC_idea_0.py: 3 wins
  variant_08_catH_idea_0.py: 2 wins
  variant_06_catF_idea_0.py: 2 wins
  variant_07_catG_idea_0.py: 1 wins
  variant_05_catE_idea_0.py: 1 wins
  variant_02_catB_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   original.py                     1.4065e-01       variant_06_catF_idea_0.py        1.5352e-01         1.1×    6.8×
     1   variant_08_catH_idea_0.py       1.8403e+00       original.py                      1.8864e+00         1.0×    2.3×
     2   variant_06_catF_idea_0.py       1.6565e-01       original.py                      2.0692e-01         1.2×   72.1×
     3   original.py                     5.5085e+00       variant_06_catF_idea_0.py        5.8005e+00         1.1×    4.3×
     4   original.py                     1.2922e+00       variant_06_catF_idea_0.py        1.2925e+00         1.0×    1.5×
     5   variant_09_catA_idea_0.py       1.8311e+02       original.py                      3.9113e+02         2.1×   12.9×
     6   variant_09_catA_idea_0.py       4.3508e+02       variant_07_catG_idea_0.py        4.7656e+02         1.1×    1.4×
     7   original.py                     6.6987e+00       variant_06_catF_idea_0.py        7.0140e+00         1.0×    2.6×
     8   original.py                     2.5605e+00       variant_06_catF_idea_0.py        2.7987e+00         1.1×    2.9×
     9   variant_07_catG_idea_0.py       3.9916e+01       variant_08_catH_idea_0.py        4.0443e+01         1.0×    1.2×
    10   original.py                     2.4854e-01       variant_06_catF_idea_0.py        2.8301e-01         1.1×   20.1×
    11   original.py                     8.5819e+01       variant_05_catE_idea_0.py        9.7877e+01         1.1×    1.4×
    12   variant_06_catF_idea_0.py       3.8411e-01       original.py                      4.5610e-01         1.2×   12.7×
    13   variant_09_catA_idea_0.py       1.8418e+01       original.py                      1.8498e+01         1.0×    1.3×
    14   variant_03_catC_idea_0.py       6.1309e+00       variant_02_catB_idea_0.py        6.4032e+00         1.0×    1.3×
    15   variant_08_catH_idea_0.py       2.9569e+00       variant_05_catE_idea_0.py        2.9822e+00         1.0×    1.1×
    16   variant_05_catE_idea_0.py       7.0320e+02       original.py                      7.7211e+02         1.1×    1.5×
    17   variant_09_catA_idea_0.py       1.5795e+04       original.py                      1.8470e+04         1.2×    1.7×
    18   original.py                     3.4928e+01       variant_06_catF_idea_0.py        3.8537e+01         1.1×    1.1×
    19   variant_03_catC_idea_0.py       6.2047e+01       variant_07_catG_idea_0.py        6.3398e+01         1.0×    1.3×
    20   original.py                     2.0538e+01       variant_06_catF_idea_0.py        2.0583e+01         1.0×    1.1×
    21   variant_03_catC_idea_0.py       4.9551e+00       variant_06_catF_idea_0.py        5.4011e+00         1.1×    1.1×
    22   variant_09_catA_idea_0.py       1.0573e+01       variant_06_catF_idea_0.py        1.0676e+01         1.0×    1.1×
    23   variant_02_catB_idea_0.py       4.5332e+01       variant_03_catC_idea_0.py        5.5833e+01         1.2×    1.3×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 8 (original.py, variant_02_catB_idea_0.py, variant_03_catC_idea_0.py, variant_05_catE_idea_0.py, variant_06_catF_idea_0.py, variant_07_catG_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 1  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 2 — variant_06_catF_idea_0.py (1.66e-01) vs original.py (2.07e-01) = 1.2× gap to runner-up, 72.1× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.86e-01, which is ~1.1× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_catB_idea_0.py (1 wins) ---
```python
def _position_update_centroid_knn(self):
        """Eigenvalue-driven anisotropic centroid correction (Category B).

        Uses eigendecomposition of population covariance to:
        1. Detect anisotropy via condition number (eigenvalue ratio)
        2. Modulate centroid attraction strength by condition number
        3. Project centroid direction onto principal subspace
        4. Weight projected direction by eigenvalue magnitude (emphasize major axes)
        5. Scale velocity per principal component proportional to eigenvalue

        Targets worst tasks (17, 16, 6, 5) with high anisotropy.
        """
        centroid = np.mean(self.population, axis=0)

        # --- Spectral decomposition of population covariance ---
        try:
            centered = self.population - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Condition number: primary anisotropy signal
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            # Effective dimensionality from cumulative eigenvalue variance
            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var
            cumvar = np.cumsum(eigenvalues_norm)
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            # Spectral signals
            cond_signal = np.log1p(cond) / np.log1p(10000.0)
            cond_signal = np.clip(cond_signal, 0.0, 1.0)

            # --- Centroid attraction modulated by spectral signals ---
            to_centroid = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist

            # Project direction onto principal subspace
            proj_coeffs = to_centroid_dir @ eigenvectors
            proj_coeffs_norm = proj_coeffs / (np.linalg.norm(proj_coeffs, axis=1, keepdims=True) + 1e-10)

            # Weight by eigenvalue magnitude (emphasize major axes)
            eigen_weights = eigenvalues / (eigenvalues[0] + 1e-10)
            projected_dir = proj_coeffs_norm * eigen_weights

            # Reconstruct aligned direction
            aligned_dir = projected_dir @ eigenvectors.T
            aligned_dir = aligned_dir / (np.linalg.norm(aligned_dir, axis=1, keepdims=True) + 1e-10)

            # Strength inversely proportional to effective dimensionality ratio
            # Low eff_dim_ratio = concentrated in few axes = stronger correction needed
            correction_strength = 0.5 * (1.0 + cond_signal) * (2.0 - eff_dim_ratio)
            correction_strength = np.clip(correction_strength, 0.2, 3.0)

            centroid_correction = correction_strength * aligned_dir

            # --- Velocity modulation in principal subspace ---
            vel_proj = self.velocity @ eigenvectors
            vel_scale = 0.5 + 0.5 * eigen_weights
            vel_scale = np.clip(vel_scale, 0.3, 1.5)
            vel_scaled = vel_proj * vel_scale

            new_population = self.population + self.inertia_weight * (vel_scaled @ eigenvectors.T) + 0.3 * centroid_correction

        except np.linalg.LinAlgError:
            to_centroid = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist
            new_population = self.population + self.inertia_weight * self.velocity + 0.3 * to_centroid_dir

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_03_catC_idea_0.py (3 wins) ---
```python
def _position_update_centroid_knn(self):
            """Gaussian likelihood entropy modulation (Category C: Information-theoretic).

            Treats population as a probability distribution. Fits a Gaussian to top
            performers, computes per-particle log-likelihood, measures entropy of the
            likelihood distribution as exploration signal. Low-likelihood particles
            in unexplored regions get stronger centroid pulls toward promising areas.
            """
            from scipy.stats import multivariate_normal

            # Fit Gaussian to top performers (distribution of good solutions)
            n_select = max(5, self.np // 4)
            top_indices = np.argsort(self.current_fitness)[:n_select]

            if n_select < 2:
                new_population = self.population + self.inertia_weight * self.velocity
                self.population = self._clip_to_bounds(new_population)
                return

            top_pop = self.population[top_indices]
            best_mean = np.mean(top_pop, axis=0)
            centered = top_pop - best_mean
            best_cov = np.cov(centered.T)
            if best_cov.ndim == 0:
                best_cov = best_cov.reshape(1, 1)
            best_cov = best_cov + 1e-6 * np.eye(self.dim)

            # Compute per-particle log-likelihood under fitted Gaussian
            log_likelihood = np.zeros(self.np)
            for i in range(self.np):
                diff = self.population[i] - best_mean
                try:
                    cov_inv = np.linalg.inv(best_cov)
                    log_det = np.log(np.linalg.det(best_cov) + 1e-10)
                    mahal = diff @ cov_inv @ diff
                    log_likelihood[i] = -0.5 * (mahal + log_det + self.dim * np.log(2 * np.pi))
                except:
                    log_likelihood[i] = -self.dim * 10.0

            # Compute entropy of likelihood distribution (information-theoretic signal)
            ll_min, ll_max = np.min(log_likelihood), np.max(log_likelihood)
            if ll_max > ll_min + 1e-10:
                norm_ll = (log_likelihood - ll_min) / (ll_max - ll_min + 1e-10)
            else:
                norm_ll = np.ones(self.np) * 0.5

            # Discretize for entropy estimation
            n_bins = min(10, max(3, self.np // 3))
            counts, _ = np.histogram(norm_ll, bins=n_bins, range=(0.0, 1.0))
            probs = counts / (self.np + 1e-10)
            probs = probs[probs > 0]
            if len(probs) > 1:
                likelihood_entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(probs))
                normalized_entropy = likelihood_entropy / (max_entropy + 1e-10)
            else:
                normalized_entropy = 0.5

            # Exploration signal from log-likelihood: low likelihood = unexplored = explore more
            likelihood = np.exp(log_likelihood - ll_max)
            exploration_factor = 1.0 / (likelihood + 1e-10)
            exploration_factor = np.clip(exploration_factor, 0.3, 5.0)

            # Modulate by entropy: high entropy = diverse distribution = more exploration globally
            global_explore = 0.5 + 0.5 * normalized_entropy
            exploration_factor *= global_explore

            # Direction toward best region (centroid of good solutions)
            to_best = best_mean - self.population
            to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_dist

            # Entropy-weighted correction toward promising distribution
            correction = 0.3 * exploration_factor[:, np.newaxis] * to_best_dir

            new_population = self.population + self.inertia_weight * self.velocity + correction
            self.population = self._clip_to_bounds(new_population)
```

# --- From variant_05_catE_idea_0.py (1 wins) ---
```python
def _position_update_centroid_knn(self):
        """MST betweenness-corrected centroid pull (Category E: Topology/graph-based).

        Builds Minimum Spanning Tree over population; uses graph centrality
        and MST edge weights to modulate per-particle centroid attraction.
        Peripheral/leaves get stronger pull; internal nodes get gentler nudge.
        """
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import pdist, squareform

        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

        # === STEP 1: Build distance matrix and MST ===
        if self.np > 2:
            # Compute pairwise distances
            try:
                pairwise_dists = squareform(pdist(self.population, metric='euclidean'))
                np.fill_diagonal(pairwise_dists, 1e-10)

                # Compute MST from distance matrix
                mst = minimum_spanning_tree(pairwise_dists)
                mst_dense = np.array(mst.todense())

                # Get MST edge weights
                mst_edges = np.triu(mst_dense)
                mst_edge_weights = mst_edges[mst_edges > 0]

                # Global MST tension (mean edge weight = typical inter-particle distance)
                global_mst_tension = np.mean(mst_edge_weights) + 1e-10
                max_mst_tension = np.max(mst_edge_weights) + 1e-10

                # === STEP 2: Compute per-particle MST centrality ===
                # Degree-based centrality: count connections in MST
                mst_degrees = np.array(np.sum(mst_dense > 0, axis=1)).ravel()

                # Betweenness approximation: particles with degree-1 (leaves) are on tree periphery
                # Higher degree = more central in MST topology
                leaf_mask = mst_degrees == 1

                # === STEP 3: MST edge weight per particle ===
                # For each particle, find its heaviest incident MST edge
                particle_mst_weight = np.zeros(self.np)
                for i in range(self.np):
                    incident_edges = mst_dense[i, :]
                    if np.any(incident_edges > 0):
                        particle_mst_weight[i] = np.max(incident_edges)
                    else:
                        particle_mst_weight[i] = global_mst_tension

                # Normalize: heavy edge = sparse region = needs more exploration
                edge_weight_modulation = np.clip(particle_mst_weight / global_mst_tension, 0.3, 3.0)

                # === STEP 4: Combine signals for correction strength ===
                # Leaf particles (degree=1): more centripetal pull needed
                # Internal particles (degree>1): gentler correction
                degree_modulation = np.where(leaf_mask, 1.5, 0.7)

                # Combined: edge weight × degree modulation
                mst_modulation = degree_modulation * edge_weight_modulation

            except Exception:
                # Fallback: uniform modulation
                mst_modulation = np.ones(self.np)
                global_mst_tension = 1.0
        else:
            mst_modulation = np.ones(self.np)
            global_mst_tension = 1.0

        # === STEP 5: k-NN density (for comparison/fusion) ===
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

        # Density from k-NN
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)

        # === STEP 6: Centroid attraction with MST + density fusion ===
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        # Fuse MST modulation with density modulation
        combined_modulation = mst_modulation[:, np.newaxis] * density_modulation
        combined_modulation = np.clip(combined_modulation, 0.2, 3.0)

        correction = centroid_attraction * to_centroid_dir * combined_modulation

        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_06_catF_idea_0.py (2 wins) ---
```python
def _position_update_centroid_knn(self):
        """Temporal stagnation-drift modulation (Category F).

        Tracks centroid movement ACROSS generations using EMA of velocity
        and k-NN density to detect stagnation/oscillation. Modulates
        correction strength dynamically rather than using static values.
        """
        centroid = np.mean(self.population, axis=0)

        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

        # --- TEMPORAL: Centroid velocity EMA (across generations) ---
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._centroid_vel_ema = 0.0
            self._prev_accel_sign = np.zeros(self.dim)
            self._knn_density_ema = 1.0
            self._oscillation_count = 0

        # Centroid velocity (instantaneous)
        centroid_vel = np.linalg.norm(centroid - self._ema_centroid)
        # EMA of centroid velocity
        self._centroid_vel_ema = 0.7 * self._centroid_vel_ema + 0.3 * centroid_vel
        self._ema_centroid = 0.7 * self._ema_centroid + 0.3 * centroid

        # Acceleration from velocity change
        acceleration = centroid_vel - self._centroid_vel_ema

        # --- TEMPORAL: k-NN density EMA ---
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        knn_density = np.mean(knn_avg_dist)

        # EMA of k-NN density
        self._knn_density_ema = 0.7 * self._knn_density_ema + 0.3 * knn_density

        # Rate of change of density (convergence detection)
        density_change = knn_density - self._knn_density_ema + 1e-10
        density_rate = density_change / (self._knn_density_ema + 1e-10)

        # --- TEMPORAL: Stagnation detection via centroid velocity EMA ---
        stagnation_signal = np.clip(1.0 - self._centroid_vel_ema / (1.0 + self._centroid_vel_ema), 0.0, 1.0)

        # --- TEMPORAL: Oscillation detection via acceleration sign changes ---
        curr_accel_sign = np.sign(acceleration) if centroid_vel > 1e-10 else np.zeros(self.dim)
        oscillation = np.mean(curr_accel_sign != self._prev_accel_sign) / max(1, self.dim)
        self._prev_accel_sign = curr_accel_sign

        # --- TEMPORAL: Convergence/divergence signals from density ---
        is_converging = knn_density < 0.8 * self._knn_density_ema
        is_diverging = knn_density > 1.2 * self._knn_density_ema

        # --- TEMPORAL: Adaptive correction strength ---
        base_correction = 0.3
        stagnation_boost = 0.2 * stagnation_signal
        oscillation_damping = 0.5 if oscillation > 0.3 else 1.0
        convergence_modulation = 0.7 if is_converging else (1.3 if is_diverging else 1.0)
        correction_strength = base_correction * (1.0 + stagnation_boost) * oscillation_damping * convergence_modulation

        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        # Density modulation enhanced with temporal rate
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
        density_modulation = density_modulation * (1.0 + 0.2 * density_rate)

        correction = centroid_attraction * to_centroid_dir * density_modulation

        new_population = self.population + self.inertia_weight * self.velocity + correction_strength * correction

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_07_catG_idea_0.py (1 wins) ---
```python
def _position_update_centroid_knn(self):
        """Monte Carlo importance-weighted centroid + random projection exploration (Category G).

        Fundamentally stochastic: uses Monte Carlo sampling of candidate directions,
        importance-weighted by estimated fitness improvement, with random projection
        exploration to escape local optima. Different from deterministic centroid_knn.
        """
        n_projections = 20
        n_importance_samples = 15

        # === MONTE CARLO: Random projection exploration ===
        # Sample random directions in latent space
        random_dirs = np.random.randn(self.dim, n_projections)
        random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=0, keepdims=True) + 1e-10)

        # Evaluate each projection: correlation between position-projection and fitness
        proj_scores = np.zeros(n_projections)
        for j in range(n_projections):
            proj = self.population @ random_dirs[:, j]
            if np.std(proj) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                proj_scores[j] = abs(np.corrcoef(proj, self.current_fitness)[0, 1])
            else:
                proj_scores[j] = 0.0

        # Normalize projection scores to weights
        proj_weights = np.exp(proj_scores * 5.0)
        proj_weights = proj_weights / (np.sum(proj_weights) + 1e-10)

        # Sample a random projection direction weighted by its score
        selected_proj_idx = np.random.choice(n_projections, p=proj_weights)
        stochastic_dir = random_dirs[:, selected_proj_idx]

        # === IMPORTANCE-WEIGHTED CENTROID ESTIMATION ===
        # Weight particles by fitness: better fitness = higher weight
        min_fit = np.min(self.current_fitness)
        max_fit = np.max(self.current_fitness)
        fit_range = max_fit - min_fit + 1e-10

        # Use inverse fitness (minimize) for importance weights
        if fit_range > 1e-10:
            inv_fitness = max_fit - self.current_fitness + 1e-10
            importance_weights = inv_fitness / (np.sum(inv_fitness) + 1e-10)
        else:
            importance_weights = np.ones(self.np) / self.np

        # Monte Carlo importance sampling of the centroid
        sampled_centroids = []
        for _ in range(n_importance_samples):
            sampled_idx = np.random.choice(self.np, size=self.np, replace=True, p=importance_weights)
            sampled_centroid = np.mean(self.population[sampled_idx], axis=0)
            sampled_centroids.append(sampled_centroid)
        sampled_centroids = np.array(sampled_centroids)

        # Select one centroid stochastically (uniform for now)
        selected_centroid = sampled_centroids[np.random.randint(n_importance_samples)]

        # === COMBINE DETERMINISTIC CENTROID ATTRACTION WITH STOCHASTIC EXPLORATION ===
        centroid = np.mean(self.population, axis=0)
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        # Project stochastic direction onto particle-wise directions
        stochastic_dir_expanded = stochastic_dir[np.newaxis, :]
        proj_onto_particle = np.sum(stochastic_dir_expanded * to_centroid_dir, axis=1, keepdims=True)
        stochastic_component = stochastic_dir_expanded * proj_onto_particle

        # Combine: 70% importance-weighted centroid + 30% stochastic projection
        alpha = np.random.uniform(0.6, 0.8, (self.np, 1))
        combined_dir = alpha * to_centroid_dir + (1 - alpha) * stochastic_component

        # Normalize combined direction
        combined_norm = np.linalg.norm(combined_dir, axis=1, keepdims=True) + 1e-10
        combined_dir = combined_dir / combined_norm

        # Apply update with inertia
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * combined_dir

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_08_catH_idea_0.py (2 wins) ---
```python
def _position_update_centroid_knn(self):
        """Hybrid: temporal-drift detection + fitness-rank FDC switching (Category H).

        Two distinct mechanisms:
        1. Temporal/dynamical: EMA centroid velocity to detect swarm drift vs convergence
        2. Fitness-rank: Spearman FDC to detect if particles moving toward better regions

        Principled switching: drift-detected → exploration mode; stable → exploitation mode.
        Targets worst tasks (17, 16, 6, 5) where multimodal landscapes trap the swarm.
        """
        centroid = np.mean(self.population, axis=0)

        # === MECHANISM 1: TEMPORAL/FLDYNAMICAL ===
        # Track centroid drift across generations
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._centroid_vel_history = []

        alpha_ema = 0.3
        self._ema_centroid = alpha_ema * centroid + (1.0 - alpha_ema) * self._ema_centroid

        # Centroid velocity = drift rate
        centroid_velocity = centroid - self._ema_centroid
        drift_magnitude = np.linalg.norm(centroid_velocity) + 1e-10

        # Historical drift for normalization
        self._centroid_vel_history.append(drift_magnitude)
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)
        avg_drift = np.mean(self._centroid_vel_history) + 1e-10
        max_drift = np.max(self._centroid_vel_history) + 1e-10

        # Drift ratio: high ratio = swarm is moving/oscillating
        drift_ratio = drift_magnitude / avg_drift

        # === MECHANISM 2: FITNESS-LANDSCAPE/RANK-BASED ===
        # Spearman fitness-distance correlation
        if self.global_best is not None and self.np > 2:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0

        # === PRINCIPLED SWITCHING LOGIC ===
        # Data-driven: use drift_ratio and FDC to determine mode
        drift_threshold = 1.5  # Adaptive based on historical drift
        is_drifting = drift_ratio > drift_threshold

        # k-NN density for local modulation
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)

        # === SWITCHING: EXPLORATION vs EXPLOITATION ===
        if is_drifting:
            # EXPLORATION MODE: centroid attraction + perturbation
            # High drift = swarm oscillating/uncertain → reduce directional pull
            dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
            centroid_attraction = 0.3 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

            to_centroid_dir = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid_dir / to_centroid_dist

            # Reduce correction strength when drifting
            exploration_strength = 0.5 / (1.0 + 0.3 * (drift_ratio - 1.0))
            correction = centroid_attraction * to_centroid_dir * density_modulation * exploration_strength

            # Add random perturbation when exploring
            random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim)) * density_modulation
            new_population = self.population + self.inertia_weight * self.velocity + 0.2 * correction + random_perturb
        else:
            # EXPLOITATION MODE: FDC-guided movement toward best
            # Stable swarm → use FDC to guide particles
            fdc_signal = np.clip(fdc, -1.0, 1.0)

            # FDC > 0: good correlation (fitter = closer to best) → exploit
            # FDC < 0: poor correlation → explore more
            exploit_weight = 0.5 * (fdc_signal + 1.0)
            explore_weight = 1.0 - exploit_weight

            # Centroid pull (reduced when exploiting)
            dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
            centroid_attraction = 0.2 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
            to_centroid_dir = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid_dir / to_centroid_dist

            # Global best pull (strong when FDC is high)
            to_best_dir = np.zeros_like(self.population)
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_dist

            # Weighted combination of centroid and best directions
            best_pull_strength = exploit_weight * 0.5
            correction = (centroid_attraction * to_centroid_dir * density_modulation * explore_weight +
                          best_pull_strength * to_best_dir * density_modulation * exploit_weight)

            new_population = self.population + self.inertia_weight * self.velocity + 0.4 * correction

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_09_catA_idea_0.py (5 wins) ---
```python
def _position_update_centroid_knn(self):
        """Convex hull boundary detection + anisotropic centroid correction.

        Category A (Geometry/spatial). Different from k-NN density by using
        TOPOLOGICAL boundary info: hull vertices = exploration frontier,
        interior points = potentially trapped. Interior particles get
        stronger outward push to escape local optima on worst tasks.
        """
        from scipy.spatial import ConvexHull

        centroid = np.mean(self.population, axis=0)
        to_centroid = centroid - self.population
        to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_norm

        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        max_dist = np.max(dists_to_centroid) + 1e-10
        normalized_dists = dists_to_centroid / max_dist

        # Geometric boundary detection via convex hull
        boundary_mask = np.zeros(self.np, dtype=bool)
        try:
            if self.np >= self.dim + 2 and self.dim <= 15:
                hull = ConvexHull(self.population)
                boundary_mask[hull.vertices] = True
            else:
                boundary_mask = normalized_dists > 0.75
        except:
            boundary_mask = normalized_dists > 0.75

        # Interior particles (trapped): stronger outward push to escape
        # Boundary particles (exploring): stabilize with mild centroid pull
        interior_strength = 0.8 * (1.0 - normalized_dists[:, np.newaxis])
        boundary_strength = 0.25 * normalized_dists[:, np.newaxis]

        correction = np.where(
            boundary_mask[:, np.newaxis],
            boundary_strength * to_centroid_dir,
            interior_strength * to_centroid_dir
        )

        new_population = self.population + self.inertia_weight * self.velocity + 0.5 * correction
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
        """Adapt neighborhood size using information-theoretic signals from population distribution.

        Treats the population as a probability distribution and uses:
        1. Entropy of discretized fitness distribution
        2. KL divergence of covariance eigenvalues from uniform (spectral entropy)

        High entropy + low KL = diverse population → reduce neighborhood for exploitation.
        Low entropy + high KL = clustered/trapped → increase neighborhood for exploration.
        """
        # Signal 1: Entropy of fitness distribution (normalized)
        fitness = self.current_fitness
        if len(fitness) > 1:
            n_bins = min(20, max(3, len(fitness) // 2))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0]
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_fitness_entropy = np.log(n_bins)
            norm_fitness_ent = fitness_entropy / (max_fitness_entropy + 1e-10)
        else:
            norm_fitness_ent = 0.5

        # Signal 2: KL divergence of covariance eigenvalues from uniform
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)

            # KL(P||U) where U is uniform
            uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
            uniform = uniform + 1e-10
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))

            max_kl = np.log(len(eigenvalues_norm))
            norm_kl = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
        except:
            norm_kl = 0.5

        # Combined exploration signal: KL pushes toward exploration, entropy pulls toward exploitation
        exploration_signal = 0.4 * (1.0 - norm_fitness_ent) + 0.6 * norm_kl
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

        # Map to neighborhood size
        min_ns = 1
        max_ns = max(3, self.np // 4)
        target_ns = int(np.round(min_ns + exploration_signal * (max_ns - min_ns)))
        target_ns = np.clip(target_ns, min_ns, max_ns)

        # Incremental adjustment
        if target_ns > self.neighborhood_size:
            self.neighborhood_size = min(target_ns, self.neighborhood_size + 1)
        elif target_ns < self.neighborhood_size:
            self.neighborhood_size = max(target_ns, self.neighborhood_size - 1)
    
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
        """Spearman-Kendall fitness-rank modulation with success-history (Category D).

        Uses ONLY fitness signals:
        - Spearman rank correlation (fitness vs distance-to-best rank)
        - Kendall tau rank stability across generations
        - Fitness percentiles (10th, 50th, 90th) for convergence detection
        - Success-history per particle
        - Temporal improvement EMA

        NO raw distances, NO covariance, NO spectral analysis.
        Targets worst tasks (17, 16, 6, 5) where rugged landscapes trap the swarm.
        """
        try:
            # === RANK COMPUTATION ===
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            # === SUCCESS HISTORY (Category D) ===
            if not hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._success_count[improved] += 1
            self._success_count[~improved] *= 0.9
            success_norm = self._success_count / (np.max(self._success_count) + 1.0)

            # === SPEARMAN FDC (fitness-distance correlation, rank-based) ===
            if self.global_best is not None and self.np > 2:
                dists = np.linalg.norm(self.population - self.global_best, axis=1)
                dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
                if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                    fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
                else:
                    fdc = 0.0
            else:
                fdc = 0.0

            # === TEMPORAL FITNESS HISTORY ===
            if not hasattr(self, '_fitness_history'):
                self._fitness_history = []
            self._fitness_history.append(np.min(self.current_fitness))
            if len(self._fitness_history) > 30:
                self._fitness_history.pop(0)

            # === KENDALL TAU RANK STABILITY ===
            if len(self._fitness_history) >= 2:
                prev_fitness = self._fitness_history[-2]
                curr_fitness = self._fitness_history[-1]
                n = len(self.current_fitness)
                concordant = np.sum((self.current_fitness < prev_fitness) & (self.current_fitness < curr_fitness))
                discordant = np.sum((self.current_fitness > prev_fitness) | (self.current_fitness > curr_fitness))
                kendall_tau = (concordant - discordant) / (n * (n - 1) / 2 + 1e-10)
            else:
                kendall_tau = 0.0

            # === FITNESS PERCENTILE CONVERGENCE DETECTION ===
            p10 = np.percentile(self.current_fitness, 10)
            p50 = np.percentile(self.current_fitness, 50)
            p90 = np.percentile(self.current_fitness, 90)
            fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)

            if not hasattr(self, '_prev_fitness_spread'):
                self._prev_fitness_spread = fitness_spread
            spread_change = abs(fitness_spread - self._prev_fitness_spread)
            self._prev_fitness_spread = fitness_spread

            is_converging = fitness_spread < 0.1 and spread_change < 0.01
            is_exploring = fitness_spread > 0.5 or spread_change > 0.05

            # === TEMPORAL IMPROVEMENT EMA ===
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_ema_improvement'):
                self._ema_improvement = 0.0
            self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

            # === SPEARMAN-BASED EXPLOIT/EXPLORE SPLIT ===
            fdc_signal = np.clip(fdc, -1.0, 1.0)
            exploit_weight = 0.5 * (fdc_signal + 1.0)
            explore_weight = 1.0 - exploit_weight

            # === GLOBAL SCALE FROM KENDALL TAU ===
            if kendall_tau < 0.2:
                global_scale = 1.4
            elif kendall_tau < 0.5:
                global_scale = 1.1
            else:
                global_scale = 0.85

            if is_converging:
                global_scale *= 0.75
            elif is_exploring:
                global_scale *= 1.25

            if self._ema_improvement > 1e-6:
                global_scale *= 0.9
            elif self._ema_improvement < 1e-10:
                global_scale *= 1.2

            # === PER-PARTICLE VELOCITY MODULATION ===
            rank_modulation = 1.0 + 0.5 * fitness_ranks
            success_modulation = 1.0 + 0.3 * success_norm
            vel_scale = rank_modulation[:, np.newaxis] * global_scale * success_modulation[:, np.newaxis]
            vel_scale = np.clip(vel_scale, 0.3, 2.5)

            # === DIRECTIONAL PERTURBATION FROM FITNESS RANK ===
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional_strength = 0.4 * (1.0 - fitness_ranks[:, np.newaxis])
                directional = directional_strength * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))

            # === RANDOM PERTURBATION INVERSELY PROPORTIONAL TO SUCCESS ===
            random_perturb = np.random.uniform(-0.25, 0.25, (self.np, self.dim))
            random_strength = (1.0 - success_norm[:, np.newaxis] * 0.5)
            random_perturb *= random_strength * explore_weight[:, np.newaxis]

            # === COMBINE ===
            new_population = self.population + vel_scale * self.velocity + directional + random_perturb

        except:
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