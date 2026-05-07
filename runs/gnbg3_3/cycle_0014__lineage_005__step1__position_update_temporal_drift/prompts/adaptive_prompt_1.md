Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_temporal_drift` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.417154e-01            1.446585e-01            1.505280e-01            1.406534e-01            1.456199e-01            1.554242e-01            1.403184e-01            1.392298e-01            1.525597e-01            
1      1.820227e+00            1.677243e+00            2.634088e+00            1.886416e+00            2.480119e+00            2.196199e+00            3.560821e+00            1.356384e+00            1.758799e+00            
2      2.009790e-01            1.984720e-01            2.062914e-01            2.069233e-01            2.100043e-01            2.098781e-01            2.083074e-01            2.034018e-01            2.075487e-01            
3      5.807786e+00            5.530825e+00            5.590998e+00            5.508458e+00            5.710789e+00            5.540118e+00            5.561110e+00            5.799556e+00            5.492547e+00            
4      1.327509e+00            1.260177e+00            1.240654e+00            1.292211e+00            1.287900e+00            1.226396e+00            1.251596e+00            1.260431e+00            1.247096e+00            
5      2.983412e+03            8.582432e+02            3.974862e+02            3.911267e+02            1.193354e+03            2.741881e+03            3.614313e+03            2.063069e+03            1.730630e+03            
6      5.491167e+02            5.812020e+02            5.701362e+02            5.371716e+02            5.562778e+02            5.260910e+02            5.585327e+02            5.211233e+02            5.489954e+02            
7      5.754320e+00            6.817479e+00            6.166267e+00            6.698704e+00            6.749523e+00            7.322517e+00            6.403367e+00            6.513596e+00            7.151794e+00            
8      2.903229e+00            2.792641e+00            2.672390e+00            2.560489e+00            2.721854e+00            2.828894e+00            2.667075e+00            2.795588e+00            2.807189e+00            
9      5.025363e+01            4.482319e+01            5.214027e+01            4.203225e+01            5.079399e+01            4.871826e+01            4.588897e+01            5.286542e+01            5.431787e+01            
10     2.493019e-01            2.369834e-01            2.614375e-01            2.485380e-01            2.601032e-01            2.540642e-01            2.710406e-01            2.427432e-01            2.797586e-01            
11     9.245367e+01            1.015209e+02            8.416278e+01            8.581919e+01            1.303815e+02            9.048230e+01            1.075511e+02            7.773071e+01            1.361141e+02            
12     4.799778e-01            2.386239e+00            5.498724e-01            4.561047e-01            1.116140e+00            5.301732e-01            3.793070e+00            4.532091e-01            5.158082e+00            
13     1.945452e+01            1.738305e+01            2.220706e+01            1.849799e+01            2.002259e+01            2.059866e+01            2.127494e+01            2.001651e+01            2.107291e+01            
14     7.856992e+00            8.067254e+00            7.944755e+00            8.655709e+00            7.245658e+00            8.272552e+00            7.433959e+00            8.439413e+00            8.594352e+00            
15     3.071738e+00            3.113193e+00            3.062016e+00            3.200086e+00            2.937924e+00            3.067144e+00            3.163591e+00            3.091653e+00            3.105385e+00            
16     9.820931e+02            1.114787e+03            7.252886e+02            7.721051e+02            9.928641e+02            9.398287e+02            1.222161e+03            9.804845e+02            1.139272e+03            
17     2.370301e+04            2.293704e+04            3.191808e+04            1.847027e+04            2.713185e+04            1.618528e+04            2.137248e+04            3.171881e+04            3.030281e+04            
18     3.720607e+01            3.952622e+01            3.694856e+01            3.492771e+01            3.992422e+01            3.913024e+01            3.954752e+01            4.335220e+01            3.646010e+01            
19     8.351462e+01            8.334959e+01            7.247090e+01            9.076116e+01            7.808170e+01            8.538112e+01            7.420056e+01            8.993778e+01            9.304846e+01            
20     2.053561e+01            2.002585e+01            2.072281e+01            2.053786e+01            2.080311e+01            2.178842e+01            2.026065e+01            2.145933e+01            2.235793e+01            
21     5.716999e+00            5.594464e+00            5.411665e+00            5.405128e+00            5.255155e+00            5.721971e+00            5.434391e+00            5.610204e+00            5.574710e+00            
22     1.205636e+01            1.127460e+01            1.203411e+01            1.126015e+01            1.160950e+01            1.117382e+01            1.118861e+01            1.161244e+01            1.101334e+01            
23     5.788320e+01            5.837633e+01            6.050128e+01            5.954521e+01            6.187971e+01            5.790258e+01            5.702441e+01            5.849408e+01            5.848324e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_08_catH_idea_0.py  (error=1.392298e-01)
Task  1: variant_08_catH_idea_0.py  (error=1.356384e+00)
Task  2: variant_02_catB_idea_0.py  (error=1.984720e-01)
Task  3: variant_10_catB_idea_0.py  (error=5.492547e+00)
Task  4: variant_06_catF_idea_0.py  (error=1.226396e+00)
Task  5: variant_04_catD_idea_0.py  (error=3.911267e+02)
Task  6: variant_08_catH_idea_0.py  (error=5.211233e+02)
Task  7: original.py  (error=5.754320e+00)
Task  8: variant_04_catD_idea_0.py  (error=2.560489e+00)
Task  9: variant_04_catD_idea_0.py  (error=4.203225e+01)
Task 10: variant_02_catB_idea_0.py  (error=2.369834e-01)
Task 11: variant_08_catH_idea_0.py  (error=7.773071e+01)
Task 12: variant_08_catH_idea_0.py  (error=4.532091e-01)
Task 13: variant_02_catB_idea_0.py  (error=1.738305e+01)
Task 14: variant_05_catE_idea_0.py  (error=7.245658e+00)
Task 15: variant_05_catE_idea_0.py  (error=2.937924e+00)
Task 16: variant_03_catC_idea_0.py  (error=7.252886e+02)
Task 17: variant_06_catF_idea_0.py  (error=1.618528e+04)
Task 18: variant_04_catD_idea_0.py  (error=3.492771e+01)
Task 19: variant_03_catC_idea_0.py  (error=7.247090e+01)
Task 20: variant_02_catB_idea_0.py  (error=2.002585e+01)
Task 21: variant_05_catE_idea_0.py  (error=5.255155e+00)
Task 22: variant_10_catB_idea_0.py  (error=1.101334e+01)
Task 23: variant_07_catG_idea_0.py  (error=5.702441e+01)

WIN COUNTS:
  variant_08_catH_idea_0.py: 5 wins
  variant_02_catB_idea_0.py: 4 wins
  variant_04_catD_idea_0.py: 4 wins
  variant_05_catE_idea_0.py: 3 wins
  variant_10_catB_idea_0.py: 2 wins
  variant_06_catF_idea_0.py: 2 wins
  variant_03_catC_idea_0.py: 2 wins
  original.py: 1 wins
  variant_07_catG_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_08_catH_idea_0.py       1.3923e-01       variant_07_catG_idea_0.py        1.4032e-01         1.0×    1.0×
     1   variant_08_catH_idea_0.py       1.3564e+00       variant_02_catB_idea_0.py        1.6772e+00         1.2×    1.5×
     2   variant_02_catB_idea_0.py       1.9847e-01       original.py                      2.0098e-01         1.0×    1.0×
     3   variant_10_catB_idea_0.py       5.4925e+00       variant_04_catD_idea_0.py        5.5085e+00         1.0×    1.0×
     4   variant_06_catF_idea_0.py       1.2264e+00       variant_03_catC_idea_0.py        1.2407e+00         1.0×    1.0×
     5   variant_04_catD_idea_0.py       3.9113e+02       variant_03_catC_idea_0.py        3.9749e+02         1.0×    4.8×
     6   variant_08_catH_idea_0.py       5.2112e+02       variant_06_catF_idea_0.py        5.2609e+02         1.0×    1.1×
     7   original.py                     5.7543e+00       variant_03_catC_idea_0.py        6.1663e+00         1.1×    1.2×
     8   variant_04_catD_idea_0.py       2.5605e+00       variant_07_catG_idea_0.py        2.6671e+00         1.0×    1.1×
     9   variant_04_catD_idea_0.py       4.2032e+01       variant_02_catB_idea_0.py        4.4823e+01         1.1×    1.2×
    10   variant_02_catB_idea_0.py       2.3698e-01       variant_08_catH_idea_0.py        2.4274e-01         1.0×    1.1×
    11   variant_08_catH_idea_0.py       7.7731e+01       variant_03_catC_idea_0.py        8.4163e+01         1.1×    1.2×
    12   variant_08_catH_idea_0.py       4.5321e-01       variant_04_catD_idea_0.py        4.5610e-01         1.0×    1.8×
    13   variant_02_catB_idea_0.py       1.7383e+01       variant_04_catD_idea_0.py        1.8498e+01         1.1×    1.2×
    14   variant_05_catE_idea_0.py       7.2457e+00       variant_07_catG_idea_0.py        7.4340e+00         1.0×    1.1×
    15   variant_05_catE_idea_0.py       2.9379e+00       variant_03_catC_idea_0.py        3.0620e+00         1.0×    1.1×
    16   variant_03_catC_idea_0.py       7.2529e+02       variant_04_catD_idea_0.py        7.7211e+02         1.1×    1.4×
    17   variant_06_catF_idea_0.py       1.6185e+04       variant_04_catD_idea_0.py        1.8470e+04         1.1×    1.6×
    18   variant_04_catD_idea_0.py       3.4928e+01       variant_10_catB_idea_0.py        3.6460e+01         1.0×    1.1×
    19   variant_03_catC_idea_0.py       7.2471e+01       variant_07_catG_idea_0.py        7.4201e+01         1.0×    1.2×
    20   variant_02_catB_idea_0.py       2.0026e+01       variant_07_catG_idea_0.py        2.0261e+01         1.0×    1.0×
    21   variant_05_catE_idea_0.py       5.2552e+00       variant_04_catD_idea_0.py        5.4051e+00         1.0×    1.1×
    22   variant_10_catB_idea_0.py       1.1013e+01       variant_06_catF_idea_0.py        1.1174e+01         1.0×    1.0×
    23   variant_07_catG_idea_0.py       5.7024e+01       original.py                      5.7883e+01         1.0×    1.0×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 9 (original.py, variant_02_catB_idea_0.py, variant_03_catC_idea_0.py, variant_04_catD_idea_0.py, variant_05_catE_idea_0.py, variant_06_catF_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 0  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_catB_idea_0.py (4 wins) ---
```python
def _position_update_temporal_drift(self):
        """SVD Subspace Collapse Recovery (Category B: Spectral/Linear-Algebraic).

        Uses singular value entropy and effective rank of the population matrix to
        detect when the population has collapsed into a low-dimensional subspace.
        Applies inverse-collapsed-direction velocity boosting and condition-number-
        based exploration scaling to recover lost degrees of freedom.

        Targets worst tasks (17, 5, 16, 6) where population collapse into disconnected
        local optima leaves exploration directions completely abandoned.
        """
        try:
            # === SPECTRAL ANALYSIS VIA SVD ===
            centered = self.population - np.mean(self.population, axis=0)

            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            # Effective rank from singular value entropy (not eigenvalue entropy)
            sv_norm = singular_values / (singular_values[0] + 1e-10)
            entropy_sv = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
            max_entropy = np.log(len(singular_values))
            effective_rank = np.exp(entropy_sv)
            eff_rank_ratio = effective_rank / len(singular_values)

            # Condition number for anisotropy
            cond = singular_values[0] / (singular_values[-1] + 1e-10)

            # === COLLAPSED DIRECTION DETECTION ===
            # Directions where sv_norm < 0.05 are considered "collapsed"
            collapse_mask = sv_norm < 0.05

            # Boost velocity in collapsed directions (inverse of normalized SV)
            # Low singular value = collapsed direction = needs strong velocity boost
            inverse_boost = np.zeros_like(sv_norm)
            inverse_boost[~collapse_mask] = 1.0 / (sv_norm[~collapse_mask] + 0.01)
            inverse_boost[collapse_mask] = 5.0 / (sv_norm[collapse_mask] + 0.01)
            inverse_boost = np.clip(inverse_boost, 0.5, 10.0)
            inverse_boost = inverse_boost / (np.max(inverse_boost) + 1e-10)

            # === EXPLORATION SCALING FROM CONDITION NUMBER ===
            # Logarithmic mapping: cond varies over many orders of magnitude
            # High cond = ill-conditioned = more exploration needed
            cond_log = np.log1p(cond) / np.log1p(1e6)
            cond_log = np.clip(cond_log, 0.0, 1.0)
            exploration_scale = 0.8 + 0.7 * cond_log

            # === COMBINED EXPLORATION SIGNAL ===
            # Low effective rank ratio = population in low-dim subspace = boost exploration
            if eff_rank_ratio < 0.3:
                exploration_scale *= 1.5
            elif eff_rank_ratio < 0.5:
                exploration_scale *= 1.2

            exploration_scale = np.clip(exploration_scale, 0.3, 2.5)

            # === VELOCITY UPDATE IN SVD BASIS ===
            vel_proj = self.velocity @ Vt.T
            vel_boosted = vel_proj * inverse_boost * exploration_scale
            new_population = self.population + (vel_boosted @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_03_catC_idea_0.py (2 wins) ---
```python
def _position_update_temporal_drift(self):
        """Information-theoretic: Entropy + KL divergence + GMM multi-modality detection (Category C).

        Key mechanisms:
        1. Fitness entropy: detects convergence from fitness distribution shape
        2. Spectral KL divergence: measures how far population is from uniform spread
        3. GMM multi-modality: detects if population is split across multiple optima
        4. Temporal entropy trend: detects stagnation via entropy rate of change

        Targets worst tasks (17, 16, 6, 11) where premature convergence to wrong
        basins causes massive errors. Information-theoretic signals detect this
        collapse before it becomes irreversible.
        """
        try:
            # === MECHANISM 1: FITNESS ENTROPY (population as probability distribution) ===
            fitness = self.current_fitness
            n_bins = min(15, max(3, self.np // 3))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0] + 1e-10
            fitness_entropy = -np.sum(hist * np.log(hist))

            if not hasattr(self, '_fitness_entropy_history'):
                self._fitness_entropy_history = []
            self._fitness_entropy_history.append(fitness_entropy)
            if len(self._fitness_entropy_history) > 10:
                self._fitness_entropy_history.pop(0)

            # Entropy trend: negative = converging/stagnating
            if len(self._fitness_entropy_history) >= 3:
                recent = np.mean(self._fitness_entropy_history[-3:])
                older = np.mean(self._fitness_entropy_history[:-3]) if len(self._fitness_entropy_history) > 3 else self._fitness_entropy_history[0]
                entropy_trend = (recent - older) / (older + 1e-10)
            else:
                entropy_trend = 0.0

            # === MECHANISM 2: KL DIVERGENCE FROM UNIFORM (spatial distribution quality) ===
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)

            # KL(P_eigen || U) - how far from isotropic
            dim_eff = len(eigenvalues_norm)
            uniform = np.ones(dim_eff) / dim_eff
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
            max_kl = np.log(dim_eff)
            kl_normalized = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)

            # High KL = anisotropic = exploration needed
            exploration_from_kl = 0.5 + 0.5 * kl_normalized

            # === MECHANISM 3: GMM MULTI-MODALITY DETECTION ===
            # Fit GMM with up to 3 components; BIC selects optimal
            from sklearn.mixture import GaussianMixture
            max_components = min(4, max(2, self.np // 10))

            best_bic = np.inf
            best_n_components = 1
            for n_comp in range(1, max_components + 1):
                try:
                    gmm = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                          n_init=3, random_state=42)
                    gmm.fit(centered)
                    bic = gmm.bic(centered)
                    if bic < best_bic:
                        best_bic = bic
                        best_n_components = n_comp
                except:
                    continue

            # Multi-modality signal: more components = more trapped in separate basins
            multimodality_signal = float(best_n_components) / float(max_components)

            # === MECHANISM 4: TEMPORAL ENTROPY RATE ===
            if not hasattr(self, '_ema_entropy'):
                self._ema_entropy = fitness_entropy
            self._ema_entropy = 0.2 * fitness_entropy + 0.8 * self._ema_entropy

            # Normalized entropy
            max_entropy = np.log(n_bins)
            norm_entropy = fitness_entropy / (max_entropy + 1e-10)
            entropy_collapse = 1.0 - np.clip(norm_entropy, 0.0, 1.0)

            # === COMBINED INFORMATION-THEORETIC SIGNAL ===
            # Entropy collapse + negative trend + high multimodality = trapped
            trapped_signal = (entropy_collapse * 0.4 +
                             max(0, -entropy_trend) * 0.3 +
                             multimodality_signal * 0.3)
            trapped_signal = np.clip(trapped_signal, 0.0, 1.0)

            # Exploration factor: high when trapped or anisotropic
            exploration_factor = (1.0 + trapped_signal * 1.5) * exploration_from_kl
            exploration_factor = np.clip(exploration_factor, 0.5, 2.5)

            # === VELOCITY SCALING FROM DISTRIBUTIONAL SIGNALS ===
            U, V = np.linalg.eigh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = V[:, idx]

            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)

            # Inverse scaling: amplify collapsed directions
            inverse_scale = 1.0 / (sv_norm + 0.1)
            inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)
            per_comp_scale = np.clip(inverse_scale, 0.2, 3.0)

            vel_scaled = vel_proj * per_comp_scale

            # === DIRECTIONAL CORRECTION FROM ENTROPY SIGNALS ===
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_dist
            else:
                to_best_dir = np.random.randn(self.np, self.dim)

            # When trapped: add random perturbation proportional to collapse
            if trapped_signal > 0.4:
                random_perturb = np.random.randn(self.np, self.dim) * trapped_signal * 0.5
            else:
                random_perturb = np.zeros((self.np, self.dim))

            # === FINAL POPULATION UPDATE ===
            new_population = self.population + exploration_factor * (vel_scaled @ V.T) + random_perturb

            # === ADAPTIVE REINITIALIZATION TRIGGER ===
            # If entropy collapsed AND we're not improving → reinitialize worst particles
            if entropy_collapse > 0.7 and entropy_trend < -0.1:
                if self.global_best is not None:
                    n_replace = max(1, int(0.2 * self.np))
                    worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
                    self.population[worst_indices] = np.random.uniform(
                        self.lower_bound, self.upper_bound, (n_replace, self.dim)
                    )
                    self.velocity[worst_indices] = np.random.uniform(
                        self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
                    )
                    self.personal_best_fitness[worst_indices] = np.inf
                    self.personal_best[worst_indices] = self.population[worst_indices]

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        except ImportError:
            # Fallback without sklearn
            try:
                centered = self.population - np.mean(self.population, axis=0)
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                eigenvalues = np.sort(eigenvalues)[::-1]
                eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
                uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
                kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
                max_kl = np.log(len(eigenvalues_norm))
                kl_normalized = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)

                U, V = np.linalg.eigh(cov)
                idx = np.argsort(U)[::-1]
                U = U[idx]
                V = V[:, idx]
                vel_proj = self.velocity @ V
                sv_scale = np.sqrt(U + 1e-10)
                sv_norm = sv_scale / (sv_scale[0] + 1e-10)
                inverse_scale = 1.0 / (sv_norm + 0.1)
                inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)
                per_comp_scale = np.clip(inverse_scale, 0.2, 3.0)
                vel_scaled = vel_proj * per_comp_scale

                exploration_factor = np.clip(0.5 + 0.5 * kl_normalized, 0.5, 2.0)
                new_population = self.population + exploration_factor * (vel_scaled @ V.T)
            except:
                new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_04_catD_idea_0.py (4 wins) ---
```python
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
```

# --- From variant_05_catE_idea_0.py (3 wins) ---
```python
def _position_update_temporal_drift(self):
        """Category E: MST bridge-breaking for topology-aware exploration.

        Builds a Minimum Spanning Tree over the population to identify bridge
        edges that connect clusters. Particles at bridge endpoints are perturbed
        to break up fragmentation and maintain population connectivity.

        Targets worst tasks (17, 16, 6, 5) where premature fragmentation
        causes the population to get trapped in disconnected local optima.
        """
        try:
            # Build Euclidean distance matrix
            diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2) + 1e-10

            # Prim's MST algorithm - O(n^2) but avoids scipy dependency
            n = self.np
            in_mst = np.zeros(n, dtype=bool)
            parent = np.full(n, -1)
            mst_edges = []

            in_mst[0] = True
            for _ in range(n - 1):
                min_edge = (np.inf, -1, -1)
                for i in range(n):
                    if not in_mst[i]:
                        continue
                    for j in range(n):
                        if in_mst[j]:
                            continue
                        if dists[i, j] < min_edge[0]:
                            min_edge = (dists[i, j], i, j)
                if min_edge[1] != -1:
                    in_mst[min_edge[2]] = True
                    parent[min_edge[2]] = min_edge[1]
                    mst_edges.append((min_edge[1], min_edge[2]))

            if len(mst_edges) == 0:
                new_population = self.population + self.velocity
                self.population = self._clip_to_bounds(new_population)
                return

            # Compute edge betweenness (simplified: edge length as proxy)
            edge_lengths = np.array([dists[u, v] for u, v in mst_edges])
            mean_len = np.mean(edge_lengths)
            std_len = np.std(edge_lengths) + 1e-10

            # Identify bridge edges (edges significantly longer than average)
            # Long edges = weak connections between clusters = bridges
            bridge_mask = edge_lengths > mean_len + 0.5 * std_len
            bridge_particles = set()
            for idx, (u, v) in enumerate(mst_edges):
                if bridge_mask[idx]:
                    bridge_particles.add(u)
                    bridge_particles.add(v)

            # Compute MST degree per particle (graph-theoretic connectivity)
            mst_degree = np.zeros(n)
            for u, v in mst_edges:
                mst_degree[u] += 1
                mst_degree[v] += 1

            # Perturbation: inverse-degree scaling
            # Low degree = isolated = needs stronger push to reconnect
            perturbation = np.zeros((n, self.dim))
            for i in range(n):
                if i not in bridge_particles:
                    continue
                # Inverse MST degree: isolated particles get stronger perturbation
                strength = 0.5 / (mst_degree[i] + 1.0)
                # Random direction for diversity
                direction = np.random.randn(self.dim)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                perturbation[i] = strength * direction

            # Velocity scaling based on MST edge properties
            # Particles in sparse regions (low degree) get velocity boost
            vel_scale = 1.0 + 0.5 * (1.0 - np.clip(mst_degree / (np.mean(mst_degree) + 1e-10), 0.0, 1.0))
            vel_scale = np.clip(vel_scale, 0.5, 2.0)

            new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + perturbation
            self.population = self._clip_to_bounds(new_population)

        except Exception:
            new_population = self.population + self.velocity
            self.population = self._clip_to_bounds(new_population)
```

# --- From variant_06_catF_idea_0.py (2 wins) ---
```python
def _position_update_temporal_drift(self):
        """Pure temporal/dynamical: Autocorrelation-stagnation momentum (Category F).

        Tracks quantities ACROSS generations — not single-generation snapshots:
          1. Fitness improvement autocorrelation (lag-1): detects correlated drift vs. exploration
          2. EMA-based centroid momentum: tracks population movement direction and magnitude
          3. Windowed stagnation detection: how long since last meaningful improvement
          4. Acceleration trend of momentum: detecting convergence phases
          5. Diversity EMA trend: detecting premature collapse

        Combined "stuckness" signal drives adaptive velocity scaling and directional kicks.
        Targets worst tasks (17, 16, 6, 5) where temporal signals reveal premature convergence.
        """
        try:
            # === MECHANISM 1: FITNESS IMPROVEMENT AUTOCORRELATION ===
            if not hasattr(self, '_fitness_history'):
                self._fitness_history = []
            self._fitness_history.append(float(np.min(self.current_fitness)))
            if len(self._fitness_history) > 30:
                self._fitness_history.pop(0)

            if len(self._fitness_history) >= 10:
                improvements = np.diff(self._fitness_history)
                mean_imp = np.mean(improvements)
                var_imp = np.var(improvements) + 1e-10
                autocorr_lag1 = np.sum(improvements[:-1] * improvements[1:]) / ((len(improvements) - 1) * var_imp)
                autocorr_lag1 = np.clip(autocorr_lag1, -1.0, 1.0)
            else:
                autocorr_lag1 = 0.0

            # === MECHANISM 2: EMA-BASED CENTROID MOMENTUM ===
            current_centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_ema_centroid'):
                self._ema_centroid = current_centroid.copy()
            self._ema_centroid = 0.7 * self._ema_centroid + 0.3 * current_centroid
            centroid_vel = current_centroid - self._ema_centroid
            centroid_vel_mag = np.linalg.norm(centroid_vel)

            if not hasattr(self, '_centroid_vel_ema_mag'):
                self._centroid_vel_ema_mag = 0.0
            self._centroid_vel_ema_mag = 0.7 * self._centroid_vel_ema_mag + 0.3 * centroid_vel_mag

            # === MECHANISM 3: WINDOWED STAGNATION DETECTION ===
            if not hasattr(self, '_stagnation_counter_temporal'):
                self._stagnation_counter_temporal = 0
            if self.global_best is not None and hasattr(self, '_prev_global_best'):
                if abs(self._prev_global_best - self.global_best_fitness) < 1e-10:
                    self._stagnation_counter_temporal += 1
                else:
                    self._stagnation_counter_temporal = 0
            self._prev_global_best = self.global_best_fitness if self.global_best is not None else 0.0

            stagnation_depth = min(self._stagnation_counter_temporal / 50.0, 1.0)

            # === MECHANISM 4: ACCELERATION TREND OF MOMENTUM ===
            if not hasattr(self, '_accel_history'):
                self._accel_history = []
            self._accel_history.append(centroid_vel_mag)
            if len(self._accel_history) > 30:
                self._accel_history.pop(0)

            if len(self._accel_history) >= 5:
                accel_array = np.array(self._accel_history)
                accel_trend = np.mean(np.diff(accel_array))
            else:
                accel_trend = 0.0
            accel_signal = -np.sign(accel_trend) * min(abs(accel_trend), 1.0)

            # === MECHANISM 5: DIVERSITY EMA TREND ===
            current_diversity = self._compute_diversity()
            if not hasattr(self, '_ema_diversity'):
                self._ema_diversity = current_diversity
            self._ema_diversity = 0.95 * self._ema_diversity + 0.05 * current_diversity
            diversity_ratio = current_diversity / (self._ema_diversity + 1e-10)
            diversity_signal = 1.0 - np.clip(diversity_ratio, 0.0, 2.0)

            # === COMBINED TEMPORAL "STUCKNESS" SIGNAL ===
            stuckness = (
                0.35 * max(0.0, autocorr_lag1) +
                0.25 * stagnation_depth +
                0.20 * max(0.0, accel_signal) +
                0.20 * max(0.0, diversity_signal)
            )
            stuckness = np.clip(stuckness, 0.0, 1.0)

            # === ADAPTIVE VELOCITY SCALING ===
            if stuckness > 0.7:
                base_scale = 1.6
            elif stuckness > 0.4:
                base_scale = 1.2
            else:
                base_scale = 1.0

            # Diversity-aware boost
            if diversity_ratio < 0.8:
                diversity_boost = 1.4
            elif diversity_ratio < 1.0:
                diversity_boost = 1.2
            else:
                diversity_boost = 1.0

            # === DIRECTIONAL KICK FOR STUCK POPULATION ===
            if stuckness > 0.5:
                if centroid_vel_mag > 1e-6:
                    vel_dir = centroid_vel / centroid_vel_mag
                else:
                    vel_dir = np.zeros(self.dim)
                kick_dir = np.random.uniform(-1, 1, self.dim)
                if centroid_vel_mag > 1e-6:
                    kick_dir -= vel_dir * np.dot(kick_dir, vel_dir)
                kick_mag = stuckness * 2.0
                directional_kick = kick_dir * kick_mag
            else:
                directional_kick = np.zeros(self.dim)

            # === FINAL UPDATE ===
            new_population = self.population + base_scale * self.velocity * diversity_boost + directional_kick

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_07_catG_idea_0.py (1 wins) ---
```python
def _position_update_temporal_drift(self):
        """Monte Carlo Bootstrap Uncertainty-Driven Exploration (Category G).

        Uses fundamentally stochastic computation:
          1. Bootstrap resampling of population to estimate uncertainty in centroid
          2. Monte Carlo sampling of random projections to estimate directional quality
          3. Latin hypercube probing of unexplored regions
          4. Uncertainty-weighted velocity scaling

        Targets worst tasks (17, 16, 6, 5) where premature convergence traps
        the population in disconnected local optima. The stochastic approach
        provides a different angle: instead of deterministic topology, we
        quantify uncertainty and explore uncertain regions more aggressively.
        """
        try:
            n_particles = self.np
            dim = self.dim

            # === MECHANISM 1: BOOTSTRAP UNCERTAINTY IN CENTROID ===
            # Resample population with replacement to get bootstrap distribution of centroid
            n_bootstrap = min(50, n_particles * 2)
            bootstrap_centroids = np.zeros((n_bootstrap, dim))

            for b in range(n_bootstrap):
                indices = np.random.randint(0, n_particles, size=n_particles)
                bootstrap_centroids[b] = np.mean(self.population[indices], axis=0)

            centroid_mean = np.mean(bootstrap_centroids, axis=0)
            centroid_std = np.std(bootstrap_centroids, axis=0) + 1e-10

            # Uncertainty radius: how sure are we about centroid location?
            uncertainty_radius = np.linalg.norm(centroid_std)
            total_spread = np.std(self.population) + 1e-10
            normalized_uncertainty = uncertainty_radius / (total_spread + 1e-10)

            # === MECHANISM 2: MONTE CARLO RANDOM PROJECTION EXPLORATION ===
            # Project population onto random directions to estimate exploration quality
            n_projections = min(20, dim)
            projection_scores = np.zeros(n_projections)

            for p in range(n_projections):
                # Random projection vector
                proj_dir = np.random.randn(dim)
                proj_dir = proj_dir / (np.linalg.norm(proj_dir) + 1e-10)

                # Project positions and velocities
                proj_pos = self.population @ proj_dir
                proj_vel = self.velocity @ proj_dir

                # Score: variance in projected position (exploration signal)
                # Higher variance = more diverse exploration in this direction
                projection_scores[p] = np.std(proj_pos) / (np.std(proj_vel) + 1e-10)

            # Best projection direction (most unexplored)
            best_proj_idx = np.argmax(projection_scores)
            best_proj_score = projection_scores[best_proj_idx]

            # Generate random projection direction for next exploration
            random_proj_dir = np.random.randn(dim)
            random_proj_dir = random_proj_dir / (np.linalg.norm(random_proj_dir) + 1e-10)

            # === MECHANISM 3: LATIN HYPERCUBE GUIDED EXPLORATION ===
            # Sample unexplored regions using Latin hypercube-like partitioning
            if not hasattr(self, '_lhc_grid') or self._lhc_generation + 5 < self.generation:
                # Rebuild grid every 5 generations
                grid_size = min(10, n_particles // 3)
                self._lhc_samples = np.random.uniform(
                    self.lower_bound, self.upper_bound, (grid_size, dim)
                )
                # Ensure Latin hypercube property: one sample per row and column bin
                for d in range(dim):
                    bins = np.linspace(self.lower_bound, self.upper_bound, grid_size + 1)
                    for i in range(grid_size):
                        self._lhc_samples[i, d] = np.random.uniform(bins[i], bins[i + 1])
                # Shuffle to break column correlations
                for d in range(dim):
                    np.random.shuffle(self._lhc_samples[:, d])
                self._lhc_generation = self.generation
                self._lhc_fitness = np.full(grid_size, np.inf)

            # Find closest LHC sample to each particle
            lhc_dists = np.sum((self.population[:, np.newaxis, :] - self._lhc_samples[np.newaxis, :, :]) ** 2, axis=2)
            closest_lhc = np.argmin(lhc_dists, axis=1)
            lhc_guidance = np.zeros((n_particles, dim))

            for i in range(n_particles):
                lhc_idx = closest_lhc[i]
                direction = self._lhc_samples[lhc_idx] - self.population[i]
                dist = np.linalg.norm(direction) + 1e-10
                lhc_guidance[i] = direction / dist

            # === MECHANISM 4: BOOTSTRAP CONFIDENCE INTERVAL VELOCITY SCALING ===
            # Use bootstrap to estimate confidence in current velocity direction
            n_vel_bootstrap = 30
            vel_std_estimate = np.zeros((n_vel_bootstrap, dim))

            for b in range(n_vel_bootstrap):
                indices = np.random.randint(0, n_particles, size=n_particles)
                vel_std_estimate[b] = np.std(self.velocity[indices], axis=0)

            velocity_confidence = 1.0 / (np.mean(vel_std_estimate, axis=0) + 1e-10)
            velocity_confidence = velocity_confidence / (np.max(velocity_confidence) + 1e-10)
            velocity_confidence = np.clip(velocity_confidence, 0.3, 2.0)

            # === COMBINE STOCHASTIC SIGNALS ===
            # Uncertainty-weighted exploration factor
            exploration_factor = 1.0 + 0.5 * normalized_uncertainty

            # Projection-guided direction (stochastic)
            proj_direction = random_proj_dir * best_proj_score

            # Bootstrap-based velocity scaling per dimension
            vel_scale = velocity_confidence

            # LHC-guided exploration boost
            lhc_strength = 0.3 * exploration_factor
            lhc_boost = lhc_strength * lhc_guidance

            # Bootstrap centroid pull (with uncertainty weighting)
            to_centroid = centroid_mean - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            centroid_pull = 0.2 * exploration_factor * (to_centroid / to_centroid_dist)

            # === APPLY STOCHASTIC UPDATE ===
            # Scale velocity by bootstrap confidence, add stochastic corrections
            scaled_velocity = self.velocity * vel_scale

            # Add projection-based exploration
            proj_correction = exploration_factor * proj_direction

            # Random perturbation scaled by uncertainty
            random_perturbation = np.random.randn(n_particles, dim) * 0.1 * exploration_factor

            # Final population update
            new_population = (
                self.population +
                scaled_velocity +
                lhc_boost +
                centroid_pull +
                random_perturbation
            )

            # Clip to bounds
            new_population = self._clip_to_bounds(new_population)

            # Update tracking variables
            self._bootstrap_centroid_mean = centroid_mean
            self._bootstrap_uncertainty = normalized_uncertainty
            self._last_random_proj_dir = random_proj_dir

        except Exception:
            # Fallback: simple stochastic perturbation
            noise = np.random.randn(n_particles, dim) * 0.5
            new_population = self.population + self.velocity + noise
            new_population = self._clip_to_bounds(new_population)

        self.population = new_population
```

# --- From variant_08_catH_idea_0.py (5 wins) ---
```python
def _position_update_temporal_drift(self):
        """Hybrid: Improvement-gated fitness-gradient + geometric-spread (Category H).

        Combines TWO distinct mechanisms with DATA-DRIVEN switching:
          1. Temporal fitness-gradient: EMA of fitness improvement rate across generations.
             High improvement → trust gradient direction; stagnation → amplify velocity.
          2. Geometric spread: Average pairwise distance as population collapse indicator.
             Low spread → inject repulsion from centroid; high spread → exploit.

        SWITCHING LOGIC (principled, data-driven):
          - improvement_rate EMA fed into a sigmoid gate that shifts weight between mechanisms
          - If fitness stagnates, geometric mechanism progressively dominates (diversity recovery)
          - If fitness improves, temporal mechanism dominates (focused exploitation)

        Targets worst tasks (17, 16, 6, 5) where severe multimodality/deception causes
        population collapse into disconnected basins. The improvement-gated switch prevents
        the algorithm from doubling down on failed strategies.
        """
        try:
            # === MECHANISM 1: TEMPORAL FITNESS-GRADIENT ===
            # Track improvement rate across generations (data-driven signal)
            if self.global_best is not None and hasattr(self, '_prev_best_fitness'):
                raw_improvement = self._prev_best_fitness - self.global_best_fitness
            else:
                raw_improvement = 0.0
            self._prev_best_fitness = self.global_best_fitness if self.global_best is not None else np.inf

            # EMA of improvement rate (temporal signal)
            if not hasattr(self, '_ema_improvement'):
                self._ema_improvement = 0.0
            self._ema_improvement = 0.15 * max(0.0, raw_improvement) + 0.85 * self._ema_improvement

            # Stagnation counter as secondary signal
            if not hasattr(self, '_stagnation_ema'):
                self._stagnation_ema = 0.0
            self._stagnation_ema = 0.1 * self.stagnation_counter + 0.9 * self._stagnation_ema

            # Normalize improvement by current fitness magnitude (scale-invariant)
            if self.global_best_fitness > 0:
                normalized_improvement = self._ema_improvement / (abs(self.global_best_fitness) + 1e-10)
            else:
                normalized_improvement = self._ema_improvement

            # Fitness-gradient velocity modulation
            # High improvement → conservative (scale down); stagnation → aggressive (scale up)
            if normalized_improvement > 1e-6:
                temporal_scale = 0.7  # Exploiting well, reduce velocity
            elif normalized_improvement > 1e-10:
                temporal_scale = 1.0  # Marginal improvement
            else:
                # Stagnation: amplify velocity for escape
                stagnation_boost = min(2.0, 1.0 + 0.1 * self._stagnation_ema)
                temporal_scale = 1.3 * stagnation_boost

            # === MECHANISM 2: GEOMETRIC SPREAD ===
            # Compute average pairwise distance (population spread)
            centered = self.population - np.mean(self.population, axis=0)
            sq_dists = np.sum(centered[:, np.newaxis, :] ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            avg_pairwise_dist = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

            # Normalize by search space diagonal
            search_space_diag = (self.upper_bound - self.lower_bound) * np.sqrt(self.dim)
            normalized_spread = avg_pairwise_dist / search_space_diag

            # Track spread history for trend detection
            if not hasattr(self, '_spread_history'):
                self._spread_history = []
            self._spread_history.append(avg_pairwise_dist)
            if len(self._spread_history) > 15:
                self._spread_history.pop(0)

            # Spread trend: is population collapsing?
            if len(self._spread_history) >= 5:
                recent_avg = np.mean(self._spread_history[-5:])
                older_avg = np.mean(self._spread_history[:-5]) if len(self._spread_history) > 5 else recent_avg
                spread_trend = (recent_avg - older_avg) / (older_avg + 1e-10)
            else:
                spread_trend = 0.0

            # Geometric correction: repulsion from centroid when collapsed
            centroid = np.mean(self.population, axis=0)
            to_centroid = self.population - centroid  # Vector FROM centroid TO particle
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist

            # Repulsion strength inversely proportional to current spread
            # Low spread → strong repulsion; high spread → weak repulsion
            repulsion_base = np.clip(1.0 - normalized_spread * 10.0, 0.0, 1.5)
            # Amplify if spread is collapsing
            if spread_trend < -0.1:
                repulsion_base *= (1.0 - spread_trend * 2.0)

            # Per-particle repulsion: particles far from centroid get pushed further (explore edges)
            # Particles near centroid get pushed outward (escape center)
            geometric_correction = repulsion_base * to_centroid_dir * to_centroid_dist

            # === DATA-DRIVEN SWITCHING LOGIC ===
            # Improvement rate gates the weight between temporal and geometric mechanisms
            # This is principled: if temporal mechanism isn't working (no improvement),
            # shift weight to geometric mechanism for diversity recovery

            # Sigmoid gate based on normalized improvement
            gate_input = np.log10(normalized_improvement + 1e-20)  # Map to [-inf, ~0]
            gate_sigmoid = 1.0 / (1.0 + np.exp(-gate_input * 2.0))  # ~0 when stagnant, ~1 when improving

            # When improving (gate_sigmoid ~1): trust temporal mechanism (focused exploitation)
            # When stagnant (gate_sigmoid ~0): trust geometric mechanism (diversity recovery)
            temporal_weight = 0.2 + 0.6 * gate_sigmoid
            geometric_weight = 1.0 - temporal_weight

            # === COMBINE MECHANISMS ===
            # Temporal contribution: scaled velocity with improvement-based modulation
            vel_magnitude = np.linalg.norm(self.velocity, axis=1, keepdims=True) + 1e-10
            vel_direction = self.velocity / vel_magnitude
            temporal_contribution = temporal_scale * vel_magnitude * vel_direction

            # Geometric contribution: repulsion-based correction
            geometric_contribution = geometric_correction

            # Combined update
            combined_update = temporal_weight * temporal_contribution + geometric_weight * geometric_contribution

            # Final population update
            new_population = self.population + combined_update

        except Exception:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_10_catB_idea_0.py (2 wins) ---
```python
def _position_update_temporal_drift(self):
        """Subspace power-law velocity modulation via SVD (Category B).

        Key insight: Tasks 17, 16, 6, 5 (worst unsolved) have errors ~1e+02 to 1e+04,
        indicating the population collapses into low-dimensional subspaces and
        cannot escape. SVD decomposition reveals this collapse via the singular
        value spectrum. A power-law modulation boosts velocity along collapsed
        directions (small singular values) while dampening dominant directions,
        re-establishing exploration in the full subspace.

        Mechanism:
        - SVD of centered population → singular values reveal effective dimensionality
        - Power-law scaling: scale_i = (sigma_i / sigma_max) ** power
          where power > 1 amplifies small singular values more aggressively
        - Condition number of singular values drives the power coefficient
        - Directional projection onto principal right singular vectors for velocity update
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)

            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            # === SPECTRAL ANALYSIS ===
            total_var = np.sum(singular_values ** 2) + 1e-10
            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Condition number for anisotropy detection
            cond = singular_values[0] / (singular_values[-1] + 1e-10)

            # Effective dimensionality from normalized singular value spectrum
            cumvar = np.cumsum(singular_values ** 2) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / len(singular_values)

            # === POWER-LAW MODULATION ===
            # Power > 1: stronger amplification of collapsed directions
            # Power scales with condition number (more anisotropic → stronger correction)
            base_power = 1.5
            power_boost = np.log1p(cond) / np.log1p(1000.0) if cond > 10.0 else 0.0
            power = base_power + 0.8 * power_boost
            power = np.clip(power, 1.0, 3.0)

            # Power-law scale: (sigma_i / sigma_max) ** power
            # Small singular values get boosted more (power > 1)
            power_scale = sv_norm ** power
            power_scale = power_scale / (np.max(power_scale) + 1e-10)

            # Blend with linear scale for stability
            blend = np.clip((cond - 5.0) / 50.0, 0.0, 1.0)
            per_dir_scale = (1.0 - blend) * sv_norm + blend * power_scale
            per_dir_scale = np.clip(per_dir_scale, 0.1, 2.5)

            # === VELOCITY UPDATE IN SVD SUBSPACE ===
            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_dir_scale

            # === SUBSPACE REHABILITATION FOR COLLAPSED POPULATION ===
            # When eff_dim_ratio < 0.3, population is severely collapsed
            # Inject random perturbations along the weakest singular directions
            collapse_threshold = 0.3
            rehabilitation = np.zeros((self.np, self.dim))
            if eff_dim_ratio < collapse_threshold:
                # Identify weak directions (smallest singular values)
                n_weak = max(1, int(len(singular_values) * (1.0 - eff_dim_ratio)))
                weak_directions = Vt[:n_weak]  # Bottom singular vectors

                # Random perturbation magnitude scaled by inverse of singular value
                # (weaker directions get stronger perturbations)
                weak_sv_norm = sv_norm[:n_weak]
                perturbation_mag = 0.3 * (1.0 - eff_dim_ratio / collapse_threshold)
                perturbation_mag = np.clip(perturbation_mag, 0.05, 0.5)

                for i in range(self.np):
                    # Perturbation proportional to 1/sigma for each weak direction
                    for j, (sv_val, wdir) in enumerate(zip(weak_sv_norm, weak_directions)):
                        inv_sv = 1.0 / (sv_val + 0.01)
                        scale = perturbation_mag * inv_sv / (np.sum(1.0 / (weak_sv_norm + 0.01)) + 1e-10)
                        rehabilitation[i] += scale * wdir * np.random.randn()

            # === COMBINE SPECTRAL UPDATE WITH SUBSPACE REHABILITATION ===
            new_population = self.population + vel_scaled + rehabilitation

            # === ANISOTROPY-DRIVEN EXPLORATION BOOST ===
            # High condition number → increase global exploration scale
            if cond > 100.0:
                exploration_scale = 1.0 + 0.3 * np.log1p(cond) / np.log1p(1000.0)
                exploration_scale = np.clip(exploration_scale, 1.0, 1.5)
                # Add small isotropic perturbation for high-anisotropy landscapes
                noise = np.random.randn(*new_population.shape) * 0.1 * exploration_scale
                new_population = new_population + noise

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