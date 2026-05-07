Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_position_update_fitness_rank` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.406534e-01            1.353480e-01            -inf                    1.413454e-01            -inf                    -inf                    1.407437e-01            1.265054e-01            1.471002e-01            1.387495e-01            
1      1.886416e+00            2.883464e+00            -inf                    1.520420e+00            -inf                    -inf                    1.608470e+00            2.241957e+00            1.991807e+00            3.204759e+00            
2      2.069233e-01            2.066830e-01            -inf                    1.950438e-01            -inf                    -inf                    1.905297e-01            2.204001e-01            2.255200e-01            1.719787e-01            
3      5.508458e+00            5.789713e+00            -inf                    5.484056e+00            -inf                    -inf                    5.741322e+00            5.673726e+00            5.481900e+00            5.716479e+00            
4      1.292211e+00            1.271073e+00            -inf                    1.294267e+00            -inf                    -inf                    1.299502e+00            1.235580e+00            1.313718e+00            1.190376e+00            
5      3.911267e+02            2.311024e+03            -inf                    1.596018e+03            -inf                    -inf                    1.326734e+04            1.076315e+03            4.527577e+02            2.387346e+03            
6      5.371716e+02            5.649247e+02            -inf                    5.579992e+02            -inf                    -inf                    5.358054e+02            6.033257e+02            4.996248e+02            5.357820e+02            
7      6.698704e+00            6.087861e+00            -inf                    6.244595e+00            -inf                    -inf                    6.167367e+00            6.235949e+00            7.021159e+00            6.583384e+00            
8      2.560489e+00            2.706876e+00            -inf                    2.835639e+00            -inf                    -inf                    2.684787e+00            2.780911e+00            2.838238e+00            2.748769e+00            
9      4.203225e+01            5.105996e+01            -inf                    3.417078e+01            -inf                    -inf                    4.881874e+01            4.489605e+01            4.243245e+01            4.337738e+01            
10     2.485380e-01            2.504958e-01            -inf                    2.754550e-01            -inf                    -inf                    2.393698e-01            2.637952e-01            2.944420e-01            2.420691e-01            
11     8.581919e+01            8.285404e+01            -inf                    9.706306e+01            -inf                    -inf                    9.579834e+01            7.781548e+01            8.481976e+01            5.992204e+01            
12     4.561047e-01            1.122360e+00            -inf                    6.350140e-01            -inf                    -inf                    3.774695e-01            6.398172e-01            9.138801e-01            6.479370e-01            
13     1.849799e+01            2.055817e+01            -inf                    1.780219e+01            -inf                    -inf                    2.017311e+01            1.642697e+01            2.221328e+01            1.856773e+01            
14     8.655709e+00            8.143296e+00            -inf                    7.574845e+00            -inf                    -inf                    8.823878e+00            8.496657e+00            7.649469e+00            8.229281e+00            
15     3.200086e+00            3.248571e+00            -inf                    3.179808e+00            -inf                    -inf                    3.206041e+00            3.077300e+00            3.103680e+00            3.169167e+00            
16     7.721051e+02            7.815695e+02            -inf                    1.256802e+03            -inf                    -inf                    9.806625e+02            9.371196e+02            9.210850e+02            9.532041e+02            
17     1.847027e+04            2.399623e+04            -inf                    2.318057e+04            -inf                    -inf                    1.389876e+04            3.583194e+04            1.680553e+04            2.851154e+04            
18     3.492771e+01            4.307200e+01            -inf                    3.351819e+01            -inf                    -inf                    4.097711e+01            3.927400e+01            3.772190e+01            3.772236e+01            
19     9.076116e+01            7.131760e+01            -inf                    8.041037e+01            -inf                    -inf                    8.994106e+01            7.355805e+01            7.775330e+01            7.482728e+01            
20     2.053786e+01            2.123874e+01            -inf                    2.038716e+01            -inf                    -inf                    2.031517e+01            2.032739e+01            2.215374e+01            2.081960e+01            
21     5.405128e+00            5.685357e+00            -inf                    5.525696e+00            -inf                    -inf                    5.541852e+00            5.492945e+00            5.415289e+00            5.715721e+00            
22     1.126015e+01            1.053704e+01            -inf                    1.114655e+01            -inf                    -inf                    1.183438e+01            1.155961e+01            1.230456e+01            1.090986e+01            
23     5.954521e+01            5.815441e+01            -inf                    5.896350e+01            -inf                    -inf                    5.790575e+01            5.774847e+01            5.683595e+01            5.935908e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_08_catH_idea_0.py  (error=1.265054e-01)
Task  1: variant_04_catD_idea_0.py  (error=1.520420e+00)
Task  2: variant_10_catB_idea_0.py  (error=1.719787e-01)
Task  3: variant_09_catA_idea_0.py  (error=5.481900e+00)
Task  4: variant_10_catB_idea_0.py  (error=1.190376e+00)
Task  5: original.py  (error=3.911267e+02)
Task  6: variant_09_catA_idea_0.py  (error=4.996248e+02)
Task  7: variant_02_catB_idea_0.py  (error=6.087861e+00)
Task  8: original.py  (error=2.560489e+00)
Task  9: variant_04_catD_idea_0.py  (error=3.417078e+01)
Task 10: variant_07_catG_idea_0.py  (error=2.393698e-01)
Task 11: variant_10_catB_idea_0.py  (error=5.992204e+01)
Task 12: variant_07_catG_idea_0.py  (error=3.774695e-01)
Task 13: variant_08_catH_idea_0.py  (error=1.642697e+01)
Task 14: variant_04_catD_idea_0.py  (error=7.574845e+00)
Task 15: variant_08_catH_idea_0.py  (error=3.077300e+00)
Task 16: original.py  (error=7.721051e+02)
Task 17: variant_07_catG_idea_0.py  (error=1.389876e+04)
Task 18: variant_04_catD_idea_0.py  (error=3.351819e+01)
Task 19: variant_02_catB_idea_0.py  (error=7.131760e+01)
Task 20: variant_07_catG_idea_0.py  (error=2.031517e+01)
Task 21: original.py  (error=5.405128e+00)
Task 22: variant_02_catB_idea_0.py  (error=1.053704e+01)
Task 23: variant_09_catA_idea_0.py  (error=5.683595e+01)

WIN COUNTS:
  variant_04_catD_idea_0.py: 4 wins
  original.py: 4 wins
  variant_07_catG_idea_0.py: 4 wins
  variant_08_catH_idea_0.py: 3 wins
  variant_10_catB_idea_0.py: 3 wins
  variant_09_catA_idea_0.py: 3 wins
  variant_02_catB_idea_0.py: 3 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_08_catH_idea_0.py       1.2651e-01       variant_02_catB_idea_0.py        1.3535e-01         1.1×    1.1×
     1   variant_04_catD_idea_0.py       1.5204e+00       variant_07_catG_idea_0.py        1.6085e+00         1.1×    1.4×
     2   variant_10_catB_idea_0.py       1.7198e-01       variant_07_catG_idea_0.py        1.9053e-01         1.1×    1.2×
     3   variant_09_catA_idea_0.py       5.4819e+00       variant_04_catD_idea_0.py        5.4841e+00         1.0×    1.0×
     4   variant_10_catB_idea_0.py       1.1904e+00       variant_08_catH_idea_0.py        1.2356e+00         1.0×    1.1×
     5   original.py                     3.9113e+02       variant_09_catA_idea_0.py        4.5276e+02         1.2×    5.0×
     6   variant_09_catA_idea_0.py       4.9962e+02       variant_10_catB_idea_0.py        5.3578e+02         1.1×    1.1×
     7   variant_02_catB_idea_0.py       6.0879e+00       variant_07_catG_idea_0.py        6.1674e+00         1.0×    1.1×
     8   original.py                     2.5605e+00       variant_07_catG_idea_0.py        2.6848e+00         1.0×    1.1×
     9   variant_04_catD_idea_0.py       3.4171e+01       original.py                      4.2032e+01         1.2×    1.3×
    10   variant_07_catG_idea_0.py       2.3937e-01       variant_10_catB_idea_0.py        2.4207e-01         1.0×    1.1×
    11   variant_10_catB_idea_0.py       5.9922e+01       variant_08_catH_idea_0.py        7.7815e+01         1.3×    1.4×
    12   variant_07_catG_idea_0.py       3.7747e-01       original.py                      4.5610e-01         1.2×    1.7×
    13   variant_08_catH_idea_0.py       1.6427e+01       variant_04_catD_idea_0.py        1.7802e+01         1.1×    1.2×
    14   variant_04_catD_idea_0.py       7.5748e+00       variant_09_catA_idea_0.py        7.6495e+00         1.0×    1.1×
    15   variant_08_catH_idea_0.py       3.0773e+00       variant_09_catA_idea_0.py        3.1037e+00         1.0×    1.0×
    16   original.py                     7.7211e+02       variant_02_catB_idea_0.py        7.8157e+02         1.0×    1.2×
    17   variant_07_catG_idea_0.py       1.3899e+04       variant_09_catA_idea_0.py        1.6806e+04         1.2×    1.7×
    18   variant_04_catD_idea_0.py       3.3518e+01       original.py                      3.4928e+01         1.0×    1.1×
    19   variant_02_catB_idea_0.py       7.1318e+01       variant_08_catH_idea_0.py        7.3558e+01         1.0×    1.1×
    20   variant_07_catG_idea_0.py       2.0315e+01       variant_08_catH_idea_0.py        2.0327e+01         1.0×    1.0×
    21   original.py                     5.4051e+00       variant_09_catA_idea_0.py        5.4153e+00         1.0×    1.0×
    22   variant_02_catB_idea_0.py       1.0537e+01       variant_10_catB_idea_0.py        1.0910e+01         1.0×    1.1×
    23   variant_09_catA_idea_0.py       5.6836e+01       variant_08_catH_idea_0.py        5.7748e+01         1.0×    1.0×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 7 (original.py, variant_02_catB_idea_0.py, variant_04_catD_idea_0.py, variant_07_catG_idea_0.py, variant_08_catH_idea_0.py, variant_09_catA_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 0  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_catB_idea_0.py (3 wins) ---
```python
def _position_update_fitness_rank(self):
        """Eigenvalue-anisotropic velocity scaling with condition-number modulation (Category B).

        Key insight: Use eigenvalue decomposition of population covariance (not SVD on centered
        data) to detect anisotropy and ill-conditioning. Scale velocity inversely along principal
        axes proportional to eigenvalue magnitude. Condition number modulates global exploration
        intensity. This is orthogonal to graph-based (E) and fitness-rank (D) approaches.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)

            # Effective dimensionality from eigenvalue spectrum
            total_var = np.sum(eigenvalues) + 1e-10
            sorted_desc = np.sort(eigenvalues)[::-1]
            cumvar = np.cumsum(sorted_desc) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1

            # Spectral signal from condition number: high cond = anisotropic landscape
            spectral_signal = np.clip(np.log1p(cond) / np.log1p(1e6), 0.0, 1.0)

            # Adaptive inertia from spectral signal
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = 0.4 + 0.55 * spectral_signal
            self.inertia_weight = np.clip(
                self.inertia_weight * 0.8 + (target_inertia * 0.6 + decay * 0.4) * 0.2,
                0.4, 0.95
            )

            # Fitness rank signal
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            # Spectral exploration factor
            # High condition number = clustered/ill-conditioned → more exploration
            spectral_explore = 1.0 + 1.0 * spectral_signal
            spectral_explore = np.clip(spectral_explore, 0.5, 2.5)

            # Velocity projection onto principal axes
            vel_proj = self.velocity @ eigenvectors

            # Inverse eigenvalue scaling: dampen high-variance directions, preserve low-variance
            sv_norm = eigenvalues / (eigenvalues[0] + 1e-10)
            # Use squared inverse for stronger effect on collapsed directions
            per_component_scale = 1.0 / (sv_norm ** 2 + 0.01)
            per_component_scale = per_component_scale / (np.max(per_component_scale) + 1e-10)

            # Blend uniform and per-component scaling based on condition number
            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            combined_scale = uniform_scale * (1.0 - blend_weight) + per_component_scale * blend_weight
            combined_scale = np.clip(combined_scale, 0.3, 2.5)

            vel_scaled = vel_proj * combined_scale * spectral_explore

            # Fitness-based directional perturbation
            fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional = fitness_perturb * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))

            # Random perturbation scaled by spectral signal
            random_perturb = spectral_explore * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

            new_population = self.population + self.inertia_weight * vel_scaled + directional + random_perturb

        except (np.linalg.LinAlgError, FloatingPointError):
            spectral_explore = 1.0 + 0.5 * np.random.random()
            new_population = self.population + self.inertia_weight * self.velocity * spectral_explore

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_04_catD_idea_0.py (4 wins) ---
```python
def _position_update_fitness_rank(self):
        """Fitness-percentile & Kendall-tau rank stability modulation (Category D).

        Uses ONLY fitness signals — no distances, no graph structures, no covariance:
        - Fitness percentile ranks per particle (scale-invariant)
        - Kendall tau rank correlation between consecutive generations (stagnation detection)
        - Fitness spread ratio (p90-p10)/p50 for convergence detection
        - Success-history EMA per particle
        - Rank-based directional perturbation toward best fitness region
        """
        try:
            # === FITNESS RANK COMPUTATION ===
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            # === FITNESS PERCENTILE SIGNALS ===
            p10 = np.percentile(self.current_fitness, 10)
            p50 = np.percentile(self.current_fitness, 50)
            p90 = np.percentile(self.current_fitness, 90)
            fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)

            # === KENDALL TAU RANK STABILITY (temporal rank correlation) ===
            if hasattr(self, '_prev_fitness') and len(self._prev_fitness) == len(self.current_fitness):
                # Concordant pairs: both ranks improved or both worsened
                f_improved = self.current_fitness < self._prev_fitness
                # Count discordant swaps in rank order
                n = len(self.current_fitness)
                concordant = 0
                discordant = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        curr_order = self.current_fitness[i] < self.current_fitness[j]
                        prev_order = self._prev_fitness[i] < self._prev_fitness[j]
                        if curr_order == prev_order:
                            concordant += 1
                        else:
                            discordant += 1
                n_pairs = n * (n - 1) / 2
                kendall_tau = (concordant - discordant) / (n_pairs + 1e-10)
            else:
                kendall_tau = 0.5  # Neutral on first evaluation
            self._prev_fitness = self.current_fitness.copy()

            # === SUCCESS HISTORY PER PARTICLE ===
            if not hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._success_count[improved] += 1
            self._success_count[~improved] *= 0.85
            success_norm = self._success_count / (np.max(self._success_count) + 1.0)

            # === CONVERGENCE PHASE DETECTION ===
            is_converged = fitness_spread < 0.05
            is_stagnant = kendall_tau > 0.75  # High tau = ranks barely changing = trapped

            # === GLOBAL EXPLORATION SCALE FROM RANK SIGNALS ===
            if is_stagnant and is_converged:
                global_scale = 1.8  # Both stuck and converged = deep trap
            elif is_stagnant:
                global_scale = 1.4  # Stuck but spread = exploring badly
            elif is_converged:
                global_scale = 0.7  # Converging well = exploit
            else:
                global_scale = 1.0  # Normal operation

            # === PER-PARTICLE VELOCITY MODULATION FROM FITNESS RANK ===
            # Higher rank (worse fitness) = more exploration boost
            rank_explore = 1.0 + 0.6 * fitness_ranks
            # Particles with low success = trapped = extra exploration
            success_explore = 1.0 + 0.4 * (1.0 - success_norm)
            # Combine: rank signal drives direction, success signal drives magnitude
            vel_scale = rank_explore[:, np.newaxis] * global_scale * success_explore[:, np.newaxis]
            vel_scale = np.clip(vel_scale, 0.2, 3.0)

            # === RANK-BASED DIRECTIONAL PERTURBATION ===
            # Compute rank-based best direction (not spatial)
            best_rank_idx = np.argmin(fitness_ranks)
            # Attraction strength inversely proportional to rank (best particles attracted less)
            attract_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])

            # Random rank-based direction using random projection (no spatial info)
            random_dir = np.random.randn(self.np, self.dim)
            random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-10)

            # Blend: attract toward best region + random exploration
            if self.global_best is not None and kendall_tau < 0.5:
                # Only attract when not too stagnant
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional = attract_strength * to_best_dir
            else:
                # Stagnant: pure random exploration in rank space
                directional = attract_strength * random_dir

            # === RANDOM PERTURBATION SCALED BY STAGNATION ===
            random_strength = 0.3 * (1.0 + kendall_tau)  # More random when stagnant
            random_perturb = random_strength * np.random.uniform(-1.0, 1.0, (self.np, self.dim))

            # === COMBINE UPDATES ===
            new_population = self.population + vel_scale * self.velocity + directional + random_perturb

        except:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_07_catG_idea_0.py (4 wins) ---
```python
def _position_update_fitness_rank(self):
        """Monte Carlo convergence-probability velocity modulation (Category G).

        Fundamentally stochastic approach using:
        1. Monte Carlo subsampling: estimate per-particle convergence probability
           by repeatedly sampling subsets and measuring how often they reach global best.
           High convergence_prob → exploit; Low → explore.
        2. Bootstrap resampling of population to estimate covariance uncertainty
           and derive per-particle variance signals for adaptive scaling.
        3. Fitness-weighted random probing for diverse exploration.

        This replaces the deterministic k-NN graph / Laplacian spectral gap (Category E)
        with a sampling-based estimate of convergence likelihood. No eigenvalues,
        no graph traversal — pure stochastic estimation.
        """
        try:
            # === MONTE CARLO CONVERGENCE PROBABILITY ===
            # Repeatedly subsample population and check if subsample reaches global best.
            # This directly estimates how "useful" each particle is for finding improvement.
            n_mc_samples = 50
            mc_subsample_size = max(3, self.np // 4)

            if not hasattr(self, '_mc_conv_prob'):
                self._mc_conv_prob = np.ones(self.np) * 0.5

            if self.global_best is not None and self.np >= mc_subsample_size:
                conv_counts = np.zeros(self.np)

                for _ in range(n_mc_samples):
                    # Random subsample
                    subsample_idx = np.random.choice(self.np, mc_subsample_size, replace=False)
                    subsample_pop = self.population[subsample_idx]

                    # Evaluate fitness of subsample
                    subsample_fitness, _ = self._evaluate_batch(subsample_pop, self._current_func)

                    # Check if subsample best is better than current global best
                    subsample_best_fitness = np.min(subsample_fitness)

                    # For each particle in subsample, record if subsample found improvement
                    for idx in subsample_idx:
                        if subsample_best_fitness < self.global_best_fitness * 1.001:
                            conv_counts[idx] += 1

                # Per-particle convergence probability
                new_conv_prob = conv_counts / n_mc_samples

                # EMA smoothing to reduce variance
                self._mc_conv_prob = 0.3 * new_conv_prob + 0.7 * self._mc_conv_prob
            else:
                self._mc_conv_prob = np.ones(self.np) * 0.5

            mc_conv_prob = np.clip(self._mc_conv_prob, 0.1, 0.9)

            # === BOOTSTRAP COVARIANCE UNCERTAINTY ===
            # Bootstrap resample population to estimate covariance uncertainty
            n_bootstrap = 30
            bootstrap_eigenvalue_samples = []

            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(self.np, self.np, replace=True)
                boot_pop = self.population[boot_idx]
                try:
                    centered = boot_pop - np.mean(boot_pop, axis=0)
                    cov = np.cov(centered.T)
                    eigenvalues = np.linalg.eigvalsh(cov)
                    eigenvalues = np.clip(eigenvalues, 1e-10, None)
                    # Record condition number as proxy for anisotropy
                    cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10) if len(eigenvalues) > 1 else 1.0
                    bootstrap_eigenvalue_samples.append(cond)
                except:
                    pass

            if len(bootstrap_eigenvalue_samples) > 5:
                bootstrap_cond_mean = np.mean(bootstrap_eigenvalue_samples)
                bootstrap_cond_std = np.std(bootstrap_eigenvalue_samples)
                # Uncertainty: high std relative to mean = uncertain population structure
                bootstrap_uncertainty = bootstrap_cond_std / (bootstrap_cond_mean + 1e-10)
            else:
                bootstrap_cond_mean = 1.0
                bootstrap_uncertainty = 0.5

            # === BOOTSTRAP-BASED GLOBAL SCALE ===
            # High uncertainty → more exploration (higher scale)
            # Low uncertainty → more exploitation (lower scale)
            uncertainty_explore = 1.0 + 0.4 * np.clip(bootstrap_uncertainty, 0.0, 2.0)

            if bootstrap_cond_mean > 10.0:
                # High condition number = anisotropic landscape → more exploration
                cond_explore = 1.3
            elif bootstrap_cond_mean < 3.0:
                cond_explore = 0.85
            else:
                cond_explore = 1.0

            global_scale = uncertainty_explore * cond_explore
            global_scale = np.clip(global_scale, 0.5, 2.0)

            # === PER-PARTICLE VELOCITY MODULATION FROM CONVERGENCE PROBABILITY ===
            # High convergence_prob → particles are in promising regions → exploit
            # Low convergence_prob → particles may be in unexplored regions → explore
            vel_scale = np.where(
                mc_conv_prob > 0.6,
                global_scale * 0.7,  # Exploit mode
                global_scale * 1.4   # Explore mode
            )
            vel_scale = np.clip(vel_scale, 0.3, 2.5)

            # === LATIN HYPERCUBE-DERIVED DIRECTIONAL PROBING ===
            # Use stratified sampling to generate diverse exploration directions
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm

                # Stratified random perturbation: divide dimension space into strata
                n_strata = max(2, self.dim // 4)
                strat_idx = np.random.randint(0, n_strata, size=(self.np,))
                stratum_width = 1.0 / n_strata
                stratum_scale = (strat_idx + np.random.random(self.np)) * stratum_width

                # Apply stratified perturbation along global-best direction
                directional = 0.4 * stratum_scale[:, np.newaxis] * to_best_dir
            else:
                # Random exploration with stratified sampling
                directional = np.random.uniform(-0.5, 0.5, (self.np, self.dim))

            # === FITNESS-WEIGHTED RANDOM PERTURBATION ===
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            # Success history
            if not hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._success_count[improved] += 1
            self._success_count[~improved] *= 0.9
            success_norm = self._success_count / (np.max(self._success_count) + 1.0)

            # Random perturbation inversely proportional to convergence probability
            random_scale = (1.0 - mc_conv_prob)[:, np.newaxis] * (1.0 - 0.3 * success_norm[:, np.newaxis])
            random_perturb = random_scale * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

            # === COMBINE ===
            new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + random_perturb

        except Exception:
            # Fallback: simple random perturbation with global scaling
            new_population = self.population + 1.2 * self.velocity + np.random.uniform(-0.3, 0.3, (self.np, self.dim))

        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_08_catH_idea_0.py (3 wins) ---
```python
def _position_update_fitness_rank(self):
        """Hybrid: Graph spectral gap + temporal improvement rate (Category H).

        Combines TWO distinct mechanisms with principled data-driven weighting:
        1. Graph-Laplacian spectral gap: detects spatial clustering of population
        2. Temporal fitness improvement rate: detects convergence/stagnation over time

        Stagnation score = (1 - spectral_gap) * (1 - norm_improvement_rate)
        determines exploration vs exploitation balance:
          - High stagnation → exploration mode (higher velocity scale, larger random perturbations)
          - Low stagnation → exploitation mode (lower velocity scale, stronger directional pull)

        Targets worst tasks (17, 16, 6, 5) where deceptive landscapes trap the swarm.
        """
        # --- Mechanism 1: Graph-Laplacian spectral gap ---
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Connected component analysis
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

        # Spectral gap from normalized Laplacian
        try:
            row_indices = np.repeat(np.arange(self.np), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
            adj = adj + adj.T
            adj.data[:] = 1.0

            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
            lap = dspy.identity(self.np) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)

            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
        except:
            spectral_gap = 1.0

        # --- Mechanism 2: Temporal fitness improvement rate ---
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(float(np.min(self.current_fitness)))
        if len(self._fitness_history) > 20:
            self._fitness_history.pop(0)

        # Compute improvement rate over sliding window
        if len(self._fitness_history) >= 5:
            recent = np.array(self._fitness_history[-5:])
            oldest, newest = recent[0], recent[-1]
            if abs(oldest) > 1e-10:
                improvement_rate = (oldest - newest) / (abs(oldest) + 1e-10)
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = 0.0

        norm_improvement = np.clip(improvement_rate, 0.0, 1.0)

        # --- Principled weighting: stagnation score drives switch ---
        stagnation_score = (1.0 - spectral_gap) * (1.0 - norm_improvement)
        stagnation_score = np.clip(stagnation_score, 0.0, 1.0)

        # Exploration mode when stagnation is high
        is_exploring = stagnation_score > 0.3
        is_stuck = stagnation_score > 0.6

        # Global scale: exploration bias when stuck
        if is_stuck:
            global_scale = 1.5
        elif is_exploring:
            global_scale = 1.15
        else:
            global_scale = 0.85

        # Fitness rank signal
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Success history
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # Velocity scaling combines stagnation-driven global scale with per-particle signals
        vel_scale = global_scale * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        # Directional perturbation toward global best (fitness-guided)
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        # Random perturbation: stronger when stuck/exploring
        random_scale = 0.5 if is_exploring else 0.25
        random_perturb = np.random.uniform(-random_scale, random_scale, (self.np, self.dim))

        # Component-size-based nudge: small component particles get extra push toward larger components
        size_factor = particle_component_size / max(1, self.np)
        component_nudge = np.zeros((self.np, self.dim))
        for comp_idx, comp in enumerate(components):
            if len(comp) < self.np * 0.3:
                centroid = np.mean(self.population[comp], axis=0)
                for p_idx in comp:
                    to_centroid = centroid - self.population[p_idx]
                    to_centroid_norm = np.linalg.norm(to_centroid) + 1e-10
                    component_nudge[p_idx] = 0.2 * (1.0 - size_factor[p_idx]) * (to_centroid / to_centroid_norm)

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb + component_nudge
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_09_catA_idea_0.py (3 wins) ---
```python
def _position_update_fitness_rank(self):
        """Axis-aligned bounding box + convex hull + pairwise distance modulation (Category A).

        Key insight: Use ONLY geometric properties of particle positions:
        - Axis-aligned spread per dimension to detect anisotropy
        - Convex hull volume to detect population collapse
        - Pairwise distance distribution to detect clustering

        Purely spatial — no graph topology, no fitness signals, no temporal tracking.
        Targets worst tasks (17, 16, 6, 5) where swarm collapses to wrong regions.
        """
        pop = self.population
        np_particles, dim = pop.shape

        # === AXIS-ALIGNED BOUNDING BOX ANALYSIS ===
        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        pop_range = pop_max - pop_min + 1e-10
        global_range = self.upper_bound - self.lower_bound

        # Per-particle coverage: how much of the search space each particle occupies
        centroid = np.mean(pop, axis=0)
        particle_to_centroid = pop - centroid
        # Signed position within bounding box [-1, 1]
        normalized_pos = 2.0 * (pop - pop_min) / pop_range - 1.0

        # Detect boundary crowding: particles near box edges
        edge_margin = 0.1
        near_edge = np.max(np.abs(normalized_pos), axis=1) > (1.0 - edge_margin)
        edge_pressure = np.sum(near_edge) / np_particles

        # Per-dimension spread ratio (detect anisotropy)
        spread_ratios = pop_range / (global_range + 1e-10)
        min_spread = np.min(spread_ratios)
        max_spread = np.max(spread_ratios)
        anisotropy = max_spread / (min_spread + 1e-10) if min_spread > 1e-10 else 1.0

        # === CONVEX HULL VOLUME (2D/3D approximation or high-D estimate) ===
        try:
            if dim <= 10 and np_particles >= dim + 1:
                # Use QHull via scipy for convex hull
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pop, qhull_options='Qt')
                hull_volume = hull.volume  # In QHull, 'volume' is (n-1)-simplex measure
            else:
                # High-dimensional approximation: product of spreads
                hull_volume = np.prod(pop_range + 1e-10)
        except:
            hull_volume = np.prod(pop_range + 1e-10)

        # Normalize by maximum possible volume
        max_volume = (global_range ** dim)
        normalized_volume = hull_volume / (max_volume + 1e-10)
        normalized_volume = np.clip(normalized_volume, 1e-10, 1.0)

        # === PAIRWISE DISTANCE DISTRIBUTION ===
        sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        all_dists = np.sqrt(sq_dists[sq_dists < np.inf])

        if len(all_dists) > 0:
            mean_dist = np.mean(all_dists)
            std_dist = np.std(all_dists)
            min_dist = np.min(all_dists)

            # Clustering ratio: mean distance vs expected uniform distance
            expected_uniform_dist = np.mean(pop_range) * 0.5
            clustering_ratio = mean_dist / (expected_uniform_dist + 1e-10)
        else:
            mean_dist = 1.0
            std_dist = 1.0
            min_dist = 1.0
            clustering_ratio = 1.0

        # === GEOMETRIC EXPLORATION SIGNALS ===
        # Small hull volume = collapsed population → boost exploration
        collapse_explore = 1.0 + 1.5 * (1.0 - normalized_volume)
        collapse_explore = np.clip(collapse_explore, 0.5, 3.0)

        # High anisotropy = elongated cluster → directional spreading
        anisotropy_explore = 1.0 + 0.3 * np.log1p(anisotropy) / np.log1p(100.0)
        anisotropy_explore = np.clip(anisotropy_explore, 0.8, 2.0)

        # Edge crowding → push particles inward
        inward_pressure = edge_pressure * 0.5

        # Small std_dist = over-clustered → add spread
        spread_explore = 1.0 + 0.5 * (1.0 - np.clip(std_dist / (mean_dist + 1e-10), 0.0, 1.0))
        spread_explore = np.clip(spread_explore, 0.5, 2.5)

        # Combined geometric exploration factor
        geo_explore = collapse_explore * anisotropy_explore * spread_explore
        geo_explore = np.clip(geo_explore, 0.3, 4.0)

        # === PER-PARTICLE GEOMETRIC MODULATION ===
        # Particles near edges get inward nudge
        edge_dist = 1.0 - np.max(np.abs(normalized_pos), axis=1)
        inward_strength = inward_pressure * (1.0 - edge_dist)[:, np.newaxis]
        to_centroid = centroid - pop
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        inward_nudge = inward_strength * (to_centroid / to_centroid_dist)

        # === VELOCITY MODULATION FROM GEOMETRY ===
        # Anisotropic scaling: extend along smallest-spread dimensions
        try:
            centered = pop - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)

            # Inverse spread weighting: boost along collapsed directions
            inv_spread = 1.0 / (eigenvalues / (eigenvalues[0] + 1e-10) + 0.01)
            inv_spread = inv_spread / (np.max(inv_spread) + 1e-10)

            vel_proj = self.velocity @ eigenvectors
            vel_corrected = vel_proj * inv_spread
            anisotropic_correction = (vel_corrected @ eigenvectors.T) - self.velocity
        except:
            anisotropic_correction = np.zeros_like(self.velocity)

        # === DIRECTIONAL PERTURBATION FROM GEOMETRY ===
        # Random perturbation scaled by geometric exploration
        random_perturb = geo_explore * np.random.uniform(-0.5, 0.5, (np_particles, dim))

        # === COMBINE ===
        new_population = pop + self.velocity * geo_explore + anisotropic_correction + inward_nudge + random_perturb
        self.population = self._clip_to_bounds(new_population)
```

# --- From variant_10_catB_idea_0.py (3 wins) ---
```python
def _position_update_fitness_rank(self):
        """Covariance-eigenvalue effective-rank velocity modulation (Category B).

        Key insight: Use effective rank (entropy of normalized covariance eigenvalues)
        as a diversity signal, combined with per-eigendirection velocity scaling.
        High effective rank = diverse population → boost exploration.
        Low effective rank = collapsed population → amplify velocity in weak directions.
        Condition number modulates blend between uniform and anisotropy-targeted scaling.
        This is orthogonal to SVD-whitening (variant_02) which used inverse-square singular
        value modulation on position, not eigenvalue-ratio velocity scaling.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            # Covariance eigenvalue decomposition
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Effective rank: entropy-based dimensionality measure
            total_eig = np.sum(eigenvalues)
            eigenvalues_norm = eigenvalues / (total_eig + 1e-10)
            eigenvalues_norm = np.clip(eigenvalues_norm, 1e-15, None)
            entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-15))
            max_entropy = np.log(len(eigenvalues_norm))
            effective_rank = np.exp(entropy) / (len(eigenvalues_norm) + 1e-10)
            effective_rank = np.clip(effective_rank, 0.0, 1.0)

            # Condition number for anisotropy detection
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)
            cond_signal = np.log1p(cond) / np.log1p(10000.0)

            # Per-eigendirection velocity scaling
            eig_ratio = eigenvalues / (eigenvalues[0] + 1e-10)
            eig_ratio = np.clip(eig_ratio, 0.0, 1.0)

            # Anisotropy-targeted scaling: penalize dominant, amplify weak
            # Low eig_ratio = weak direction → amplify
            # High eig_ratio = dominant direction → dampen
            spectral_scale = np.sqrt(eig_ratio + 0.01)
            spectral_scale = spectral_scale / (np.max(spectral_scale) + 1e-10)

            # Blend uniform and anisotropy-targeted based on condition number
            uniform_scale = 1.0
            blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            per_component_scale = uniform_scale * (1.0 - blend) + spectral_scale * blend

            # Project velocity to eigenspace, scale, project back
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_component_scale
            velocity_correction = vel_scaled @ eigenvectors.T - self.velocity

            # Exploration modulation from effective rank
            # High effective_rank → population diverse → more exploration
            # Low effective_rank → population collapsed → more exploration to escape
            if effective_rank < 0.3:
                explore_boost = 1.5
            elif effective_rank > 0.7:
                explore_boost = 1.2
            else:
                explore_boost = 1.0

            # Add scaled velocity correction + random perturbation
            random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
            random_perturb *= explore_boost * (1.0 + 0.5 * cond_signal)

            new_population = self.population + self.velocity + 0.5 * velocity_correction + random_perturb

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