Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or routes between) the best `_adapt_neighborhood_size` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.631587e-01            -inf                    1.328386e-01            1.470174e-01            1.359389e-01            1.447163e-01            1.610435e-01            1.400823e-01            1.427722e-01            1.665369e-01            
1      2.701172e+00            -inf                    1.878269e+00            2.981502e+00            2.609748e+00            2.874549e+00            2.120599e+00            3.301404e+00            2.918820e+00            2.642750e+00            
2      4.252344e-01            -inf                    1.831771e-01            1.909204e-01            1.803783e-01            2.203713e-01            2.372786e-01            7.561509e-01            2.067881e-01            4.096725e-01            
3      6.780165e+00            -inf                    5.441582e+00            5.779058e+00            5.585310e+00            5.597714e+00            5.876529e+00            5.432250e+00            5.434965e+00            6.330331e+00            
4      1.188679e+00            -inf                    1.165388e+00            1.257042e+00            1.317906e+00            1.316692e+00            1.204913e+00            1.333625e+00            1.179427e+00            1.307177e+00            
5      2.633829e+01            -inf                    6.390744e+02            1.040301e+03            1.064555e+03            1.768949e+03            8.915312e+02            4.049603e+03            1.717194e+03            5.957466e+00            
6      6.861531e+02            -inf                    3.676451e+02            5.914251e+02            4.583121e+02            3.293139e+02            6.104011e+02            3.223608e+02            4.080798e+02            7.025159e+02            
7      1.071404e+01            -inf                    5.982451e+00            6.035192e+00            6.664039e+00            5.146922e+00            6.510873e+00            5.828153e+00            4.590174e+00            1.158352e+01            
8      2.950122e+00            -inf                    2.682287e+00            2.635673e+00            2.800514e+00            2.683283e+00            2.894798e+00            2.596880e+00            2.598540e+00            2.885475e+00            
9      1.048843e+02            -inf                    2.564968e+01            5.223695e+01            2.589039e+01            2.719463e+01            7.499093e+01            2.324300e+01            2.143226e+01            1.038544e+02            
10     1.416184e+01            -inf                    2.432874e-01            2.567364e-01            3.870201e-01            4.043410e-01            3.191033e-01            1.780866e+00            4.721952e-01            1.348334e+01            
11     1.965966e+02            -inf                    1.762710e+02            6.491211e+01            1.683359e+02            1.987943e+02            9.958366e+01            4.221116e+02            2.066121e+02            1.799043e+02            
12     7.250700e+01            -inf                    1.000238e+00            4.296942e-01            4.615679e+00            4.579523e+00            4.756559e+00            5.697658e+00            4.237269e+00            5.475891e+01            
13     1.684541e+01            -inf                    2.439617e+01            1.506949e+01            2.391385e+01            2.451778e+01            1.626558e+01            2.791730e+01            2.468937e+01            1.354814e+01            
14     8.846425e+00            -inf                    4.313992e+00            8.189363e+00            4.242228e+00            3.957964e+00            8.291559e+00            3.873390e+00            3.819644e+00            9.522675e+00            
15     3.680040e+00            -inf                    3.376360e+00            3.229560e+00            3.634450e+00            3.529852e+00            3.310862e+00            3.804428e+00            3.396370e+00            3.725141e+00            
16     2.122926e+03            -inf                    2.198538e+03            1.235170e+03            3.677297e+03            2.843284e+03            1.027829e+03            4.598842e+03            2.648237e+03            1.919753e+03            
17     5.020825e+04            -inf                    3.975042e+04            1.585805e+04            5.117905e+04            4.970247e+04            2.329897e+04            1.261308e+05            5.176318e+04            7.576968e+04            
18     5.119816e+01            -inf                    3.916027e+01            3.624564e+01            4.400560e+01            4.477200e+01            3.765090e+01            6.242163e+01            4.563243e+01            4.636813e+01            
19     1.401594e+02            -inf                    7.654822e+01            9.296075e+01            7.071187e+01            7.301086e+01            9.712767e+01            1.074155e+02            1.327138e+02            1.554255e+02            
20     2.550005e+01            -inf                    2.085252e+01            2.087584e+01            2.241248e+01            2.066203e+01            2.149368e+01            2.328372e+01            2.128571e+01            2.477619e+01            
21     5.474189e+00            -inf                    5.814449e+00            5.407747e+00            5.431484e+00            4.946241e+00            5.517821e+00            5.859321e+00            5.350192e+00            5.384589e+00            
22     1.376412e+01            -inf                    1.231917e+01            1.129259e+01            1.122861e+01            1.186850e+01            1.244069e+01            1.365841e+01            1.232746e+01            1.378339e+01            
23     6.506394e+01            -inf                    4.451957e+01            5.818064e+01            4.212177e+01            4.196428e+01            6.149883e+01            4.064289e+01            4.294083e+01            6.884862e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_02_catB_idea_0.py  (error=1.328386e-01)
Task  1: variant_02_catB_idea_0.py  (error=1.878269e+00)
Task  2: variant_04_catD_idea_0.py  (error=1.803783e-01)
Task  3: variant_08_catH_idea_0.py  (error=5.432250e+00)
Task  4: variant_02_catB_idea_0.py  (error=1.165388e+00)
Task  5: variant_10_catB_idea_0.py  (error=5.957466e+00)
Task  6: variant_08_catH_idea_0.py  (error=3.223608e+02)
Task  7: variant_09_catA_idea_0.py  (error=4.590174e+00)
Task  8: variant_08_catH_idea_0.py  (error=2.596880e+00)
Task  9: variant_09_catA_idea_0.py  (error=2.143226e+01)
Task 10: variant_02_catB_idea_0.py  (error=2.432874e-01)
Task 11: variant_03_catC_idea_0.py  (error=6.491211e+01)
Task 12: variant_03_catC_idea_0.py  (error=4.296942e-01)
Task 13: variant_10_catB_idea_0.py  (error=1.354814e+01)
Task 14: variant_09_catA_idea_0.py  (error=3.819644e+00)
Task 15: variant_03_catC_idea_0.py  (error=3.229560e+00)
Task 16: variant_07_catG_idea_0.py  (error=1.027829e+03)
Task 17: variant_03_catC_idea_0.py  (error=1.585805e+04)
Task 18: variant_03_catC_idea_0.py  (error=3.624564e+01)
Task 19: variant_04_catD_idea_0.py  (error=7.071187e+01)
Task 20: variant_05_catE_idea_0.py  (error=2.066203e+01)
Task 21: variant_05_catE_idea_0.py  (error=4.946241e+00)
Task 22: variant_04_catD_idea_0.py  (error=1.122861e+01)
Task 23: variant_08_catH_idea_0.py  (error=4.064289e+01)

WIN COUNTS:
  variant_03_catC_idea_0.py: 5 wins
  variant_02_catB_idea_0.py: 4 wins
  variant_08_catH_idea_0.py: 4 wins
  variant_04_catD_idea_0.py: 3 wins
  variant_09_catA_idea_0.py: 3 wins
  variant_10_catB_idea_0.py: 2 wins
  variant_05_catE_idea_0.py: 2 wins
  variant_07_catG_idea_0.py: 1 wins

────────────────────────────────────────────────────────────────────────
PER-TASK WINNER vs RUNNER-UP — gap table (this is the most important
piece of data in the prompt; it tells you whether ensemble blending is
mathematically viable):

  Task   Winner                          WinnerErr        2ndBest                          2ndErr           Gap(2nd)  Gap(med)
  -----  ------------------------------  ---------------  ------------------------------   ---------------  --------  --------
     0   variant_02_catB_idea_0.py       1.3284e-01       variant_04_catD_idea_0.py        1.3594e-01         1.0×    1.1×
     1   variant_02_catB_idea_0.py       1.8783e+00       variant_07_catG_idea_0.py        2.1206e+00         1.1×    1.5×
     2   variant_04_catD_idea_0.py       1.8038e-01       variant_02_catB_idea_0.py        1.8318e-01         1.0×    1.3×
     3   variant_08_catH_idea_0.py       5.4322e+00       variant_09_catA_idea_0.py        5.4350e+00         1.0×    1.0×
     4   variant_02_catB_idea_0.py       1.1654e+00       variant_09_catA_idea_0.py        1.1794e+00         1.0×    1.1×
     5   variant_10_catB_idea_0.py       5.9575e+00       original.py                      2.6338e+01         4.4×  176.7×
     6   variant_08_catH_idea_0.py       3.2236e+02       variant_05_catE_idea_0.py        3.2931e+02         1.0×    1.6×
     7   variant_09_catA_idea_0.py       4.5902e+00       variant_05_catE_idea_0.py        5.1469e+00         1.1×    1.4×
     8   variant_08_catH_idea_0.py       2.5969e+00       variant_09_catA_idea_0.py        2.5985e+00         1.0×    1.1×
     9   variant_09_catA_idea_0.py       2.1432e+01       variant_08_catH_idea_0.py        2.3243e+01         1.1×    1.9×
    10   variant_02_catB_idea_0.py       2.4329e-01       variant_03_catC_idea_0.py        2.5674e-01         1.1×    1.8×
    11   variant_03_catC_idea_0.py       6.4912e+01       variant_07_catG_idea_0.py        9.9584e+01         1.5×    2.9×
    12   variant_03_catC_idea_0.py       4.2969e-01       variant_02_catB_idea_0.py        1.0002e+00         2.3×   10.9×
    13   variant_10_catB_idea_0.py       1.3548e+01       variant_03_catC_idea_0.py        1.5069e+01         1.1×    1.8×
    14   variant_09_catA_idea_0.py       3.8196e+00       variant_08_catH_idea_0.py        3.8734e+00         1.0×    1.6×
    15   variant_03_catC_idea_0.py       3.2296e+00       variant_07_catG_idea_0.py        3.3109e+00         1.0×    1.1×
    16   variant_07_catG_idea_0.py       1.0278e+03       variant_03_catC_idea_0.py        1.2352e+03         1.2×    2.4×
    17   variant_03_catC_idea_0.py       1.5858e+04       variant_07_catG_idea_0.py        2.3299e+04         1.5×    3.2×
    18   variant_03_catC_idea_0.py       3.6246e+01       variant_07_catG_idea_0.py        3.7651e+01         1.0×    1.2×
    19   variant_04_catD_idea_0.py       7.0712e+01       variant_05_catE_idea_0.py        7.3011e+01         1.0×    1.4×
    20   variant_05_catE_idea_0.py       2.0662e+01       variant_02_catB_idea_0.py        2.0853e+01         1.0×    1.1×
    21   variant_05_catE_idea_0.py       4.9462e+00       variant_09_catA_idea_0.py        5.3502e+00         1.1×    1.1×
    22   variant_04_catD_idea_0.py       1.1229e+01       variant_03_catC_idea_0.py        1.1293e+01         1.0×    1.1×
    23   variant_08_catH_idea_0.py       4.0643e+01       variant_05_catE_idea_0.py        4.1964e+01         1.0×    1.3×

DISPATCH-SIGNAL DIAGNOSIS:
  - Tasks with a clear per-task winner: 24
  - Distinct winners across tasks: 8 (variant_02_catB_idea_0.py, variant_03_catC_idea_0.py, variant_04_catD_idea_0.py, variant_05_catE_idea_0.py, variant_07_catG_idea_0.py, variant_08_catH_idea_0.py ...)
  - BLEND-HOSTILE tasks (winner beats runner-up by >=10× OR beats median competitor by >=50×): 1  (linear blending CANNOT preserve these advantages)
  - DISPATCH-MANDATORY tasks (winner beats runner-up by >=100× OR beats median competitor by >=500×): 0  (must route to the specific winner; ensemble or weighted blending is forbidden)
  - Worst-case task: 5 — variant_10_catB_idea_0.py (5.96e+00) vs original.py (2.63e+01) = 4.4× gap to runner-up, 176.7× gap to median competitor. A 50/50 blend with the runner-up would yield ~1.61e+01, which is ~2.7× WORSE than the winner alone.

RECOMMENDED MECHANISM: pick from the menu using the DECISION GUIDE below, based on the win-count distribution. Multiple mechanisms are viable here — there are no extreme per-task gaps that mathematically rule any out.
────────────────────────────────────────────────────────────────────────

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_catB_idea_0.py (4 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on spectral condition number.

        Category B (Spectral / linear-algebraic):
        - Computes eigenvalues of population covariance matrix
        - Derives condition number as anisotropy measure
        - High condition number (collapsed along eigenvectors) → expand neighborhood
          to accelerate cross-subpopulation communication and prevent fragmentation
        - Low condition number (isotropic) → contract neighborhood for faster convergence
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / eigenvalues[-1]
            cond = np.clip(cond, 1.0, 10000.0)

            # Map condition number to spectral signal in [0, 1]
            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            # High anisotropy (high cond) → expand neighborhood for faster info spread
            # Low anisotropy (low cond) → contract neighborhood for faster convergence
            expansion_bias = 2.0 * spectral_signal - 1.0  # in [-1, 1]
            expansion_bias = np.clip(expansion_bias, -0.5, 0.5)

            min_size = max(1, self.np // 10)
            max_size = max(3, self.np // 3)

            if expansion_bias > 0:
                self.neighborhood_size = min(max_size, self.neighborhood_size + 1)
            elif expansion_bias < 0:
                self.neighborhood_size = max(min_size, self.neighborhood_size - 1)
        except np.linalg.LinAlgError:
            pass
```

# --- From variant_03_catC_idea_0.py (5 wins) ---
```python
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
```

# --- From variant_04_catD_idea_0.py (3 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size using fitness-landscape signals.

        Category D: Uses ONLY fitness signals (ranks, entropy, success-history).
        - Fitness rank entropy: detects multimodal vs unimodal landscape
        - Spearman-like rank correlation: detects landscape smoothness across gens
        - Success-history: tracks which neighborhood sizes yielded improvements
        """
        # Fitness rank entropy: measure of fitness distribution uniformity
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        # Clip to avoid log(0)
        fr_clipped = np.clip(fitness_ranks, 1e-10, 1.0 - 1e-10)
        rank_entropy = -np.mean(fr_clipped * np.log(fr_clipped) - (1 - fr_clipped) * np.log(1 - fr_clipped + 1e-10))
        max_entropy = -np.log(0.5)
        entropy_ratio = np.clip(rank_entropy / (max_entropy + 1e-10), 0.0, 1.0)

        # Spearman-like rank correlation across generations (landscape smoothness)
        if hasattr(self, '_prev_fitness_ranks') and self._prev_fitness_ranks is not None:
            if len(self._prev_fitness_ranks) == len(fitness_ranks):
                if np.std(fitness_ranks) > 1e-10 and np.std(self._prev_fitness_ranks) > 1e-10:
                    rank_corr = np.corrcoef(fitness_ranks, self._prev_fitness_ranks)[0, 1]
                    rank_corr = np.clip(rank_corr, -1.0, 1.0)
                else:
                    rank_corr = 0.0
            else:
                rank_corr = 0.0
        else:
            rank_corr = 0.0
        self._prev_fitness_ranks = fitness_ranks.copy()

        # Success-history for neighborhood sizes
        if not hasattr(self, '_neighborhood_history'):
            self._neighborhood_history = []
        if not hasattr(self, '_neighborhood_success'):
            self._neighborhood_success = {}

        # Track improvement this generation
        if self.global_best_fitness < getattr(self, '_prev_best_for_history', np.inf):
            improvement = True
        else:
            improvement = False
        self._prev_best_for_history = self.global_best_fitness

        current_ns = self.neighborhood_size
        if current_ns not in self._neighborhood_success:
            self._neighborhood_success[current_ns] = []
        self._neighborhood_success[current_ns].append(1.0 if improvement else 0.0)
        if len(self._neighborhood_success[current_ns]) > 10:
            self._neighborhood_success[current_ns].pop(0)

        # Compute success rate for current and neighboring sizes
        def success_rate(ns):
                if ns in self._neighborhood_success and len(self._neighborhood_success[ns]) >= 3:
                    return np.mean(self._neighborhood_success[ns])
                return 0.5

        current_success = success_rate(current_ns)
        smaller_success = success_rate(max(1, current_ns - 1))
        larger_success = success_rate(min(self.np // 2, current_ns + 1))

        # Decision: combine entropy, rank correlation, and success-history
        # High entropy (rugged) OR low rank_corr (unstable) OR low success → increase neighborhood
        # Low entropy (converged) AND high rank_corr (stable) AND high success → decrease neighborhood

        # Exploration signal: rugged landscape needs larger neighborhoods
        exploration_signal = (1.0 - entropy_ratio) * 0.4 + (1.0 - rank_corr) * 0.3 + (1.0 - current_success) * 0.3
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

        # If neighboring size has higher success rate, consider switching
        if larger_success > current_success + 0.1:
            exploration_signal = min(1.0, exploration_signal + 0.3)
        if smaller_success > current_success + 0.1 and exploration_signal < 0.5:
            exploration_signal = max(0.0, exploration_signal - 0.2)

        # Adapt neighborhood size based on exploration signal
        if exploration_signal > 0.6:
            # High exploration need → increase neighborhood
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif exploration_signal < 0.3:
            # Low exploration need → decrease neighborhood
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
```

# --- From variant_05_catE_idea_0.py (2 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on k-NN graph connectivity.

        Category E: Build k-NN graph over population; use connected component count,
        spectral gap, and graph density to detect fragmentation vs convergence.
        """
        n = self.np
        k = min(5, n - 1)

        # Build k-NN graph
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Connected component analysis via BFS
        visited = np.zeros(n, dtype=bool)
        components = []
        for start in range(n):
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
        max_component_size = np.max(component_sizes)

        # Fragmentation ratio: how unevenly distributed is the population?
        fragmentation_ratio = 1.0 - (max_component_size / (n + 1e-10))

        # Compute graph density
        max_edges = n * (n - 1) / 2
        actual_edges = n * k / 2
        graph_density = actual_edges / (max_edges + 1e-10)

        # Spectral gap from normalized graph Laplacian
        try:
            row_indices = np.repeat(np.arange(n), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
            adj = adj + adj.T
            adj.data[:] = 1.0

            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
            lap = dspy.identity(n) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, n - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)

            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
        except:
            spectral_gap = 1.0

        # Decision logic based on graph topology
        # Fragmented population (multiple components or high fragmentation) → increase neighborhood
        is_fragmented = (n_components > 1) or (fragmentation_ratio > 0.3)
        # Poor mixing (low spectral gap) → increase neighborhood
        poor_mixing = spectral_gap < 0.3
        # Well-connected and converging → decrease neighborhood
        is_well_connected = (n_components == 1) and (spectral_gap > 0.6) and (graph_density > 0.3)

        if is_fragmented or poor_mixing:
            # Population graph is fragmented → expand neighborhood for better information flow
            self.neighborhood_size = min(n // 2, self.neighborhood_size + 1)
        elif is_well_connected:
            # Population is well-mixed → shrink neighborhood for faster exploitation
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
```

# --- From variant_07_catG_idea_0.py (1 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology via stochastic sampling of neighborhood candidates.

        Category G: Uses Monte Carlo sampling and bootstrap estimation to
        probabilistically select neighborhood size based on empirical performance.
        """
        # Generate candidate neighborhood sizes (stochastic candidate generation)
        candidates = list(range(1, max(2, min(self.np // 2, self.neighborhood_size + 3)) + 1))
        if len(candidates) < 2:
            return  # Not enough candidates to sample meaningfully

        # Bootstrap sampling: estimate expected improvement for each candidate
        n_bootstrap = 20
        bootstrap_scores = {c: [] for c in candidates}

        for _ in range(n_bootstrap):
            # Subsample population for Monte Carlo estimate
            subsample_size = max(5, self.np // 2)
            subsample_idx = np.random.choice(self.np, size=subsample_size, replace=False)

            for candidate in candidates:
                # Estimate local best quality for this neighborhood size
                local_best_qualities = []
                for idx in subsample_idx:
                    # Ring neighborhood around this particle
                    left = (idx - candidate) % self.np
                    right = (idx + candidate + 1) % self.np
                    if left < right:
                        nb_indices = np.arange(left, right)
                    else:
                        nb_indices = np.concatenate([np.arange(left, self.np), np.arange(0, right)])

                    if len(nb_indices) > 0:
                        nb_fitness = self.personal_best_fitness[nb_indices]
                        local_best_qualities.append(np.min(nb_fitness))

                if local_best_qualities:
                    # Bootstrap estimate: quality of local bests accessible
                    bootstrap_scores[candidate].append(np.mean(local_best_qualities))

        # Compute expected performance and uncertainty for each candidate
        candidate_stats = {}
        for candidate in candidates:
            scores = bootstrap_scores[candidate]
            if scores:
                mean_perf = np.mean(scores)
                std_perf = np.std(scores) + 1e-10
                # UCB-style: higher is better, penalize high variance
                candidate_stats[candidate] = mean_perf - 0.5 * std_perf
            else:
                candidate_stats[candidate] = 0.0

        # Diversity signal as additional stochastic probe
        diversity = self._compute_diversity()
        diversity_norm = np.clip(diversity / (self.diversity_threshold_high + 1e-10), 0.0, 1.0)

        # Probabilistic selection via softmax sampling
        temperatures = [0.5 + 0.5 * diversity_norm for _ in candidates]

        # Convert stats to probabilities (lower fitness = better)
        min_stat = min(candidate_stats.values())
        max_stat = max(candidate_stats.values())
        stat_range = max_stat - min_stat + 1e-10

        weights = []
        for i, candidate in enumerate(candidates):
            normalized = (candidate_stats[candidate] - min_stat) / stat_range
            # Temperature controls stochasticity: high diversity → more exploration
            weight = np.exp(normalized / temperatures[i])
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-10)

        # Sample neighborhood size stochastically
        chosen_idx = np.random.choice(len(candidates), p=weights)
        chosen_size = candidates[chosen_idx]

        # Update with momentum for stability
        if not hasattr(self, '_neighborhood_momentum'):
            self._neighborhood_momentum = chosen_size
        self._neighborhood_momentum = 0.7 * self._neighborhood_momentum + 0.3 * chosen_size

        # Final decision with bounded update
        target = int(np.round(self._neighborhood_momentum))
        self.neighborhood_size = max(1, min(self.np // 2, target))
```

# --- From variant_08_catH_idea_0.py (4 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Hybrid: combine temporal convergence signals with topology fragmentation detection."""
        # === MECHANISM 1: TEMPORAL CONVERGENCE SIGNALS ===
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        # Stagnation signal
        stagnation_ratio = min(self.stagnation_counter / 100.0, 1.0)

        # Temporal signal: high improvement + low stagnation → exploit (smaller neighborhood)
        # Low improvement + high stagnation → explore (larger neighborhood)
        temporal_signal = stagnation_ratio - min(self._ema_improvement, 1.0)
        temporal_signal = np.clip(temporal_signal, -1.0, 1.0)

        # === MECHANISM 2: TOPOLOGY FRAGMENTATION SIGNALS ===
        k = min(5, self.np - 1)
        sq_dists = np.sum(
            (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2
        )
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

        component_sizes = np.array([len(c) for c in components])
        n_components = len(components)

        # Fragmentation ratio: how uneven is the distribution?
        max_component_size = np.max(component_sizes)
        fragmentation_ratio = 1.0 - (max_component_size / (self.np + 1e-10))

        # Topology signal: more components + higher fragmentation → explore (larger neighborhood)
        topology_signal = (n_components - 1) / max(self.np // 10, 1) + fragmentation_ratio
        topology_signal = np.clip(topology_signal, 0.0, 2.0)

        # === PRINCIPLED WEIGHTING SWITCHING ===
        is_fragmented = n_components > 1 and fragmentation_ratio > 0.3

        if is_fragmented:
            topology_weight = 0.7
            temporal_weight = 0.3
        else:
            topology_weight = 0.3
            temporal_weight = 0.7

        # === COMBINE MECHANISMS ===
        combined_signal = (
            temporal_weight * (0.5 + 0.5 * temporal_signal) +
            topology_weight * topology_signal
        )

        # Map combined signal to neighborhood size
        # signal > 1 → need more exploration → larger neighborhood
        # signal < 1 → need more exploitation → smaller neighborhood
        target_ratio = combined_signal
        target_size = int(self.np * np.clip(target_ratio, 0.05, 0.5))
        target_size = max(1, min(self.np // 2, target_size))

        # Gradual adjustment
        if target_size > self.neighborhood_size:
            self.neighborhood_size = min(target_size, self.neighborhood_size + 2)
        else:
            self.neighborhood_size = max(target_size, self.neighborhood_size - 2)

        self.neighborhood_size = max(1, min(self.np // 2, self.neighborhood_size))
```

# --- From variant_09_catA_idea_0.py (3 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on k-NN local density structure.

        Category A (Geometry / spatial): Uses k-NN nearest-neighbor distances and
        axis-aligned spread ratio to detect clustering vs dispersion. Different from
        centroid-distance approach in prior variant — focuses on LOCAL neighborhood
        density rather than global centroid scatter.
        """
        if self.np < 5:
            return

        # k-NN structure: average distance to k nearest neighbors per particle
        k = min(3, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        sorted_dists = np.sort(sq_dists, axis=1)
        knn_avg_dist = np.mean(sorted_dists[:, :k], axis=1)

        # Global spread via axis-aligned bounding box extent
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        bbox_extent = np.mean(pop_max - pop_min) + 1e-10

        # Local density ratio: small knn_avg_dist → dense/clustered
        density_ratio = knn_avg_dist.mean() / bbox_extent

        # Spread signal: how much of the bbox is occupied
        occupancy = np.std(self.population) / (bbox_extent * 0.5 + 1e-10)
        occupancy = np.clip(occupancy, 0.0, 2.0)

        # Cluster detection: variance of knn distances (high variance = fragmented)
        knn_variance = np.var(knn_avg_dist) / (np.mean(knn_avg_dist) ** 2 + 1e-10)
        knn_variance = np.clip(knn_variance, 0.0, 10.0)

        # Determine adjustment based on density structure
        # Dense population (low ratio) → shrink neighborhood to maintain local search
        # Sparse/fragmented (high ratio or high variance) → expand neighborhood
        if density_ratio < 0.1 or knn_variance > 2.0:
            # Clustered or fragmented: shrink to favor local exploitation
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
        elif density_ratio > 0.5 and occupancy < 0.5:
            # Sparse and under-spread: expand to encourage exploration
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif density_ratio > 0.3:
            # Moderately sparse: gradual expansion
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
```

# --- From variant_10_catB_idea_0.py (2 wins) ---
```python
def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size using spectral properties of population.

        Uses condition number and spectral entropy of population covariance to detect:
        - High condition number → anisotropic/ill-conditioned → larger neighborhood
        - Low spectral entropy → collapsed subspace → larger neighborhood
        - Isotropic distribution → smaller neighborhood for exploitation

        This is orthogonal to diversity-based approaches (Category A) because it
        directly measures the eigenvalue structure of the population distribution.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Condition number: ratio of max/min eigenvalue
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)

            # Spectral entropy: how uniformly spread are eigenvalues?
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                eig_norm = eigenvalues / total_var
                eig_norm = eig_norm[eig_norm > 1e-10]  # avoid log(0)
                spectral_entropy = -np.sum(eig_norm * np.log(eig_norm))
                max_entropy = np.log(self.dim)
                entropy_ratio = spectral_entropy / (max_entropy + 1e-10)
            else:
                entropy_ratio = 0.0

            # Combined spectral signal: high cond OR low entropy → need larger neighborhood
            cond_signal = np.log1p(cond) / np.log1p(1e6)
            spectral_signal = 0.5 * cond_signal + 0.5 * (1.0 - entropy_ratio)

            # Nonlinear mapping: spectral signal -> neighborhood scale factor
            # Range: [0.4, 1.5] (well-conditioned -> ill-conditioned)
            neighborhood_scale = 0.4 + 1.1 * spectral_signal

            target_size = int(self.neighborhood_size * neighborhood_scale)
            target_size = np.clip(target_size, 1, self.np // 2)

            if target_size > self.neighborhood_size:
                self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
            elif target_size < self.neighborhood_size:
                self.neighborhood_size = max(1, self.neighborhood_size - 1)

        except np.linalg.LinAlgError:
            # Fallback: use simple diversity-based rule
            diversity = self._compute_diversity()
            if diversity < self.diversity_threshold_low:
                self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
            elif diversity > self.diversity_threshold_high:
                self.neighborhood_size = max(1, self.neighborhood_size - 1)
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