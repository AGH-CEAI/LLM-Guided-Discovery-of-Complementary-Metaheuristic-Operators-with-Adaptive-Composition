Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_sample_population_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      8.143601e-08            8.080851e-08            8.738083e-08            5.222991e+01            9.705600e-08            1.163242e-07            8.533500e-08            8.270154e-08            1.121488e+00            3.251266e+01            1.227267e+02            
1      2.331501e-07            2.167765e-07            2.308238e-07            1.402259e-06            2.702834e-07            3.151109e-07            2.459056e-07            2.404252e-07            3.156236e+00            6.138951e+01            2.416152e+02            
2      2.081050e-05            1.965974e-05            2.197864e-05            8.155544e+02            2.401214e-05            2.849681e-05            2.205300e-05            2.147446e-05            2.078818e+00            3.281797e+02            1.005086e+03            
3      3.368466e-04            3.175667e-04            3.341728e-04            3.291424e+02            3.559632e-04            4.009945e-04            3.537883e-04            3.520052e-04            2.494467e+00            1.618885e+02            3.911037e+02            
4      2.358636e-02            2.320714e-02            2.357829e-02            5.845563e+00            2.449461e-02            2.588691e-02            2.406730e-02            2.401635e-02            1.556180e+00            4.318121e+00            6.255586e+00            
5      1.000000e-08            1.000000e-08            1.000000e-08            6.238442e+09            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.355439e+02            2.294667e+08            1.125627e+10            
6      4.002108e-07            3.659795e-07            3.908246e-07            1.241412e+04            4.656180e-07            5.472041e-07            4.239501e-07            3.961639e-07            4.457498e+02            3.022358e+03            1.374035e+04            
7      1.355345e-05            1.250934e-05            1.386767e-05            4.612768e+02            1.467991e-05            1.660613e-05            1.436736e-05            1.407760e-05            5.142127e+00            1.899745e+02            5.440734e+02            
8      1.708490e-03            1.657382e-03            1.705012e-03            8.323099e-03            1.823988e-03            1.921930e-03            1.753601e-03            1.815183e-03            1.394351e+00            3.231400e+01            6.264941e+01            
9      4.872533e-04            4.551851e-04            4.886144e-04            5.014863e+02            5.089720e-04            5.720825e-04            4.978152e-04            1.371562e-01            3.264474e+01            2.262265e+02            4.963521e+02            
10     3.461544e+00            4.461581e+00            4.127952e-06            4.723320e+02            2.970100e+00            4.065589e+00            2.674670e+00            6.160971e+00            2.356049e+02            1.792503e+02            5.282309e+02            
11     7.138172e-01            1.018631e+00            1.073698e-03            1.980869e+03            5.264734e-01            1.659682e+00            7.938556e-01            4.508296e+00            1.070221e+03            7.229174e+02            2.091975e+03            
12     4.326057e-01            5.371752e-01            4.525422e-03            3.308042e+02            8.826205e-01            2.042528e+00            4.944861e-01            1.715020e+00            2.256708e+02            1.554762e+02            3.457521e+02            
13     9.595551e-01            1.251214e+00            1.087223e+00            6.014928e+01            9.392296e-01            8.968355e-01            9.909404e-01            1.370055e+00            4.223755e+01            3.587584e+01            6.397978e+01            
14     2.547958e+00            3.245484e+00            1.620980e+00            2.187948e+01            2.500848e+00            2.604377e+00            2.433609e+00            2.832377e+00            8.137600e+00            1.661001e+01            2.152174e+01            
15     9.241363e-01            8.802905e-01            9.561087e-01            4.378861e+00            9.110957e-01            1.050103e+00            1.031813e+00            8.966427e-01            4.176628e+00            3.854704e+00            4.481635e+00            
16     5.096286e+01            2.998519e+02            3.263178e+01            4.353954e+04            5.859042e+01            1.653501e+02            5.625005e+01            5.982593e+02            1.148084e+04            9.249573e+03            4.517311e+04            
17     3.120103e+00            3.120103e+00            3.120103e+00            2.668501e+05            3.120103e+00            3.120104e+00            3.120103e+00            3.120104e+00            1.793749e+05            7.623226e+04            3.528653e+05            
18     2.674433e+00            2.674661e+00            2.674813e+00            9.675229e+01            2.674653e+00            2.675426e+00            2.674590e+00            1.480097e+01            7.683508e+01            5.957072e+01            1.003787e+02            
19     2.354865e+01            2.558055e+01            1.193529e+01            6.193574e+02            6.189787e+00            6.546273e+00            6.698815e+00            3.958530e+01            2.874955e+02            4.306068e+02            6.130034e+02            
20     5.641609e+00            1.747075e+01            1.950201e+00            7.561749e+01            3.865142e+00            6.425611e+00            3.233376e+00            2.339736e+01            4.776139e+01            4.935496e+01            7.813758e+01            
21     4.406281e+00            4.344807e+00            4.227724e+00            6.064064e+00            4.204066e+00            4.380975e+00            4.196915e+00            4.377805e+00            5.778902e+00            5.778127e+00            6.078369e+00            
22     2.142187e+00            2.118615e+00            2.389391e+00            2.410965e+01            2.177051e+00            2.261292e+00            2.213544e+00            2.752526e+00            2.002875e+01            2.002948e+01            2.443270e+01            
23     5.656724e+00            2.237811e+01            5.133041e+00            1.630486e+02            5.572287e+00            6.469361e+00            4.866678e+00            2.619235e+01            6.975835e+01            9.793198e+01            1.602321e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_01_idea_0.py  (error=2.167765e-07)
Task  2: variant_01_idea_0.py  (error=1.965974e-05)
Task  3: variant_01_idea_0.py  (error=3.175667e-04)
Task  4: variant_01_idea_0.py  (error=2.320714e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_01_idea_0.py  (error=3.659795e-07)
Task  7: variant_01_idea_0.py  (error=1.250934e-05)
Task  8: variant_01_idea_0.py  (error=1.657382e-03)
Task  9: variant_01_idea_0.py  (error=4.551851e-04)
Task 10: variant_02_idea_0.py  (error=4.127952e-06)
Task 11: variant_02_idea_0.py  (error=1.073698e-03)
Task 12: variant_02_idea_0.py  (error=4.525422e-03)
Task 13: variant_05_idea_0.py  (error=8.968355e-01)
Task 14: variant_02_idea_0.py  (error=1.620980e+00)
Task 15: variant_01_idea_0.py  (error=8.802905e-01)
Task 16: variant_02_idea_0.py  (error=3.263178e+01)
Task 17: variant_01_idea_0.py  (error=3.120103e+00)
Task 18: original.py  (error=2.674433e+00)
Task 19: variant_04_idea_0.py  (error=6.189787e+00)
Task 20: variant_02_idea_0.py  (error=1.950201e+00)
Task 21: variant_06_idea_0.py  (error=4.196915e+00)
Task 22: variant_01_idea_0.py  (error=2.118615e+00)
Task 23: variant_06_idea_0.py  (error=4.866678e+00)

WIN COUNTS:
  variant_01_idea_0.py: 11 wins
  variant_02_idea_0.py: 6 wins
  variant_06_idea_0.py: 2 wins
  variant_05_idea_0.py: 1 wins
  original.py: 1 wins
  variant_04_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (11 wins) ---
```python
def _sample_population_batch(self):
        """Sample a new population using antithetic variates for variance reduction."""
        dim = self.dim
        lam = self.pop_size

        # Determine number of half-samples (must be even for pairs)
        half_lam = (lam // 2) * 1  # integer division
        if lam % 2 == 0:
            n_pairs = lam // 2
        else:
            n_pairs = (lam - 1) // 2  # one extra sample will be added

        # Sample n_pairs from standard normal
        z_pairs = np.random.randn(n_pairs, dim)

        # Create antithetic pairs: stack original and negated samples
        z = np.vstack([z_pairs, -z_pairs])

        # If odd population size, generate one additional independent sample
        if lam % 2 == 1:
            z_extra = np.random.randn(1, dim)
            z = np.vstack([z, z_extra])

        # Transform: x = mean + sigma * B * D * z
        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T

        return population, z
```

# --- From variant_02_idea_0.py (6 wins) ---
```python
def _sample_population_batch(self):
        """Sample population using hybrid center = blend of mean and best solution."""
        dim = self.dim
        lam = self.pop_size

        # Sample standard normal
        z = np.random.randn(lam, dim)

        # Hybrid center: if best solution exists and has improved, bias toward it
        if self.best_x is not None and np.isfinite(self.best_f):
            # Adaptive weight: more pull toward best when mean has drifted
            diff_norm = np.linalg.norm(self.mean - self.best_x)
            drift_scale = min(diff_norm / (self.sigma + 1e-20), 1.0)
            w_best = 0.3 * drift_scale
            center = (1.0 - w_best) * self.mean + w_best * self.best_x
        else:
            center = self.mean

        # Transform: x = center + sigma * B * D * z
        sqrt_eig = np.sqrt(np.maximum(self.eigenvalues, 1e-20))
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = center[None, :] + self.sigma * y.T

        return population, z
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
        """Sample with restart-triggered diversity injection to escape local optima."""
        dim = self.dim
        lam = self.pop_size

        # Determine random injection fraction based on stagnation depth
        if self.stagnation_counter > 8:
            random_frac = 0.5  # 50% random when deeply stuck
        elif self.stagnation_counter > 4:
            random_frac = 0.3  # 30% random when stuck
        elif self.stagnation_counter > 2:
            random_frac = 0.2  # 20% random when beginning to stagnate
        else:
            random_frac = 0.1  # 10% random for mild diversity maintenance

        n_random = max(1, int(lam * random_frac))
        n_cma = lam - n_random

        # Standard CMA-ES sampling
        z_all = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues + 1e-20)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z_all.T)
        cma_samples = self.mean[None, :] + self.sigma * y.T

        # Inject uniform random samples for diversity
        random_samples = np.random.uniform(self.lb, self.ub, (n_random, dim))

        # Combine CMA-ES and random samples
        if n_random > 0 and n_cma > 0:
            population = np.vstack([cma_samples[:n_cma], random_samples])
        elif n_random > 0:
            population = random_samples
        else:
            population = cma_samples

        # Clip to bounds
        population = np.clip(population, self.lb, self.ub)

        return population, z_all
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
        """Sample a new population with restart-driven diversity injection."""
        dim = self.dim
        lam = self.pop_size

        # Sample standard normal
        z = np.random.randn(lam, dim)

        # Transform: x = mean + sigma * B * D * z
        sqrt_eig = np.sqrt(self.eigenvalues)
        # y = B * D * z^T => shape (dim, lam)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        # population = mean + sigma * y^T
        population = self.mean[None, :] + self.sigma * y.T

        # Inject restart-based diversity: replace ~50% of population with fresh samples
        num_restart = max(1, int(0.5 * lam))
        restart_indices = np.random.choice(lam, num_restart, replace=False)

        for idx in restart_indices:
            # Sample uniformly from the full search space (escape local basin)
            restart_point = np.random.uniform(self.lb, self.ub, dim)
            population[idx] = restart_point
            # Approximate z for covariance update (map restart point back to z-space)
            diff = restart_point - self.mean
            z[idx] = (self.eigenvectors.T @ diff) / (self.sigma * sqrt_eig + 1e-20)

        return population, z
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _sample_population_batch(self):
        """Sample from CMA-ES distribution with archive-guided diversity for escaping local optima."""
        dim = self.dim
        lam = self.pop_size

        # Maintain archive of recent good solutions
        if not hasattr(self, 'sampling_archive'):
            self.sampling_archive = []
        if len(self.sampling_archive) > 0 and len(self.sampling_archive[0]) != dim:
            self.sampling_archive = []

        # Check stagnation to trigger archive-based sampling
        is_stagnant = self.stagnation_counter > max(3, int(5 * self.dim / self.pop_size))

        # Determine how many to draw from archive
        n_archive = 0
        if is_stagnant and len(self.sampling_archive) >= 5:
            # Draw 40-70% from archive when stagnant
            n_archive = int(lam * np.random.uniform(0.4, 0.7))
            n_archive = min(n_archive, len(self.sampling_archive))

        n_cmaes = lam - n_archive

        population = np.empty((lam, dim))
        z = np.empty((lam, dim))

        # Standard CMA-ES sampling for regular portion
        if n_cmaes > 0:
            z[:n_cmaes] = np.random.randn(n_cmaes, dim)
            # Add heavy-tailed perturbations (t-distribution, df=2) for 20% of samples
            heavy_tail_mask = np.random.rand(n_cmaes) < 0.2
            if np.any(heavy_tail_mask):
                z[:n_cmaes][heavy_tail_mask] = np.random.standard_t(2.0, size=(np.sum(heavy_tail_mask), dim))

            sqrt_eig = np.sqrt(self.eigenvalues)
            y = self.eigenvectors @ (sqrt_eig[:, None] * z[:n_cmaes].T)
            population[:n_cmaes] = self.mean[None, :] + self.sigma * y.T

        # Archive-based sampling for diversity
        if n_archive > 0:
            archive_array = np.array(self.sampling_archive)
            # Sample with replacement from archive
            archive_indices = np.random.randint(0, len(archive_array), size=n_archive)
            sampled_from_archive = archive_array[archive_indices]

            # Add exploration noise scaled to current sigma
            exploration_noise = np.random.randn(n_archive, dim)
            # Also add heavy-tailed component
            heavy_tail_pert = np.random.standard_t(2.0, size=(n_archive, dim))
            combined_noise = 0.7 * exploration_noise + 0.3 * heavy_tail_pert

            # Scale noise by sigma and eigenvalue structure
            sqrt_eig = np.sqrt(self.eigenvalues)
            noise = self.eigenvectors @ (sqrt_eig[:, None] * combined_noise.T)
            population[n_cmaes:] = sampled_from_archive + self.sigma * 2.0 * noise.T

            # Record z vectors for covariance update (use the combined noise)
            z[n_cmaes:] = combined_noise

        return population, z
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance matrix adaptation using a rank-1 + rank-mu update
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update with rank-weighted recombination
    - Periodic restarts on stagnation detection
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))  # CMA-ES lambda
        self.pop_size = max(self.pop_size, 2 * dim)
        self.mu = self.pop_size // 2  # number of parents
        self.best_f = np.inf
        self.best_x = None
        self._initialize_strategy_params()

    def _initialize_strategy_params(self):
        """Initialize all strategy parameters for covariance adaptation."""
        dim = self.dim
        mu = self.mu
        lam = self.pop_size

        # Recombination weights (log-linear)
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size adaptation parameters
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        # Covariance adaptation parameters
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1,
                            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        # Expected norm of N(0,I)
        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        """Initialize the evolutionary state: mean, covariance, paths, step-size."""
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0  # initial step size
        self.C = np.eye(dim)  # covariance matrix
        self.p_sigma = np.zeros(dim)  # evolution path for sigma
        self.p_c = np.zeros(dim)  # evolution path for C
        self.eigenvalues = np.ones(dim)
        self.eigenvectors = np.eye(dim)
        self.gen_since_eigen = 0
        self.stagnation_counter = 0
        self.prev_best_f = np.inf

    def _decompose_covariance(self):
        """Eigendecompose the covariance matrix for sampling."""
        try:
            self.C = 0.5 * (self.C + self.C.T)  # enforce symmetry
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            # Clamp eigenvalues to avoid numerical issues
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.gen_since_eigen = 0
        except np.linalg.LinAlgError:
            # Reset covariance if decomposition fails
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.gen_since_eigen = 0

    def _sample_population_batch(self):
        """Sample a new population from the current distribution N(mean, sigma^2 * C)."""
        dim = self.dim
        lam = self.pop_size

        # Sample standard normal
        z = np.random.randn(lam, dim)

        # Transform: x = mean + sigma * B * D * z
        sqrt_eig = np.sqrt(self.eigenvalues)
        # y = B * D * z^T => shape (dim, lam)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        # population = mean + sigma * y^T
        population = self.mean[None, :] + self.sigma * y.T

        return population, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        """Sort population by fitness (ascending = better) and return sorted arrays."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, z_vectors

        # Put invalid entries at the end with inf fitness
        sort_fitness = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fitness)
        return population[order], sort_fitness[order], z_vectors[order]

    def _update_best(self, population, fitness):
        """Track the global best solution found so far."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_idx = np.where(valid_mask)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.best_f:
            self.best_f = fitness[best_idx]
            self.best_x = population[best_idx].copy()

    def _update_mean(self, sorted_population):
        """Update the distribution mean using weighted recombination of the mu best."""
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]  # (mu, dim)
        self.mean = self.weights @ selected  # weighted sum
        return old_mean

    def _update_evolution_path_sigma(self, old_mean):
        """Update the cumulative step-size adaptation path."""
        dim = self.dim
        # Inverse square root of C: B * D^{-1} * B^T
        inv_sqrt_eig = 1.0 / np.sqrt(self.eigenvalues + 1e-20)
        inv_sqrt_C = self.eigenvectors @ np.diag(inv_sqrt_eig) @ self.eigenvectors.T

        displacement = (self.mean - old_mean) / self.sigma
        self.p_sigma = ((1.0 - self.c_sigma) * self.p_sigma +
                        np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) *
                        inv_sqrt_C @ displacement)

    def _update_evolution_path_c(self, old_mean):
        """Update the cumulative covariance adaptation path."""
        dim = self.dim
        # Check if step-size path is not too long (h_sigma)
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (dim + 1.0)) * self.chi_n
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        displacement = (self.mean - old_mean) / self.sigma
        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) *
                     displacement)
        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update the covariance matrix with rank-1 and rank-mu updates."""
        dim = self.dim
        selected = sorted_population[:self.mu]

        # Rank-1 update component
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        # Combined update
        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu)

        # Enforce symmetry
        self.C = 0.5 * (self.C + self.C.T)

    def _adapt_step_size(self):
        """Adapt the global step size sigma using cumulative path length control."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
        # Clamp sigma to reasonable range
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)

    def _check_stagnation(self, fitness):
        """Detect stagnation and trigger restart if needed."""
        valid = fitness[np.isfinite(fitness)]
        if len(valid) == 0:
            return False

        current_best = np.min(valid)
        improvement = self.prev_best_f - current_best

        if improvement < 1e-12 * (1.0 + abs(self.prev_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        self.prev_best_f = min(self.prev_best_f, current_best)

        # Stagnation threshold: restart after many generations without improvement
        stagnation_limit = 10 + int(30 * self.dim / self.pop_size)
        return self.stagnation_counter > stagnation_limit

    def _check_condition_number(self):
        """Check if covariance matrix condition number is too high."""
        ratio = np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-30)
        return ratio > 1e14

    def _restart(self):
        """Perform a restart with potentially increased population size."""
        # Increase pop size for IPOP-like restarts
        self.pop_size = min(int(self.pop_size * 1.5), 800)
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        # Bias restart toward best known solution with some probability
        if self.best_x is not None and np.random.rand() < 0.3:
            self.mean = self.best_x + np.random.randn(self.dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        """Handle case where func returns fewer values than expected."""
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            z_vectors = z_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, z_vectors

    def _should_decompose(self):
        """Determine if eigendecomposition is needed this generation."""
        self.gen_since_eigen += 1
        # Decompose every few generations to save compute
        decompose_freq = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        decompose_freq = min(decompose_freq, self.pop_size)
        return self.gen_since_eigen >= decompose_freq

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_state()
        self._decompose_covariance()
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)

        # Initial evaluation to seed the best
        initial_pop = self._clip_to_bounds_batch(
            np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        )
        if not stopping_condition():
            initial_fit = func(initial_pop)
            if stopping_condition():
                self._update_best(initial_pop[:len(initial_fit)], initial_fit)
                return self.best_f, self.best_x
            self._update_best(initial_pop[:len(initial_fit)], initial_fit)

            # Set mean to best initial point
            valid_mask = np.isfinite(initial_fit)
            if np.any(valid_mask):
                best_idx = np.argmin(np.where(valid_mask, initial_fit, np.inf))
                self.mean = initial_pop[best_idx].copy()

        generation = 0
        while not stopping_condition():
            # Sample new population
            population, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            # Evaluate
            fitness = func(population)

            if stopping_condition():
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._update_best(population, fitness)
                break

            population, fitness, z_vectors = self._handle_truncated_fitness(
                population, fitness, z_vectors
            )
            self._update_best(population, fitness)

            if len(fitness) < self.mu:
                break

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(
                population, fitness, z_vectors
            )

            # Update distribution
            old_mean = self._update_mean(sorted_pop)
            self._update_evolution_path_sigma(old_mean)
            h_sigma = self._update_evolution_path_c(old_mean)
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)
            self._adapt_step_size()

            # Eigendecomposition if needed
            if self._should_decompose():
                self._decompose_covariance()

            # Check for restart conditions
            need_restart = self._check_stagnation(sorted_fit)
            if not need_restart:
                need_restart = self._check_condition_number()

            if need_restart:
                self._restart()
                self._decompose_covariance()

            generation += 1

        return self.best_f, self.best_x

```