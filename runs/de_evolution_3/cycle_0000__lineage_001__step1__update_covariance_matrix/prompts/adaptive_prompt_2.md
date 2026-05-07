Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_covariance_matrix` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      8.143601e-08            7.954607e-08            8.033880e-08            6.891462e-06            5.894260e-03            9.374120e-08            1.174297e-04            1.014570e-07            8.181107e-08            8.348056e-08            -inf                    
1      2.331501e-07            2.358150e-07            2.244308e-07            1.145364e+00            2.034200e+00            2.515075e-07            1.307106e+00            2.663361e-07            2.188521e-07            2.498496e-07            -inf                    
2      2.081050e-05            2.168728e-05            1.992969e-05            1.948476e+00            3.611059e+00            2.435092e-05            2.363126e+00            2.243901e-05            2.003289e-05            2.042313e-05            -inf                    
3      3.368466e-04            3.395137e-04            3.328931e-04            1.585582e+00            1.735501e+00            3.586879e-04            2.341991e+00            3.486967e-04            3.511348e-04            3.391127e-04            -inf                    
4      2.358636e-02            2.343147e-02            2.331430e-02            1.102918e+00            1.113049e+00            2.389302e-02            1.424438e+00            2.453438e-02            3.212874e-02            2.356210e-02            -inf                    
5      1.000000e-08            1.000000e-08            1.000000e-08            2.461862e+00            1.941885e+01            1.000000e-08            9.878456e+01            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
6      4.002108e-07            4.047127e-07            1.470896e-02            4.331734e+02            2.545866e+02            5.840406e+00            5.998702e+01            3.935829e-07            1.912533e+01            4.031679e-07            -inf                    
7      1.355345e-05            1.345468e-05            1.329200e-05            1.827532e+00            4.361883e+00            1.452896e-05            3.470598e+00            1.432793e-05            1.412341e-05            1.336407e-05            -inf                    
8      1.708490e-03            1.707526e-03            1.682779e-03            1.772306e+00            1.615663e+00            1.789918e-03            2.093400e+00            1.797068e-03            1.141986e+00            1.713526e-03            -inf                    
9      4.872533e-04            4.984748e-04            2.808080e+00            2.050954e+01            1.774956e+01            3.742905e+00            9.183660e+00            4.673650e-04            7.145115e+00            4.720814e-04            -inf                    
10     3.461544e+00            4.385179e+00            1.523926e+00            9.024223e+00            1.008427e+01            3.130950e+00            4.290238e+00            3.833576e+00            3.220476e+00            3.885058e+00            -inf                    
11     7.138172e-01            1.057092e+00            8.859128e-01            2.782517e+01            6.976209e+00            2.099717e+00            1.921822e+00            1.537287e+00            8.510243e-01            1.783027e+00            -inf                    
12     4.326057e-01            7.143217e-01            7.219857e-01            1.124254e+01            3.018877e+00            2.453275e+00            1.919903e+00            1.282687e+00            6.572294e-01            1.155855e+00            -inf                    
13     9.595551e-01            8.994553e-01            1.098967e+00            5.932116e+00            3.333475e+00            1.085857e+00            1.357766e+00            1.335304e+00            9.320179e-01            1.023455e+00            -inf                    
14     2.547958e+00            2.468053e+00            2.789507e+00            4.036309e+00            3.985160e+00            3.871311e+00            2.548779e+00            3.866622e+00            3.425929e+00            2.942540e+00            -inf                    
15     9.241363e-01            1.023363e+00            1.040373e+00            1.817234e+00            1.646389e+00            1.264403e+00            1.301378e+00            1.192860e+00            9.630985e-01            1.054512e+00            -inf                    
16     5.096286e+01            8.328804e+01            3.257227e+02            1.621101e+03            1.060877e+03            5.878590e+02            3.891009e+01            4.739526e+02            3.630602e+02            3.675169e+02            -inf                    
17     3.120103e+00            3.120103e+00            3.120103e+00            6.282789e+02            1.450564e+01            3.120103e+00            3.150017e+00            3.120103e+00            3.120103e+00            3.120103e+00            -inf                    
18     2.674433e+00            2.674447e+00            2.843492e+00            2.099744e+01            1.657835e+01            1.251753e+01            3.708711e+00            4.638126e+00            2.674363e+00            2.675438e+00            -inf                    
19     2.354865e+01            1.239077e+01            2.038189e+01            5.742356e+01            5.045771e+01            4.352687e+01            7.236436e+00            3.696385e+01            1.516944e+01            2.511438e+01            -inf                    
20     5.641609e+00            6.715814e+00            1.977803e+01            2.452009e+01            2.513569e+01            2.265722e+01            5.293363e+00            2.273348e+01            5.780647e+00            1.893831e+01            -inf                    
21     4.406281e+00            4.387981e+00            4.323875e+00            4.531877e+00            4.530522e+00            4.461173e+00            4.367613e+00            4.443689e+00            4.374068e+00            4.415384e+00            -inf                    
22     2.142187e+00            2.169643e+00            2.563368e+00            6.491011e+00            8.117472e+00            6.912209e+00            2.385263e+00            6.631937e+00            2.522822e+00            2.281925e+00            -inf                    
23     5.656724e+00            5.557762e+00            2.185882e+01            2.836862e+01            2.960002e+01            2.770702e+01            5.379685e+00            2.635104e+01            6.580733e+00            7.587657e+00            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_08_idea_0.py  (error=2.188521e-07)
Task  2: variant_02_idea_0.py  (error=1.992969e-05)
Task  3: variant_02_idea_0.py  (error=3.328931e-04)
Task  4: variant_02_idea_0.py  (error=2.331430e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_07_idea_0.py  (error=3.935829e-07)
Task  7: variant_02_idea_0.py  (error=1.329200e-05)
Task  8: variant_02_idea_0.py  (error=1.682779e-03)
Task  9: variant_07_idea_0.py  (error=4.673650e-04)
Task 10: variant_02_idea_0.py  (error=1.523926e+00)
Task 11: original.py  (error=7.138172e-01)
Task 12: original.py  (error=4.326057e-01)
Task 13: variant_01_idea_0.py  (error=8.994553e-01)
Task 14: variant_01_idea_0.py  (error=2.468053e+00)
Task 15: original.py  (error=9.241363e-01)
Task 16: variant_06_idea_0.py  (error=3.891009e+01)
Task 17: variant_02_idea_0.py  (error=3.120103e+00)
Task 18: variant_08_idea_0.py  (error=2.674363e+00)
Task 19: variant_06_idea_0.py  (error=7.236436e+00)
Task 20: variant_06_idea_0.py  (error=5.293363e+00)
Task 21: variant_02_idea_0.py  (error=4.323875e+00)
Task 22: original.py  (error=2.142187e+00)
Task 23: variant_06_idea_0.py  (error=5.379685e+00)

WIN COUNTS:
  variant_02_idea_0.py: 8 wins
  original.py: 4 wins
  variant_06_idea_0.py: 4 wins
  variant_08_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_01_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
            """Update covariance using weighted rank-mu + population-wide covariance."""
            dim = self.dim
            lam = self.pop_size
            selected = sorted_population[:self.mu]

            # Rank-1 update component
            rank1 = np.outer(self.p_c, self.p_c)
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            # Weighted rank-mu update (current approach)
            diff = (selected - old_mean[None, :]) / self.sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(self.mu):
                rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

            # NEW: Population-wide covariance for robustness
            pop_diff = (sorted_population - old_mean[None, :]) / self.sigma
            pop_cov = np.cov(pop_diff.T, bias=True) if lam > 1 else np.zeros((dim, dim))

            # Combined update with population-wide component
            # c_pop controls contribution of population-wide covariance
            c_pop = min(0.1, self.c_mu_cov * 0.5)
            self.C = ((1.0 - self.c_1 - self.c_mu_cov - c_pop + delta_h * self.c_1) * self.C +
                       self.c_1 * rank1 +
                       self.c_mu_cov * rank_mu +
                       c_pop * pop_cov)

            # Enforce symmetry and positive definiteness
            self.C = 0.5 * (self.C + self.C.T)
            # Ensure diagonal is positive
            diag = np.diag(self.C)
            np.fill_diagonal(self.C, np.maximum(diag, 1e-20))
```

# --- From variant_02_idea_0.py (8 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update the covariance matrix with rank-1 and active rank-mu updates."""
        dim = self.dim
        lam = self.pop_size

        # Rank-1 update component
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Active CMA: use ALL individuals with positive/negative weights
        # Positive weights for top half, negative weights for bottom half
        diff = (sorted_population - old_mean[None, :]) / self.sigma  # (lambda, dim)

        # Compute positive weights (top half)
        mu_pos = self.mu
        raw_pos = np.log(mu_pos + 0.5) - np.log(np.arange(1, mu_pos + 1))
        pos_weights = raw_pos / np.sum(raw_pos)

        # Compute negative weights (bottom half) with smaller magnitude
        mu_neg = lam - self.mu
        raw_neg = np.log(mu_neg + 0.5) - np.log(np.arange(1, mu_neg + 1))
        neg_weights = -raw_neg / np.sum(raw_neg) * 0.4  # scaled negative weights

        # Combine weights for all lambda individuals
        all_weights = np.concatenate([pos_weights, neg_weights])
        all_weights = np.clip(all_weights, -0.5, None)  # clip negative extremes

        # Normalize so sum of abs(weights) = 1 for consistent learning rate
        weight_sum = np.sum(np.abs(all_weights))
        if weight_sum > 1e-30:
            all_weights = all_weights / weight_sum

        # Compute rank-mu update using full population
        rank_mu = np.zeros((dim, dim))
        for i in range(lam):
            w = all_weights[i]
            if abs(w) > 1e-30:
                rank_mu += w * np.outer(diff[i], diff[i])

        # Combined update
        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu)

        # Enforce symmetry and positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        # Add minimal diagonal if needed for numerical stability
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-15:
            self.C += (1e-14 - min_eig) * np.eye(dim)
```

# --- From variant_06_idea_0.py (4 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update the covariance matrix using exponential fitness-weighted rank-mu update."""
        dim = self.dim
        selected = sorted_population[:self.mu]

        # Rank-1 update component (unchanged)
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Exponential fitness-based weights for covariance: emphasize best solutions
        # This is fundamentally different from log-linear weights used elsewhere
        inv_ranks = 1.0 / np.arange(1, self.mu + 1)
        exp_weights = np.exp(-1.5 * inv_ranks)  # exponential decay over rank
        exp_weights = exp_weights / np.sum(exp_weights)
        mu_eff_exp = 1.0 / np.sum(exp_weights ** 2)

        # Adaptive learning rate based on effective weight concentration
        c_mu_adaptive = min(0.9, self.c_mu_cov * (1.0 + 0.5 * (mu_eff_exp - 1.0)))

        # Rank-mu update with exponential weights (more aggressive selection)
        diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += exp_weights[i] * np.outer(diff[i], diff[i])

        # Combined update with adaptive learning rate
        self.C = ((1.0 - self.c_1 - c_mu_adaptive + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   c_mu_adaptive * rank_mu)

        # Enforce symmetry and add slight diagonal push for numerical stability
        self.C = 0.5 * (self.C + self.C.T)
        self.C.flat[::dim + 1] += 1e-10 * np.ones(dim)
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update covariance with exponentially weighted rank-mu history."""
        dim = self.dim
        selected = sorted_population[:self.mu]

        # Rank-1 update component
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Compute current rank-mu update
        diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        # Initialize historical rank-mu memory if needed
        if not hasattr(self, '_rank_mu_hist') or self._rank_mu_hist is None:
            self._rank_mu_hist = rank_mu.copy()

        # Exponentially weighted moving average for rank-mu history
        # c_mu_hist is the learning rate for the EWMA (use c_mu_cov * 0.1 for slow adaptation)
        c_mu_hist = self.c_mu_cov * 0.1
        self._rank_mu_hist = (1.0 - c_mu_hist) * self._rank_mu_hist + c_mu_hist * rank_mu

        # Blend current rank-mu with historical average (50/50 split)
        rank_mu_blend = 0.5 * rank_mu + 0.5 * self._rank_mu_hist

        # Combined update
        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu_blend)

        # Enforce symmetry and positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        # Ensure diagonal dominance for numerical stability
        diag = np.diag(self.C)
        min_diag = 1e-20
        if np.any(diag < min_diag):
            np.fill_diagonal(self.C, np.maximum(diag, min_diag))
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update covariance with adaptive learning rates and eigenvalue bounding."""
        dim = self.dim
        selected = sorted_population[:self.mu]

        # Rank-1 update component
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component with variance stabilization
        diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        # Add diagonal variance from selected solutions for stability
        selected_var = np.var(selected, axis=0) / (self.sigma ** 2 + 1e-20)
        diag_contribution = np.sum(self.weights) * np.mean(self.weights) * np.diag(selected_var)

        # Compute condition number for adaptive learning
        eigenvalues = self.eigenvalues
        cond = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-30)
        cond = min(cond, 1e14)

        # Adaptive learning rates: increase for ill-conditioned problems
        adapt_factor = 1.0 + 2.0 * np.log1p(cond) / np.log1p(1e14)
        c_1_adapt = min(self.c_1 * adapt_factor, 0.5)
        c_mu_adapt = min(self.c_mu_cov * adapt_factor, 0.5)

        # Combined update with adaptive rates and diagonal contribution
        self.C = ((1.0 - c_1_adapt - c_mu_adapt + delta_h * c_1_adapt) * self.C +
                   c_1_adapt * rank1 +
                   c_mu_adapt * (rank_mu + 0.1 * diag_contribution))

        # Symmetrize
        self.C = 0.5 * (self.C + self.C.T)

        # Ensure positive definiteness via eigenvalue bounding
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        self.C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        self.C = 0.5 * (self.C + self.C.T)
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