Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_add_to_archive` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_idea_1.py       variant_idea_2.py       variant_idea_3.py       variant_idea_4.py       variant_idea_5.py       variant_idea_6.py       variant_idea_7.py       variant_idea_8.py       
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      4.961975e+03            9.661229e+03            6.444769e+03            6.195698e+03            6.122150e+03            8.020133e+03            7.466172e+03            -inf                    6.702842e+03            
1      1.542589e+00            1.561574e+00            1.534072e+00            1.560495e+00            1.556814e+00            1.566719e+00            1.543184e+00            -inf                    1.542355e+00            
2      2.329247e+09            2.129110e+09            3.837108e+09            2.527655e+09            3.581001e+09            3.230612e+09            2.267423e+09            -inf                    2.826567e+09            
3      2.583890e+04            3.780674e+04            3.490660e+04            2.527710e+04            2.310046e+04            2.704689e+04            1.900576e+04            -inf                    2.850959e+04            
4      2.914634e+00            2.927135e+00            2.966257e+00            2.978052e+00            2.954659e+00            2.920997e+00            2.971382e+00            -inf                    2.943860e+00            
5      2.900267e+00            2.906782e+00            2.858614e+00            2.906891e+00            2.924593e+00            2.870072e+00            2.898166e+00            -inf                    2.880441e+00            
6      1.357718e+04            1.222469e+04            8.234157e+03            1.055220e+04            1.083527e+04            7.921156e+03            9.910069e+03            -inf                    1.107139e+04            
7      8.244093e+03            1.219286e+04            9.448934e+03            8.782674e+03            9.866056e+03            1.227122e+04            8.605855e+03            -inf                    9.505258e+03            
8      3.900785e+03            4.806853e+03            3.279165e+03            4.941328e+03            5.136420e+03            3.234887e+03            3.852807e+03            -inf                    3.848511e+03            
9      9.199047e+03            1.086891e+04            1.381437e+04            1.170307e+04            1.068169e+04            1.175131e+04            1.245156e+04            -inf                    9.399112e+03            
10     9.740912e+03            1.112840e+04            1.156561e+04            9.579463e+03            1.160314e+04            1.110787e+04            1.177756e+04            -inf                    1.221412e+04            
11     9.441209e+03            1.357325e+04            1.213469e+04            1.119476e+04            1.123879e+04            1.213813e+04            1.280487e+04            -inf                    1.498902e+04            
12     4.769813e+03            6.014439e+03            4.223251e+03            4.358923e+03            4.622310e+03            5.726230e+03            4.933762e+03            -inf                    4.769313e+03            
13     1.346200e+04            9.443828e+03            1.166406e+04            1.189329e+04            1.181756e+04            1.187370e+04            1.199755e+04            -inf                    1.412895e+04            
14     6.513653e+00            6.489874e+00            6.428844e+00            6.667184e+00            6.789966e+00            6.576064e+00            6.720098e+00            -inf                    6.571921e+00            
15     2.698635e+03            4.754151e+03            4.181695e+03            4.182080e+03            3.243523e+03            6.400288e+03            2.074553e+03            -inf                    3.345387e+03            
16     1.041192e+05            1.390271e+05            9.925807e+04            8.735723e+04            9.928761e+04            1.052014e+05            9.461620e+04            -inf                    1.221117e+05            
17     5.699810e+03            6.956039e+03            6.084590e+03            7.628167e+03            6.223272e+03            9.798710e+03            5.875714e+03            -inf                    7.998904e+03            
18     8.895416e+03            7.586100e+03            5.418552e+03            1.134043e+04            7.827110e+03            7.990153e+03            6.315520e+03            -inf                    9.776223e+03            
19     9.610284e+00            1.124443e+01            1.014587e+01            1.038230e+01            1.028341e+01            1.025020e+01            1.141143e+01            -inf                    1.091168e+01            
20     6.868018e+01            7.614959e+01            6.732428e+01            7.649255e+01            8.249364e+01            7.327366e+01            8.500053e+01            -inf                    6.868513e+01            
21     4.046187e+04            4.828287e+04            5.976777e+04            4.503373e+04            4.957720e+04            5.161752e+04            4.628416e+04            -inf                    5.153611e+04            
22     3.856544e+01            3.636118e+01            3.570577e+01            3.543326e+01            3.761996e+01            3.478608e+01            3.534315e+01            -inf                    3.533074e+01            
23     1.172903e+02            1.231249e+02            1.130783e+02            1.217162e+02            1.113504e+02            1.197011e+02            1.269396e+02            -inf                    1.121876e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=4.961975e+03)
Task  1: variant_idea_2.py  (error=1.534072e+00)
Task  2: variant_idea_1.py  (error=2.129110e+09)
Task  3: variant_idea_6.py  (error=1.900576e+04)
Task  4: original.py  (error=2.914634e+00)
Task  5: variant_idea_2.py  (error=2.858614e+00)
Task  6: variant_idea_5.py  (error=7.921156e+03)
Task  7: original.py  (error=8.244093e+03)
Task  8: variant_idea_5.py  (error=3.234887e+03)
Task  9: original.py  (error=9.199047e+03)
Task 10: variant_idea_3.py  (error=9.579463e+03)
Task 11: original.py  (error=9.441209e+03)
Task 12: variant_idea_2.py  (error=4.223251e+03)
Task 13: variant_idea_1.py  (error=9.443828e+03)
Task 14: variant_idea_2.py  (error=6.428844e+00)
Task 15: variant_idea_6.py  (error=2.074553e+03)
Task 16: variant_idea_3.py  (error=8.735723e+04)
Task 17: original.py  (error=5.699810e+03)
Task 18: variant_idea_2.py  (error=5.418552e+03)
Task 19: original.py  (error=9.610284e+00)
Task 20: variant_idea_2.py  (error=6.732428e+01)
Task 21: original.py  (error=4.046187e+04)
Task 22: variant_idea_5.py  (error=3.478608e+01)
Task 23: variant_idea_4.py  (error=1.113504e+02)

WIN COUNTS:
  original.py: 8 wins
  variant_idea_2.py: 6 wins
  variant_idea_5.py: 3 wins
  variant_idea_1.py: 2 wins
  variant_idea_6.py: 2 wins
  variant_idea_3.py: 2 wins
  variant_idea_4.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_idea_1.py (2 wins) ---
```python
def _add_to_archive(self, new_elites):
        """Add new elite individuals using exponential crossover."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        low, high = self.bounds
        n_new = len(new_elites)
        n_archive = len(self.elite_archive)
        indices = self._rng.choice(n_archive, size=n_new, replace=True)
        parents = self.elite_archive[indices]
        offspring = np.empty_like(new_elites)
        for i in range(n_new):
            child = new_elites[i].copy()
            k = self._rng.integers(0, self.dim)
            for j in range(self.dim):
                idx = (k + j) % self.dim
                if self._rng.random() < 0.5:
                    child[idx] = parents[i, idx]
            offspring[i] = np.clip(child, low, high)
        self.elite_archive = np.vstack([self.elite_archive, offspring])
```

# --- From variant_idea_2.py (6 wins) ---
```python
def _add_to_archive(self, new_elites):
        """Add new elite individuals using BLX-α arithmetic blend crossover."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        low, high = self.bounds
        n_new = len(new_elites)
        alpha = 0.5
        indices = self._rng.choice(len(self.elite_archive), size=n_new, replace=True)
        parents = self.elite_archive[indices]
        offspring = np.empty_like(new_elites)
        for i in range(n_new):
            p_min = np.minimum(new_elites[i], parents[i])
            p_max = np.maximum(new_elites[i], parents[i])
            rng = p_max - p_min + 1e-12
            gamma = 1 + 2 * alpha
            child = p_min - alpha * rng + self._rng.uniform(0, gamma, size=self.dim) * (p_max - p_min + alpha * rng)
            offspring[i] = np.clip(child, low, high)
        self.elite_archive = np.vstack([self.elite_archive, offspring])
```

# --- From variant_idea_3.py (2 wins) ---
```python
def _add_to_archive(self, new_elites):
        """Add new elite individuals using segment-based crossover."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        low, high = self.bounds
        n_new = len(new_elites)
        num_segments = max(2, self.dim // 5)
        seg_size = self.dim // num_segments
        indices = self._rng.choice(len(self.elite_archive), size=n_new, replace=True)
        parents = self.elite_archive[indices]
        offspring = np.empty_like(new_elites)
        for i in range(n_new):
            child = new_elites[i].copy()
            boundaries = self._rng.choice(self.dim - 1, size=num_segments - 1, replace=False)
            boundaries = np.sort(boundaries)
            starts = np.concatenate([[0], boundaries + 1])
            ends = np.concatenate([boundaries, [self.dim]])
            for s, e in zip(starts, ends):
                if self._rng.random() < 0.5:
                    child[s:e] = parents[i, s:e]
            offspring[i] = np.clip(child, low, high)
        self.elite_archive = np.vstack([self.elite_archive, offspring])
```

# --- From variant_idea_4.py (1 wins) ---
```python
def _add_to_archive(self, new_elites):
        """Add new elite individuals using eigenvector-rotated crossover."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        low, high = self.bounds
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            T = eigenvectors * sqrt_eigenvalues
            T_inv = eigenvectors.T / sqrt_eigenvalues
            new_rotated = new_elites @ T
            indices = self._rng.choice(len(self.elite_archive), size=len(new_elites), replace=True)
            parents_rotated = self.elite_archive[indices] @ T
            mask = self._rng.random((len(new_elites), self.dim)) < 0.5
            offspring_rotated = np.where(mask, new_rotated, parents_rotated)
            offspring = offspring_rotated @ T_inv
        except np.linalg.LinAlgError:
            indices = self._rng.choice(len(self.elite_archive), size=len(new_elites), replace=True)
            offspring = np.where(
                self._rng.random((len(new_elites), self.dim)) < 0.5,
                new_elites, self.elite_archive[indices]
            )
        self.elite_archive = np.vstack([self.elite_archive, np.clip(offspring, low, high)])
```

# --- From variant_idea_5.py (3 wins) ---
```python
def _add_to_archive(self, new_elites):
        """Add new elite individuals using fitness-weighted probabilistic merge."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        low, high = self.bounds
        fitnesses = self._evaluate_archive_members()
        valid_fitness = np.where(np.isfinite(fitnesses), fitnesses, np.max(fitnesses) + 1e6)
        inv_fitness = 1.0 / (valid_fitness - np.min(valid_fitness) + 1e-6)
        probs = inv_fitness / np.sum(inv_fitness)
        indices = self._rng.choice(len(self.elite_archive), size=len(new_elites), replace=True, p=probs)
        parents = self.elite_archive[indices]
        weights = self._rng.uniform(0.3, 0.7, size=(len(new_elites), 1))
        offspring = weights * new_elites + (1 - weights) * parents
        self.elite_archive = np.vstack([self.elite_archive, np.clip(offspring, low, high)])
```

# --- From variant_idea_6.py (2 wins) ---
```python
def _add_to_archive(self, new_elites):
        """Add new elite individuals using crowding distance for diversity."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        low, high = self.bounds
        all_points = np.vstack([self.elite_archive, new_elites])
        n = len(all_points)
        distances = np.zeros(n)
        for d in range(self.dim):
            sorted_idx = np.argsort(all_points[:, d])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            span = all_points[sorted_idx[-1], d] - all_points[sorted_idx[0], d] + 1e-12
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (all_points[sorted_idx[i + 1], d] - all_points[sorted_idx[i - 1], d]) / span
        archive_indices = set(range(len(self.elite_archive)))
        candidates = list(range(len(self.elite_archive), n))
        sorted_candidates = sorted(candidates, key=lambda i: distances[i], reverse=True)
        threshold = np.percentile(distances[distances < np.inf], 50) if np.any(distances < np.inf) else 0.0
        selected = [c for c in sorted_candidates if distances[c] > threshold]
        if not selected:
            selected = sorted_candidates[:max(1, len(sorted_candidates) // 2)]
        self.elite_archive = np.vstack([self.elite_archive, np.clip(new_elites[[c - len(self.elite_archive) for c in selected]], low, high)])
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


class CovarianceGuidedEvolutionStrategy:
    """
    Covariance-Guided Evolution Strategy (CG-ES)
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with a simplified rank-based update and adaptive population management.
    Uses a small elite archive and restart mechanism for robustness.
    """
    
    def __init__(self, dim=30, pop_size=None, bounds=(-100, 100)):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size if pop_size else max(50, 4 + int(3 * np.log(dim)))
        self._rng = np.random.default_rng()
        
        # Distribution parameters
        self.mean = None
        self.covariance = None
        self.step_size = 1.0
        
        # Adaptive parameters
        self.mu_eff = 1.0
        self.cc = 0.01
        self.c1 = 0.01
        self.cmu = 0.01
        self.damping = 1.0
        
        # Archive for elitism
        self.elite_archive = None
        self.archive_size = 10
        
        # Diversity and stagnation tracking
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        self._initialize_optimizer()
        
        while not stopping_condition():
            self._sample_and_evaluate(func)
            self._update_distribution_parameters()
            self._adapt_step_size()
            self._update_elite_archive()
            self._manage_stagnation()
            
            if self._should_restart():
                self._restart_optimizer()
                
        return self._get_best_result()
    
    def _initialize_optimizer(self):
        """Initialize all optimizer state variables."""
        self._initialize_distribution()
        self._initialize_archive()
        self._initialize_tracking()
        
    def _initialize_distribution(self):
        """Initialize mean vector and covariance matrix."""
        self.mean = self._rng.uniform(
            self.bounds[0], self.bounds[1], size=self.dim
        )
        self.covariance = np.eye(self.dim)
        self.step_size = 0.5 * (self.bounds[1] - self.bounds[0])
        
    def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        parent1 = self._rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
        parent2 = self._rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
        alpha = 0.5
        offspring = []
        for _ in range(n):
            gene_min = np.minimum(parent1, parent2)
            gene_max = np.maximum(parent1, parent2)
            range_len = gene_max - gene_min + 1e-8  # Avoid division by zero
            factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
            child = gene_min + factor * range_len
            offspring.append(child)
        offspring = np.array(offspring)
        offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
        self.elite_archive = offspring
        
    def _initialize_tracking(self):
        """Initialize stagnation and diversity tracking."""
        self.best_fitness_history = []
        self.stagnation_counter = 0
        
    def _sample_and_evaluate(self, func):
        """Sample candidates, clip bounds, and evaluate fitness."""
        candidates = self._sample_candidates()
        clipped = self._clip_to_bounds(candidates)
        self.current_fitness = func(clipped)
        self.current_population = clipped
        
        self._handle_evaluation_errors()
        
    def _sample_candidates(self):
        """Sample new candidates from the adapted distribution."""
        num_samples = self.pop_size - len(self.elite_archive)
        samples = self._sample_from_distribution(num_samples)
        return samples
        
    def _sample_from_distribution(self, num_samples):
        """Draw samples using simplified covariance sampling."""
        try:
            L = np.linalg.cholesky(self.covariance)
            noise = self._rng.standard_normal((num_samples, self.dim))
            samples = self.mean + self.step_size * (noise @ L.T)
        except np.linalg.LinAlgError:
            samples = self._fallback_sampling(num_samples)
        return samples
        
    def _fallback_sampling(self, num_samples):
        """Fallback to diagonal covariance sampling."""
        std = np.sqrt(np.diag(self.covariance)) + 1e-8
        samples = self.mean + self.step_size * self._rng.standard_normal(
            (num_samples, self.dim)
        ) * std
        return samples
        
    def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        rng = high - low
        rng = rng if rng != 0 else 1e-12
        above = candidates > high
        below = candidates < low
        d_above = candidates - high
        d_below = low - candidates
        repaired = np.where(above, high - d_above * np.exp(-d_above / rng), candidates)
        repaired = np.where(below, low + d_below * np.exp(-d_below / rng), repaired)
        return np.clip(repaired, low, high)
        
    def _handle_evaluation_errors(self):
        """Replace NaN/Inf fitness values with worst possible fitness."""
        valid = np.isfinite(self.current_fitness)
        if not np.all(valid):
            worst = np.nanmax(self.current_fitness[np.isfinite(self.current_fitness)])
            worst = worst if np.isfinite(worst) else 1e10
            self.current_fitness[~valid] = worst
            
    def _update_distribution_parameters(self):
        """Update mean and covariance based on selected individuals."""
        selected_indices = self._select_indices()
        selected_pop = self._get_combined_population()[selected_indices]
        selected_fitness = self.current_fitness[selected_indices]
        
        self._update_mean(selected_pop, selected_fitness)
        self._update_covariance(selected_pop)
        
    def _select_indices(self):
        """Select top individuals based on fitness (minimization)."""
        num_selected = max(1, self.pop_size // 4)
        sorted_indices = np.argsort(self.current_fitness)
        return sorted_indices[:num_selected]
        
    def _get_combined_population(self):
        """Combine current population with elite archive."""
        if len(self.elite_archive) > 0:
            return np.vstack([self.current_population, self.elite_archive])
        return self.current_population
        
    def _update_mean(self, selected_pop, selected_fitness):
        """Update the distribution mean using weighted average."""
        weights = self._compute_selection_weights(selected_fitness)
        old_mean = self.mean.copy()
        self.mean = np.average(selected_pop, axis=0, weights=weights)
        self.mean_diff = self.mean - old_mean
        
    def _compute_selection_weights(self, fitness):
        """Compute weights for selected individuals (softer than strict selection)."""
        normalized = fitness - fitness.min() + 1e-6
        weights = 1.0 / normalized
        weights /= weights.sum()
        return weights
        
    def _update_covariance(self, selected_pop):
        """Simplified rank-based covariance update."""
        self._compute_adaptation_learning_rates()
        
        # Rank-1 update
        rank_one = np.outer(self.mean_diff, self.mean_diff)
        
        # Rank-mu update (simplified)
        rank_mu = np.zeros((self.dim, self.dim))
        for i, ind in enumerate(selected_pop):
            diff = (ind - self.mean).reshape(-1, 1)
            rank_mu += np.dot(diff, diff.T)
        rank_mu /= len(selected_pop)
        
        # Combine updates
        self.covariance = (
            (1 - self.c1 - self.cmu) * self.covariance +
            self.c1 * rank_one +
            self.cmu * rank_mu
        )
        
        # Ensure positive definiteness
        self._ensure_covariance_validity()
        
    def _compute_adaptation_learning_rates(self):
        """Compute learning rates for covariance adaptation."""
        pop_ratio = self.pop_size / self.dim
        self.c1 = 2.0 / (pop_ratio + 2.0 * self.dim)
        self.cmu = min(0.6, 2.0 / (pop_ratio + 2.0 * self.dim**0.5))
        
    def _ensure_covariance_validity(self):
        """Ensure covariance matrix is valid (symmetric, positive semi-definite)."""
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue < 1e-10:
            self.covariance += (1e-9 - min_eigenvalue) * np.eye(self.dim)
            
    def _adapt_step_size(self):
        """Adapt step size based on successful mutations."""
        current_best = np.min(self.current_fitness)
        self._update_stagnation_tracking(current_best)
        
        # Simple step size adaptation
        if len(self.best_fitness_history) >= 5:
            improvement_rate = self._compute_improvement_rate()
            self._adjust_step_size(improvement_rate)
            
    def _update_stagnation_tracking(self, current_best):
        """Track fitness history for stagnation detection."""
        self.best_fitness_history.append(current_best)
        if len(self.best_fitness_history) > 20:
            self.best_fitness_history.pop(0)
            
        if len(self.best_fitness_history) >= 2:
            if self.best_fitness_history[-1] >= self.best_fitness_history[-2]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
    def _compute_improvement_rate(self):
        """Compute recent improvement rate."""
        history = np.array(self.best_fitness_history[-5:])
        if len(history) < 2:
            return 0.0
        return np.mean(np.diff(history))
        
    def _adjust_step_size(self, improvement_rate):
        """Adjust step size based on improvement rate."""
        if improvement_rate >= 0:
            self.step_size *= 1.1
        else:
            self.step_size *= 0.9
        self.step_size = np.clip(self.step_size, 1e-10, 
                                  (self.bounds[1] - self.bounds[0]) * 0.5)
        
    def _update_elite_archive(self):
        """Update the elite archive with best individuals."""
        num_elites = min(self.archive_size, self.pop_size // 5)
        sorted_indices = np.argsort(self.current_fitness)
        new_elites = self.current_population[sorted_indices[:num_elites]]
        
        self._add_to_archive(new_elites)
        self._prune_archive()
        
    def _add_to_archive(self, new_elites):
        """Add new elite individuals to archive."""
        self.elite_archive = np.vstack([self.elite_archive, new_elites])
        
    def _prune_archive(self):
        """Maintain archive size by keeping only best individuals."""
        if len(self.elite_archive) > self.archive_size:
            fitness_values = self._evaluate_archive_members()
            best_indices = np.argsort(fitness_values)[:self.archive_size]
            self.elite_archive = self.elite_archive[best_indices]
            
    def _evaluate_archive_members(self):
        """Evaluate archive members (for pruning)."""
        fitness_values = np.array([
            self.current_fitness[np.argmin(
                np.sum((self.current_population - elite)**2, axis=1)
            )]
            for elite in self.elite_archive
        ])
        return fitness_values
        
    def _manage_stagnation(self):
        """Check and respond to stagnation conditions."""
        diversity = self._compute_diversity()
        self.diversity_threshold = 1e-6 * (self.bounds[1] - self.bounds[0])
        
        if diversity < self.diversity_threshold:
            self._boost_diversity()
            
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        population = self._get_combined_population()
        if len(population) < 2:
            return 1.0
            
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid)**2, axis=1))
        return np.mean(distances)
        
    def _boost_diversity(self):
        """Boost diversity by perturbing the mean and increasing step size."""
        perturbation = self._rng.standard_normal(self.dim)
        self.mean += 0.3 * self.step_size * perturbation
        self.step_size *= 1.5
        self.step_size = np.clip(self.step_size, 1e-10,
                                  (self.bounds[1] - self.bounds[0]) * 0.5)
        
    def _should_restart(self):
        """Determine if optimizer should restart."""
        return (
            self.stagnation_counter > self.stagnation_threshold or
            self.step_size < 1e-10 or
            np.any(np.isnan(self.mean)) or
            np.any(np.isinf(self.mean))
        )
        
    def _restart_optimizer(self):
        """Perform strategic restart of the optimizer."""
        self._reinitialize_distribution()
        self._clear_archive()
        self._reset_tracking()
        
    def _reinitialize_distribution(self):
        """Reinitialize distribution around current best."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        
        self.mean = best_position.copy()
        self.covariance = np.eye(self.dim) * (self.step_size**2)
        self.step_size = 0.5 * (self.bounds[1] - self.bounds[0])
        
    def _clear_archive(self):
        """Clear the elite archive."""
        self.elite_archive = np.zeros((0, self.dim))
        
    def _reset_tracking(self):
        """Reset tracking variables after restart."""
        self.stagnation_counter = 0
        self.best_fitness_history = []
        
    def _get_best_result(self):
        """Extract best solution found from current population."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        best_fitness = self.current_fitness[best_idx]
        
        return best_fitness, best_position.copy()

```