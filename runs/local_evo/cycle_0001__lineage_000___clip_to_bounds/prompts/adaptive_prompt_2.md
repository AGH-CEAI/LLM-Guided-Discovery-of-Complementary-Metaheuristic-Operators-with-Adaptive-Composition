Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_clip_to_bounds` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_idea_1.py       variant_idea_2.py       variant_idea_3.py       variant_idea_4.py       variant_idea_5.py       variant_idea_6.py       variant_idea_7.py       variant_idea_8.py       
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.114927e+04            2.680819e+03            5.048561e+03            3.922321e+03            8.394264e+04            6.747810e+03            5.183741e+03            1.533385e+04            5.787969e+03            
1      1.582901e+00            1.520724e+00            1.574439e+00            1.557318e+00            1.753591e+00            1.539771e+00            1.570162e+00            1.618721e+00            1.526771e+00            
2      3.583001e+09            1.281276e+09            1.925620e+09            2.941362e+09            3.524817e+10            2.371255e+09            1.947193e+09            7.766413e+09            1.863370e+09            
3      2.007709e+04            2.155056e+04            2.417261e+04            2.087355e+04            3.531471e+05            3.021912e+04            2.587552e+04            6.715446e+04            2.527913e+04            
4      2.991093e+00            2.903893e+00            2.917002e+00            2.953661e+00            3.357812e+00            2.966522e+00            2.918328e+00            3.054319e+00            2.925640e+00            
5      2.993520e+00            2.935736e+00            2.912427e+00            2.885165e+00            3.356267e+00            2.964497e+00            2.898440e+00            3.030141e+00            2.878819e+00            
6      1.529972e+04            7.729247e+03            1.449990e+04            1.488994e+04            7.622032e+04            1.118781e+04            1.369692e+04            1.658765e+04            9.718550e+03            
7      1.264577e+04            1.214641e+04            1.137452e+04            9.218718e+03            1.771958e+04            9.650126e+03            7.474518e+03            1.492809e+04            9.273117e+03            
8      8.778418e+03            4.064272e+03            3.367684e+03            4.080868e+03            2.046082e+04            3.099366e+03            3.400814e+03            6.658088e+03            3.983841e+03            
9      1.495628e+04            1.315202e+04            9.068487e+03            1.086846e+04            1.352706e+04            9.260988e+03            8.503039e+03            9.796814e+03            1.180322e+04            
10     2.303039e+04            1.327822e+04            9.697387e+03            1.358855e+04            1.361650e+04            1.149689e+04            1.428193e+04            1.576792e+04            1.096252e+04            
11     1.485501e+04            9.673282e+03            1.115795e+04            1.303954e+04            1.852434e+04            1.549297e+04            8.025902e+03            1.126319e+04            1.114125e+04            
12     9.550312e+03            6.096629e+03            4.677016e+03            3.064785e+03            6.231240e+03            4.740527e+03            5.604005e+03            1.051641e+04            2.500335e+03            
13     2.828238e+04            1.255028e+04            2.888619e+04            1.496359e+04            1.198691e+04            1.130324e+04            1.481762e+04            1.223990e+04            9.495319e+03            
14     8.169992e+00            6.314379e+00            7.973536e+00            6.334634e+00            7.811498e+00            6.581940e+00            6.697313e+00            6.777506e+00            6.581994e+00            
15     2.985887e+03            3.374036e+03            2.560226e+03            2.496632e+03            6.602487e+04            2.971648e+03            1.846808e+03            7.098868e+03            2.730255e+03            
16     3.018309e+05            9.988224e+04            1.165079e+05            8.914514e+04            2.998874e+06            1.059174e+05            8.602055e+04            2.850290e+05            1.520842e+05            
17     1.155477e+04            9.306716e+03            7.544798e+03            9.968508e+03            1.284175e+04            8.434956e+03            1.313558e+04            7.858710e+03            6.779061e+03            
18     1.111992e+04            7.217137e+03            1.349393e+04            1.400556e+04            1.010514e+04            1.345453e+04            1.290035e+04            7.252380e+03            8.296690e+03            
19     1.232993e+01            1.079049e+01            9.955274e+00            1.028374e+01            1.342633e+01            1.066310e+01            1.092869e+01            1.067009e+01            1.115356e+01            
20     8.621199e+01            7.602859e+01            7.908223e+01            7.427855e+01            2.343058e+02            6.473131e+01            7.531164e+01            9.731082e+00            6.138959e+01            
21     6.491630e+04            5.465569e+04            5.676861e+04            7.031299e+04            3.020271e+04            7.495665e+04            5.124124e+04            4.667089e+04            5.352461e+04            
22     4.402652e+01            3.286601e+01            3.274556e+01            3.591446e+01            4.847840e+01            3.825473e+01            3.376333e+01            3.883084e+01            3.639807e+01            
23     1.223122e+02            1.208291e+02            1.340202e+02            1.193873e+02            1.530368e+02            1.366840e+02            1.236465e+02            1.339448e+02            1.374286e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_idea_1.py  (error=2.680819e+03)
Task  1: variant_idea_1.py  (error=1.520724e+00)
Task  2: variant_idea_1.py  (error=1.281276e+09)
Task  3: original.py  (error=2.007709e+04)
Task  4: variant_idea_1.py  (error=2.903893e+00)
Task  5: variant_idea_8.py  (error=2.878819e+00)
Task  6: variant_idea_1.py  (error=7.729247e+03)
Task  7: variant_idea_6.py  (error=7.474518e+03)
Task  8: variant_idea_5.py  (error=3.099366e+03)
Task  9: variant_idea_6.py  (error=8.503039e+03)
Task 10: variant_idea_2.py  (error=9.697387e+03)
Task 11: variant_idea_6.py  (error=8.025902e+03)
Task 12: variant_idea_8.py  (error=2.500335e+03)
Task 13: variant_idea_8.py  (error=9.495319e+03)
Task 14: variant_idea_1.py  (error=6.314379e+00)
Task 15: variant_idea_6.py  (error=1.846808e+03)
Task 16: variant_idea_6.py  (error=8.602055e+04)
Task 17: variant_idea_8.py  (error=6.779061e+03)
Task 18: variant_idea_1.py  (error=7.217137e+03)
Task 19: variant_idea_2.py  (error=9.955274e+00)
Task 20: variant_idea_7.py  (error=9.731082e+00)
Task 21: variant_idea_4.py  (error=3.020271e+04)
Task 22: variant_idea_2.py  (error=3.274556e+01)
Task 23: variant_idea_3.py  (error=1.193873e+02)

WIN COUNTS:
  variant_idea_1.py: 7 wins
  variant_idea_6.py: 5 wins
  variant_idea_8.py: 4 wins
  variant_idea_2.py: 3 wins
  original.py: 1 wins
  variant_idea_5.py: 1 wins
  variant_idea_7.py: 1 wins
  variant_idea_4.py: 1 wins
  variant_idea_3.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_idea_1.py (7 wins) ---
```python
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
```

# --- From variant_idea_2.py (3 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        above = candidates > high
        below = candidates < low
        w = self._rng.uniform(0.5, 1.5, size=candidates.shape)
        repaired = np.where(above, high - w * (candidates - high), candidates)
        repaired = np.where(below, low + w * (low - candidates), repaired)
        return np.clip(repaired, low, high)
```

# --- From variant_idea_3.py (1 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        segment = high - low
        if segment == 0:
            return np.full_like(candidates, low)
        period = 2 * segment
        shifted = candidates - low
        t = shifted % period
        reflected = np.where(t <= segment, low + t, high - (t - segment))
        return np.clip(reflected, low, high)
```

# --- From variant_idea_4.py (1 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        if self.covariance is not None:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                V = eigenvectors
                centered = candidates - self.mean
                y = centered @ V
                sigma = np.sqrt(eigenvalues)
                y_clipped = np.clip(y, -sigma, sigma)
                repaired = y_clipped @ V.T + self.mean
                return np.clip(repaired, low, high)
            except np.linalg.LinAlgError:
                pass
        return np.clip(candidates, low, high)
```

# --- From variant_idea_5.py (1 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        repaired = candidates.copy()
        max_iter = 10
        for _ in range(max_iter):
            above = repaired > high
            below = repaired < low
            if not (np.any(above) or np.any(below)):
                break
            diff = repaired[above] - high
            repaired[above] = high - diff
            diff = low - repaired[below]
            repaired[below] = low + diff
        return np.clip(repaired, low, high)
```

# --- From variant_idea_6.py (5 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        repaired = candidates.copy()
        if self.covariance is not None:
            std = np.sqrt(np.diag(self.covariance))
        else:
            std = np.full(self.dim, self.step_size)
        std = np.maximum(std, 1e-12)
        noise = self._rng.normal(0, std, size=candidates.shape)
        above = repaired > high
        below = repaired < low
        if np.any(above):
            repaired[above] = np.clip(high + noise[above], low, high)
        if np.any(below):
            repaired[below] = np.clip(low - noise[below], low, high)
        return repaired
```

# --- From variant_idea_7.py (1 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        midpoint = (high + low) / 2.0
        rng = high - low
        if rng == 0:
            return np.full_like(candidates, low)
        z = (candidates - midpoint) / (rng / 2.0)
        compressed = np.tanh(z)
        repaired = low + (compressed + 1.0) * 0.5 * rng
        return np.clip(repaired, low, high)
```

# --- From variant_idea_8.py (4 wins) ---
```python
def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        rng = high - low
        if rng == 0:
            return np.full_like(candidates, low)
        above = candidates > high
        below = candidates < low
        d_above = candidates - high
        d_below = low - candidates
        shrink_above = 1.0 / (1.0 + d_above / rng)
        shrink_below = 1.0 / (1.0 + d_below / rng)
        repaired = np.where(above, high - d_above * shrink_above, candidates)
        repaired = np.where(below, low + d_below * shrink_below, repaired)
        return np.clip(repaired, low, high)
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
        """Clip all candidates to the search bounds."""
        return np.clip(candidates, self.bounds[0], self.bounds[1])
        
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