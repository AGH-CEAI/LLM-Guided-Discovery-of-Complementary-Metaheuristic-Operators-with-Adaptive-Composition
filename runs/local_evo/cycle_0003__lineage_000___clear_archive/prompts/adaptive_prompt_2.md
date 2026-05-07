Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_clear_archive` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_idea_1.py       variant_idea_2.py       variant_idea_3.py       variant_idea_4.py       variant_idea_5.py       variant_idea_6.py       
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      4.629205e+03            5.927031e+03            4.261711e+03            4.009197e+03            5.352217e+03            3.605464e+03            4.885494e+03            
1      1.565110e+00            1.552692e+00            1.530983e+00            1.559042e+00            1.521496e+00            1.539319e+00            1.567540e+00            
2      3.273365e+09            2.347583e+09            1.946373e+09            2.386119e+09            1.624356e+09            2.179667e+09            2.991354e+09            
3      1.842929e+04            2.440247e+04            2.234700e+04            2.050464e+04            2.937947e+04            1.358939e+04            2.409666e+04            
4      2.984865e+00            2.960414e+00            2.950158e+00            2.967094e+00            2.902041e+00            2.910689e+00            2.914824e+00            
5      2.869887e+00            2.927208e+00            2.919266e+00            2.903000e+00            2.866120e+00            2.938052e+00            2.917211e+00            
6      8.052013e+03            1.317989e+04            1.290301e+04            1.172213e+04            1.039610e+04            1.256259e+04            1.269608e+04            
7      9.361273e+03            1.053277e+04            6.528280e+03            9.581290e+03            1.288902e+04            1.147998e+04            1.068029e+04            
8      4.225940e+03            2.847769e+03            3.770030e+03            3.584572e+03            4.049161e+03            5.501674e+03            6.178636e+03            
9      9.617165e+03            1.223569e+04            1.025476e+04            7.764241e+03            1.408258e+04            1.224461e+04            9.996611e+03            
10     1.740837e+04            1.073268e+04            1.175689e+04            1.415323e+04            1.125874e+04            1.270641e+04            1.032311e+04            
11     8.307930e+03            1.135961e+04            1.029167e+04            1.038969e+04            9.201991e+03            1.316463e+04            1.586077e+04            
12     4.411893e+03            3.796517e+03            4.395677e+03            4.302581e+03            5.980533e+03            4.301728e+03            5.371683e+03            
13     1.242910e+04            9.450220e+03            1.157739e+04            1.286554e+04            1.151806e+04            1.128380e+04            1.182251e+04            
14     6.649406e+00            6.624113e+00            6.475723e+00            6.648652e+00            6.585647e+00            6.851968e+00            6.528831e+00            
15     3.066451e+03            2.193818e+03            2.638700e+03            2.395791e+03            2.698683e+03            2.105498e+03            3.266427e+03            
16     6.576938e+04            9.459894e+04            7.916058e+04            6.071284e+04            1.109473e+05            8.116780e+04            9.621928e+04            
17     8.405028e+03            8.383328e+03            4.820168e+03            7.573378e+03            6.458777e+03            5.319294e+03            7.940485e+03            
18     8.474447e+03            1.040931e+04            1.213901e+04            1.014784e+04            7.509971e+03            6.997312e+03            9.892396e+03            
19     1.113913e+01            1.081531e+01            1.032866e+01            1.010502e+01            1.038724e+01            1.032667e+01            1.082203e+01            
20     6.721060e+01            8.375251e+01            6.657143e+01            6.770852e+01            7.371588e+01            7.429768e+01            6.576004e+01            
21     5.111231e+04            5.082357e+04            6.799172e+04            4.316804e+04            6.764887e+04            5.610468e+04            4.701908e+04            
22     3.406642e+01            3.687493e+01            3.430496e+01            3.707525e+01            3.515195e+01            3.375797e+01            3.806677e+01            
23     1.271318e+02            1.198211e+02            1.171884e+02            1.257225e+02            1.276994e+02            1.182184e+02            1.241545e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_idea_5.py  (error=3.605464e+03)
Task  1: variant_idea_4.py  (error=1.521496e+00)
Task  2: variant_idea_4.py  (error=1.624356e+09)
Task  3: variant_idea_5.py  (error=1.358939e+04)
Task  4: variant_idea_4.py  (error=2.902041e+00)
Task  5: variant_idea_4.py  (error=2.866120e+00)
Task  6: original.py  (error=8.052013e+03)
Task  7: variant_idea_2.py  (error=6.528280e+03)
Task  8: variant_idea_1.py  (error=2.847769e+03)
Task  9: variant_idea_3.py  (error=7.764241e+03)
Task 10: variant_idea_6.py  (error=1.032311e+04)
Task 11: original.py  (error=8.307930e+03)
Task 12: variant_idea_1.py  (error=3.796517e+03)
Task 13: variant_idea_1.py  (error=9.450220e+03)
Task 14: variant_idea_2.py  (error=6.475723e+00)
Task 15: variant_idea_5.py  (error=2.105498e+03)
Task 16: variant_idea_3.py  (error=6.071284e+04)
Task 17: variant_idea_2.py  (error=4.820168e+03)
Task 18: variant_idea_5.py  (error=6.997312e+03)
Task 19: variant_idea_3.py  (error=1.010502e+01)
Task 20: variant_idea_6.py  (error=6.576004e+01)
Task 21: variant_idea_3.py  (error=4.316804e+04)
Task 22: variant_idea_5.py  (error=3.375797e+01)
Task 23: variant_idea_2.py  (error=1.171884e+02)

WIN COUNTS:
  variant_idea_5.py: 5 wins
  variant_idea_4.py: 4 wins
  variant_idea_2.py: 4 wins
  variant_idea_3.py: 4 wins
  variant_idea_1.py: 3 wins
  original.py: 2 wins
  variant_idea_6.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_idea_1.py (3 wins) ---
```python
def _clear_archive(self):
        # Exponential crossover operator applied to archive members before discarding
        if self.elite_archive.shape[0] >= 2:
            # Randomly select two distinct individuals from the archive
            idx = self._rng.choice(self.elite_archive.shape[0], 2, replace=False)
            p1 = self.elite_archive[idx[0]]
            p2 = self.elite_archive[idx[1]]
            # Exponential crossover: probability of taking gene from p1 decays exponentially with dimension index
            lam = max(1, self.dim // 2)
            probs = np.exp(-np.arange(self.dim) / lam) / self.dim
            mask = self._rng.random(self.dim) < probs
            offspring = np.where(mask, p1, p2)
            # Clip to bounds for robustness
            low, high = self.bounds
            offspring = np.clip(offspring, low, high)
        # Clear the archive
        self.elite_archive = np.zeros((0, self.dim))
```

# --- From variant_idea_2.py (4 wins) ---
```python
def _clear_archive(self):
        # Arithmetic blend (BLX-α) crossover between archive centroid and current mean before clearing
        if self.elite_archive.shape[0] > 0:
            alpha = 0.5
            # Use current mean as second parent
            parent1 = np.mean(self.elite_archive, axis=0)
            parent2 = self.mean
            # Compute min and max for each gene
            gene_min = np.minimum(parent1, parent2)
            gene_max = np.maximum(parent1, parent2)
            range_len = gene_max - gene_min
            # Avoid division by zero
            range_len = np.where(range_len == 0, 1e-8, range_len)
            factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
            blended = gene_min + factor * range_len
            low, high = self.bounds
            blended = np.clip(blended, low, high)
        # Clear archive
        self.elite_archive = np.zeros((0, self.dim))
```

# --- From variant_idea_3.py (4 wins) ---
```python
def _clear_archive(self):
        # Segment-based crossover: swap a random contiguous segment between two archive members
        if self.elite_archive.shape[0] >= 2:
            idx = self._rng.choice(self.elite_archive.shape[0], 2, replace=False)
            p1 = self.elite_archive[idx[0]]
            p2 = self.elite_archive[idx[1]]
            # Choose segment start and length
            seg_start = self._rng.integers(0, self.dim)
            max_len = max(1, self.dim - seg_start)
            seg_len = self._rng.integers(1, max_len + 1)
            # Create offspring by swapping segment
            offspring = p1.copy()
            offspring[seg_start:seg_start + seg_len] = p2[seg_start:seg_start + seg_len]
            # Clip to bounds
            low, high = self.bounds
            offspring = np.clip(offspring, low, high)
        # Clear archive
        self.elite_archive = np.zeros((0, self.dim))
```

# --- From variant_idea_4.py (4 wins) ---
```python
def _clear_archive(self):
        # Rotate archive using eigenvectors of the current covariance matrix before clearing
        if self.elite_archive.shape[0] > 0:
            try:
                # Compute eigenvalues and eigenvectors (sorted)
                eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
                # Ensure eigenvectors are in correct shape (dim x dim)
                # Rotate: project archive onto eigenvector basis
                centered = self.elite_archive - self.mean
                rotated = centered @ eigenvectors
                # Clip rotated values to bounds (approximate)
                low, high = self.bounds
                rotated = np.clip(rotated, low, high)
            except np.linalg.LinAlgError:
                # If covariance is invalid, skip rotation
                pass
        # Clear archive
        self.elite_archive = np.zeros((0, self.dim))
```

# --- From variant_idea_5.py (5 wins) ---
```python
def _clear_archive(self):
        # Deterministic crowding: replace the worst archive member with the best before clearing
        if self.elite_archive.shape[0] > 0:
            # Compute distance to current mean as fitness proxy
            distances = np.linalg.norm(self.elite_archive - self.mean, axis=1)
            best_idx = np.argmin(distances)
            worst_idx = np.argmax(distances)
            # Replace worst with best (crowding effect)
            self.elite_archive[worst_idx] = self.elite_archive[best_idx]
            # Clip to bounds for robustness
            low, high = self.bounds
            self.elite_archive = np.clip(self.elite_archive, low, high)
        # Clear archive
        self.elite_archive = np.zeros((0, self.dim))
```

# --- From variant_idea_6.py (2 wins) ---
```python
def _clear_archive(self):
        # Clustering-based compression: run simple k-means on archive before clearing
        if self.elite_archive.shape[0] > 1:
            k = max(2, self.elite_archive.shape[0] // 2)
            # Initialize centroids by randomly sampling k individuals from archive
            centroids = self.elite_archive[self._rng.choice(self.elite_archive.shape[0], k, replace=False)]
            # Simple k-means for a few iterations
            for _ in range(5):
                # Compute squared distances between each point and each centroid
                dists = np.sum((self.elite_archive[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
                assignments = np.argmin(dists, axis=1)
                # Update centroids
                for c in range(k):
                    mask = assignments == c
                    if np.any(mask):
                        centroids[c] = np.mean(self.elite_archive[mask], axis=0)
            # Clip centroids to bounds (optional)
            low, high = self.bounds
            centroids = np.clip(centroids, low, high)
        # Clear archive
        self.elite_archive = np.zeros((0, self.dim))
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