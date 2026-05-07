Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_initialize_archive` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_idea_1.py       variant_idea_2.py       variant_idea_3.py       variant_idea_4.py       variant_idea_5.py       variant_idea_6.py       variant_idea_7.py       variant_idea_8.py       
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      6.665466e+03            1.018221e+04            1.092807e+04            1.154161e+04            7.327146e+03            8.157949e+03            8.048200e+03            1.073403e+04            9.295288e+03            
1      1.582598e+00            1.603304e+00            1.603567e+00            1.586294e+00            1.596491e+00            1.598087e+00            1.605489e+00            1.600021e+00            1.599422e+00            
2      3.460653e+09            3.005813e+09            4.026233e+09            4.337634e+09            4.894199e+09            3.344719e+09            2.308025e+09            5.655018e+09            4.041109e+09            
3      3.044369e+04            2.896847e+04            2.792976e+04            2.491793e+04            2.732866e+04            4.224890e+04            3.963751e+04            2.105698e+04            3.238838e+04            
4      3.045805e+00            3.050738e+00            2.966787e+00            3.011074e+00            3.023915e+00            3.056149e+00            2.992749e+00            3.005556e+00            3.037682e+00            
5      3.068953e+00            3.007514e+00            2.971756e+00            3.008679e+00            2.981167e+00            2.980981e+00            3.001231e+00            2.995751e+00            3.030595e+00            
6      2.082719e+04            2.113174e+04            1.815055e+04            1.729732e+04            1.619631e+04            1.557985e+04            1.531785e+04            1.475228e+04            1.617372e+04            
7      1.377486e+04            1.113168e+04            1.586664e+04            1.604537e+04            1.767547e+04            1.434209e+04            1.367348e+04            1.560011e+04            1.137042e+04            
8      1.881525e+04            6.765680e+03            2.821846e+03            6.230786e+03            6.102785e+03            6.036486e+03            7.260808e+03            5.786428e+03            4.775727e+03            
9      1.242799e+04            1.138464e+04            1.906150e+04            1.313568e+04            1.247329e+04            1.783482e+04            1.170874e+04            1.205469e+04            1.149833e+04            
10     2.280780e+04            1.450554e+04            1.511357e+04            1.920798e+04            2.063374e+04            2.119903e+04            1.035951e+04            2.030030e+04            1.907674e+04            
11     1.796886e+04            2.651944e+04            1.634089e+04            8.757308e+03            1.782616e+04            1.412918e+04            2.701699e+04            3.086720e+04            4.290856e+04            
12     5.876919e+03            9.050022e+03            1.417663e+04            1.133371e+04            8.807238e+04            6.355682e+04            1.235798e+04            5.835369e+03            3.456209e+04            
13     2.855731e+04            2.878851e+04            3.020766e+04            3.020520e+04            2.996230e+04            2.772846e+04            2.283906e+04            3.008504e+04            2.855361e+04            
14     8.224086e+00            8.005694e+00            7.989778e+00            8.352580e+00            8.389256e+00            8.051324e+00            8.219188e+00            8.395583e+00            8.264216e+00            
15     3.573648e+03            3.731923e+03            4.211401e+03            3.636209e+03            5.173527e+03            4.824469e+03            3.168417e+03            4.315215e+03            4.066895e+03            
16     1.721784e+05            1.766670e+05            1.428573e+05            1.265300e+05            1.649814e+05            1.395901e+05            1.658181e+05            1.497179e+05            1.700514e+05            
17     1.105952e+04            2.788541e+04            1.264162e+04            8.819916e+03            2.014954e+04            8.892314e+03            1.053355e+04            1.807252e+04            1.281366e+04            
18     1.422447e+04            1.098667e+04            1.466614e+04            2.123154e+04            1.586750e+04            1.012637e+04            2.654100e+04            1.066540e+04            1.478636e+04            
19     1.162738e+01            1.151046e+01            1.157266e+01            1.168880e+01            1.146299e+01            1.159214e+01            1.166554e+01            1.160219e+01            1.243955e+01            
20     8.466778e+01            8.959132e+01            9.736867e+01            7.925391e+01            8.752835e+01            1.017555e+02            9.391214e+01            1.047221e+02            8.780221e+01            
21     3.426291e+04            5.654729e+04            3.486067e+04            4.483676e+04            4.542628e+04            5.842202e+04            4.105011e+04            6.305459e+04            5.999653e+04            
22     3.751842e+01            4.286153e+01            4.010062e+01            4.558618e+01            5.046521e+01            3.911159e+01            4.062796e+01            3.614035e+01            3.954261e+01            
23     1.400498e+02            1.339664e+02            1.400542e+02            1.309121e+02            1.356505e+02            1.432584e+02            1.357072e+02            1.334424e+02            1.300533e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=6.665466e+03)
Task  1: original.py  (error=1.582598e+00)
Task  2: variant_idea_6.py  (error=2.308025e+09)
Task  3: variant_idea_7.py  (error=2.105698e+04)
Task  4: variant_idea_2.py  (error=2.966787e+00)
Task  5: variant_idea_2.py  (error=2.971756e+00)
Task  6: variant_idea_7.py  (error=1.475228e+04)
Task  7: variant_idea_1.py  (error=1.113168e+04)
Task  8: variant_idea_2.py  (error=2.821846e+03)
Task  9: variant_idea_1.py  (error=1.138464e+04)
Task 10: variant_idea_6.py  (error=1.035951e+04)
Task 11: variant_idea_3.py  (error=8.757308e+03)
Task 12: variant_idea_7.py  (error=5.835369e+03)
Task 13: variant_idea_6.py  (error=2.283906e+04)
Task 14: variant_idea_2.py  (error=7.989778e+00)
Task 15: variant_idea_6.py  (error=3.168417e+03)
Task 16: variant_idea_3.py  (error=1.265300e+05)
Task 17: variant_idea_3.py  (error=8.819916e+03)
Task 18: variant_idea_5.py  (error=1.012637e+04)
Task 19: variant_idea_4.py  (error=1.146299e+01)
Task 20: variant_idea_3.py  (error=7.925391e+01)
Task 21: original.py  (error=3.426291e+04)
Task 22: variant_idea_7.py  (error=3.614035e+01)
Task 23: variant_idea_8.py  (error=1.300533e+02)

WIN COUNTS:
  variant_idea_6.py: 4 wins
  variant_idea_7.py: 4 wins
  variant_idea_2.py: 4 wins
  variant_idea_3.py: 4 wins
  original.py: 3 wins
  variant_idea_1.py: 2 wins
  variant_idea_5.py: 1 wins
  variant_idea_4.py: 1 wins
  variant_idea_8.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_idea_1.py (2 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        # Create LHS points by dividing each dimension into n equal intervals and randomly sampling one point per interval.
        intervals = np.linspace(0, 1, n + 1)
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = self._rng.permutation(n)
            randoms = self._rng.random(n)
            points[:, d] = intervals[perm] + randoms[perm] * (intervals[1] - intervals[0])
        # Scale to bounds and clip.
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
```

# --- From variant_idea_2.py (4 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        # Two parents: mean and a random point.
        parent1 = self.mean.copy()
        parent2 = self._rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
        offspring = []
        for _ in range(n):
            child = parent2.copy()
            # Random start index and length for segment from parent1.
            start = self._rng.integers(0, self.dim)
            length = self._rng.integers(0, self.dim + 1)
            indices = np.arange(start, start + length) % self.dim
            child[indices] = parent1[indices]
            offspring.append(child)
        offspring = np.array(offspring)
        offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
        self.elite_archive = offspring
```

# --- From variant_idea_3.py (4 wins) ---
```python
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
```

# --- From variant_idea_4.py (1 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        # Generate centers of lower and upper halves for each dimension.
        centers = []
        mid = (self.bounds[0] + self.bounds[1]) / 2.0
        lower_center = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * 0.25
        upper_center = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * 0.75
        for d in range(self.dim):
            point = np.full(self.dim, mid)
            point[d] = lower_center
            centers.append(point)
            point = np.full(self.dim, mid)
            point[d] = upper_center
            centers.append(point)
        centers = np.array(centers)
        # Select n points randomly without replacement.
        if n >= len(centers):
            selected = centers
        else:
            selected = self._rng.choice(centers, size=n, replace=False)
        selected = np.clip(selected, self.bounds[0], self.bounds[1])
        self.elite_archive = selected
```

# --- From variant_idea_5.py (1 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        # Compute eigenvectors of covariance matrix.
        try:
            _, eigenvectors = np.linalg.eigh(self.covariance)
        except np.linalg.LinAlgError:
            eigenvectors = np.eye(self.dim)
        # Generate points by perturbing mean along eigenvectors.
        points = []
        step = self.step_size * 0.5
        for i in range(n):
            if i < self.dim:
                point = self.mean.copy() + step * eigenvectors[:, i]
            else:
                point = self.mean.copy() + step * self._rng.standard_normal(self.dim)
            points.append(point)
        points = np.array(points)
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
```

# --- From variant_idea_6.py (4 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        def halton_sequence(index, base):
                result = 0.0
                f = 1.0
                while index > 0:
                    f /= base
                    result += f * (index % base)
                    index //= base
                return result
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            base = bases[d % len(bases)]
            for i in range(n):
                points[i, d] = halton_sequence(i + 1, base)
        # Scale to bounds and clip.
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
```

# --- From variant_idea_7.py (4 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        # Generate random vectors and orthogonalize via Gram-Schmidt.
        random_vectors = self._rng.standard_normal((n, self.dim))
        orthogonal_vectors = []
        for i in range(n):
            v = random_vectors[i].copy()
            for j in range(len(orthogonal_vectors)):
                v = v - np.dot(orthogonal_vectors[j], v) * orthogonal_vectors[j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
                orthogonal_vectors.append(v)
            else:
                # Generate new random vector if linearly dependent.
                v = self._rng.standard_normal(self.dim)
                for j in range(len(orthogonal_vectors)):
                    v = v - np.dot(orthogonal_vectors[j], v) * orthogonal_vectors[j]
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v = v / norm
                    orthogonal_vectors.append(v)
        orthogonal_vectors = np.array(orthogonal_vectors)
        # Generate points along orthogonal directions from the mean.
        points = []
        step = self.step_size * 0.5
        for i in range(min(len(orthogonal_vectors), n)):
            points.append(self.mean.copy() + step * orthogonal_vectors[i])
        while len(points) < n:
            points.append(self.mean.copy() + step * self._rng.standard_normal(self.dim))
        points = np.array(points)
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
```

# --- From variant_idea_8.py (1 wins) ---
```python
def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        points = []
        # Center point.
        center = (self.bounds[0] + self.bounds[1]) / 2.0
        points.append(np.full(self.dim, center))
        # Midpoints of edges: for each dimension, set that dimension to midpoint and others to bounds.
        for d in range(self.dim):
            point = np.full(self.dim, self.bounds[0])
            point[d] = center
            points.append(point)
            point = np.full(self.dim, self.bounds[1])
            point[d] = center
            points.append(point)
        # Add corners until we have n points.
        for bits in range(1 << self.dim):
            if len(points) >= n:
                break
            corner = np.array([self.bounds[1] if (bits >> d) & 1 else self.bounds[0] for d in range(self.dim)])
            points.append(corner)
        points = np.array(points)
        if len(points) > n:
            points = points[:n]
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
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
        """Initialize empty elite archive."""
        self.elite_archive = np.zeros((0, self.dim))
        
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