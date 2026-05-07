Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_initialize_tracking` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_idea_1.py       variant_idea_2.py       variant_idea_3.py       variant_idea_4.py       variant_idea_5.py       variant_idea_6.py       variant_idea_7.py       variant_idea_8.py       
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.921906e+03            4.062977e+03            4.619153e+03            5.338394e+03            4.381543e+03            6.140270e+03            4.242759e+03            5.603563e+03            5.033300e+03            
1      1.562710e+00            1.567059e+00            1.555364e+00            1.544585e+00            1.547830e+00            1.573781e+00            1.539824e+00            1.519209e+00            1.530874e+00            
2      1.986323e+09            1.701489e+09            1.998162e+09            1.419478e+09            1.847012e+09            2.284341e+09            2.141872e+09            2.646476e+09            2.517900e+09            
3      1.867045e+04            2.364404e+04            2.171956e+04            2.905664e+04            2.751719e+04            2.345705e+04            2.817969e+04            2.306992e+04            2.084410e+04            
4      2.896940e+00            2.965540e+00            2.915866e+00            2.874065e+00            2.933315e+00            2.897914e+00            2.912871e+00            2.868781e+00            2.896893e+00            
5      2.916394e+00            2.902268e+00            2.947396e+00            2.878838e+00            2.919088e+00            2.905105e+00            2.879077e+00            2.904457e+00            2.934761e+00            
6      1.120922e+04            8.880062e+03            1.316111e+04            9.387224e+03            8.900018e+03            9.388739e+03            1.130702e+04            1.030026e+04            1.081198e+04            
7      8.135992e+03            7.890992e+03            9.213866e+03            1.084819e+04            5.815871e+03            9.391100e+03            8.132296e+03            9.815124e+03            1.086833e+04            
8      4.097683e+03            4.471476e+03            5.999089e+03            2.902585e+03            3.690675e+03            4.060780e+03            3.813654e+03            3.451727e+03            3.023757e+03            
9      9.497941e+03            1.162812e+04            9.673830e+03            8.538685e+03            6.946104e+03            1.041582e+04            1.085383e+04            9.246808e+03            1.007468e+04            
10     1.226282e+04            1.073856e+04            1.198208e+04            1.366488e+04            1.299340e+04            1.381197e+04            7.709842e+03            1.098742e+04            1.605628e+04            
11     8.305572e+03            1.214685e+04            1.090236e+04            1.188671e+04            9.797078e+03            9.225932e+03            1.249187e+04            1.255460e+04            1.429983e+04            
12     3.500652e+03            3.973674e+03            4.783102e+03            3.769241e+03            2.781750e+03            3.894025e+03            6.347492e+03            8.654461e+03            5.680706e+03            
13     1.103286e+04            1.294575e+04            1.076082e+04            1.193881e+04            1.007289e+04            1.044241e+04            1.097667e+04            1.502971e+04            1.035193e+04            
14     6.459077e+00            6.473717e+00            6.410178e+00            6.573924e+00            6.755082e+00            6.493793e+00            6.378417e+00            6.621672e+00            6.854539e+00            
15     2.451391e+03            2.525064e+03            2.809238e+03            2.894755e+03            2.700199e+03            2.386707e+03            1.660524e+03            2.578436e+03            1.774537e+03            
16     9.059711e+04            7.900894e+04            1.067990e+05            6.653501e+04            1.078192e+05            8.735147e+04            7.352403e+04            8.774602e+04            1.056554e+05            
17     6.612334e+03            8.359917e+03            6.547224e+03            8.879948e+03            8.183615e+03            6.455594e+03            6.290406e+03            7.732058e+03            5.770740e+03            
18     9.529433e+03            6.453656e+03            7.234660e+03            8.562395e+03            8.692199e+03            5.374762e+03            5.874423e+03            8.927688e+03            1.146603e+04            
19     1.066332e+01            1.077230e+01            1.086749e+01            1.095769e+01            1.074151e+01            1.015774e+01            1.012587e+01            9.947649e+00            1.018818e+01            
20     6.879922e+01            8.198515e+01            6.503657e+01            7.749094e+01            7.462259e+01            6.514212e+01            6.818312e+01            6.552394e+01            7.462237e+01            
21     3.573693e+04            5.152396e+04            5.835385e+04            3.448796e+04            6.053712e+04            4.621473e+04            5.839521e+04            3.805128e+04            5.669698e+04            
22     3.111218e+01            3.320487e+01            3.687664e+01            3.276157e+01            2.996893e+01            3.274090e+01            3.286529e+01            3.006427e+01            3.378784e+01            
23     1.182659e+02            1.243799e+02            1.208153e+02            1.070034e+02            1.180827e+02            1.152449e+02            1.243701e+02            1.330635e+02            1.224870e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_idea_1.py  (error=4.062977e+03)
Task  1: variant_idea_7.py  (error=1.519209e+00)
Task  2: variant_idea_3.py  (error=1.419478e+09)
Task  3: original.py  (error=1.867045e+04)
Task  4: variant_idea_7.py  (error=2.868781e+00)
Task  5: variant_idea_3.py  (error=2.878838e+00)
Task  6: variant_idea_1.py  (error=8.880062e+03)
Task  7: variant_idea_4.py  (error=5.815871e+03)
Task  8: variant_idea_3.py  (error=2.902585e+03)
Task  9: variant_idea_4.py  (error=6.946104e+03)
Task 10: variant_idea_6.py  (error=7.709842e+03)
Task 11: original.py  (error=8.305572e+03)
Task 12: variant_idea_4.py  (error=2.781750e+03)
Task 13: variant_idea_4.py  (error=1.007289e+04)
Task 14: variant_idea_6.py  (error=6.378417e+00)
Task 15: variant_idea_6.py  (error=1.660524e+03)
Task 16: variant_idea_3.py  (error=6.653501e+04)
Task 17: variant_idea_8.py  (error=5.770740e+03)
Task 18: variant_idea_5.py  (error=5.374762e+03)
Task 19: variant_idea_7.py  (error=9.947649e+00)
Task 20: variant_idea_2.py  (error=6.503657e+01)
Task 21: variant_idea_3.py  (error=3.448796e+04)
Task 22: variant_idea_4.py  (error=2.996893e+01)
Task 23: variant_idea_3.py  (error=1.070034e+02)

WIN COUNTS:
  variant_idea_3.py: 6 wins
  variant_idea_4.py: 5 wins
  variant_idea_7.py: 3 wins
  variant_idea_6.py: 3 wins
  variant_idea_1.py: 2 wins
  original.py: 2 wins
  variant_idea_8.py: 1 wins
  variant_idea_5.py: 1 wins
  variant_idea_2.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_idea_1.py (2 wins) ---
```python
def _initialize_tracking(self):
        # Exponential crossover initialization
        # Sample two individuals from current distribution
        samples = self._sample_from_distribution(2)
        p1, p2 = samples[0], samples[1]
        # Choose random start index
        start = self._rng.integers(0, self.dim)
        # Determine segment length using geometric distribution (mean 1/p)
        pGeom = 0.5
        L = int(self._rng.geometric(pGeom))
        L = min(self.dim, max(1, L))
        # Build mask for contiguous segment
        indices = (np.arange(self.dim) + start) % self.dim
        mask = np.zeros(self.dim, dtype=bool)
        mask[indices[:L]] = True
        # Create offspring
        offspring = np.where(mask, p1, p2)
        # Clip to bounds
        low, high = self.bounds
        offspring = np.clip(offspring, low, high)
        # Pseudo‑fitness: Euclidean distance to mean
        pseudo_fit = float(np.linalg.norm(offspring - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        # Additional diversity tracking: distance between parents
        self.diversity_history = [float(np.linalg.norm(p1 - p2))]
```

# --- From variant_idea_2.py (1 wins) ---
```python
def _initialize_tracking(self):
        # Arithmetic blend (BLX‑α) initialization
        samples = self._sample_from_distribution(2)
        p1, p2 = samples[0], samples[1]
        alpha = 0.5
        gene_min = np.minimum(p1, p2)
        gene_max = np.maximum(p1, p2)
        range_len = gene_max - gene_min
        # Avoid zero range
        range_len = np.where(range_len == 0, 1e-8, range_len)
        factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
        child = gene_min + factor * range_len
        low, high = self.bounds
        child = np.clip(child, low, high)
        pseudo_fit = float(np.linalg.norm(child - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.linalg.norm(p1 - p2))]
```

# --- From variant_idea_3.py (6 wins) ---
```python
def _initialize_tracking(self):
        # Segment‑based crossover initialization
        samples = self._sample_from_distribution(2)
        p1, p2 = samples[0], samples[1]
        seg_start = self._rng.integers(0, self.dim)
        max_len = max(1, self.dim - seg_start)
        seg_len = self._rng.integers(1, max_len + 1)
        child = p1.copy()
        child[seg_start:seg_start + seg_len] = p2[seg_start:seg_start + seg_len]
        low, high = self.bounds
        child = np.clip(child, low, high)
        pseudo_fit = float(np.linalg.norm(child - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.linalg.norm(p1 - p2))]
```

# --- From variant_idea_4.py (5 wins) ---
```python
def _initialize_tracking(self):
        # Eigenvector‑rotated initialization
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            # Sample random vector in rotated space
            z = self._rng.standard_normal(self.dim)
            # Rotate back and scale by step size
            rotated = self.mean + self.step_size * (z @ eigenvectors.T)
        except np.linalg.LinAlgError:
            # Fallback: sample from distribution directly
            rotated = self._sample_from_distribution(1)[0]
        low, high = self.bounds
        rotated = np.clip(rotated, low, high)
        pseudo_fit = float(np.linalg.norm(rotated - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.linalg.norm(rotated - self.mean))]
```

# --- From variant_idea_5.py (1 wins) ---
```python
def _initialize_tracking(self):
        # Quasi‑random (Van der Corput) initialization
        def van_der_corput(n, base=2):
                seq = np.zeros(n)
                for i in range(n):
                    q = 0.0
                    denom = 1.0
                    idx = i + 1
                    while idx > 0:
                        denom *= base
                        q += (idx % base) / denom
                        idx //= base
                    seq[i] = q
                return seq
        low, high = self.bounds
        # Generate two quasi‑random sequences with different bases
        seq2 = van_der_corput(self.dim, base=2)
        seq3 = van_der_corput(self.dim, base=3)
        # Combine them to form a vector
        quasi = (seq2 + seq3) / 2.0
        range_len = high - low
        range_len = range_len if range_len != 0 else 1e-12
        quasi = low + quasi * range_len
        pseudo_fit = float(np.linalg.norm(quasi - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.linalg.norm(quasi - self.mean))]
```

# --- From variant_idea_6.py (3 wins) ---
```python
def _initialize_tracking(self):
        # Self‑adversarial initialization
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            # Direction of largest variance
            max_idx = np.argmax(eigenvalues)
            direction = eigenvectors[:, max_idx]
            # Compute step as half of the search range
            search_range = self.bounds[1] - self.bounds[0]
            step = 0.5 * search_range
            adversarial = self.mean + step * direction
        except np.linalg.LinAlgError:
            # Fallback: random direction
            direction = self._rng.standard_normal(self.dim)
            norm = np.linalg.norm(direction)
            direction = direction / (norm + 1e-12)
            search_range = self.bounds[1] - self.bounds[0]
            step = 0.5 * search_range
            adversarial = self.mean + step * direction
        low, high = self.bounds
        adversarial = np.clip(adversarial, low, high)
        pseudo_fit = float(np.linalg.norm(adversarial - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.linalg.norm(adversarial - self.mean))]
```

# --- From variant_idea_7.py (3 wins) ---
```python
def _initialize_tracking(self):
        # Hybrid multi‑operator initialization
        samples = self._sample_from_distribution(3)
        p1, p2, p3 = samples[0], samples[1], samples[2]
        # Operator 1: exponential crossover between p1 and p2
        start = self._rng.integers(0, self.dim)
        pGeom = 0.5
        L = int(self._rng.geometric(pGeom))
        L = min(self.dim, max(1, L))
        indices = (np.arange(self.dim) + start) % self.dim
        mask = np.zeros(self.dim, dtype=bool)
        mask[indices[:L]] = True
        child1 = np.where(mask, p1, p2)
        # Operator 2: arithmetic blend between p2 and p3
        alpha = 0.5
        gene_min = np.minimum(p2, p3)
        gene_max = np.maximum(p2, p3)
        range_len = gene_max - gene_min
        range_len = np.where(range_len == 0, 1e-8, range_len)
        factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
        child2 = gene_min + factor * range_len
        # Operator 3: segment‑based between p3 and p1
        seg_start = self._rng.integers(0, self.dim)
        max_len = max(1, self.dim - seg_start)
        seg_len = self._rng.integers(1, max_len + 1)
        child3 = p3.copy()
        child3[seg_start:seg_start + seg_len] = p1[seg_start:seg_start + seg_len]
        low, high = self.bounds
        child1 = np.clip(child1, low, high)
        child2 = np.clip(child2, low, high)
        child3 = np.clip(child3, low, high)
        # Compute pseudo‑fitness as min distance to mean
        distances = np.array([np.linalg.norm(c - self.mean) for c in [child1, child2, child3]])
        pseudo_fit = float(np.min(distances))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.mean(distances))]
```

# --- From variant_idea_8.py (1 wins) ---
```python
def _initialize_tracking(self):
        # Random projection initialization using orthogonal matrix
        # Generate random orthogonal matrix via QR decomposition
        Z = self._rng.standard_normal((self.dim, self.dim))
        try:
            Q, R = np.linalg.qr(Z)
            # Ensure columns have positive sign for deterministic orientation
            signs = np.sign(np.diag(R))
            Q = Q * signs
        except np.linalg.LinAlgError:
            Q = np.eye(self.dim)
        # Sample a random vector and project onto orthogonal basis
        z = self._rng.standard_normal(self.dim)
        projected = self.mean + self.step_size * (z @ Q.T)
        low, high = self.bounds
        projected = np.clip(projected, low, high)
        pseudo_fit = float(np.linalg.norm(projected - self.mean))
        self.best_fitness_history = [pseudo_fit]
        self.stagnation_counter = 0
        self.diversity_history = [float(np.linalg.norm(projected - self.mean))]
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
    Adaptive Covariance-Guided Evolution Strategy (CG-ES)
    
    Automatically selects the best _clear_archive strategy using Thompson Sampling
    based on sliding-window credit assignment of restart improvement rewards.
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
        
        # Adaptive operator selection (Thompson Sampling with sliding window)
        self.num_operators = 7
        self._operator_rewards = [[] for _ in range(self.num_operators)]
        self._reward_window_size = 10
        self._selected_operator = 0
        self._restart_history = []
        self._last_restart_best_fitness = None
        self._total_restarts = 0
        
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
                self._select_clear_archive_operator()
                self._record_pre_restart_fitness()
                self._restart_optimizer()
                self._total_restarts += 1
                
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
            range_len = gene_max - gene_min + 1e-8
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
            finite_values = self.current_fitness[np.isfinite(self.current_fitness)]
            if len(finite_values) > 0:
                worst = np.nanmax(finite_values)
            else:
                worst = 1e10
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
        
        rank_one = np.outer(self.mean_diff, self.mean_diff)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i, ind in enumerate(selected_pop):
            diff = (ind - self.mean).reshape(-1, 1)
            rank_mu += np.dot(diff, diff.T)
        rank_mu /= len(selected_pop)
        
        self.covariance = (
            (1 - self.c1 - self.cmu) * self.covariance +
            self.c1 * rank_one +
            self.cmu * rank_mu
        )
        
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
        
        if len(self.best_fitness_history) >= 5:
            improvement_rate = self._compute_improvement_rate()
            self._adjust_step_size(improvement_rate)
            
    def _update_stagnation_tracking(self, current_best):
        """Track fitness history for stagnation detection."""
        self.best_fitness_history.append(float(current_best))
        if len(self.best_fitness_history) > 20:
            self.best_fitness_history.pop(0)
            
        if len(self.best_fitness_history) >= 2:
            if self.best_fitness_history[-1] >= self.best_fitness_history[-2]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
    def _compute_improvement_rate(self):
        """Compute recent improvement rate."""
        history = np.array(self.best_fitness_history[-5:], dtype=float)
        if len(history) < 2:
            return 0.0
        return float(np.mean(np.diff(history)))
        
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
        return float(np.mean(distances))
        
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
        
    def _select_clear_archive_operator(self):
        """Select operator using Thompson Sampling with sliding window rewards."""
        min_trials = 2
        
        if self._total_restarts < min_trials:
            self._selected_operator = self._rng.integers(0, self.num_operators)
        else:
            samples = np.zeros(self.num_operators)
            for i in range(self.num_operators):
                rewards = self._operator_rewards[i]
                if len(rewards) > 0:
                    mean_reward = float(np.mean(rewards))
                    std_reward = float(np.std(rewards)) if len(rewards) > 1 else 0.1
                    std_reward = max(0.01, std_reward)
                    samples[i] = float(self._rng.normal(mean_reward, std_reward))
                else:
                    samples[i] = float(self._rng.normal(-0.5, 1.0))
            
            self._selected_operator = int(np.argmax(samples))
        
        self._restart_history.append({
            'operator_idx': int(self._selected_operator),
            'iteration': len(self._restart_history)
        })
        
    def _record_pre_restart_fitness(self):
        """Record fitness before restart for reward calculation."""
        curr_fitness = np.min(self.current_fitness)
        if np.isfinite(curr_fitness):
            self._last_restart_best_fitness = float(curr_fitness)
        else:
            self._last_restart_best_fitness = 1e10
            
    def _restart_optimizer(self):
        """Perform strategic restart of the optimizer."""
        self._reinitialize_distribution()
        self._clear_archive()
        self._reset_tracking()
        self._update_operator_reward_after_restart()
        
    def _update_operator_reward_after_restart(self):
        """Update operator reward based on improvement after restart."""
        if self._last_restart_best_fitness is None or len(self._restart_history) == 0:
            return
            
        last_entry = self._restart_history[-1]
        operator_idx = last_entry['operator_idx']
        
        curr_fitness = np.min(self.current_fitness)
        if not np.isfinite(curr_fitness):
            curr_fitness = 1e10
            
        prev_fitness = self._last_restart_best_fitness
        improvement = prev_fitness - curr_fitness
        
        baseline = max(1e-10, abs(prev_fitness) + abs(curr_fitness) + 1.0)
        reward = float(np.clip(improvement / baseline, -1.0, 1.0))
        
        rewards = self._operator_rewards[operator_idx]
        if len(rewards) >= self._reward_window_size:
            rewards.pop(0)
        rewards.append(reward)
        
    def _clear_archive(self):
        """Clear the elite archive using the selected adaptive operator."""
        op = self._selected_operator
        
        if op == 0:
            self._clear_archive_original()
        elif op == 1:
            self._clear_archive_variant_1()
        elif op == 2:
            self._clear_archive_variant_2()
        elif op == 3:
            self._clear_archive_variant_3()
        elif op == 4:
            self._clear_archive_variant_4()
        elif op == 5:
            self._clear_archive_variant_5()
        elif op == 6:
            self._clear_archive_variant_6()
        else:
            self._clear_archive_original()
            
    def _clear_archive_original(self):
        """Original: Clear the elite archive without any processing."""
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_1(self):
        """Variant 1: Exponential crossover operator applied to archive members."""
        if self.elite_archive.shape[0] >= 2:
            idx = self._rng.choice(self.elite_archive.shape[0], 2, replace=False)
            p1 = self.elite_archive[idx[0]]
            p2 = self.elite_archive[idx[1]]
            lam = max(1, self.dim // 2)
            probs = np.exp(-np.arange(self.dim) / lam) / self.dim
            mask = self._rng.random(self.dim) < probs
            offspring = np.where(mask, p1, p2)
            low, high = self.bounds
            offspring = np.clip(offspring, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_2(self):
        """Variant 2: Arithmetic blend (BLX-α) crossover between archive centroid and current mean."""
        if self.elite_archive.shape[0] > 0:
            alpha = 0.5
            parent1 = np.mean(self.elite_archive, axis=0)
            parent2 = self.mean.copy()
            gene_min = np.minimum(parent1, parent2)
            gene_max = np.maximum(parent1, parent2)
            range_len = gene_max - gene_min
            range_len = np.where(range_len == 0, 1e-8, range_len)
            factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
            blended = gene_min + factor * range_len
            low, high = self.bounds
            blended = np.clip(blended, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_3(self):
        """Variant 3: Segment-based crossover: swap a random contiguous segment."""
        if self.elite_archive.shape[0] >= 2:
            idx = self._rng.choice(self.elite_archive.shape[0], 2, replace=False)
            p1 = self.elite_archive[idx[0]]
            p2 = self.elite_archive[idx[1]]
            seg_start = self._rng.integers(0, self.dim)
            max_len = max(1, self.dim - seg_start)
            seg_len = self._rng.integers(1, max_len + 1)
            offspring = p1.copy()
            offspring[seg_start:seg_start + seg_len] = p2[seg_start:seg_start + seg_len]
            low, high = self.bounds
            offspring = np.clip(offspring, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_4(self):
        """Variant 4: Rotate archive using eigenvectors of the current covariance matrix."""
        if self.elite_archive.shape[0] > 0:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
                centered = self.elite_archive - self.mean
                rotated = centered @ eigenvectors
                low, high = self.bounds
                rotated = np.clip(rotated, low, high)
            except np.linalg.LinAlgError:
                pass
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_5(self):
        """Variant 5: Deterministic crowding: replace worst archive member with best."""
        if self.elite_archive.shape[0] > 0:
            distances = np.linalg.norm(self.elite_archive - self.mean, axis=1)
            best_idx = int(np.argmin(distances))
            worst_idx = int(np.argmax(distances))
            self.elite_archive[worst_idx] = self.elite_archive[best_idx].copy()
            low, high = self.bounds
            self.elite_archive = np.clip(self.elite_archive, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_6(self):
        """Variant 6: Clustering-based compression using k-means on archive."""
        if self.elite_archive.shape[0] > 1:
            k = max(2, self.elite_archive.shape[0] // 2)
            centroids = self.elite_archive[self._rng.choice(
                self.elite_archive.shape[0], k, replace=False
            )]
            for _ in range(5):
                dists = np.sum(
                    (self.elite_archive[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                    axis=2
                )
                assignments = np.argmin(dists, axis=1)
                for c in range(k):
                    mask = assignments == c
                    if np.any(mask):
                        centroids[c] = np.mean(self.elite_archive[mask], axis=0)
            low, high = self.bounds
            centroids = np.clip(centroids, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _reset_tracking(self):
        """Reset tracking variables after restart."""
        self.stagnation_counter = 0
        self.best_fitness_history = []
        
    def _get_best_result(self):
        """Extract best solution found from current population."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        best_fitness = float(self.current_fitness[best_idx])
        
        return best_fitness, best_position.copy()

```