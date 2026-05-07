**Idea 1: Exponential Crossover Initialization**  
Uses exponential crossover to create archive members from random parents.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    num_parents = max(2, self.pop_size)
    parents = self._rng.uniform(low, high, size=(num_parents, self.dim))
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        lam = max(1, self.dim // 2)
        probs = np.exp(-np.arange(self.dim) / lam)
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / self.dim
        else:
            probs = probs / probs_sum
        offspring = []
        for _ in range(n):
            idx = self._rng.choice(num_parents, 2, replace=False)
            p1, p2 = parents[idx[0]], parents[idx[1]]
            mask = self._rng.random(self.dim) < probs
            child = np.where(mask, p1, p2)
            child = np.clip(child, low, high)
            offspring.append(child)
        self.elite_archive = np.array(offspring)
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 2: Arithmetic Blend (BLX‑α) Initialization**  
Applies BLX‑α crossover between random parents to generate archive members.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        alpha = 0.5
        parents = self._rng.uniform(low, high, size=(2 * n, self.dim))
        offspring = []
        for i in range(n):
            p1 = parents[2 * i]
            p2 = parents[2 * i + 1]
            gene_min = np.minimum(p1, p2)
            gene_max = np.maximum(p1, p2)
            range_len = gene_max - gene_min
            range_len = np.where(range_len == 0, 1e-8, range_len)
            factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
            child = gene_min + factor * range_len
            child = np.clip(child, low, high)
            offspring.append(child)
        self.elite_archive = np.array(offspring)
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 3: Segment‑Based Crossover Initialization**  
Uses a random contiguous segment swap between parents to produce offspring.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        parents = self._rng.uniform(low, high, size=(2 * n, self.dim))
        offspring = []
        for i in range(n):
            p1 = parents[2 * i]
            p2 = parents[2 * i + 1]
            if self.dim == 1:
                seg_start = 0
                seg_len = 1
            else:
                seg_start = self._rng.integers(0, self.dim)
                max_len = max(1, self.dim - seg_start)
                seg_len = self._rng.integers(1, max_len + 1)
            child = p1.copy()
            end = min(seg_start + seg_len, self.dim)
            child[seg_start:end] = p2[seg_start:end]
            child = np.clip(child, low, high)
            offspring.append(child)
        self.elite_archive = np.array(offspring)
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 4: Eigenvector‑Rotated Initialization**  
Rotates standard‑normal points by a random orthogonal matrix before scaling and shifting.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        points = self._rng.standard_normal((n, self.dim))
        random_matrix = self._rng.standard_normal((self.dim, self.dim))
        try:
            Q, R = np.linalg.qr(random_matrix)
            signs = np.sign(np.diag(R))
            Q = Q * signs
            rotated = points @ Q.T
            rotated = self.mean + self.step_size * rotated
            rotated = np.clip(rotated, low, high)
            self.elite_archive = rotated
        except np.linalg.LinAlgError:
            self.elite_archive = self._rng.uniform(low, high, size=(n, self.dim))
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 5: Latin Hypercube Sampling Initialization**  
Generates archive members with Latin Hypercube Sampling for uniform coverage.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        intervals = n
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            u = self._rng.uniform(0, 1, size=n)
            perm = self._rng.permuted(np.arange(intervals))
            interval_width = 1.0 / intervals
            values = (perm + u) * interval_width
            points[:, d] = values
        self.elite_archive = low + points * (high - low)
        self.elite_archive = np.clip(self.elite_archive, low, high)
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 6: PCA‑Based Initialization**  
Computes principal components from a random sample and rotates points along eigenvectors.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        sample_size = max(self.dim, n)
        sample = self._rng.uniform(low, high, size=(sample_size, self.dim))
        sample_centered = sample - sample.mean(axis=0)
        if sample_size > 1:
            cov = np.cov(sample_centered, rowvar=False)
        else:
            cov = np.eye(self.dim)
        cov = 0.5 * (cov + cov.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if np.any(np.isnan(eigenvectors)) or np.any(np.isinf(eigenvectors)):
            self.elite_archive = self._rng.uniform(low, high, size=(n, self.dim))
        else:
            points = self._rng.standard_normal((n, self.dim))
            rotated = points @ eigenvectors.T
            rotated = self.mean + self.step_size * rotated
            rotated = np.clip(rotated, low, high)
            self.elite_archive = rotated
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 7: Adaptive Grid Initialization**  
Covers the search space with a grid and samples each cell, with jitter for diversity.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive = np.zeros((0, self.dim))
    else:
        k = max(2, int(np.round(n ** (1.0 / self.dim))))
        grids = [np.linspace(low, high, k + 1) for _ in range(self.dim)]
        mg = np.meshgrid(*grids, indexing='ij')
        points = np.stack([g.ravel() for g in mg], axis=1)
        jitter = self._rng.uniform(-rng / (2 * k), rng / (2 * k), size=points.shape)
        points = points + jitter
        points = np.clip(points, low, high)
        if len(points) > n:
            idx = self._rng.choice(len(points), n, replace=False)
            points = points[idx]
        elif len(points) < n:
            extra = self._rng.uniform(low, high, size=(n - len(points), self.dim))
            points = np.vstack([points, extra])
        self.elite_archive = points
    pseudo_fit = float(np.linalg.norm(self.elite_archive - self.mean) if len(self.elite_archive) > 0 else 0.0)
    self.best_fitness_history = [pseudo_fit]
    self.stagnation_counter = 0
    self.diversity_history = [pseudo_fit]
```

**Idea 8: Memetic Local Search Initialization**  
Starts with random points and refines them via a simple hill‑climbing local search.  
```python
def _initialize_optimizer(self):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        rng = 1.0
        low = low - 0.5
        high = high + 0.5
    self.mean = self._rng.uniform(low, high, size=self.dim)
    self.covariance = np.eye(self.dim)
    self.step_size = 0.5 * rng
    n = self.archive_size
    if n <= 0:
        self.elite_archive =