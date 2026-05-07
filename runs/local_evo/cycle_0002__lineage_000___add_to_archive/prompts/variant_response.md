**Idea 1: Exponential Crossover**
Uses exponential crossover operator (common in Differential Evolution) where each dimension is exchanged with probability decreasing exponentially from the first dimension.

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

---

**Idea 2: Arithmetic Blend Crossover (BLX-α)**
Blend crossover creates offspring as convex combinations with extrapolation (α=0.5), allowing exploration beyond parent boundaries.

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

---

**Idea 3: Segment-Based Crossover**
Divides the genome into contiguous blocks and randomly assigns each block to either the new elite or a randomly chosen archive member, preserving linkage between adjacent genes.

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

---

**Idea 4: Eigenvector-Rotated Crossover**
Projects into the eigenspace of the covariance matrix, performs crossover in rotated coordinates (exploiting the learned correlation structure), then projects back.

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

---

**Idea 5: Fitness-Weighted Probabilistic Merge**
Uses inverse-fitness weighting to probabilistically merge new elites with archive members, giving preference to better solutions during combination.

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

---

**Idea 6: Crowding Distance Diversity Maintenance**
Inspired by NSGA-II, computes crowding distance to prioritize adding elites that occupy sparse regions of the search space, maintaining diversity.

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

---

**Idea 7: Adaptive Archive with Exponential Age Decay**
Maintains an age counter for each archive member; new elites are blended with older members using age-dependent weights, promoting turnover and preventing stagnation.

```python
def _add_to_archive(self, new_elites):
    """Add new elite individuals using adaptive age-decay blending."""
    if not hasattr(self, '_archive_age'):
        self._archive_age = np.array([], dtype=float)
    if new_elites is None or len(new_elites) == 0:
        return
    if self.elite_archive is None or len(self.elite_archive) == 0:
        self.elite_archive = np.atleast_2d(new_elites)
        self._archive_age = np.zeros(len(new_elites))
        return
    low, high = self.bounds
    self._archive_age += 1.0
    decay = 0.9
    new_ages = np.zeros(len(new_elites))
    n_new = len(new_elites)
    indices = self._rng.choice(len(self.elite_archive), size=n_new, replace=True)
    parents = self.elite_archive[indices]
    parent_ages = self._archive_age[indices]
    combined_ages = parent_ages + 1.0
    age_weights = np.exp(-decay * (combined_ages - 1) / (np.max(combined_ages) + 1e-12))
    weights = (1 - age_weights).reshape(-1, 1)
    offspring = weights * new_elites + (1 - weights) * parents
    self.elite_archive = np.vstack([self.elite_archive, np.clip(offspring, low, high)])
    self._archive_age = np.concatenate([self._archive_age, new_ages])
```

---

**Idea 8: Space-Partitioning KD-Tree Insertion**
Uses a KD-tree to spatially index existing archive members and only inserts new elites if they are sufficiently distant from all current members, maintaining uniform coverage.

```python
def _add_to_archive(self, new_elites):
    """Add new elite individuals using KD-tree space-partitioning."""
    if new_elites is None or len(new_elites) == 0:
        return
    if self.elite_archive is None or len(self.elite_archive) == 0:
        self.elite_archive = np.atleast_2d(new_elites)
        return
    low, high = self.bounds
    rng = high - low + 1e-12
    min_distance = rng / (2 * (self.archive_size ** (1.0 / self.dim)) + 1e-12)
    to_add = []
    if len(self.elite_archive) >= 2 * self.dim:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.elite_archive)
            for elite in new_elites:
                dist, _ = tree.query(elite, k=1)
                if dist > min_distance:
                    to_add.append(elite)
        except ImportError:
            for elite in new_elites:
                dists = np.sqrt(np.sum((self.elite_archive - elite) ** 2, axis=1))
                if np.min(dists) > min_distance:
                    to_add.append(elite)
    else:
        for elite in new_elites:
            dists = np.sqrt(np.sum((self.elite_archive - elite) ** 2, axis=1))
            if len(dists) == 0 or np.min(dists) > min_distance:
                to_add.append(elite)
    if not to_add:
        to_add = new_elites[:max(1, len(new_elites) // 2)].tolist()
    self.elite_archive = np.vstack([self.elite_archive, np.clip(np.array(to_add), low, high)])
```