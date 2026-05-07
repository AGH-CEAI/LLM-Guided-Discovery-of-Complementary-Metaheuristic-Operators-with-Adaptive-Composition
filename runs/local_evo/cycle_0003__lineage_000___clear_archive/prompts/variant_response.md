**Idea 1: Exponential Crossover Reset**
Use exponential crossover to mix archive members before clearing.
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

**Idea 2: Arithmetic Blend Reset**
Apply BLX-α arithmetic blend between archive centroid and mean before clearing.
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

**Idea 3: Segment-Based Crossover Reset**
Swap a random segment between two archive individuals before clearing.
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

**Idea 4: Eigenvector-Rotated Reset**
Rotate archive using eigenvectors of covariance matrix before clearing.
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

**Idea 5: Deterministic Crowding Reset**
Replace worst archive member with best before clearing.
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

**Idea 6: Clustering-Based Compression Reset**
Perform simple k-means compression on archive before clearing.
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

**Idea 7: Random Walk Reinitialization Reset**
Generate a random walk from current mean before clearing.
```python
def _clear_archive(self):
    # Random walk reinitialization: generate a few steps from the current mean before clearing
    if self.elite_archive.shape[0] > 0:
        low, high = self.bounds
        step