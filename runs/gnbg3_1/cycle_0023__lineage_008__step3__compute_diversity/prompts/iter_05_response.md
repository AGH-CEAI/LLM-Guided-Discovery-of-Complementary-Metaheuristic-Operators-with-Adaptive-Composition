**Idea: k-NN Distance Distribution Entropy**
Computes entropy of k-nearest-neighbor distance distribution across population members, capturing geometric structure that mean-std misses. This detects population collapse and deceptive clustering more reliably for multi-modal tasks 16-23.

```python
def _compute_diversity(self):
        """Compute diversity via entropy of k-nearest-neighbor distance distribution."""
        n = len(self.population)
        if n < 10:
            return 1e-10
        
        k = min(5, n - 1)
        bounds_range = self.ub - self.lb
        bounds_range = np.maximum(bounds_range, 1e-10)
        
        # Compute k-NN distances for each point
        knn_distances = []
        for i in range(n):
            dists = []
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum(((self.population[i] - self.population[j]) / bounds_range) ** 2))
                    dists.append(d)
            dists = np.sort(dists)[:k]
            knn_distances.extend(dists)
        
        knn_distances = np.array(knn_distances)
        
        # Compute expected distance for uniform distribution
        dim = self.dim
        expected_dist = np.sqrt(dim) * 0.5
        
        # Normalize by expected
        normalized = knn_distances / (expected_dist + 1e-10)
        normalized = np.clip(normalized, 1e-10, 10.0)
        
        # Compute entropy of normalized distances
        hist, _ = np.histogram(normalized, bins=20, density=True)
        hist = np.maximum(hist, 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Scale: higher entropy = more diverse, max ~ln(20) for uniform
        max_entropy = np.log(20.0)
        diversity = entropy / (max_entropy + 1e-10)
        
        return float(np.clip(diversity, 1e-10, 1.0))
```