Looking at the worst unsolved tasks (17, 16, 6 with errors 1e+04 to 1e+03), the optimizer is failing to either explore broadly enough or maintain connectivity across the search space. Category E (Topology/graph-based) gives me a fundamentally different angle: I can measure the *algebraic connectivity* of the population via the graph Laplacian's Fiedler value, which is more informative than raw spatial diversity.

**Idea: Fiedler Spectral Connectivity**
Adapt inertia weight using the Fiedler eigenvalue (second-smallest Laplacian eigenvalue) of the k-NN population graph. High Fiedler value → well-connected swarm → increase exploitation (higher inertia). Low Fiedler value → fragmented/poorly-connected → increase exploration (lower inertia).

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using graph Laplacian spectral gap (Fiedler value)."""
    if self.np < 3:
        self.inertia_weight = 0.729
        return
    
    k = min(max(3, self.neighborhood_size), self.np - 1)
    
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    adjacency = np.zeros((self.np, self.np), dtype=np.float64)
    for i in range(self.np):
        adjacency[i, knn_indices[i]] = 1.0
        adjacency[knn_indices[i], i] = 1.0
    
    degree = np.sum(adjacency, axis=1) + 1e-10
    d_sqrt_inv = np.diag(1.0 / np.sqrt(degree))
    laplacian = np.eye(self.np) - d_sqrt_inv @ adjacency @ d_sqrt_inv
    
    try:
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)
        fiedler = eigenvalues[1] if len(eigenvalues) > 1 else eigenvalues[0]
    except np.linalg.LinAlgError:
        fiedler = 0.1
    
    if fiedler < 0.1:
        self.inertia_weight = min(0.95, self.inertia_weight * 1.1)
    elif fiedler > 0.25:
        self.inertia_weight = max(0.4, self.inertia_weight * 0.92)
    else:
        self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
    
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```