**Idea: Entropy-Clustered Velocity Modulation**
Use KL-divergence-based Jensen-Shannon divergence between local and global k-NN distance distributions to detect when particles are in low-entropy clustered regions, then amplify exploration by scaling velocity inversely to this divergence.

```python
def _position_update_batch(self):
    """Update positions using entropy-based velocity modulation via KL divergence."""
    k = max(3, min(10, self.np // 3))
    
    # Compute pairwise distances for k-NN entropy estimation
    from scipy.spatial.distance import pdist, squareform
    pairwise_dists = squareform(pdist(self.population + 1e-10))
    
    # Global entropy: average log of k-NN distance (per particle)
    sorted_dists = np.sort(pairwise_dists, axis=1)
    knn_dists_global = sorted_dists[:, k] + 1e-12
    global_entropy = np.mean(np.log(knn_dists_global))
    
    # Local entropy per particle (k-NN within neighborhood)
    local_entropy = np.zeros(self.np)
    for i in range(self.np):
        neighbors = np.argsort(pairwise_dists[i])[1:k+1]
        local_knn = sorted_dists[i, neighbors[k-1]] + 1e-12
        local_entropy[i] = np.log(local_knn)
    
    # KL divergence proxy: JS-style measure of local vs global entropy mismatch
    # High divergence = particle is in atypical local region (needs exploration)
    entropy_ratio = (local_entropy - global_entropy + 1e-8)
    kl_div = np.abs(entropy_ratio)
    
    # Normalize to [0, 1] per generation for robustness
    max_kl = np.max(kl_div) + 1e-10
    kl_normalized = kl_div / max_kl
    
    # Scale velocity inversely to KL divergence
    # Low KL (particle in typical region) -> moderate velocity
    # High KL (particle in outlier region) -> amplified velocity for exploration
    velocity_scale = 1.0 + 2.0 * kl_normalized
    
    # Apply scaled velocity update
    new_population = self.population + self.velocity * velocity_scale[:, np.newaxis]
    self.population = self._clip_to_bounds(new_population)
```