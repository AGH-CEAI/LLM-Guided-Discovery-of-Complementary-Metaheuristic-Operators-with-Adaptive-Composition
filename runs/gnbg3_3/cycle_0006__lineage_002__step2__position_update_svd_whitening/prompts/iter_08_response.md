**Idea: SVD-Whitening + Graph-Clustering Hybrid**
Combine SVD spectral conditioning (mechanism 1) with graph-based clustering detection (mechanism 2). Weighting is data-driven: stagnation counter drives exploration vs exploitation, while k-NN component count modulates spectral strength based on population fragmentation. This targets the worst tasks (17, 16, 6) which likely suffer from both anisotropy and deceptive clustering.

```python
def _position_update_svd_whitening(self):
    """Hybrid: SVD spectral conditioning + graph-based clustering detection.
    
    Combines two distinct mechanisms:
      1) SVD whitening: counteracts population anisotropy via per-component scaling
      2) Graph clustering: k-NN connected components detect fragmentation/stagnation
    
    Weighting is principled:
      - Stagnation counter (data-driven) → switches between exploration and exploitation
      - k-NN component count (data-driven) → modulates spectral strength based on fragmentation
    
    Targets worst tasks (17, 16, 6, 11) which likely have both anisotropy and
    deceptive local optima that existing single-mechanism operators miss.
    """
    centered = self.population - np.mean(self.population, axis=0)

    try:
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        singular_values = np.clip(singular_values, 1e-10, None)

        # --- MECHANISM 1: SVD spectral conditioning ---
        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        sv_norm = singular_values / (singular_values[0] + 1e-10)
        
        # Inverse-square-law spectral modulation
        spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
        spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

        # --- MECHANISM 2: Graph-based clustering detection ---
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Connected component analysis via BFS
        visited = np.zeros(self.np, dtype=bool)
        num_components = 0
        for start in range(self.np):
            if visited[start]:
                continue
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                for neighbor in knn_indices[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            num_components += 1

        # Fragmentation signal: more components = more fragmented population
        fragmentation = num_components / max(1, self.np)
        fragmentation = np.clip(fragmentation, 0.0, 1.0)

        # --- PRINCIPLED WEIGHTING: stagnation-driven switching ---
        # Stagnation counter (data-driven) detects convergence stalls
        stagnation_signal = np.clip(np.log1p(self.stagnation_counter) / np.log1p(100), 0.0, 1.0)
        
        # When stagnant: boost exploration (higher spectral modulation, more uniform scaling)
        # When not stagnant: rely more on SVD's natural spectral scaling
        exploration_weight = stagnation_signal * 0.5 + 0.25
        
        # Fragmentation amplifies spectral modulation when population is fragmented
        cluster_scale = 1.0 + fragmentation * stagnation_signal

        # --- COMBINED SCALING ---
        uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
        per_component_scale = (
            uniform_scale * (1.0 - exploration_weight) + 
            spectral_modulation * exploration_weight * cluster_scale
        )

        # Stagnation-driven velocity magnitude boost for escaping local optima
        vel_magnitude_scale = 1.0 + stagnation_signal * 0.5
        vel_magnitude_scale = np.clip(vel_magnitude_scale, 1.0, 1.5)

        vel_proj = self.velocity @ Vt.T
        vel_scaled = vel_proj * per_component_scale * vel_magnitude_scale
        new_population = self.population + (vel_scaled @ Vt)

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```