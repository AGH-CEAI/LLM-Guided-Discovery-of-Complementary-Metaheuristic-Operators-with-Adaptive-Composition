**Idea: Entropy-Guided Gaussian Distribution Mutation**

One-line description: Fit a Gaussian to the population, compute differential entropy of fitness distribution to decide exploration/exploitation balance, use Mahalanobis-scaled mutations and Gaussian-weighted bias from top performers.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using information-theoretic / distributional strategy:
        - Fit Gaussian to population, use Mahalanobis distance for shape-aware scaling
        - Compute fitness entropy to decide exploration vs exploitation balance
        - Weight mutation toward high-likelihood regions of top performers
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # 1. Fit Gaussian to population for distributional modeling
        centroid = np.mean(population, axis=0)
        cov_matrix = np.cov(population, rowvar=False)
        cov_matrix = cov_matrix + 1e-6 * np.eye(dim)
        
        # 2. Compute fitness entropy for adaptive exploration/exploitation
        fitness_normalized = fitness - np.min(fitness) + 1e-10
        p = fitness_normalized / np.sum(fitness_normalized)
        p = np.clip(p, 1e-10, 1.0)
        fitness_entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(np_pop)
        entropy_ratio = np.clip(fitness_entropy / (max_entropy + 1e-10), 0.0, 1.0)
        
        # Low entropy = converged population = increase exploration weight
        exploration_weight = 0.3 + 0.4 * (1.0 - entropy_ratio)
        
        # 3. Compute Mahalanobis distance for each individual (covariance-aware scaling)
        diff = population - centroid
        cov_inv = np.linalg.pinv(cov_matrix)
        mahal_dist = np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))
        mahal_scale = np.clip(mahal_dist / (np.mean(mahal_dist) + 1e-10), 0.3, 2.5)
        
        # 4. Fit Gaussian to top performers and compute log-likelihood weights
        n_top = max(3, np_pop // 4)
        top_indices = np.argpartition(fitness, n_top)[:n_top]
        top_pop = population[top_indices]
        top_centroid = np.mean(top_pop, axis=0)
        top_cov = np.cov(top_pop, rowvar=False) + 1e-4 * np.eye(dim)
        
        # Compute log-likelihood of each individual under top-performer Gaussian
        diff_top = population - top_centroid
        try:
            cho_info = np.linalg.cho_solve(np.linalg.cholesky(top_cov), np.eye(dim))
            mahal_top = np.sum(diff_top @ cho_info * diff_top, axis=1)
            log_likelihood = -0.5 * mahal_top - 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(top_cov) + 1e-10)
        except:
            log_likelihood = np.zeros(np_pop)
        
        # Convert to weights via softmax
        ll_weights = np.exp(log_likelihood - np.max(log_likelihood))
        ll_weights = ll_weights / (np.sum(ll_weights) + 1e-10)
        
        # Weighted centroid from top performers
        weighted_centroid = np.sum(population * ll_weights[:, np.newaxis], axis=0)
        
        # 5. Ring-based DE mutation component
        r1 = ring_prev
        r2 = ring_next
        current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
        base_mutation = population[r1] + current_F * (population[r2] - population)
        
        # 6. Covariance-aware differential: scale by Mahalanobis distance
        cov_aware_diff = base_mutation * mahal_scale[:, np.newaxis]
        
        # 7. Composite mutation with entropy-guided blending
        # High entropy (diverse fitness) → favor exploitation (centroid bias)
        # Low entropy (converged) → favor exploration (covariance-scaled DE)
        centroid_bias = centroid + 0.5 * (weighted_centroid - centroid)
        
        trials = (1.0 - exploration_weight) * cov_aware_diff + exploration_weight * centroid_bias
        
        # 8. Small perturbation based on distribution spread (information-guided noise)
        eigenvals = np.linalg.eigvalsh(cov_matrix)
        max_eigenval = np.max(eigenvals) + 1e-10
        info_noise = np.random.randn(np_pop, dim) * np.sqrt(max_eigenval) * 0.1
        trials = trials + info_noise
        
        return trials, current_F
```