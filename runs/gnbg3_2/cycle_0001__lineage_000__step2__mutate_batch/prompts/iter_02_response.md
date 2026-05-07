**Idea: SVD Condition-Number-Weighted Mutation**

Leverage the population covariance's singular value decomposition to detect anisotropy and condition number, then weight mutation components by inverse singular values to escape collapsed subspaces while dampening exploration in already-explored high-variance directions.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using SVD-based spectral mutation.
        Category B: Uses singular values and condition number of population
        covariance to weight mutation components, targeting anisotropic
        and ill-conditioned search landscapes.
        """
        np_pop, dim = population
        
        # Compute SVD of population matrix to analyze spectral structure
        try:
            U, s, Vt = np.linalg.svd(population, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback on SVD failure
            return self._mutate_batch_fallback(population, fitness, ring_prev, ring_next)
        
        # Detect near-zero singular values (numerical rank deficiency)
        tol = dim * np.spacing(np.max(s))
        rank = np.sum(s > tol)
        
        if rank < min(np_pop, dim) // 2:
            # Population collapsed to low-rank subspace - inject diversity
            # Generate perturbation orthogonal to current subspace
            V_ortho = Vt[rank:].T if rank < dim else np.eye(dim)
            if V_ortho.shape[1] > 0:
                random_dir = V_ortho[:, np.random.randint(V_ortho.shape[1])]
                subspace_injection = 0.5 * random_dir * np.random.uniform(0.8, 1.2)
            else:
                subspace_injection = np.zeros(dim)
        else:
            subspace_injection = np.zeros(dim)
        
        # Condition number as anisotropy measure
        s_nonzero = s[s > tol]
        if len(s_nonzero) >= 2:
            cond_num = s_nonzero[0] / s_nonzero[-1]
        else:
            cond_num = 1.0
        
        # Adaptive scaling: high cond_num -> more aggressive mutation to escape
        spectral_scale = np.clip(1.0 + 0.1 * np.log(1 + cond_num), 0.8, 2.5)
        
        # Project population onto principal components for mutation
        pop_centered = population - np.mean(population, axis=0)
        pop_pca = pop_centered @ Vt.T  # (np_pop, rank)
        
        # Compute inverse-variance weights for mutation scaling
        s_weighted = s_nonzero[:len(pop_pca[0])]
        var_weights = s_weighted / (np.sum(s_weighted) + 1e-20)
        inv_var_weights = 1.0 / (var_weights + 0.01)
        inv_var_weights = inv_var_weights / np.sum(inv_var_weights)
        
        # Compute fitness-weighted centroid
        centroid = self._compute_population_centroid(population, fitness)
        
        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next
        
        # Adaptive F
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Compute differential in PCA space
        rand1_pca = pop_pca[r1]
        rand2_pca = pop_pca[r2]
        pop_pca_centered = pop_pca - np.mean(pop_pca, axis=0)
        
        # Differential mutation in PCA space, weighted by inverse variance
        diff_pca = current_F * (rand2_pca - pop_pca_centered)
        diff_weighted = diff_pca * inv_var_weights[np.newaxis, :len(diff_pca[0])]
        
        # Project back to original space
        diff_original = diff_weighted @ Vt[:len(diff_weighted[0])]
        
        # Blend with centroid component
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        base_mutation = rand1_pca + diff_weighted
        base_mutation_orig = base_mutation @ Vt[:len(base_mutation[0])]
        
        trials = ((1 - blend_weight) * (population[r1] + spectral_scale * diff_original) +
                  blend_weight * (centroid + subspace_injection))
        
        trials = np.clip(trials, self.lower, self.upper)
        
        return trials, current_F
```