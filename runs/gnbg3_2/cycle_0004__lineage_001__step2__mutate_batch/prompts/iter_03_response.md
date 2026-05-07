**Idea: GMM Entropy-Directed Mutation**

One-line description: Fit a Gaussian Mixture Model to the population, compute distribution entropy, and sample mutation vectors from high-fitness density regions weighted by KL divergence from the full population.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using information-theoretic GMM approach:
        - Fit GMM to population to characterize distribution
        - Compute entropy as diversity metric
        - Sample mutations from high-fitness density regions
        - Use KL divergence weighting between elite and full population
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Compute fitness-based weights
        fitness_min = np.min(fitness)
        fitness_range = np.max(fitness) - fitness_min + 1e-10
        fit_weights = 1.0 - (fitness - fitness_min) / fitness_range
        fit_weights = np.clip(fit_weights, 0.01, 1.0)
        
        # Select elite individuals (top 30% by fitness)
        n_elite = max(3, int(np_pop * 0.3))
        elite_order = np.argsort(fitness)[:n_elite]
        elite_pop = population[elite_order]
        elite_weights = fit_weights[elite_order]
        elite_weights = elite_weights / (np.sum(elite_weights) + 1e-10)
        
        # Fit GMM to elite population for high-fitness region characterization
        n_components = min(3, n_elite)
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=n_components, covariance_reg=1e-6,
                                  random_state=42, max_iter=50)
            gmm.fit(elite_pop)
            
            # Compute entropy of GMM (measure of exploration potential)
            # H = -sum_k w_k * log(det(S_k) + eps)
            covariances = gmm.covariances_ + np.eye(dim) * 1e-8
            dets = np.clip(np.linalg.det(covariances), 1e-20, None)
            entropies = -gmm.weights_ * np.log(dets + 1e-10)
            gmm_entropy = np.sum(entropies)
            
            # Scale exploration rate by entropy (high entropy = more exploration)
            exploration_rate = np.clip(gmm_entropy / (dim * 3.0), 0.1, 0.7)
        except Exception:
            gmm = None
            exploration_rate = 0.3
        
        # Compute KL divergence weighting for mutation guidance
        # Weight each elite by its relative contribution to exploration
        if gmm is not None:
            # Sample from GMM to estimate density under elite distribution
            gmm_samples = gmm.sample(np_pop * 2)[0]
            gmm_samples = np.clip(gmm_samples, self.lower, self.upper)
            
            # Compute log-likelihood under GMM for each population member
            try:
                log_lik_full = gmm.score_samples(population)
                log_lik_elite = gmm.score_samples(elite_pop)
                
                # KL-like weighting: favor regions where elite > full population density
                full_density = np.exp(log_lik_full - np.max(log_lik_full))
                elite_density = np.exp(log_lik_elite - np.max(log_lik_elite))
                
                # Regions where elite has higher relative density
                kl_weights = np.zeros(np_pop)
                for i in range(np_pop):
                    # How much does this region favor elite over full?
                    region_density_ratio = np.mean(elite_density) / (full_density[i] + 1e-10)
                    kl_weights[i] = np.clip(region_density_ratio, 0.1, 5.0)
                kl_weights = kl_weights / (np.sum(kl_weights) + 1e-10)
            except Exception:
                kl_weights = np.ones(np_pop) / np_pop
        else:
            kl_weights = np.ones(np_pop) / np_pop
        
        # Compute weighted centroid of elite (target for exploitation)
        elite_centroid = np.sum(elite_pop * elite_weights[:, np.newaxis], axis=0)
        
        # Compute weighted centroid using KL weights (high-fitness regions)
        kl_centroid = np.sum(population * kl_weights[:, np.newaxis], axis=0)
        
        # Adaptive F based on population spread
        pop_spread = np.std(population, axis=0)
        current_F = self.F * (1.0 + 0.05 * np.random.randn())
        current_F = np.clip(current_F * (1.0 + 0.1 * np.mean(pop_spread)), 0.1, 2.0)
        
        # Ring-based parents for rand/1 component
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Base DE mutation
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # GMM-guided component: sample from fitted distribution
        if gmm is not None:
            gmm_mutations = gmm.sample(np_pop)[0]
            gmm_mutations = np.clip(gmm_mutations, self.lower, self.upper)
        else:
            gmm_mutations = elite_centroid + 0.5 * np.random.randn(np_pop, dim)
        
        # Blend components based on exploration rate
        # High entropy -> more GMM exploration; low entropy -> more exploitation
        blend_gmm = exploration_rate * 0.4
        blend_centroid = 0.3
        blend_base = 1.0 - blend_gmm - blend_centroid
        
        trials = (blend_base * base_mutation + 
                  blend_centroid * kl_centroid + 
                  blend_gmm * gmm_mutations)
        
        # Add small perturbation proportional to fitness improvement potential
        perturbation_scale = 0.1 * (1.0 - exploration_rate)
        trials = trials + perturbation_scale * np.random.randn(np_pop, dim)
        
        return trials, current_F
```