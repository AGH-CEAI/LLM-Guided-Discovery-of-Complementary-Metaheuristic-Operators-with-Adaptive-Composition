**Idea: Gaussian Likelihood Entropy Modulation**

Information-theoretic approach: fit a Gaussian to the best-performing particles, compute per-particle log-likelihood under this distribution, measure entropy of the likelihood distribution, and use this KL-divergence-inspired signal to modulate exploration strength. Particles in low-density regions relative to good solutions get stronger nudges toward promising areas.

```python
def _position_update_centroid_knn(self):
        """Gaussian likelihood entropy modulation (Category C: Information-theoretic).
        
        Treats population as a probability distribution. Fits a Gaussian to top
        performers, computes per-particle log-likelihood, measures entropy of the
        likelihood distribution as exploration signal. Low-likelihood particles
        in unexplored regions get stronger centroid pulls toward promising areas.
        """
        from scipy.stats import multivariate_normal
        
        # Fit Gaussian to top performers (distribution of good solutions)
        n_select = max(5, self.np // 4)
        top_indices = np.argsort(self.current_fitness)[:n_select]
        
        if n_select < 2:
            new_population = self.population + self.inertia_weight * self.velocity
            self.population = self._clip_to_bounds(new_population)
            return
        
        top_pop = self.population[top_indices]
        best_mean = np.mean(top_pop, axis=0)
        centered = top_pop - best_mean
        best_cov = np.cov(centered.T)
        if best_cov.ndim == 0:
            best_cov = best_cov.reshape(1, 1)
        best_cov = best_cov + 1e-6 * np.eye(self.dim)
        
        # Compute per-particle log-likelihood under fitted Gaussian
        log_likelihood = np.zeros(self.np)
        for i in range(self.np):
            diff = self.population[i] - best_mean
            try:
                cov_inv = np.linalg.inv(best_cov)
                log_det = np.log(np.linalg.det(best_cov) + 1e-10)
                mahal = diff @ cov_inv @ diff
                log_likelihood[i] = -0.5 * (mahal + log_det + self.dim * np.log(2 * np.pi))
            except:
                log_likelihood[i] = -self.dim * 10.0
        
        # Compute entropy of likelihood distribution (information-theoretic signal)
        ll_min, ll_max = np.min(log_likelihood), np.max(log_likelihood)
        if ll_max > ll_min + 1e-10:
            norm_ll = (log_likelihood - ll_min) / (ll_max - ll_min + 1e-10)
        else:
            norm_ll = np.ones(self.np) * 0.5
        
        # Discretize for entropy estimation
        n_bins = min(10, max(3, self.np // 3))
        counts, _ = np.histogram(norm_ll, bins=n_bins, range=(0.0, 1.0))
        probs = counts / (self.np + 1e-10)
        probs = probs[probs > 0]
        if len(probs) > 1:
            likelihood_entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(probs))
            normalized_entropy = likelihood_entropy / (max_entropy + 1e-10)
        else:
            normalized_entropy = 0.5
        
        # Exploration signal from log-likelihood: low likelihood = unexplored = explore more
        likelihood = np.exp(log_likelihood - ll_max)
        exploration_factor = 1.0 / (likelihood + 1e-10)
        exploration_factor = np.clip(exploration_factor, 0.3, 5.0)
        
        # Modulate by entropy: high entropy = diverse distribution = more exploration globally
        global_explore = 0.5 + 0.5 * normalized_entropy
        exploration_factor *= global_explore
        
        # Direction toward best region (centroid of good solutions)
        to_best = best_mean - self.population
        to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_dist
        
        # Entropy-weighted correction toward promising distribution
        correction = 0.3 * exploration_factor[:, np.newaxis] * to_best_dir
        
        new_population = self.population + self.inertia_weight * self.velocity + correction
        self.population = self._clip_to_bounds(new_population)
```