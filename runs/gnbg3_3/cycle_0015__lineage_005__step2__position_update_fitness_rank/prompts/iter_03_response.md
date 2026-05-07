Looking at the current `_position_update_fitness_rank`, it's actually Category E (graph-based with Laplacian eigenvalues), but I'm assigned Category C (Information-theoretic / distributional). The worst unsolved tasks (17, 16, 6, 5) have errors 1e+2 to 1e+4, indicating severe multi-modality traps where the population collapses to local optima.

**My angle for Category C**: Fit a Gaussian Mixture Model (GMM) to the population to capture multi-modality directly. Use:
1. **GMM log-likelihood per particle** to identify particles in low-density (exploration-needed) regions
2. **Fitness entropy** to detect convergence (low entropy = converged = need more exploration)
3. **Within-cluster vs between-cluster variance ratio** to detect population fragmentation
4. **Mutual information between position and fitness** to detect deceptive landscapes

This is fundamentally different from graph-based (k-NN/Laplacian) because it models the population as a probability distribution and derives exploration signals from density estimation rather than graph topology.

**Idea: GMM Entropy-Based Velocity Modulation**
```python
def _position_update_fitness_rank(self):
    """GMM entropy-based velocity modulation (Category C: Information-theoretic).

    Key insight: Fit a Gaussian Mixture Model to the population to detect
    multi-modality. Use entropy, GMM log-likelihood, and MI-based signals
    to modulate exploration vs exploitation. This is orthogonal to graph-based
    (Laplacian eigenvalues) and spectral (SVD/PCA) approaches.
    """
    from sklearn.mixture import GaussianMixture
    
    # === GMM FITTING: capture multi-modality ===
    np_pop = self.np
    n_components = min(5, max(2, np_pop // 10))
    
    try:
        gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type='diag',
            max_iter=50,
            n_init=2,
            random_state=42
        )
        gmm.fit(self.population)
        component_probs = gmm.predict_proba(self.population)  # (np, n_components)
        per_particle_ll = gmm.score_samples(self.population)  # log-likelihood per particle
        
        # GMM component weight entropy: high entropy = evenly spread = diverse
        weights = gmm.weights_
        weights = np.clip(weights, 1e-10, 1.0)
        gmm_entropy = -np.sum(weights * np.log(weights))
        max_gmm_entropy = np.log(len(weights))
        norm_gmm_entropy = gmm_entropy / (max_gmm_entropy + 1e-10)
        
        # Per-particle assignment confidence (max component probability)
        max_component_prob = np.max(component_probs, axis=1)  # (np,)
        
        # Within-cluster variance vs between-cluster variance (multi-modality signal)
        centers = gmm.means_  # (n_components, dim)
        center_dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        between_var = np.var(center_dists) if len(center_dists) > 1 else 1.0
        
        # Per-particle within-cluster variance
        assignments = np.argmax(component_probs, axis=1)
        within_var = 0.0
        for i in range(n_components):
            mask = assignments == i
            if np.sum(mask) > 1:
                cluster_pts = self.population[mask]
                cluster_center = gmm.means_[i]
                within_var += np.sum(np.var(cluster_pts - cluster_center, axis=0))
        within_var = within_var / (n_components + 1e-10)
        
        # Fragmentation ratio: high ratio = fragmented population = need cohesion
        fragmentation_ratio = within_var / (between_var + 1e-10)
        fragmentation_signal = np.clip(fragmentation_ratio / 10.0, 0.0, 1.0)
        
    except:
        norm_gmm_entropy = 0.5
        max_component_prob = np.ones(np_pop)
        per_particle_ll = np.zeros(np_pop)
        fragmentation_signal = 0.5
    
    # === FITNESS ENTROPY: convergence detection ===
    fitness = self.current_fitness
    n_bins = min(15, max(3, np_pop // 3))
    hist, _ = np.histogram(fitness, bins=n_bins)
    hist = hist / (np.sum(hist) + 1e-10)
    hist = hist[hist > 0]
    fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
    max_fitness_entropy = np.log(n_bins)
    norm_fitness_entropy = fitness_entropy / (max_fitness_entropy + 1e-10)
    
    # === EXPLORATION SIGNAL: entropy + fragmentation + GMM likelihood ===
    # Low fitness entropy = converged = need exploration
    # High fragmentation = scattered = need cohesion
    # Low per-particle GMM likelihood = in sparse region = need exploration
    exploration_signal = (
        0.35 * (1.0 - norm_fitness_entropy) +  # convergence detection
        0.35 * fragmentation_signal +            # fragmentation detection
        0.30 * (1.0 - np.clip(np.mean(max_component_prob), 0.0, 1.0))  # multi-modality
    )
    exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
    
    # === PER-PARTICLE LIKELIHOOD-BASED SCALING ===
    ll_ranks = np.argsort(np.argsort(-per_particle_ll)) / max(1, np_pop - 1)  # inverse: low ll = high rank
    ll_modulation = 1.0 + 0.4 * (1.0 - ll_ranks)  # low-likelihood particles get boost
    
    # === FITNESS RANK SIGNAL ===
    fitness_ranks = np.argsort(np.argsort(fitness)) / max(1, np_pop - 1)
    
    # === SUCCESS HISTORY ===
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(np_pop)
    improved = self.personal_best_fitness >= fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)
    
    # === VELOCITY SCALING ===
    global_scale = 1.0 + 0.5 * exploration_signal
    global_scale *= (1.0 + 0.3 * success_norm.mean())
    
    vel_scale = global_scale * ll_modulation * (1.0 + 0.3 * (1.0 - fitness_ranks))
    vel_scale = np.clip(vel_scale, 0.4, 2.5)
    
    # === DIRECTIONAL PERTURBATION ===
    directional_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        directional = directional_strength * to_best_dir
    else:
        directional = directional_strength * np.random.uniform(-1, 1, (np_pop, self.dim))
    
    # === RANDOM PERTURBATION ===
    random_perturb = np.random.uniform(-0.5, 0.5, (np_pop, self.dim))
    random_strength = exploration_signal * (1.0 - 0.3 * success_norm[:, np.newaxis])
    random_perturb *= random_strength
    
    # === UPDATE ===
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
    self.population = self._clip_to_bounds(new_population)
```