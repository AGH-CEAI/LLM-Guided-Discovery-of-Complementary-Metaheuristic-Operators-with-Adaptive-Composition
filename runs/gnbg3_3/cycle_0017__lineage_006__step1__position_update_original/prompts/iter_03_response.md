Looking at the worst unsolved tasks (17, 5, 16, 6, 11, 19, 23, 9), all with errors 10+ orders of magnitude above target, the swarm is clearly getting trapped in local optima. The current spectral approach (eigenvalue-based) doesn't capture the full distributional state of the population.

**Idea: Entropy-Based Distribution Modulation**
Use information-theoretic measures on the population as a probability distribution: fitness entropy, spatial entropy via kernel density estimation, KL divergence from uniform reference, and mutual information between positions and fitness. This gives fundamentally different signals than spectral or fitness-rank approaches.

```python
def _position_update_original(self):
    """Information-theoretic: entropy, KL divergence, and mutual information modulation.
    
    Category C — Treats the population as a probability distribution to characterize.
    Uses fitness entropy, spatial entropy, KL divergence from uniform, and
    mutual information between position and fitness to modulate velocity.
    """
    # === Signal 1: Fitness Distribution Entropy ===
    fitness = self.current_fitness
    n_bins = min(15, max(5, len(fitness) // 3))
    hist, _ = np.histogram(fitness, bins=n_bins)
    hist = hist / (len(fitness) + 1e-10)
    hist = hist[hist > 0]
    fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
    max_fitness_entropy = np.log(n_bins)
    norm_fitness_ent = fitness_entropy / (max_fitness_entropy + 1e-10)
    
    # === Signal 2: Spatial Distribution Entropy (per-dimension) ===
    pop = self.population
    n_bins_spatial = 10
    spatial_entropies = []
    for d in range(self.dim):
        col = pop[:, d]
        hist_d, _ = np.histogram(col, bins=n_bins_spatial, range=(self.lower_bound, self.upper_bound))
        hist_d = hist_d / (len(col) + 1e-10)
        hist_d = hist_d[hist_d > 0]
        if len(hist_d) > 1:
            spatial_entropies.append(-np.sum(hist_d * np.log(hist_d + 1e-10)))
        else:
            spatial_entropies.append(0.0)
    spatial_entropy = np.mean(spatial_entropies)
    max_spatial_entropy = np.log(n_bins_spatial)
    norm_spatial_ent = spatial_entropy / (max_spatial_entropy + 1e-10)
    
    # === Signal 3: KL Divergence from Uniform Reference ===
    uniform_ref = 1.0 / n_bins_spatial
    kl_divs = []
    for d in range(self.dim):
        col = pop[:, d]
        hist_d, _ = np.histogram(col, bins=n_bins_spatial, range=(self.lower_bound, self.upper_bound))
        p_d = hist_d / (len(col) + 1e-10)
        p_d = np.clip(p_d, 1e-10, 1.0)
        kl_d = np.sum(p_d * np.log(p_d / uniform_ref))
        kl_divs.append(kl_d)
    total_kl = np.sum(kl_divs)
    max_kl = self.dim * np.log(n_bins_spatial)
    norm_kl = np.clip(total_kl / (max_kl + 1e-10), 0.0, 1.0)
    
    # === Signal 4: Mutual Information between positions and fitness ===
    try:
        n_bins_mi = 5
        fitness_hist, f_edges = np.histogram(fitness, bins=n_bins_mi)
        f_probs = fitness_hist / (len(fitness) + 1e-10)
        f_probs = np.clip(f_probs, 1e-10, 1.0)
        h_fitness = -np.sum(f_probs * np.log(f_probs))
        
        h_joint = 0.0
        for d in range(min(self.dim, 3)):
            col = pop[:, d]
            x_hist, x_edges = np.histogram(col, bins=n_bins_mi, range=(self.lower_bound, self.upper_bound))
            x_probs = x_hist / (len(col) + 1e-10)
            x_probs = np.clip(x_probs, 1e-10, 1.0)
            h_joint += -np.sum(x_probs * np.log(x_probs))
        
        h_joint /= min(self.dim, 3)
        mi_estimate = max(0.0, h_fitness + norm_spatial_ent * np.log(n_bins_mi) - h_joint - np.log(n_bins_mi))
        norm_mi = np.clip(mi_estimate / (np.log(n_bins_mi) + 1e-10), 0.0, 1.0)
    except:
        norm_mi = 0.5
    
    # === Signal 5: Entropy-based velocity scaling ===
    exploration_signal = 0.3 * norm_fitness_ent + 0.3 * norm_spatial_ent + 0.2 * (1.0 - norm_kl) + 0.2 * norm_mi
    exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
    
    # === Global velocity scale ===
    global_scale = 0.7 + 0.6 * (1.0 - exploration_signal)
    global_scale = np.clip(global_scale, 0.4, 1.6)
    
    # === Convergence detection ===
    is_converging = (norm_spatial_ent < 0.2 and norm_kl > 0.6)
    is_exploring = (norm_fitness_ent > 0.6 and norm_spatial_ent > 0.5)
    
    if is_converging:
        global_scale *= 1.3
    elif is_exploring:
        global_scale *= 0.85
    
    # === Per-particle modulation based on local density entropy ===
    try:
        k_local = min(7, self.np - 1)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k_local]
        local_entropies = []
        for i in range(self.np):
            d_i = knn_dists[i]
            d_min, d_max = np.min(d_i), np.max(d_i)
            if d_max > d_min + 1e-10:
                bins_local = 5
                counts, _ = np.histogram(d_i, bins=bins_local, range=(d_min, d_max + 1e-10))
                probs = counts / (k_local + 1e-10)
                probs = probs[probs > 0]
                local_ent = -np.sum(probs * np.log(probs + 1e-10))
            else:
                local_ent = 0.0
            local_entropies.append(local_ent)
        local_entropies = np.array(local_entropies)
        max_local_ent = np.log(5)
        norm_local_ent = local_entropies / (max_local_ent + 1e-10)
        
        particle_scale = 1.0 + 0.4 * (1.0 - norm_local_ent)
        particle_scale = np.clip(particle_scale, 0.5, 2.0)
    except:
        particle_scale = np.ones(self.np)
    
    vel_scale = particle_scale * global_scale
    vel_scale = np.clip(vel_scale, 0.3, 2.0)
    
    # === Entropy-based perturbation for trapped particles ===
    if is_converging:
        try:
            centered = pop - np.mean(pop, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            principal_dir = eigvals[0] / (np.sum(eigvals) + 1e-10)
        except:
            principal_dir = 0.5
        
        entropy_perturb = (1.0 - norm_spatial_ent) * principal_dir
        random_perturb = entropy_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
    else:
        random_perturb = np.zeros((self.np, self.dim))
    
    new_population = pop + self.velocity * vel_scale[:, np.newaxis] + random_perturb
    self.population = self._clip_to_bounds(new_population)
```