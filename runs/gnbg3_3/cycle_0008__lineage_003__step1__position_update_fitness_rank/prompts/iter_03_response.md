**Idea: Entropy-KL Divergence Velocity Modulation**
One-line description: Fit population to Gaussian; use fitness entropy to detect convergence and KL divergence between consecutive generations to measure exploration-exploitation balance, then scale velocity accordingly.
```python
def _position_update_fitness_rank(self):
    """Entropy-KL divergence velocity modulation (Category C: Information-theoretic).

    Key insight: Model population as a probability distribution. Fitness entropy
    detects convergence (low entropy = clustered fitness = premature convergence).
    KL divergence between consecutive populations measures exploration progress.
    Gaussian log-likelihood provides scale-adaptive directional signal.
    This distributional reasoning is orthogonal to graph-topology (E) and
    spectral-matrix (B) approaches used previously.
    """
    import scipy.stats as stats
    
    # === ENTROPY SIGNAL: fitness distribution convergence detection ===
    # Normalize fitness to [0,1] for probability computation
    f_min = np.min(self.current_fitness)
    f_max = np.max(self.current_fitness) + 1e-10
    f_norm = (self.current_fitness - f_min) / (f_max - f_min)
    f_norm = np.clip(f_norm, 1e-10, 1.0 - 1e-10)
    
    # Shannon entropy of fitness bins (convergence detector)
    n_bins = min(10, self.np)
    if self.np >= n_bins:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.histogram(f_norm, bins=bin_edges)[0] + 1
        p = bin_counts / np.sum(bin_counts)
        fitness_entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(n_bins)
        entropy_ratio = np.clip(fitness_entropy / max_entropy, 0.0, 1.0)
    else:
        entropy_ratio = 1.0
    
    # === GAUSSIAN FIT: population distribution characterization ===
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T) + 1e-10 * np.eye(self.dim)
        
        # Log-likelihood under fitted Gaussian (per-particle score)
        sign, logdet = np.linalg.slogdet(cov)
        if sign > 0:
            cov_inv = np.linalg.inv(cov)
            mahal_sq = np.sum(centered @ cov_inv * centered, axis=1)
            log_likelihood = -0.5 * (self.dim * np.log(2 * np.pi) + logdet + mahal_sq)
            
            # Normalize to [0,1] range per generation
            ll_min, ll_max = np.min(log_likelihood), np.max(log_likelihood)
            if ll_max > ll_min:
                ll_norm = (log_likelihood - ll_min) / (ll_max - ll_min + 1e-10)
            else:
                ll_norm = np.ones(self.np) * 0.5
        else:
            ll_norm = np.ones(self.np) * 0.5
    except:
        ll_norm = np.ones(self.np) * 0.5
    
    # === KL DIVERGENCE: population shift between generations ===
    if hasattr(self, '_prev_population') and self._prev_population is not None:
        try:
            prev_centered = self._prev_population - np.mean(self._prev_population, axis=0)
            prev_cov = np.cov(prev_centered.T) + 1e-10 * np.eye(self.dim)
            curr_cov = cov if 'cov' in dir() else np.eye(self.dim) * 1e-2
            
            # Simplified KL(prev || curr) using trace-based approximation
            # KL(N0||N1) ≈ 0.5 * (tr(curr_inv * cov0) + (mu1-mu0)^T curr_inv*(mu1-mu0) - d + log(det(curr)/det(cov0)))
            prev_eigvals = np.linalg.eigvalsh(prev_cov)
            curr_eigvals = np.linalg.eigvalsh(curr_cov)
            prev_eigvals = np.clip(prev_eigvals, 1e-10, None)
            curr_eigvals = np.clip(curr_eigvals, 1e-10, None)
            
            # Geometric mean of eigenvalues as proxy for KL
            kl_trace = 0.5 * np.mean(prev_eigvals / curr_eigvals + curr_eigvals / prev_eigvals - 1)
            kl_signal = np.clip(kl_trace, 0.0, 10.0)
        except:
            kl_signal = 1.0
    else:
        kl_signal = 1.0
    
    # Store current population for next-generation KL computation
    self._prev_population = self.population.copy()
    
    # === MUTUAL INFORMATION: coordinate dependency signal ===
    try:
        # Approximate mutual information between first two principal components
        U = np.linalg.eigvalsh(cov)[::-1]
        U_norm = U / (np.sum(U) + 1e-10)
        cumvar = np.cumsum(U_norm)
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        mi_signal = eff_dim / self.dim  # Low = concentrated, High = spread
    except:
        mi_signal = 0.5
    
    # === VELOCITY SCALING based on information-theoretic signals ===
    # Low entropy = converging = need more exploration
    exploration_boost = 1.0 + 1.5 * (1.0 - entropy_ratio)
    # High KL signal = population shifting = good exploration
    kl_modulation = 1.0 + 0.3 * np.tanh(kl_signal - 1.0)
    # Low MI = concentrated distribution = anisotropic correction needed
    anisotropy_correction = 1.0 / (mi_signal + 0.2)
    
    vel_scale = exploration_boost * kl_modulation * np.clip(anisotropy_correction, 0.5, 2.0)
    vel_scale = np.clip(vel_scale, 0.4, 2.5)
    
    # === DIRECTIONAL PERTURBATION using Gaussian log-likelihood ===
    # Particles with LOW log-likelihood (far from Gaussian center) get stronger pull
    ll_perturb_strength = 0.5 * (1.0 - ll_norm)
    
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        directional = ll_perturb_strength[:, np.newaxis] * to_best_dir
    else:
        directional = ll_perturb_strength[:, np.newaxis] * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # === ENTROPY-DRIVEN RANDOM PERTURBATION ===
    # High entropy = diverse population = less random perturbation
    random_strength = 0.3 * (1.0 - entropy_ratio)
    random_perturb = random_strength * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # === POPULATION UPDATE ===
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
    self.population = self._clip_to_bounds(new_population)
```