**Idea: Entropy-Driven Distribution Adaptation with Multi-Modal Detection**

Category C approach: Treat the population as a probability distribution and use entropy, KL divergence, and GMM-based multi-modality detection to drive exploration. The key insight is that worst tasks (17, 16, 6) likely have deceptive multi-modal landscapes where the population collapses to a local optimum — detecting this via entropy collapse and GMM cluster count gives a fundamentally different signal than temporal or spectral approaches.

```python
def _position_update_temporal_drift(self):
    """Information-theoretic: Entropy + KL divergence + GMM multi-modality detection (Category C).

    Key mechanisms:
    1. Fitness entropy: detects convergence from fitness distribution shape
    2. Spectral KL divergence: measures how far population is from uniform spread
    3. GMM multi-modality: detects if population is split across multiple optima
    4. Temporal entropy trend: detects stagnation via entropy rate of change

    Targets worst tasks (17, 16, 6, 11) where premature convergence to wrong
    basins causes massive errors. Information-theoretic signals detect this
    collapse before it becomes irreversible.
    """
    try:
        # === MECHANISM 1: FITNESS ENTROPY (population as probability distribution) ===
        fitness = self.current_fitness
        n_bins = min(15, max(3, self.np // 3))
        hist, _ = np.histogram(fitness, bins=n_bins)
        hist = hist / (len(fitness) + 1e-10)
        hist = hist[hist > 0] + 1e-10
        fitness_entropy = -np.sum(hist * np.log(hist))

        if not hasattr(self, '_fitness_entropy_history'):
            self._fitness_entropy_history = []
        self._fitness_entropy_history.append(fitness_entropy)
        if len(self._fitness_entropy_history) > 10:
            self._fitness_entropy_history.pop(0)

        # Entropy trend: negative = converging/stagnating
        if len(self._fitness_entropy_history) >= 3:
            recent = np.mean(self._fitness_entropy_history[-3:])
            older = np.mean(self._fitness_entropy_history[:-3]) if len(self._fitness_entropy_history) > 3 else self._fitness_entropy_history[0]
            entropy_trend = (recent - older) / (older + 1e-10)
        else:
            entropy_trend = 0.0

        # === MECHANISM 2: KL DIVERGENCE FROM UNIFORM (spatial distribution quality) ===
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)

        # KL(P_eigen || U) - how far from isotropic
        dim_eff = len(eigenvalues_norm)
        uniform = np.ones(dim_eff) / dim_eff
        kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
        max_kl = np.log(dim_eff)
        kl_normalized = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)

        # High KL = anisotropic = exploration needed
        exploration_from_kl = 0.5 + 0.5 * kl_normalized

        # === MECHANISM 3: GMM MULTI-MODALITY DETECTION ===
        # Fit GMM with up to 3 components; BIC selects optimal
        from sklearn.mixture import GaussianMixture
        max_components = min(4, max(2, self.np // 10))

        best_bic = np.inf
        best_n_components = 1
        for n_comp in range(1, max_components + 1):
            try:
                gmm = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                      n_init=3, random_state=42)
                gmm.fit(centered)
                bic = gmm.bic(centered)
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n_comp
            except:
                continue

        # Multi-modality signal: more components = more trapped in separate basins
        multimodality_signal = float(best_n_components) / float(max_components)

        # === MECHANISM 4: TEMPORAL ENTROPY RATE ===
        if not hasattr(self, '_ema_entropy'):
            self._ema_entropy = fitness_entropy
        self._ema_entropy = 0.2 * fitness_entropy + 0.8 * self._ema_entropy

        # Normalized entropy
        max_entropy = np.log(n_bins)
        norm_entropy = fitness_entropy / (max_entropy + 1e-10)
        entropy_collapse = 1.0 - np.clip(norm_entropy, 0.0, 1.0)

        # === COMBINED INFORMATION-THEORETIC SIGNAL ===
        # Entropy collapse + negative trend + high multimodality = trapped
        trapped_signal = (entropy_collapse * 0.4 +
                         max(0, -entropy_trend) * 0.3 +
                         multimodality_signal * 0.3)
        trapped_signal = np.clip(trapped_signal, 0.0, 1.0)

        # Exploration factor: high when trapped or anisotropic
        exploration_factor = (1.0 + trapped_signal * 1.5) * exploration_from_kl
        exploration_factor = np.clip(exploration_factor, 0.5, 2.5)

        # === VELOCITY SCALING FROM DISTRIBUTIONAL SIGNALS ===
        U, V = np.linalg.eigh(cov)
        idx = np.argsort(U)[::-1]
        U = U[idx]
        V = V[:, idx]

        vel_proj = self.velocity @ V
        sv_scale = np.sqrt(U + 1e-10)
        sv_norm = sv_scale / (sv_scale[0] + 1e-10)

        # Inverse scaling: amplify collapsed directions
        inverse_scale = 1.0 / (sv_norm + 0.1)
        inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)
        per_comp_scale = np.clip(inverse_scale, 0.2, 3.0)

        vel_scaled = vel_proj * per_comp_scale

        # === DIRECTIONAL CORRECTION FROM ENTROPY SIGNALS ===
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_dist
        else:
            to_best_dir = np.random.randn(self.np, self.dim)

        # When trapped: add random perturbation proportional to collapse
        if trapped_signal > 0.4:
            random_perturb = np.random.randn(self.np, self.dim) * trapped_signal * 0.5
        else:
            random_perturb = np.zeros((self.np, self.dim))

        # === FINAL POPULATION UPDATE ===
        new_population = self.population + exploration_factor * (vel_scaled @ V.T) + random_perturb

        # === ADAPTIVE REINITIALIZATION TRIGGER ===
        # If entropy collapsed AND we're not improving → reinitialize worst particles
        if entropy_collapse > 0.7 and entropy_trend < -0.1:
            if self.global_best is not None:
                n_replace = max(1, int(0.2 * self.np))
                worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
                self.population[worst_indices] = np.random.uniform(
                    self.lower_bound, self.upper_bound, (n_replace, self.dim)
                )
                self.velocity[worst_indices] = np.random.uniform(
                    self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
                )
                self.personal_best_fitness[worst_indices] = np.inf
                self.personal_best[worst_indices] = self.population[worst_indices]

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    except ImportError:
        # Fallback without sklearn
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
            uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
            max_kl = np.log(len(eigenvalues_norm))
            kl_normalized = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)

            U, V = np.linalg.eigh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = V[:, idx]
            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            inverse_scale = 1.0 / (sv_norm + 0.1)
            inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)
            per_comp_scale = np.clip(inverse_scale, 0.2, 3.0)
            vel_scaled = vel_proj * per_comp_scale

            exploration_factor = np.clip(0.5 + 0.5 * kl_normalized, 0.5, 2.0)
            new_population = self.population + exploration_factor * (vel_scaled @ V.T)
        except:
            new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```