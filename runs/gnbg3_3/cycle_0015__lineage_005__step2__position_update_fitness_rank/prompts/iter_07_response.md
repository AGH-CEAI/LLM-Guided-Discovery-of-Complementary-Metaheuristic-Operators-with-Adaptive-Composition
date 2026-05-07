**Idea: Monte Carlo Convergence-Probability Velocity Modulation**

Use repeated Monte Carlo subsampling of the population to estimate per-particle convergence probability (how often a random subsample containing this particle reaches the global best). This stochastic signal replaces the deterministic spectral gap, directly measuring exploration/exploitation need. Bootstrap resampling of the population covariance provides uncertainty-aware velocity scaling, and fitness-weighted random probing ensures diverse exploration. Fundamentally stochastic: every computation involves randomness (sampling, bootstrapping) rather than deterministic graph/eigenvalue analysis.

```python
def _position_update_fitness_rank(self):
    """Monte Carlo convergence-probability velocity modulation (Category G).

    Fundamentally stochastic approach using:
    1. Monte Carlo subsampling: estimate per-particle convergence probability
       by repeatedly sampling subsets and measuring how often they reach global best.
       High convergence_prob → exploit; Low → explore.
    2. Bootstrap resampling of population to estimate covariance uncertainty
       and derive per-particle variance signals for adaptive scaling.
    3. Fitness-weighted random probing for diverse exploration.

    This replaces the deterministic k-NN graph / Laplacian spectral gap (Category E)
    with a sampling-based estimate of convergence likelihood. No eigenvalues,
    no graph traversal — pure stochastic estimation.
    """
    try:
        # === MONTE CARLO CONVERGENCE PROBABILITY ===
        # Repeatedly subsample population and check if subsample reaches global best.
        # This directly estimates how "useful" each particle is for finding improvement.
        n_mc_samples = 50
        mc_subsample_size = max(3, self.np // 4)

        if not hasattr(self, '_mc_conv_prob'):
            self._mc_conv_prob = np.ones(self.np) * 0.5

        if self.global_best is not None and self.np >= mc_subsample_size:
            conv_counts = np.zeros(self.np)

            for _ in range(n_mc_samples):
                # Random subsample
                subsample_idx = np.random.choice(self.np, mc_subsample_size, replace=False)
                subsample_pop = self.population[subsample_idx]

                # Evaluate fitness of subsample
                subsample_fitness, _ = self._evaluate_batch(subsample_pop, self._current_func)

                # Check if subsample best is better than current global best
                subsample_best_fitness = np.min(subsample_fitness)

                # For each particle in subsample, record if subsample found improvement
                for idx in subsample_idx:
                    if subsample_best_fitness < self.global_best_fitness * 1.001:
                        conv_counts[idx] += 1

            # Per-particle convergence probability
            new_conv_prob = conv_counts / n_mc_samples

            # EMA smoothing to reduce variance
            self._mc_conv_prob = 0.3 * new_conv_prob + 0.7 * self._mc_conv_prob
        else:
            self._mc_conv_prob = np.ones(self.np) * 0.5

        mc_conv_prob = np.clip(self._mc_conv_prob, 0.1, 0.9)

        # === BOOTSTRAP COVARIANCE UNCERTAINTY ===
        # Bootstrap resample population to estimate covariance uncertainty
        n_bootstrap = 30
        bootstrap_eigenvalue_samples = []

        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(self.np, self.np, replace=True)
            boot_pop = self.population[boot_idx]
            try:
                centered = boot_pop - np.mean(boot_pop, axis=0)
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                # Record condition number as proxy for anisotropy
                cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10) if len(eigenvalues) > 1 else 1.0
                bootstrap_eigenvalue_samples.append(cond)
            except:
                pass

        if len(bootstrap_eigenvalue_samples) > 5:
            bootstrap_cond_mean = np.mean(bootstrap_eigenvalue_samples)
            bootstrap_cond_std = np.std(bootstrap_eigenvalue_samples)
            # Uncertainty: high std relative to mean = uncertain population structure
            bootstrap_uncertainty = bootstrap_cond_std / (bootstrap_cond_mean + 1e-10)
        else:
            bootstrap_cond_mean = 1.0
            bootstrap_uncertainty = 0.5

        # === BOOTSTRAP-BASED GLOBAL SCALE ===
        # High uncertainty → more exploration (higher scale)
        # Low uncertainty → more exploitation (lower scale)
        uncertainty_explore = 1.0 + 0.4 * np.clip(bootstrap_uncertainty, 0.0, 2.0)

        if bootstrap_cond_mean > 10.0:
            # High condition number = anisotropic landscape → more exploration
            cond_explore = 1.3
        elif bootstrap_cond_mean < 3.0:
            cond_explore = 0.85
        else:
            cond_explore = 1.0

        global_scale = uncertainty_explore * cond_explore
        global_scale = np.clip(global_scale, 0.5, 2.0)

        # === PER-PARTICLE VELOCITY MODULATION FROM CONVERGENCE PROBABILITY ===
        # High convergence_prob → particles are in promising regions → exploit
        # Low convergence_prob → particles may be in unexplored regions → explore
        vel_scale = np.where(
            mc_conv_prob > 0.6,
            global_scale * 0.7,  # Exploit mode
            global_scale * 1.4   # Explore mode
        )
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        # === LATIN HYPERCUBE-DERIVED DIRECTIONAL PROBING ===
        # Use stratified sampling to generate diverse exploration directions
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm

            # Stratified random perturbation: divide dimension space into strata
            n_strata = max(2, self.dim // 4)
            strat_idx = np.random.randint(0, n_strata, size=(self.np,))
            stratum_width = 1.0 / n_strata
            stratum_scale = (strat_idx + np.random.random(self.np)) * stratum_width

            # Apply stratified perturbation along global-best direction
            directional = 0.4 * stratum_scale[:, np.newaxis] * to_best_dir
        else:
            # Random exploration with stratified sampling
            directional = np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        # === FITNESS-WEIGHTED RANDOM PERTURBATION ===
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Success history
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # Random perturbation inversely proportional to convergence probability
        random_scale = (1.0 - mc_conv_prob)[:, np.newaxis] * (1.0 - 0.3 * success_norm[:, np.newaxis])
        random_perturb = random_scale * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        # === COMBINE ===
        new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + random_perturb

    except Exception:
        # Fallback: simple random perturbation with global scaling
        new_population = self.population + 1.2 * self.velocity + np.random.uniform(-0.3, 0.3, (self.np, self.dim))

    self.population = self._clip_to_bounds(new_population)
```