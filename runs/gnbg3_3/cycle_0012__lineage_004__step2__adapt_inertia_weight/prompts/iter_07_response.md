**Idea: Bootstrap Uncertainty + Random Direction Probing**
Monte-Carlo estimation of inertia: bootstrap confidence intervals on population diversity plus random directional fitness probing to detect gradient strength, enabling uncertainty-aware exploration/exploitation balancing.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using stochastic sampling (Category G).
    
    Uses TWO Monte-Carlo mechanisms:
    1. Bootstrap resampling of population diversity → uncertainty quantification
    2. Random directional fitness probing → gradient strength estimation
    
    High uncertainty in diversity → favor exploration (higher inertia).
    Strong local gradient → favor exploitation (lower inertia).
    """
    np = np if 'np' in dir() else numpy  # safety
    
    # === MECHANISM 1: BOOTSTRAP CONFIDENCE ON DIVERSITY ===
    n_bootstrap = 50
    diversity_samples = []
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, self.np, size=self.np)
        boot_pop = self.population[indices]
        boot_centroid = np.mean(boot_pop, axis=0)
        boot_diversity = np.mean(np.linalg.norm(boot_pop - boot_centroid, axis=1))
        diversity_samples.append(boot_diversity)
    
    diversity_samples = np.array(diversity_samples)
    mean_diversity = np.mean(diversity_samples)
    diversity_std = np.std(diversity_samples)
    
    # 95% confidence interval half-width, normalized
    ci_half_width = 1.96 * diversity_std / np.sqrt(n_bootstrap)
    confidence_uncertainty = ci_half_width / (mean_diversity + 1e-10)
    confidence_uncertainty = np.clip(confidence_uncertainty, 0.0, 2.0)
    
    # === MECHANISM 2: RANDOM DIRECTION FITNESS PROBING ===
    # Sample random directions, probe fitness along them
    n_probes = 20
    probe_deltas = np.random.uniform(-1.0, 1.0, (n_probes, self.dim))
    probe_norms = np.linalg.norm(probe_deltas, axis=1, keepdims=True) + 1e-10
    probe_deltas = probe_deltas / probe_norms
    
    # Evaluate at centroid + small steps along each direction
    centroid = np.mean(self.population, axis=0)
    step_size = 0.5 * (self.upper_bound - self.lower_bound) / 100.0
    
    fitness_at_centroid = np.mean(self.current_fitness)
    fitness_changes = []
    for delta in probe_deltas:
        probe_pos = self._clip_to_bounds(centroid + step_size * delta)
        probe_pop = np.tile(probe_pos, (min(3, self.np), 1))
        probe_fit, _ = self._evaluate_batch(probe_pop, lambda x: self.current_fitness[:len(x)])
        if len(probe_fit) > 0:
            fitness_changes.append(np.mean(probe_fit) - fitness_at_centroid)
    
    if len(fitness_changes) >= 3:
        fitness_changes = np.array(fitness_changes)
        # Gradient strength: std of fitness changes indicates local ruggedness
        gradient_strength = np.std(np.abs(fitness_changes))
        gradient_strength = np.clip(gradient_strength / (np.std(self.current_fitness) + 1e-10), 0.0, 1.0)
    else:
        gradient_strength = 0.5
    
    # === COMBINE STOCHASTIC SIGNALS ===
    # Uncertainty signal → exploration (high uncertainty = might be in unexplored region)
    # Gradient signal → exploitation (high gradient = in a basin, exploit it)
    exploration_signal = confidence_uncertainty
    exploitation_signal = gradient_strength
    
    # Map to inertia range [0.4, 0.95]
    # High exploration_need → higher inertia (trust momentum, explore)
    # High exploitation_need → lower inertia (follow gradient)
    net_signal = exploration_signal - 0.5 * exploitation_signal
    net_signal = np.clip(net_signal, -0.5, 1.0)
    
    # Normalize to [0, 1] range for inertia mapping
    normalized = (net_signal + 0.5) / 1.5
    target_inertia = 0.4 + 0.55 * normalized
    
    # Blend with time-based baseline for stability
    decay = 0.729 - 0.15 * (self.generation / 1000)
    target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
    
    # EMA smoothing
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```