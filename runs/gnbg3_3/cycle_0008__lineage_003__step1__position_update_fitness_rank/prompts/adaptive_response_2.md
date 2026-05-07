```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT (mechanism #1) with rank-normalized rewards.
    
    The gap table is BLEND-FRIENDLY (all gaps < 2×), so ensemble blending is
    mathematically viable but NOT optimal here. The 5 winners are spread
    evenly (7/6/5/4/2 wins), and each uses a DIFFERENT core signal:
      - original: eigenvalue-anisotropic modulation
      - variant_02_catB: Rayleigh-quotient velocity modulation
      - variant_04_catD: Spearman-rank stagnation modulation (fitness-only)
      - variant_08_catH: Hybrid fitness-rank + spectral-subspace escape
      - variant_10_catB: Condition-number-gated anisotropic damping
    
    Because the gaps are small, a bandit can reliably credit-assign. Thompson
    Sampling is preferred over UCB1 here because:
      (a) it handles correlated operators better (explores more smoothly),
      (b) rank-based rewards are bounded [0,1], so the Beta posterior is well-
          calibrated without scale-dependent priors.
    
    Reward signal: rank-normalized fitness improvement within a sliding window
    of 5*num_arms=25 generations. This is scale-independent and handles any
    black-box func without magic constants.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self.current_fitness = None
        
        # --- BANDIT STATE: Beta posterior for Thompson Sampling ---
        # 5 arms, each with alpha (successes) and beta (failures)
        # Initialize with a mild optimistic prior: alpha=2, beta=2
        # This says "each arm is ~50/50 with slight optimism"
        self._arm_names = [
            'original',      # eigenvalue-anisotropic modulation
            'catB_rayleigh', # Rayleigh-quotient velocity modulation (variant_02)
            'catD_spearman', # Spearman-rank stagnation modulation (variant_04)
            'catH_hybrid',   # Hybrid fitness-rank + spectral escape (variant_08)
            'catB_cond',     # Condition-number-gated damping (variant_10)
        ]
        self._arm_alpha = {name: 2.0 for name in self._arm_names}
        self._arm_beta = {name: 2.0 for name in self._arm_names}
        
        # Sliding window for rank-normalized rewards
        # K >= 3 * num_arms = 15; use 25 for stability
        self._reward_window_size = max(25, 3 * len(self._arm_names))
        self._reward_history = []  # list of (arm, rank_score) tuples
        
        # Running best-fitness history for rank normalization
        self._window_best_fitness = []  # best fitness per generation in window
        self._window_worst_fitness = []  # worst fitness per generation in window
        
        # Current selected arm
        self._current_arm = self._arm_names[0]
        
        # Per-arm running statistics (for logging/debugging)
        self._arm_total_reward = {name: 0.0 for name in self._arm_names}
        self._arm_count = {name: 0 for name in self._arm_names}
    
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight using condition number of population covariance."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            # Spectral signal: map condition number to inertia adjustment
            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            # Map spectral signal to inertia range [0.4, 0.95]
            target_inertia = 0.4 + 0.55 * spectral_signal

            # Blend with time-based baseline
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (shared across all position update strategies)
    # -------------------------------------------------------------------------
    
    def _velocity_update_base(self):
        """Base velocity update: cognitive + social + DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # BANDIT: Thompson Sampling with rank-normalized rewards
    # -------------------------------------------------------------------------
    
    def _thompson_sample(self):
        """Sample an arm from the Beta posterior using Thompson Sampling.
        
        Returns the arm name with the highest sampled value from its posterior.
        This naturally balances exploration (uncertain arms) vs exploitation
        (high-mean arms) without needing an explicit exploration constant.
        """
        samples = {}
        for name in self._arm_names:
            alpha = max(1.0, self._arm_alpha[name])
            beta = max(1.0, self._arm_beta[name])
            samples[name] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
    def _record_reward(self, arm, best_fitness):
        """Record a rank-normalized reward for the given arm.
        
        The reward is computed as:
          rank_score = (worst_in_window - best_fitness) / (worst - best + eps)
        
        This is SCALE-INDEPENDENT: it measures where this arm's best fitness
        falls in the distribution of all fitness values seen in the window.
        A score of 1.0 means this was the best fitness in the window; 0.0 means
        it was the worst. No magic constants, no scale-dependent multipliers.
        """
        self._window_best_fitness.append(best_fitness)
        self._window_worst_fitness.append(best_fitness)
        
        # Maintain sliding window
        if len(self._window_best_fitness) > self._reward_window_size:
            self._window_best_fitness.pop(0)
            self._window_worst_fitness.pop(0)
        
        # Compute rank-normalized reward
        window_best = min(self._window_best_fitness)
        window_worst = max(self._window_worst_fitness)
        
        if window_worst > window_best:
            # Higher score = better (lower fitness)
            rank_score = (window_worst - best_fitness) / (window_worst - window_best + 1e-10)
            rank_score = np.clip(rank_score, 0.0, 1.0)
        else:
            rank_score = 1.0  # All same, give max credit
        
        # Update sliding window history
        self._reward_history.append((arm, rank_score))
        if len(self._reward_history) > self._reward_window_size:
            self._reward_history.pop(0)
        
        # Update Beta posterior using rank score as binary feedback
        # rank_score in [0,1] is treated as probability of success
        # This is a "soft" update: we add rank_score to alpha and (1-rank_score) to beta
        self._arm_alpha[arm] += rank_score
        self._arm_beta[arm] += (1.0 - rank_score)
        
        # Running statistics
        self._arm_total_reward[arm] += rank_score
        self._arm_count[arm] += 1
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (the 5 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation."""
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB_rayleigh(self):
        """Rayleigh-quotient velocity modulation using covariance eigendecomposition (Category B).

        Key insight: Compute Rayleigh quotient R = v^T Cov v / v^T v to measure velocity
        alignment with population covariance principal axes. High Rayleigh quotient on
        collapsed eigendirections → population stuck in subspace → apply inverse scaling
        to velocity along dominant eigencomponents. Targets worst tasks (17, 16, 6) where
        large condition numbers indicate severe anisotropy and subspace collapse.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            rq_values = np.zeros(self.np)
            for i in range(self.np):
                v = self.velocity[i]
                v_cov_v = np.dot(v, cov @ v)
                v_v = np.dot(v, v) + 1e-10
                rq_values[i] = v_cov_v / v_v
            rq_mean = np.mean(rq_values)
            rq_std = np.std(rq_values) + 1e-10

            vel_proj = self.velocity @ eigenvectors
            vel_energy_per_comp = vel_proj ** 2
            total_vel_energy = np.sum(vel_energy_per_comp) + 1e-10
            vel_energy_fraction = vel_energy_per_comp / total_vel_energy

            alignment = vel_energy_fraction / (eigenvalues_norm + 1e-10)
            alignment = np.clip(alignment, 0.0, 10.0)

            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                per_comp_scale = 1.0 + 0.5 * log_damp * (1.0 - alignment)
            else:
                per_comp_scale = 1.0 + 0.3 * spectral_signal * (1.0 - alignment)

            per_comp_scale = np.clip(per_comp_scale, 0.1, 2.5)
            vel_scaled = vel_proj * per_comp_scale

            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
                directional = 0.4 * (1.0 - fitness_ranks[:, np.newaxis]) * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))

            random_perturb = 0.3 * np.random.uniform(-1, 1, (self.np, self.dim))

            new_population = self.population + (vel_scaled @ eigenvectors.T) + directional + random_perturb

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catD_spearman(self):
        """Spearman-rank stagnation modulation (Category D: fitness-landscape only).

        Key insight: Use ONLY fitness signals — no position distances, no graph
        structures, no covariance. Spearman correlation between current fitness
        rank and personal-best rank identifies particles stuck at local optima.
        Fitness percentiles detect convergence. Success-history tracks improvement.
        This is orthogonal to graph-based (E) and spectral (B) approaches.
        """
        # Pure fitness-based rank computation
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Personal best fitness rank (captures "stagnation trap" detection)
        personal_best_rank = np.argsort(np.argsort(self.personal_best_fitness)) / max(1, self.np - 1)

        # Spearman-like correlation: high correlation = particle stuck at local optimum
        if np.std(fitness_ranks) > 1e-10 and np.std(personal_best_rank) > 1e-10:
            spearman_corr = np.corrcoef(fitness_ranks, personal_best_rank)[0, 1]
            spearman_corr = np.clip(spearman_corr, -1.0, 1.0)
        else:
            spearman_corr = 0.0

        # Fitness percentile signals
        bottom_percentile = np.percentile(self.current_fitness, 10)
        top_percentile = np.percentile(self.current_fitness, 90)

        stagnant_particles = self.current_fitness <= bottom_percentile
        top_particles = self.current_fitness >= top_percentile

        # Success history
        if not hasattr(self, '_fitness_success_ema'):
            self._fitness_success_ema = np.zeros(self.np)

        improvement = self.personal_best_fitness - self.current_fitness
        improved_mask = improvement > 0
        self._fitness_success_ema[improved_mask] = self._fitness_success_ema[improved_mask] * 0.9 + 0.1
        self._fitness_success_ema[~improved_mask] = self._fitness_success_ema[~improved_mask] * 0.9 - 0.05
        self._fitness_success_ema = np.clip(self._fitness_success_ema, -0.5, 1.0)

        success_norm = (self._fitness_success_ema - np.min(self._fitness_success_ema) + 1e-10)
        success_norm = success_norm / (np.max(success_norm) + 1e-10)

        # Fitness-distance correlation in FITNESS space
        if self.global_best_fitness < np.inf:
            fitness_distance_to_best = np.abs(self.current_fitness - self.global_best_fitness)
            fitness_dist_rank = np.argsort(np.argsort(fitness_distance_to_best)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(fitness_dist_rank) > 1e-10:
                fitness_dist_corr = np.corrcoef(fitness_ranks, fitness_dist_rank)[0, 1]
                fitness_dist_corr = np.clip(fitness_dist_corr, -1.0, 1.0)
            else:
                fitness_dist_corr = 0.0
        else:
            fitness_dist_corr = 0.0

        stagnation_signal = spearman_corr * (1.0 - success_norm)
        exploration_signal = np.clip(stagnation_signal * 0.5, 0.0, 1.0)
        exploitation_signal = success_norm * stagnant_particles.astype(float)

        vel_scale = 1.0 + 0.5 * exploration_signal - 0.3 * exploitation_signal
        vel_scale = vel_scale * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.4, 2.5)

        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional_strength = (1.0 - fitness_ranks[:, np.newaxis]) * (1.0 + success_norm[:, np.newaxis])
            directional = 0.3 * directional_strength * to_best_dir
        else:
            directional = 0.3 * (1.0 - fitness_ranks[:, np.newaxis]) * np.random.uniform(-1, 1, (self.np, self.dim))

        random_scale = 0.3 + 0.4 * exploration_signal[:, np.newaxis]
        random_perturb = random_scale * np.random.uniform(-1, 1, (self.np, self.dim))

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catH_hybrid(self):
        """Hybrid: Fitness-rank gradient + spectral-subspace escape with condition-driven switching (Category H).

        Mechanism 1 (Category D): Fitness-rank gradient - particles follow weighted
        attraction toward global best based on their rank.

        Mechanism 2 (Category B): Spectral-subspace escape - when condition number
        exceeds threshold, project velocity onto minor eigenvectors.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            eigenvalues_norm = eigenvalues / total_var
            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
        except:
            cond = 1.0
            cond_log = 0.0
            eff_dim_ratio = 1.0
            entropy_ratio = 0.5
            eigenvalues = np.ones(self.dim)
            eigenvectors = np.eye(self.dim)

        cond_threshold = 100.0
        spectral_weight = np.clip((cond - cond_threshold) / (cond_threshold * 10), 0.0, 0.8)
        fitness_weight = 1.0 - spectral_weight

        collapse_signal = 1.0 - eff_dim_ratio
        spectral_weight = np.clip(spectral_weight + 0.2 * collapse_signal, 0.0, 0.8)
        fitness_weight = 1.0 - spectral_weight

        # --- MECHANISM 1: Fitness-rank gradient ---
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        if not hasattr(self, '_h_success_count'):
            self._h_success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._h_success_count[improved] += 1
        self._h_success_count[~improved] *= 0.9
        success_norm = self._h_success_count / (np.max(self._h_success_count) + 1e-10)

        rank_boost = 1.0 + 0.5 * (1.0 - fitness_ranks) + 0.3 * success_norm
        rank_boost = np.clip(rank_boost, 0.8, 2.0)

        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            pull_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
            fitness_gradient = pull_strength * to_best_dir * rank_boost[:, np.newaxis]
        else:
            fitness_gradient = np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        # --- MECHANISM 2: Spectral-subspace escape ---
        try:
            _, eigenvectors = np.linalg.eigh(cov)
            eigenvectors = eigenvectors[:, np.argsort(np.linalg.eigvalsh(cov))[::-1]]

            vel_proj = self.velocity @ eigenvectors
            sv_scale = np.sqrt(eigenvalues + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)

            inverse_scale = 1.0 / (sv_norm + 0.1)
            inverse_scale = inverse_scale / (np.max(inverse_scale) + 1e-10)

            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_comp_scale = uniform_scale * (1.0 - spectral_weight) + inverse_scale * spectral_weight
            per_comp_scale = np.clip(per_comp_scale, 0.1, 2.5)

            vel_scaled = vel_proj * per_comp_scale

            if cond > cond_threshold:
                escape_strength = np.clip(cond_log / 10.0, 0.0, 0.5)
                minor_perturb = np.zeros_like(self.velocity)
                n_minor = max(1, int(0.3 * self.dim))
                for i in range(n_minor):
                    minor_dir = eigenvectors[:, -(i + 1)]
                    minor_perturb += escape_strength * np.outer(
                        np.random.uniform(-1, 1, self.np), minor_dir
                    )
                spectral_component = vel_scaled @ eigenvectors.T + minor_perturb
            else:
                spectral_component = vel_scaled @ eigenvectors.T

        except np.linalg.LinAlgError:
            spectral_component = self.velocity
            per_comp_scale = np.ones(self.np)

        velocity_contribution = (
            fitness_weight * self.velocity * rank_boost[:, np.newaxis] +
            spectral_weight * spectral_component
        )

        gradient_perturbation = fitness_weight * fitness_gradient

        explore_scale = 0.5 + 0.3 * np.clip(cond_log / 5.0, 0.0, 1.0)
        random_perturb = explore_scale * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + velocity_contribution + gradient_perturbation + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB_cond(self):
        """Condition-number-gated anisotropic velocity damping (Category B).

        Key insight: Use condition number of population covariance as a gating
        signal. When cond > threshold (high anisotropy), apply per-eigenvector
        damping that boosts velocity in collapsed directions.
        """
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            cond_threshold = 50.0
            if cond > cond_threshold:
                sv_scale = np.sqrt(eigenvalues)
                sv_norm = sv_scale / (sv_scale[0] + 1e-10)
                per_comp_scale = 1.0 / (sv_norm + 0.05)
                per_comp_scale = per_comp_scale / (np.max(per_comp_scale) + 1e-10)
                blend = np.clip((cond - cond_threshold) / 200.0, 0.0, 1.0)
                uniform_scale = np.clip(cond_log / np.log1p(1000.0), 0.5, 2.0)
                final_scale = uniform_scale * (1.0 - blend) + per_comp_scale * blend
            else:
                final_scale = np.clip(cond_log / np.log1p(100.0), 0.7, 1.5)

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * final_scale

            explore_boost = 1.0 + 0.3 * (1.0 - eff_dim_ratio)

            new_population = self.population + explore_boost * (vel_scaled @ eigenvectors.T)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        fitness_perturb = 0.4 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
        new_population = new_population + directional + random_perturb

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'catB_rayleigh':
            self._position_update_catB_rayleigh()
        elif operator == 'catD_spearman':
            self._position_update_catD_spearman()
        elif operator == 'catH_hybrid':
            self._position_update_catH_hybrid()
        elif operator == 'catB_cond':
            self._position_update_catB_cond()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # RESTART & LOCAL BEST
    # -------------------------------------------------------------------------
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            if self.global_best is not None:
                perturbation = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + perturbation)
                self.personal_best[0] = self.population[0]
                self.personal_best_fitness[0] = np.inf
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with Thompson Sampling bandit for position-update selection.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Reset bandit state
        self._reward_history = []
        self._window_best_fitness = []
        self._window_worst_fitness = []
        for name in self._arm_names:
            self._arm_alpha[name] = 2.0
            self._arm_beta[name] = 2.0
            self._arm_total_reward[name] = 0.0
            self._arm_count[name] = 0
        
        # Reset EMA states
        if hasattr(self, '_fitness_success_ema'):
            self._fitness_success_ema = np.zeros(self.np)
        if hasattr(self, '_h_success_count'):
            self._h_success_count = np.zeros(self.np)
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
            if hasattr(self, '_fitness_success_ema'):
                self._fitness_success_ema = np.zeros(self.np)
            if hasattr(self, '_h_success_count'):
                self._h_success_count = np.zeros(self.np)
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Initial bandit selection (uniform random for first few gens to warm up)
        if self.generation == 0:
            self._current_arm = self._arm_names[self.generation % len(self._arm_names)]
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Thompson Sampling: select arm for this generation
            self._current_arm = self._thompson_sample()
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED BY BANDIT)
            self._velocity_update_base()
            self._dispatch_position_update(self._current_arm)
            
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            
            # Record rank-normalized reward for the selected arm
            gen_best = float(np.min(self.current_fitness))
            self._record_reward(self._current_arm, gen_best)
            
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```