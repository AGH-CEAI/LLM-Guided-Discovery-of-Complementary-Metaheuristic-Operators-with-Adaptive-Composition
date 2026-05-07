import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING BANDIT (mechanism #1) with rank-based UCB.
    
    Gap table analysis (CRITICAL):
      - ALL 24 tasks have winner-runner-up gaps < 2× (max: 1.5× on task 5).
      - NO tasks have gap ≥ 10× → ensemble blending is NOT mathematically excluded,
        but a bandit is still preferred because:
        (a) wins are spread evenly across 7 variants (4/4/4/3/3/3/3),
        (b) different operators win on different tasks,
        (c) rank-based bandit avoids scale-dependent reward shaping.
    
    The mechanism:
      - 7 arms = the 7 benchmark-winning position update strategies.
      - Rank-based reward: each generation, score = fraction of last K generations
        where this operator produced the best global-best improvement.
      - Thompson Sampling for selection (handles exploration-exploitation naturally).
      - UCB1 bonus on top to ensure sufficient exploration of all arms.
      - Sliding window of K=35 (≥ 3× num_arms=7) so each arm is sampled enough.
      - No per-generation probe overhead (unlike probe-and-commit).
    
    Why Thompson Sampling over probe-and-commit:
      - Probe wastes ~15-20% of budget when gaps are small.
      - Bandit adapts continuously; probe only learns once at start.
      - Rank-based rewards are scale-invariant — no magic constants.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.np = min(max(5 * dim, 20), 300)
        self.neighborhood_size = max(3, dim // 5)
        
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
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
        
        # --- THOMPSON SAMPLING BANDIT STATE ---
        # 7 arms = the 7 benchmark-winning position update strategies
        self._operators = [
            'original',          # original.py wins: tasks 5, 8, 16, 21
            'eigenvalue_aniso',  # variant_02_catB_idea_0 wins: tasks 7, 19, 22
            'fitness_rank',      # variant_04_catD_idea_0 wins: tasks 1, 9, 14, 18
            'mc_conv',           # variant_07_catG_idea_0 wins: tasks 10, 12, 17, 20
            'graph_spectral',    # variant_08_catH_idea_0 wins: tasks 0, 13, 15
            'geo_bounding',      # variant_09_catA_idea_0 wins: tasks 3, 6, 23
            'eff_rank',          # variant_10_catB_idea_0 wins: tasks 2, 4, 11
        ]
        
        # Sliding window reward tracking
        self._window_size = 35  # ≥ 3× num_arms=7
        self._op_rewards = {op: [] for op in self._operators}
        self._op_gen_scores = []  # List of (gen, operator, score)
        
        # UCB tracking
        self._op_total_reward = {op: 0.0 for op in self._operators}
        self._op_count = {op: 0 for op in self._operators}
        self._total_selections = 0
        
        # Global-best tracking for rank-based scoring
        self._prev_global_best = None
        self._prev_global_best_fitness = np.inf
        self._gen_best_history = []  # (gen, best_fitness, operator)
        
        # Success history for per-particle signals
        self._success_count = None
        self._fitness_history = []
        self._prev_fitness = None
    
    def _initialize_population(self):
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
        
        self._success_count = np.zeros(self.np)
        self._fitness_history = []
        self._prev_fitness = None
    
    def _clip_to_bounds(self, population):
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
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
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * spectral_signal
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        fitness = self.current_fitness
        if len(fitness) > 1:
            n_bins = min(20, max(3, len(fitness) // 2))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0]
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_fitness_entropy = np.log(n_bins)
            norm_fitness_ent = fitness_entropy / (max_fitness_entropy + 1e-10)
        else:
            norm_fitness_ent = 0.5

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)

            uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
            uniform = uniform + 1e-10
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))

            max_kl = np.log(len(eigenvalues_norm))
            norm_kl = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
        except:
            norm_kl = 0.5

        exploration_signal = 0.4 * (1.0 - norm_fitness_ent) + 0.6 * norm_kl
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

        min_ns = 1
        max_ns = max(3, self.np // 4)
        target_ns = int(np.round(min_ns + exploration_signal * (max_ns - min_ns)))
        target_ns = np.clip(target_ns, min_ns, max_ns)

        if target_ns > self.neighborhood_size:
            self.neighborhood_size = min(target_ns, self.neighborhood_size + 1)
        elif target_ns < self.neighborhood_size:
            self.neighborhood_size = max(target_ns, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    def _velocity_update_base(self):
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
    # BANDIT: Thompson Sampling + UCB selection
    # -------------------------------------------------------------------------
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling with UCB bonus.
        
        Thompson Sampling: sample from posterior Beta distribution for each arm,
        pick arm with highest sample. UCB bonus ensures sufficient exploration.
        
        Uses rank-based average reward (already in [0,1]) so no scale issues.
        """
        # First, ensure all arms have been tried at least once
        for op in self._operators:
            if self._op_count[op] == 0:
                return op
        
        # Compute UCB-adjusted Thompson samples
        samples = {}
        for op in self._operators:
            # Rank-based average reward over sliding window
            if len(self._op_rewards[op]) > 0:
                avg_reward = np.mean(self._op_rewards[op])
            else:
                avg_reward = 0.5
            
            # UCB1 exploration bonus on the same [0,1] scale
            n = self._op_count[op]
            ucb_bonus = np.sqrt(2.0 * np.log(max(self._total_selections, 1)) / max(n, 1))
            
            # Effective mean for sampling: blend of empirical mean and UCB bonus
            # Scale bonus to be meaningful but not dominant
            effective_mean = avg_reward + 0.3 * np.clip(ucb_bonus, 0.0, 0.5)
            
            # Thompson Sampling: sample from Normal(mean=empirical, std=exploration)
            # Use empirical std if we have enough data, else fixed std
            if len(self._op_rewards[op]) >= 5:
                std = np.std(self._op_rewards[op]) + 1e-10
            else:
                std = 0.3  # Prior: moderate uncertainty
            
            sample = np.random.normal(effective_mean, std)
            samples[op] = sample
        
        return max(samples.keys(), key=lambda op: samples[op])
    
    def _record_reward(self, operator, current_best_fitness):
        """Record reward using rank-based scoring within sliding window.
        
        Reward = rank of this generation's global-best improvement among
        the last K generations. This is scale-invariant — no magic constants.
        
        Improvement = max(0, prev_best - current_best). Rank-normalized to [0,1].
        """
        if self._prev_global_best_fitness == np.inf:
            self._prev_global_best_fitness = current_best_fitness
            reward = 0.5  # Neutral first reward
        else:
            improvement = max(0.0, self._prev_global_best_fitness - current_best_fitness)
            
            # Rank-based scoring: look at last K generations' improvements
            window_improvements = []
            for i in range(max(0, len(self._gen_best_history) - self._window_size), 
                          len(self._gen_best_history)):
                if i > 0:
                    prev_f = self._gen_best_history[i-1][1]
                    curr_f = self._gen_best_history[i][1]
                    imp = max(0.0, prev_f - curr_f)
                    window_improvements.append(imp)
            
            window_improvements.append(improvement)
            
            if len(window_improvements) > 1 and max(window_improvements) > 0:
                # Rank: what fraction of the window did this improvement beat?
                rank = sum(1 for x in window_improvements if x <= improvement) / len(window_improvements)
                reward = np.clip(rank, 0.0, 1.0)
            else:
                reward = 0.5
        
        # Update sliding window
        self._op_rewards[operator].append(reward)
        if len(self._op_rewards[operator]) > self._window_size:
            self._op_rewards[operator].pop(0)
        
        # Update UCB stats
        self._op_total_reward[operator] += reward
        self._op_count[operator] += 1
        self._total_selections += 1
        
        # Track history
        self._gen_best_history.append((self.generation, current_best_fitness, operator))
        if len(self._gen_best_history) > self._window_size * 2:
            self._gen_best_history.pop(0)
        
        self._prev_global_best_fitness = current_best_fitness
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (7 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation (original.py wins: 5, 8, 16, 21)."""
        centered = self.population - np.mean(self.population, axis=0)
        try:
            cov = np.cov(centered.T)
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
    
    def _position_update_eigenvalue_aniso(self):
        """Eigenvalue-anisotropic velocity scaling (variant_02_catB, wins: 7, 19, 22)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)

            total_var = np.sum(eigenvalues) + 1e-10
            sorted_desc = np.sort(eigenvalues)[::-1]
            cumvar = np.cumsum(sorted_desc) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1

            spectral_signal = np.clip(np.log1p(cond) / np.log1p(1e6), 0.0, 1.0)

            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = 0.4 + 0.55 * spectral_signal
            self.inertia_weight = np.clip(
                self.inertia_weight * 0.8 + (target_inertia * 0.6 + decay * 0.4) * 0.2,
                0.4, 0.95
            )

            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            spectral_explore = 1.0 + 1.0 * spectral_signal
            spectral_explore = np.clip(spectral_explore, 0.5, 2.5)

            vel_proj = self.velocity @ eigenvectors

            sv_norm = eigenvalues / (eigenvalues[0] + 1e-10)
            per_component_scale = 1.0 / (sv_norm ** 2 + 0.01)
            per_component_scale = per_component_scale / (np.max(per_component_scale) + 1e-10)

            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            combined_scale = uniform_scale * (1.0 - blend_weight) + per_component_scale * blend_weight
            combined_scale = np.clip(combined_scale, 0.3, 2.5)

            vel_scaled = vel_proj * combined_scale * spectral_explore

            fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional = fitness_perturb * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))

            random_perturb = spectral_explore * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

            new_population = self.population + self.inertia_weight * vel_scaled + directional + random_perturb

        except (np.linalg.LinAlgError, FloatingPointError):
            spectral_explore = 1.0 + 0.5 * np.random.random()
            new_population = self.population + self.inertia_weight * self.velocity * spectral_explore

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_fitness_rank(self):
        """Fitness-percentile & Kendall-tau rank stability (variant_04_catD, wins: 1, 9, 14, 18)."""
        try:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            p10 = np.percentile(self.current_fitness, 10)
            p50 = np.percentile(self.current_fitness, 50)
            p90 = np.percentile(self.current_fitness, 90)
            fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)

            if hasattr(self, '_prev_fitness') and len(self._prev_fitness) == len(self.current_fitness):
                f_improved = self.current_fitness < self._prev_fitness
                n = len(self.current_fitness)
                concordant = 0
                discordant = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        curr_order = self.current_fitness[i] < self.current_fitness[j]
                        prev_order = self._prev_fitness[i] < self._prev_fitness[j]
                        if curr_order == prev_order:
                            concordant += 1
                        else:
                            discordant += 1
                n_pairs = n * (n - 1) / 2
                kendall_tau = (concordant - discordant) / (n_pairs + 1e-10)
            else:
                kendall_tau = 0.5
            self._prev_fitness = self.current_fitness.copy()

            if self._success_count is None:
                self._success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._success_count[improved] += 1
            self._success_count[~improved] *= 0.85
            success_norm = self._success_count / (np.max(self._success_count) + 1.0)

            is_converged = fitness_spread < 0.05
            is_stagnant = kendall_tau > 0.75

            if is_stagnant and is_converged:
                global_scale = 1.8
            elif is_stagnant:
                global_scale = 1.4
            elif is_converged:
                global_scale = 0.7
            else:
                global_scale = 1.0

            rank_explore = 1.0 + 0.6 * fitness_ranks
            success_explore = 1.0 + 0.4 * (1.0 - success_norm)
            vel_scale = rank_explore[:, np.newaxis] * global_scale * success_explore[:, np.newaxis]
            vel_scale = np.clip(vel_scale, 0.2, 3.0)

            best_rank_idx = np.argmin(fitness_ranks)
            attract_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])

            random_dir = np.random.randn(self.np, self.dim)
            random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-10)

            if self.global_best is not None and kendall_tau < 0.5:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional = attract_strength * to_best_dir
            else:
                directional = attract_strength * random_dir

            random_strength = 0.3 * (1.0 + kendall_tau)
            random_perturb = random_strength * np.random.uniform(-1.0, 1.0, (self.np, self.dim))

            new_population = self.population + vel_scale * self.velocity + directional + random_perturb

        except:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_mc_conv(self):
        """Monte Carlo convergence-probability velocity modulation (variant_07_catG, wins: 10, 12, 17, 20)."""
        try:
            n_mc_samples = 50
            mc_subsample_size = max(3, self.np // 4)

            if not hasattr(self, '_mc_conv_prob'):
                self._mc_conv_prob = np.ones(self.np) * 0.5

            if self.global_best is not None and self.np >= mc_subsample_size:
                conv_counts = np.zeros(self.np)

                for _ in range(n_mc_samples):
                    subsample_idx = np.random.choice(self.np, mc_subsample_size, replace=False)
                    subsample_pop = self.population[subsample_idx]

                    subsample_fitness, _ = self._evaluate_batch(subsample_pop, self._current_func)

                    subsample_best_fitness = np.min(subsample_fitness)

                    for idx in subsample_idx:
                        if subsample_best_fitness < self.global_best_fitness * 1.001:
                            conv_counts[idx] += 1

                new_conv_prob = conv_counts / n_mc_samples

                self._mc_conv_prob = 0.3 * new_conv_prob + 0.7 * self._mc_conv_prob
            else:
                self._mc_conv_prob = np.ones(self.np) * 0.5

            mc_conv_prob = np.clip(self._mc_conv_prob, 0.1, 0.9)

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
                    cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10) if len(eigenvalues) > 1 else 1.0
                    bootstrap_eigenvalue_samples.append(cond)
                except:
                    pass

            if len(bootstrap_eigenvalue_samples) > 5:
                bootstrap_cond_mean = np.mean(bootstrap_eigenvalue_samples)
                bootstrap_cond_std = np.std(bootstrap_eigenvalue_samples)
                bootstrap_uncertainty = bootstrap_cond_std / (bootstrap_cond_mean + 1e-10)
            else:
                bootstrap_cond_mean = 1.0
                bootstrap_uncertainty = 0.5

            uncertainty_explore = 1.0 + 0.4 * np.clip(bootstrap_uncertainty, 0.0, 2.0)

            if bootstrap_cond_mean > 10.0:
                cond_explore = 1.3
            elif bootstrap_cond_mean < 3.0:
                cond_explore = 0.85
            else:
                cond_explore = 1.0

            global_scale = uncertainty_explore * cond_explore
            global_scale = np.clip(global_scale, 0.5, 2.0)

            vel_scale = np.where(
                mc_conv_prob > 0.6,
                global_scale * 0.7,
                global_scale * 1.4
            )
            vel_scale = np.clip(vel_scale, 0.3, 2.5)

            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm

                n_strata = max(2, self.dim // 4)
                strat_idx = np.random.randint(0, n_strata, size=(self.np,))
                stratum_width = 1.0 / n_strata
                stratum_scale = (strat_idx + np.random.random(self.np)) * stratum_width

                directional = 0.4 * stratum_scale[:, np.newaxis] * to_best_dir
            else:
                directional = np.random.uniform(-0.5, 0.5, (self.np, self.dim))

            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

            if self._success_count is None:
                self._success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._success_count[improved] += 1
            self._success_count[~improved] *= 0.9
            success_norm = self._success_count / (np.max(self._success_count) + 1.0)

            random_scale = (1.0 - mc_conv_prob)[:, np.newaxis] * (1.0 - 0.3 * success_norm[:, np.newaxis])
            random_perturb = random_scale * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

            new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + random_perturb

        except Exception:
            new_population = self.population + 1.2 * self.velocity + np.random.uniform(-0.3, 0.3, (self.np, self.dim))

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_graph_spectral(self):
        """Graph spectral gap + temporal improvement rate (variant_08_catH, wins: 0, 13, 15)."""
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        visited = np.zeros(self.np, dtype=bool)
        components = []
        for start in range(self.np):
            if visited[start]:
                continue
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for neighbor in knn_indices[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            components.append(component)

        component_sizes = np.array([len(c) for c in components])
        particle_component_size = np.array([component_sizes[np.argmax([i in c for c in components])] for i in range(self.np)])

        try:
            row_indices = np.repeat(np.arange(self.np), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
            adj = adj + adj.T
            adj.data[:] = 1.0

            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
            lap = dspy.identity(self.np) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)

            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
        except:
            spectral_gap = 1.0

        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(float(np.min(self.current_fitness)))
        if len(self._fitness_history) > 20:
            self._fitness_history.pop(0)

        if len(self._fitness_history) >= 5:
            recent = np.array(self._fitness_history[-5:])
            oldest, newest = recent[0], recent[-1]
            if abs(oldest) > 1e-10:
                improvement_rate = (oldest - newest) / (abs(oldest) + 1e-10)
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = 0.0

        norm_improvement = np.clip(improvement_rate, 0.0, 1.0)

        stagnation_score = (1.0 - spectral_gap) * (1.0 - norm_improvement)
        stagnation_score = np.clip(stagnation_score, 0.0, 1.0)

        is_exploring = stagnation_score > 0.3
        is_stuck = stagnation_score > 0.6

        if is_stuck:
            global_scale = 1.5
        elif is_exploring:
            global_scale = 1.15
        else:
            global_scale = 0.85

        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        if self._success_count is None:
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        vel_scale = global_scale * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        random_scale = 0.5 if is_exploring else 0.25
        random_perturb = np.random.uniform(-random_scale, random_scale, (self.np, self.dim))

        size_factor = particle_component_size / max(1, self.np)
        component_nudge = np.zeros((self.np, self.dim))
        for comp_idx, comp in enumerate(components):
            if len(comp) < self.np * 0.3:
                centroid = np.mean(self.population[comp], axis=0)
                for p_idx in comp:
                    to_centroid = centroid - self.population[p_idx]
                    to_centroid_norm = np.linalg.norm(to_centroid) + 1e-10
                    component_nudge[p_idx] = 0.2 * (1.0 - size_factor[p_idx]) * (to_centroid / to_centroid_norm)

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb + component_nudge
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_geo_bounding(self):
        """Axis-aligned bounding box + convex hull + pairwise distance (variant_09_catA, wins: 3, 6, 23)."""
        pop = self.population
        np_particles, dim = pop.shape

        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        pop_range = pop_max - pop_min + 1e-10
        global_range = self.upper_bound - self.lower_bound

        centroid = np.mean(pop, axis=0)
        particle_to_centroid = pop - centroid
        normalized_pos = 2.0 * (pop - pop_min) / pop_range - 1.0

        edge_margin = 0.1
        near_edge = np.max(np.abs(normalized_pos), axis=1) > (1.0 - edge_margin)
        edge_pressure = np.sum(near_edge) / np_particles

        spread_ratios = pop_range / (global_range + 1e-10)
        min_spread = np.min(spread_ratios)
        max_spread = np.max(spread_ratios)
        anisotropy = max_spread / (min_spread + 1e-10) if min_spread > 1e-10 else 1.0

        try:
            if dim <= 10 and np_particles >= dim + 1:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pop, qhull_options='Qt')
                hull_volume = hull.volume
            else:
                hull_volume = np.prod(pop_range + 1e-10)
        except:
            hull_volume = np.prod(pop_range + 1e-10)

        max_volume = (global_range ** dim)
        normalized_volume = hull_volume / (max_volume + 1e-10)
        normalized_volume = np.clip(normalized_volume, 1e-10, 1.0)

        sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        all_dists = np.sqrt(sq_dists[sq_dists < np.inf])

        if len(all_dists) > 0:
            mean_dist = np.mean(all_dists)
            std_dist = np.std(all_dists)
            min_dist = np.min(all_dists)

            expected_uniform_dist = np.mean(pop_range) * 0.5
            clustering_ratio = mean_dist / (expected_uniform_dist + 1e-10)
        else:
            mean_dist = 1.0
            std_dist = 1.0
            min_dist = 1.0
            clustering_ratio = 1.0

        collapse_explore = 1.0 + 1.5 * (1.0 - normalized_volume)
        collapse_explore = np.clip(collapse_explore, 0.5, 3.0)

        anisotropy_explore = 1.0 + 0.3 * np.log1p(anisotropy) / np.log1p(100.0)
        anisotropy_explore = np.clip(anisotropy_explore, 0.8, 2.0)

        inward_pressure = edge_pressure * 0.5

        spread_explore = 1.0 + 0.5 * (1.0 - np.clip(std_dist / (mean_dist + 1e-10), 0.0, 1.0))
        spread_explore = np.clip(spread_explore, 0.5, 2.5)

        geo_explore = collapse_explore * anisotropy_explore * spread_explore
        geo_explore = np.clip(geo_explore, 0.3, 4.0)

        edge_dist = 1.0 - np.max(np.abs(normalized_pos), axis=1)
        inward_strength = inward_pressure * (1.0 - edge_dist)[:, np.newaxis]
        to_centroid = centroid - pop
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        inward_nudge = inward_strength * (to_centroid / to_centroid_dist)

        try:
            centered = pop - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)

            inv_spread = 1.0 / (eigenvalues / (eigenvalues[0] + 1e-10) + 0.01)
            inv_spread = inv_spread / (np.max(inv_spread) + 1e-10)

            vel_proj = self.velocity @ eigenvectors
            vel_corrected = vel_proj * inv_spread
            anisotropic_correction = (vel_corrected @ eigenvectors.T) - self.velocity
        except:
            anisotropic_correction = np.zeros_like(self.velocity)

        random_perturb = geo_explore * np.random.uniform(-0.5, 0.5, (np_particles, dim))

        new_population = pop + self.velocity * geo_explore + anisotropic_correction + inward_nudge + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_eff_rank(self):
        """Covariance-eigenvalue effective-rank velocity modulation (variant_10_catB, wins: 2, 4, 11)."""
        centered = self.population - np.mean(self.population, axis=0)

        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            total_eig = np.sum(eigenvalues)
            eigenvalues_norm = eigenvalues / (total_eig + 1e-10)
            eigenvalues_norm = np.clip(eigenvalues_norm, 1e-15, None)
            entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-15))
            max_entropy = np.log(len(eigenvalues_norm))
            effective_rank = np.exp(entropy) / (len(eigenvalues_norm) + 1e-10)
            effective_rank = np.clip(effective_rank, 0.0, 1.0)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)
            cond_signal = np.log1p(cond) / np.log1p(10000.0)

            eig_ratio = eigenvalues / (eigenvalues[0] + 1e-10)
            eig_ratio = np.clip(eig_ratio, 0.0, 1.0)

            spectral_scale = np.sqrt(eig_ratio + 0.01)
            spectral_scale = spectral_scale / (np.max(spectral_scale) + 1e-10)

            uniform_scale = 1.0
            blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            per_component_scale = uniform_scale * (1.0 - blend) + spectral_scale * blend

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_component_scale
            velocity_correction = vel_scaled @ eigenvectors.T - self.velocity

            if effective_rank < 0.3:
                explore_boost = 1.5
            elif effective_rank > 0.7:
                explore_boost = 1.2
            else:
                explore_boost = 1.0

            random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
            random_perturb *= explore_boost * (1.0 + 0.5 * cond_signal)

            new_population = self.population + self.velocity + 0.5 * velocity_correction + random_perturb

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        dispatch_map = {
            'original': self._position_update_original,
            'eigenvalue_aniso': self._position_update_eigenvalue_aniso,
            'fitness_rank': self._position_update_fitness_rank,
            'mc_conv': self._position_update_mc_conv,
            'graph_spectral': self._position_update_graph_spectral,
            'geo_bounding': self._position_update_geo_bounding,
            'eff_rank': self._position_update_eff_rank,
        }
        
        method = dispatch_map.get(operator, self._position_update_original)
        method()
    
    # -------------------------------------------------------------------------
    # RESTART & LOCAL BEST
    # -------------------------------------------------------------------------
    
    def _restart_if_stagnant(self):
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
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with THOMPSON SAMPLING BANDIT position-update selection.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._current_func = func
        
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        for op in self._operators:
            self._op_rewards[op] = []
            self._op_total_reward[op] = 0.0
            self._op_count[op] = 0
        self._total_selections = 0
        self._gen_best_history = []
        self._prev_global_best_fitness = np.inf
        
        self._mc_conv_prob = np.ones(self.np) * 0.5
        self._prev_fitness = None
        if hasattr(self, '_fitness_history'):
            self._fitness_history = []
        
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
            self._success_count = np.zeros(self.np)
            self._mc_conv_prob = np.ones(self.np) * 0.5
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Select operator using Thompson Sampling
            current_operator = self._select_operator_thompson()
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
            self._velocity_update_base()
            self._dispatch_position_update(current_operator)
            
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
            
            # Record rank-based reward for the selected operator
            self._record_reward(current_operator, self.global_best_fitness)
            
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
