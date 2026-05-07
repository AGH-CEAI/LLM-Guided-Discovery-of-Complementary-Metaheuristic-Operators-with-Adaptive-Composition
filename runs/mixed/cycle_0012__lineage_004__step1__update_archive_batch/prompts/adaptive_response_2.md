```python
import numpy as np


class CulturalDEWithAdaptiveArchive:
    """
    Cultural Differential Evolution with Adaptive Archive, Velocity-Guided
    Mutation, and Targeted Local Search.
    
    Key innovations:
    - Current-to-pbest/mean mutation with cultural guidance
    - Velocity-based momentum for exploration continuity
    - Adaptive F/CR based on exponential moving average success
    - Bounded archive for diversity maintenance
    - Stagnation-triggered local search on best candidate
    - Diversity-aware restart mechanism
    - Adaptive operator selection using Multi-Armed Bandit with sliding window
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(4 * dim, 60), 300)
        self.lower = -100.0
        self.upper = 100.0
        self.p_best_frac = 0.15
        self.archive_max = self.NP // 2
        self.F_init = 0.6
        self.CR_init = 0.85
        self.F_min, self.F_max = 0.3, 1.2
        self.CR_min, self.CR_max = 0.3, 0.95
        self.local_search_interval = 50
        self.local_search_radius = 0.1
        self.diversity_threshold = 1e-6
        self.stagnation_limit = 80
        
        # Adaptive operator selection (Multi-Armed Bandit with UCB)
        self.num_operators = 7
        self.operator_names = ['v01', 'v02', 'v03', 'v05', 'v06', 'v09', 'v10']
        self.operator_rewards = np.zeros(self.num_operators)
        self.operator_counts = np.zeros(self.num_operators)
        self.sliding_window_size = 15
        self.operator_history = [[] for _ in range(self.num_operators)]
        self.exploration_constant = 1.0
        self.current_operator = 0
        self.total_selections = 0.5  # Start slightly above 0 for UCB
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = self._eval_wrapper(func, population)
        archive = self._init_archive(population, fitness)
        velocity = self._init_velocity()
        F, CR = self._init_parameters()
        F_ema, CR_ema = F, CR
        success_F, success_CR = [], []
        best_idx = self._argmin(fitness)
        f_best, x_best = fitness[best_idx], population[best_idx].copy()
        stagnation = 0
        gen = 0
        
        while not stopping_condition():
            trials, velocity = self._build_trials_batched(population, velocity, F, CR, archive, fitness)
            trial_fit = self._eval_wrapper(func, trials)
            
            if len(trial_fit) < len(trials):
                valid = len(trial_fit)
                trials, trial_fit = trials[:valid], trial_fit[:valid]
                if valid == 0:
                    break
            
            if stopping_condition():
                break
            
            population, fitness, archive, success_F, success_CR = self._select_survivors_batch(
                population, fitness, trials, trial_fit, archive, success_F, success_CR
            )
            
            F, CR, F_ema, CR_ema = self._adapt_parameters_batch(
                F, CR, F_ema, CR_ema, success_F, success_CR
            )
            
            new_best_idx = self._argmin(fitness)
            if fitness[new_best_idx] < f_best - 1e-12:
                f_best, x_best = fitness[new_best_idx], population[new_best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1
            
            velocity = self._update_velocity_batch(population, velocity, x_best)
            
            if stagnation >= self.local_search_interval:
                x_best, f_best = self._apply_adaptive_local_search(
                    func, x_best, f_best, self.local_search_radius
                )
                stagnation = 0
            
            if self._compute_diversity(population) < self.diversity_threshold:
                population = self._restart_if_stagnant(population, fitness, x_best)
                velocity = self._init_velocity()
            
            gen += 1
        
        return f_best, x_best
    
    def _select_operator_ucb(self):
        """Select operator using Upper Confidence Bound (UCB) with softmax exploration."""
        if self.total_selections < 2:
            self.current_operator = 0
            self.operator_counts[0] += 1
            self.total_selections += 1
            return
        
        ucb_scores = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if self.operator_counts[i] > 0:
                avg_reward = self.operator_rewards[i] / max(1e-9, self.operator_counts[i])
            else:
                avg_reward = 0.0
            exploration_bonus = self.exploration_constant * np.sqrt(
                np.log(self.total_selections) / max(1e-9, self.operator_counts[i])
            )
            ucb_scores[i] = avg_reward + exploration_bonus
        
        # Softmax with temperature for probabilistic selection
        temperature = 0.5 + 0.3 * min(1.0, gen / 100) if hasattr(self, 'gen') else 0.5
        exp_scores = np.exp((ucb_scores - np.max(ucb_scores)) / max(temperature, 1e-3))
        probs = exp_scores / np.sum(exp_scores)
        
        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        self.total_selections += 1
    
    def _update_operator_reward(self, reward):
        """Update reward for current operator with sliding window."""
        idx = self.current_operator
        reward = float(np.clip(reward, -1.0, 1.0))
        self.operator_history[idx].append(reward)
        if len(self.operator_history[idx]) > self.sliding_window_size:
            self.operator_history[idx] = self.operator_history[idx][-self.sliding_window_size:]
        self.operator_rewards[idx] = float(np.mean(self.operator_history[idx]))
    
    # =========================================================================
    # All 7 winning archive update strategies
    # =========================================================================
    
    def _update_archive_batch(self, archive, new_solutions, new_fitness):
        """Adaptive archive update using Multi-Armed Bandit selection."""
        if len(new_solutions) == 0:
            return archive
        
        self._select_operator_ucb()
        old_archive = archive.copy() if len(archive) > 0 else np.array([]).reshape(0, self.dim)
        
        operators = [
            self._update_archive_v01,
            self._update_archive_v02,
            self._update_archive_v03,
            self._update_archive_v05,
            self._update_archive_v06,
            self._update_archive_v09,
            self._update_archive_v10,
        ]
        
        new_archive = operators[self.current_operator](archive, new_solutions, new_fitness)
        
        # Compute reward based on archive improvement
        reward = self._compute_archive_reward(old_archive, new_archive)
        self._update_operator_reward(reward)
        
        return new_archive
    
    def _compute_archive_reward(self, old_archive, new_archive):
        """Compute reward based on improvement in archive quality."""
        if len(new_archive) == 0:
            return -0.5
        
        if len(old_archive) == 0:
            return 0.3
        
        old_best = np.min([np.min(old_archive), np.min(new_archive)])
        new_best = np.min(new_archive)
        
        # Bounded relative improvement
        denom = max(abs(old_best), 1e-8, abs(new_best))
        improvement = (old_best - new_best) / denom
        reward = float(np.clip(improvement * 0.5, -0.5, 0.5))
        
        return reward
    
    def _update_archive_v01(self, archive, new_solutions, new_fitness):
        """Variant 01: Hypergrid with diversity bonus (4 wins)."""
        if len(new_solutions) == 0:
            return archive
        
        n_cells = max(2, int(np.sqrt(self.archive_max)))
        cell_capacity = max(1, self.archive_max // (n_cells ** self.dim) if self.dim <= 5 else 1)
        
        def normalize(solutions):
            range_val = self.upper - self.lower
            if range_val == 0:
                return np.zeros_like(solutions)
            return (solutions - self.lower) / range_val
        
        current_sols = archive
        current_fits = np.array([-np.inf] * len(archive)) if len(archive) > 0 else np.array([])
        
        all_solutions = np.vstack([current_sols, new_solutions]) if len(current_sols) > 0 else new_solutions
        all_fitness = np.concatenate([current_fits, new_fitness]) if len(current_fits) > 0 else new_fitness
        
        if len(all_solutions) == 0:
            return archive
        
        normalized = normalize(all_solutions)
        normalized = np.clip(normalized, 0.0, 0.9999)
        grid_indices = (normalized * n_cells).astype(int)
        
        grid = {}
        for idx in range(len(all_solutions)):
            key = tuple(grid_indices[idx])
            if key not in grid:
                grid[key] = []
            grid[key].append((all_fitness[idx], idx))
        
        selected_indices = []
        cell_best_fits = {}
        
        for key, members in grid.items():
            members_sorted = sorted(members, key=lambda x: x[0])
            keep_count = min(len(members_sorted), cell_capacity)
            
            for rank, (fit, sol_idx) in enumerate(members_sorted[:keep_count]):
                selected_indices.append(sol_idx)
                cell_best_fits[sol_idx] = fit - rank * 1e-10
        
        if len(selected_indices) > self.archive_max:
            selected_fitness = np.array([cell_best_fits[i] for i in selected_indices])
            keep_top = np.argsort(selected_fitness)[-self.archive_max:]
            selected_indices = [selected_indices[i] for i in keep_top]
        
        return all_solutions[selected_indices]
    
    def _update_archive_v02(self, archive, new_solutions, new_fitness):
        """Variant 02: Elite + greedy diversity selection (1 win)."""
        if len(new_solutions) == 0:
            return archive
        
        combined = np.vstack([archive, new_solutions])
        n_combined = len(combined)
        
        all_fitness = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        all_fitness = all_fitness[unique_mask]
        
        n_unique = len(combined)
        if n_unique == 0:
            return archive
        
        new_mask = np.arange(len(combined)) >= len(archive)
        new_combined = combined[new_mask]
        new_fitness_filtered = all_fitness[new_mask]
        
        n_elite = min(self.archive_max // 2, len(new_combined))
        elite_indices = np.argsort(new_fitness_filtered)[:n_elite]
        elite_solutions = new_combined[elite_indices]
        
        n_diversity = self.archive_max - n_elite
        remaining = combined[~new_mask] if len(archive) > 0 else combined
        
        if len(remaining) == 0:
            return elite_solutions
        
        selected = list(elite_solutions)
        remaining_list = list(remaining)
        diversity_solutions = []
        
        for _ in range(min(n_diversity, len(remaining_list))):
            if len(remaining_list) == 0:
                break
            dists = np.array([min(np.linalg.norm(p - s) for s in selected) 
                             if selected else np.linalg.norm(p) 
                             for p in remaining_list])
            farthest_idx = np.argmax(dists)
            diversity_solutions.append(remaining_list.pop(farthest_idx))
        
        if len(diversity_solutions) == 0:
            return elite_solutions
        
        return np.vstack([elite_solutions, np.array(diversity_solutions)])
    
    def _update_archive_v03(self, archive, new_solutions, new_fitness):
        """Variant 03: Fitness-distance Pareto selection (2 wins)."""
        if len(new_solutions) == 0:
            return archive
        
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        if len(combined) <= self.archive_max:
            return combined
        
        n_keep = self.archive_max
        sorted_indices = np.argsort(combined_fit)
        
        selected = []
        selected_vectors = []
        
        for idx in sorted_indices:
            if len(selected) >= n_keep:
                break
            
            if len(selected_vectors) == 0:
                selected.append(idx)
                selected_vectors.append(combined[idx])
                continue
            
            diffs = combined[idx] - np.array(selected_vectors)
            min_dist = np.min(np.sqrt(np.sum(diffs**2, axis=1)))
            
            dist_threshold = 1e-2 * self.dim + 1e-3
            if min_dist >= dist_threshold or len(selected) < n_keep // 3:
                selected.append(idx)
                selected_vectors.append(combined[idx])
        
        if len(selected) < n_keep:
            remaining = [i for i in sorted_indices if i not in selected]
            needed = n_keep - len(selected)
            selected.extend(remaining[:needed])
        
        return combined[selected]
    
    def _update_archive_v05(self, archive, new_solutions, new_fitness):
        """Variant 05: Diversity-weighted selection (1 win)."""
        if len(new_solutions) == 0:
            return archive
        
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        if len(combined) <= self.archive_max:
            return combined
        
        n_new = len(new_solutions)
        n_arch = len(combined) - n_new
        
        arch_best_mask = np.zeros(len(combined), dtype=bool)
        if n_arch > 0:
            arch_best_count = min(self.archive_max // 4, n_arch)
            arch_best_indices = np.argsort(combined_fit[:n_arch])[-arch_best_count:]
            arch_best_mask[arch_best_indices] = True
        
        new_best_mask = np.zeros(len(combined), dtype=bool)
        if n_new > 0:
            new_solutions_trimmed = combined[n_arch:]
            new_fitness_trimmed = combined_fit[n_arch:]
            
            if n_arch > 0 and arch_best_count > 0:
                arch_best_sols = combined[arch_best_indices]
                distances = np.linalg.norm(
                    new_solutions_trimmed[:, np.newaxis, :] - arch_best_sols[np.newaxis, :, :],
                    axis=2
                )
                min_distances = np.maximum(np.min(distances, axis=1), 1e-10)
                fitness_range = max(np.max(new_fitness_trimmed) - np.min(new_fitness_trimmed), 1e-10)
                norm_fitness = (new_fitness_trimmed - np.min(new_fitness_trimmed)) / fitness_range
                diversity_scores = 0.4 * norm_fitness + 0.6 * (min_distances / (min_distances + 1.0))
            else:
                fitness_range = max(np.max(new_fitness_trimmed) - np.min(new_fitness_trimmed), 1e-10)
                diversity_scores = (new_fitness_trimmed - np.min(new_fitness_trimmed)) / fitness_range
            
            new_best_count = self.archive_max - arch_best_count
            if n_new <= new_best_count:
                new_best_mask[n_arch:] = True
            else:
                new_best_indices = np.argsort(diversity_scores)[-new_best_count:]
                new_best_mask[n_arch + new_best_indices] = True
        
        selected_mask = arch_best_mask
        if n_new > 0:
            selected_mask = np.logical_or(selected_mask, new_best_mask)
        
        return combined[selected_mask]
    
    def _update_archive_v06(self, archive, new_solutions, new_fitness):
        """Variant 06: Fitness-distance clustering (1 win)."""
        if len(new_solutions) == 0:
            return archive
        
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        if len(combined) <= self.archive_max:
            return combined
        
        n_keep = self.archive_max
        n_combined = len(combined)
        
        fit_min, fit_max = combined_fit.min(), combined_fit.max()
        fit_range = fit_max - fit_min + 1e-20
        norm_fitness = (combined_fit - fit_min) / fit_range
        
        centroid = np.mean(combined, axis=0)
        distances = np.linalg.norm(combined - centroid, axis=1)
        dist_max = distances.max() + 1e-20
        norm_dist = distances / dist_max
        
        scores = np.sqrt(norm_fitness ** 2 + (1 - norm_dist) ** 2)
        
        selected_indices = []
        remaining = np.arange(n_combined)
        
        best_idx = np.argmin(combined_fit)
        selected_indices.append(best_idx)
        remaining = np.delete(remaining, np.where(remaining == best_idx)[0])
        
        while len(selected_indices) < n_keep and len(remaining) > 0:
            selected_arr = np.array(selected_indices)
            min_dists = np.zeros(len(remaining))
            for i, rem_idx in enumerate(remaining):
                rem_point = combined[rem_idx]
                rem_fit = combined_fit[rem_idx]
                sol_dists = np.linalg.norm(combined[selected_arr] - rem_point, axis=1)
                fit_diffs = np.abs(combined_fit[selected_arr] - rem_fit) / (fit_range + 1e-20)
                min_dists[i] = np.sqrt(np.min(sol_dists) ** 2 + np.min(fit_diffs) ** 2)
            
            weighted = min_dists * (1.0 + scores[remaining])
            pick_pos = np.argmax(weighted)
            pick_idx = remaining[pick_pos]
            selected_indices.append(pick_idx)
            remaining = np.delete(remaining, pick_pos)
        
        return combined[np.array(selected_indices)]
    
    def _update_archive_v09(self, archive, new_solutions, new_fitness):
        """Variant 09: Crowding distance selection (2 wins)."""
        if len(new_solutions) == 0:
            return archive
        
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([
            np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
            new_fitness
        ])
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        if len(combined) <= self.archive_max:
            return combined
        
        n = len(combined)
        crowding = np.zeros(n)
        
        for i in range(n):
            distances = np.abs(combined_fit - combined_fit[i])
            distances[i] = np.inf
            nearest_idx = np.argmin(distances)
            crowding[i] = distances[nearest_idx] if np.isfinite(distances[nearest_idx]) else 0.0
        
        fit_range = combined_fit.max() - combined_fit.min()
        fit_norm = np.zeros(n) if fit_range <= 1e-15 else (combined_fit - combined_fit.min()) / fit_range
        
        crow_range = crowding.max() - crowding.min()
        crow_norm = np.zeros(n) if crow_range <= 1e-15 else (crowding - crowding.min()) / crow_range
        
        score = 0.5 * (1.0 - fit_norm) + 0.5 * crow_norm
        
        keep_count = min(self.archive_max, n)
        keep_indices = np.argsort(score)[-keep_count:]
        
        return combined[keep_indices]
    
    def _update_archive_v10(self, archive, new_solutions, new_fitness):
        """Variant 10: Nearest-neighbor diversity + fitness composite (1 win)."""
        if len(new_solutions) == 0:
            return archive
        
        if len(archive) > 0:
            combined = np.vstack([archive, new_solutions])
            combined_fit = np.concatenate([np.full(len(archive), -np.inf), new_fitness])
        else:
            combined = new_solutions
            combined_fit = new_fitness.copy()
        
        unique_mask = self._find_unique_solutions(combined)
        combined = combined[unique_mask]
        combined_fit = combined_fit[unique_mask]
        
        n = len(combined)
        if n <= self.archive_max:
            return combined
        
        n_combined = len(combined)
        min_dists = np.full(n_combined, np.inf)
        
        chunk_size = 500
        for i in range(0, n_combined, chunk_size):
            end_i = min(i + chunk_size, n_combined)
            chunk = combined[i:end_i]
            diffs = combined[np.newaxis, :, :] - chunk[:, np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists[i:i+chunk_size, :][:end_i-i], np.inf)
            min_dists[i:end_i] = np.min(dists, axis=1)
        
        fit_min, fit_max = np.min(combined_fit), np.max(combined_fit)
        fit_range = fit_max - fit_min + 1e-15
        fitness_score = 1.0 - (combined_fit - fit_min) / fit_range
        
        dist_max = np.max(min_dists) + 1e-15
        diversity_score = min_dists / dist_max
        
        composite = 0.7 * fitness_score + 0.3 * diversity_score
        
        top_indices = np.argsort(composite)[-self.archive_max:]
        
        return combined[top_indices]
    
    # =========================================================================
    # Original algorithm methods (unchanged signatures)
    # =========================================================================
    
    def _initialize_population(self):
        return np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
    
    def _init_archive(self, population, fitness):
        top_k = min(self.archive_max, self.NP // 4)
        top_indices = np.argpartition(fitness, top_k)[:top_k]
        return population[top_indices].copy()
    
    def _init_velocity(self):
        range_val = self.upper - self.lower
        return np.random.uniform(-0.1 * range_val, 0.1 * range_val, (self.NP, self.dim))
    
    def _init_parameters(self):
        return self.F_init, self.CR_init
    
    def _eval_wrapper(self, func, population):
        clipped = self._clip_to_bounds_batch(population)
        return func(clipped)
    
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.lower, self.upper)
    
    def _argmin(self, arr):
        return int(np.argmin(arr))
    
    def _build_trials_batched(self, population, velocity, F, CR, archive, fitness):
        mutated = self._mutate_current_to_pbest_with_culture(
            population, fitness, archive, F
        )
        vel_scaled = velocity * 0.1 * np.random.uniform(0.8, 1.2)
        mutated = mutated + vel_scaled
        trials = self._crossover_batch(population, mutated, CR)
        new_velocity = mutated - population
        return trials, new_velocity
    
    def _mutate_current_to_pbest_with_culture(self, population, fitness, archive, F):
        NP, dim = population.shape
        pbest_count = max(1, int(self.p_best_frac * NP))
        pbest_indices = np.argpartition(fitness, pbest_count)[:pbest_count]
        pbest = population[np.random.choice(pbest_indices)]
        
        r1_idx = np.random.choice(NP, NP, replace=True)
        r2_idx = np.random.choice(NP, NP, replace=True)
        while np.any(r1_idx == np.arange(NP)) or np.any(r2_idx == np.arange(NP)):
            r1_idx = np.random.choice(NP, NP, replace=True)
            r2_idx = np.random.choice(NP, NP, replace=True)
        
        r1, r2 = population[r1_idx], population[r2_idx]
        
        if len(archive) > 0:
            arch_idx = np.random.choice(len(archive), NP, replace=True)
            r3 = archive[arch_idx]
        else:
            r3 = population[np.random.choice(NP, NP, replace=True)]
        
        mean_target = np.mean(population, axis=0)
        cultural_weight = np.random.uniform(0.0, 0.3, (NP, 1))
        
        mutated = population + F * (pbest - population) + F * (r1 - r2) + F * cultural_weight * (mean_target - population)
        mutated = self._clip_to_bounds_batch(mutated)
        return mutated
    
    def _crossover_batch(self, target, donor, CR):
        NP, dim = target.shape
        j_rand = np.random.randint(0, dim, NP)
        mask = np.random.random((NP, dim)) < CR
        mask[np.arange(NP), j_rand] = True
        trial = np.where(mask, donor, target)
        return trial
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fit, 
                                 archive, success_F, success_CR):
        NP = len(population)
        improved = trial_fit < fitness
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        improve_idx = np.where(improved)[0]
        for idx in improve_idx:
            new_pop[idx] = trials[idx]
            new_fit[idx] = trial_fit[idx]
        
        for idx in improve_idx:
            F_i = np.random.uniform(self.F_min, self.F_max)
            CR_i = np.random.uniform(self.CR_min, self.CR_max)
            success_F.append(F_i)
            success_CR.append(CR_i)
        
        new_archive = self._update_archive_batch(archive, new_pop[improve_idx], 
                                                  new_fit[improve_idx])
        
        if len(success_F) > 20:
            trim = len(success_F) - 20
            success_F = success_F[trim:]
            success_CR = success_CR[trim:]
        
        return new_pop, new_fit, new_archive, success_F, success_CR
    
    def _find_unique_solutions(self, solutions, tol=1e-3):
        if len(solutions) <= 1:
            return np.ones(len(solutions), dtype=bool)
        
        n = len(solutions)
        is_unique = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_unique[i]:
                continue
            diffs = np.abs(solutions[i] - solutions[i+1:]) if i+1 < n else np.array([])
            if len(diffs) > 0 and np.any(np.all(diffs < tol, axis=1)):
                is_unique[i+1:] = False
        return is_unique
    
    def _adapt_parameters_batch(self, F, CR, F_ema, CR_ema, success_F, success_CR):
        if len(success_F) >= 5:
            F_ema = 0.9 * F_ema + 0.1 * np.mean(success_F[-10:])
            CR_ema = 0.9 * CR_ema + 0.1 * np.mean(success_CR[-10:])
            F = np.clip(F_ema + np.random.uniform(-0.1, 0.1), self.F_min, self.F_max)
            CR = np.clip(CR_ema + np.random.uniform(-0.1, 0.1), self.CR_min, self.CR_max)
        return F, CR, F_ema, CR_ema
    
    def _update_velocity_batch(self, population, velocity, x_best):
        NP = len(population)
        inertia = 0.7
        cognitive = 1.5
        social = 1.5
        
        r1 = np.random.uniform(0, 1, (NP, self.dim))
        r2 = np.random.uniform(0, 1, (NP, self.dim))
        
        new_vel = inertia * velocity + \
                  cognitive * r1 * (x_best - population) + \
                  social * r2 * (np.mean(population, axis=0) - population)
        
        max_vel = (self.upper - self.lower) * 0.2
        new_vel = np.clip(new_vel, -max_vel, max_vel)
        return new_vel
    
    def _apply_adaptive_local_search(self, func, x_best, f_best, radius):
        history = [x_best.copy()]
        current = x_best.copy()
        current_f = f_best
        
        for _ in range(3):
            step = np.random.uniform(-radius, radius, self.dim)
            candidate = self._clip_to_bounds_batch(current + step)
            cand_f = self._eval_wrapper(func, candidate.reshape(1, -1))[0]
            
            if cand_f < current_f:
                current = candidate
                current_f = cand_f
                radius *= 1.2
            else:
                radius *= 0.5
            
            history.append(current.copy())
        
        if current_f < f_best:
            return current, current_f
        return x_best, f_best
    
    def _compute_diversity(self, population):
        if len(population) < 2:
            return 1.0
        
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return float(np.mean(distances))
    
    def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        worst_count = NP // 3
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        for idx in worst_indices:
            new_pop[idx] = x_best + np.random.uniform(-10, 10, self.dim)
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```