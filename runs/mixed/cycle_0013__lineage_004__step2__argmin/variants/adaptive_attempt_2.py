import numpy as np


class CulturalDEWithAdaptiveArchive:
    """
    Cultural Differential Evolution with Adaptive Archive, Velocity-Guided
    Mutation, Targeted Local Search, and Adaptive Operator Selection for _argmin.
    
    Key innovations:
    - Current-to-pbest/mean mutation with cultural guidance
    - Velocity-based momentum for exploration continuity
    - Adaptive F/CR based on exponential moving average success
    - Bounded archive for diversity maintenance
    - Stagnation-triggered local search on best candidate
    - Diversity-aware restart mechanism
    - Adaptive _argmin operator selection via Thompson Sampling
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
        
        # Adaptive operator selection for _argmin
        self._num_operators = 8
        self._op_alpha = np.ones(self._num_operators)  # Beta prior alpha
        self._op_beta = np.ones(self._num_operators)   # Beta prior beta
        self._op_attempts = np.zeros(self._num_operators, dtype=float)
        self._op_successes = np.zeros(self._num_operators, dtype=float)
        self._op_rewards = [[] for _ in range(self._num_operators)]
        self._window_size = 20
        self._selected_op = 0
        self._current_pop = None
        self._last_pop = None
        self._generation = 0
        
        # Temperature for variant_08
        self._argmin_temp = 0.5
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        self._current_pop = population.copy()
        fitness = self._eval_wrapper(func, population)
        archive = self._init_archive(population, fitness)
        velocity = self._init_velocity()
        F, CR = self._init_parameters()
        F_ema, CR_ema = F, CR
        success_F, success_CR = [], []
        best_idx = self._argmin(fitness)
        f_best, x_best = float(fitness[best_idx]), population[best_idx].copy()
        stagnation = 0
        gen = 0
        self._generation = 0
        
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
            new_f_best = float(fitness[new_best_idx])
            
            # Update operator selection based on improvement
            if new_f_best < f_best - 1e-12:
                improvement = (f_best - new_f_best) / (abs(f_best) + 1e-10)
                self._update_operator_reward(self._selected_op, min(improvement, 1.0))
                f_best = new_f_best
                x_best = population[new_best_idx].copy()
                stagnation = 0
            else:
                self._update_operator_reward(self._selected_op, 0.0)
                stagnation += 1
            
            self._last_pop = self._current_pop.copy()
            self._current_pop = population.copy()
            velocity = self._update_velocity_batch(population, velocity, x_best)
            
            if stagnation >= self.local_search_interval:
                x_best, f_best = self._apply_adaptive_local_search(
                    func, x_best, f_best, self.local_search_radius
                )
                stagnation = 0
            
            if self._compute_diversity(population) < self.diversity_threshold:
                population = self._restart_if_stagnant(population, fitness, x_best)
                self._current_pop = population.copy()
                velocity = self._init_velocity()
            
            gen += 1
            self._generation = gen
        
        return float(f_best), x_best.copy()
    
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
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling with Beta distribution."""
        samples = np.array([
            np.random.beta(self._op_alpha[i], self._op_beta[i]) 
            for i in range(self._num_operators)
        ])
        selected = int(np.argmax(samples))
        self._selected_op = selected
        return selected
    
    def _update_operator_reward(self, op_idx, reward):
        """Update operator statistics with sliding window credit assignment."""
        reward = float(reward)
        self._op_attempts[op_idx] += 1.0
        self._op_rewards[op_idx].append(reward)
        
        # Maintain sliding window
        if len(self._op_rewards[op_idx]) > self._window_size:
            self._op_rewards[op_idx] = self._op_rewards[op_idx][-self._window_size:]
        
        # Update Beta distribution parameters
        if len(self._op_rewards[op_idx]) >= 5:
            mean_reward = np.mean(self._op_rewards[op_idx])
            self._op_successes[op_idx] = mean_reward
            # Update Beta parameters (add-prior smoothing)
            self._op_alpha[op_idx] = 1.0 + mean_reward * 10.0
            self._op_beta[op_idx] = 1.0 + (1.0 - mean_reward) * 10.0
    
    def _argmin(self, arr):
        """Adaptive argmin that selects best strategy using Thompson Sampling."""
        op_idx = self._select_operator_thompson()
        
        if op_idx == 0:
            return self._argmin_v1_nanargmin(arr)
        elif op_idx == 1:
            return self._argmin_v2_median_based(arr)
        elif op_idx == 2:
            return self._argmin_v3_diversity_fitness(arr)
        elif op_idx == 3:
            return self._argmin_v4_softmax_temp(arr)
        elif op_idx == 4:
            return self._argmin_v5_rank_based(arr)
        elif op_idx == 5:
            return self._argmin_v6_adaptive_composite(arr)
        elif op_idx == 6:
            return self._argmin_v7_exp_fitness_diversity(arr)
        else:
            return self._argmin_v8_ucb_biased(arr)
    
    def _argmin_v1_nanargmin(self, arr):
        """Variant 01: Simple nanargmin approach."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        idx = np.nanargmin(arr)
        return int(idx)
    
    def _argmin_v2_median_based(self, arr):
        """Variant 03: Median-based selection with diversity."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        median_val = np.median(arr)
        median_mask = arr <= median_val
        
        if np.any(median_mask):
            candidates = np.where(median_mask)[0]
            if len(candidates) == 1:
                return int(candidates[0])
            sub_arr = arr[candidates]
            best_sub_idx = int(np.argmin(sub_arr))
            return int(candidates[best_sub_idx])
        else:
            return int(np.argmin(arr))
    
    def _argmin_v3_diversity_fitness(self, arr):
        """Variant 04: Diversity and fitness composite with random exploration."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        centroid = np.mean(arr)
        distances = np.abs(arr - centroid)
        max_dist = np.max(distances) + 1e-15
        diversity = distances / max_dist
        
        arr_min, arr_max = np.min(arr), np.max(arr)
        arr_range = arr_max - arr_min + 1e-15
        fitness_norm = 1.0 - (arr - arr_min) / arr_range
        
        composite = 0.7 * fitness_norm + 0.3 * diversity
        
        if np.random.random() < 0.05:
            return int(np.random.randint(len(arr)))
        
        return int(np.argmax(composite))
    
    def _argmin_v4_softmax_temp(self, arr):
        """Variant 07: Softmax with adaptive temperature."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        best_idx = int(np.argmin(arr))
        best_val = arr[best_idx]
        
        perf_diff = arr - best_val
        max_diff = np.max(perf_diff)
        
        if max_diff < 1e-15:
            return best_idx
        
        norm_perf = perf_diff / (max_diff + 1e-15)
        variance = np.var(arr)
        temp = 0.1 + 0.5 * np.tanh(variance * 1e-6)
        
        exponent = -norm_perf / (temp + 1e-15)
        exponent -= np.max(exponent)
        probs = np.exp(exponent)
        probs = probs / (np.sum(probs) + 1e-15)
        
        if np.random.random() < 0.7:
            return best_idx
        else:
            return int(np.random.choice(len(arr), p=probs))
    
    def _argmin_v5_rank_based(self, arr):
        """Variant 08: Rank-based selection with adaptive temperature."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        fitness_std = np.std(arr)
        if fitness_std < 1e-6:
            self._argmin_temp = min(3.0, self._argmin_temp * 1.2)
        else:
            self._argmin_temp = max(0.1, self._argmin_temp * 0.95)
        
        ranks = np.argsort(np.argsort(arr))
        n = len(arr)
        
        weights = np.exp(-ranks / self._argmin_temp)
        probs = weights / np.sum(weights)
        
        selected_idx = np.random.choice(n, p=probs)
        return int(selected_idx)
    
    def _argmin_v6_adaptive_composite(self, arr):
        """Variant 06: Adaptive composite with probabilistic top-k selection."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        fitness_ranks = np.argsort(np.argsort(arr))
        fitness_score = 1.0 - fitness_ranks / (len(arr) - 1)
        
        pop = getattr(self, '_current_pop', None)
        if pop is not None and len(pop) == len(arr):
            centroid = np.mean(pop, axis=0)
            dists = np.linalg.norm(pop - centroid, axis=1)
            max_dist = np.max(dists) + 1e-15
            diversity_score = dists / max_dist
        else:
            diversity_score = np.ones(len(arr)) * 0.5
        
        gen = getattr(self, '_generation', 0)
        adapt = min(0.6 + 0.3 * np.tanh(gen / 100), 0.9)
        
        composite = (adapt * fitness_score + (1 - adapt) * 0.5 * diversity_score)
        
        k = max(3, len(arr) // 4)
        top_indices = np.argsort(composite)[-k:]
        top_probs = composite[top_indices]
        top_probs = np.clip(top_probs, 1e-15, None)
        top_probs = top_probs / np.sum(top_probs)
        
        return int(np.random.choice(top_indices, p=top_probs))
    
    def _argmin_v7_exp_fitness_diversity(self, arr):
        """Variant 09: Exponential fitness with diversity scoring."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        best_idx = int(np.argmin(arr))
        best_val = float(arr[best_idx])
        worst_val = float(np.max(arr))
        fit_range = worst_val - best_val + 1e-30
        fit_score = np.exp(-2.0 * (arr - best_val) / fit_range)
        
        try:
            pop = getattr(self, '_last_pop', None)
            if pop is not None and len(pop) == len(arr):
                dists = np.linalg.norm(pop - pop[best_idx], axis=1)
                max_dist = np.max(dists) + 1e-30
                div_score = dists / max_dist
            else:
                div_score = np.ones(len(arr)) * 0.5
        except:
            div_score = np.ones(len(arr)) * 0.5
        
        composite = 0.7 * fit_score + 0.3 * div_score
        
        return int(np.argmax(composite))
    
    def _argmin_v8_ucb_biased(self, arr):
        """Fallback: UCB-inspired selection favoring unexplored operators."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0
        if arr.size == 1:
            return 0
        
        # Compute UCB scores for each index
        n = len(arr)
        ucb_scores = np.zeros(n)
        
        for i in range(n):
            # Exploitation term
            exploit = -arr[i]  # Negative because lower is better
            # Exploration term (favor diverse solutions)
            centroid = np.mean(arr)
            explore = np.abs(arr[i] - centroid) / (np.max(np.abs(arr - centroid)) + 1e-15)
            ucb_scores[i] = 0.7 * exploit + 0.3 * explore
        
        # Add small random noise for exploration
        ucb_scores += np.random.uniform(0, 0.01, n)
        
        return int(np.argmax(ucb_scores))
    
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
            success_F.append(float(F_i))
            success_CR.append(float(CR_i))
        
        new_archive = self._update_archive_batch(archive, new_pop[improve_idx], 
                                                  new_fit[improve_idx])
        
        if len(success_F) > 20:
            trim = len(success_F) - 20
            success_F = success_F[trim:]
            success_CR = success_CR[trim:]
        
        return new_pop, new_fit, new_archive, success_F, success_CR
    
    def _update_archive_batch(self, archive, new_solutions, new_fitness):
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
        current_f = float(f_best)
        
        for _ in range(3):
            step = np.random.uniform(-radius, radius, self.dim)
            candidate = self._clip_to_bounds_batch(current + step)
            cand_f = float(self._eval_wrapper(func, candidate.reshape(1, -1))[0])
            
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
