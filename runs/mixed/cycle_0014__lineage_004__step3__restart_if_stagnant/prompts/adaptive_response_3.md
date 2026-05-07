```python
import numpy as np


class CulturalDEWithAdaptiveArchive:
    """
    Cultural Differential Evolution with Adaptive Archive, Velocity-Guided
    Mutation, Targeted Local Search, and Adaptive Restart Strategy Selection.
    
    Key innovations:
    - Current-to-pbest/mean mutation with cultural guidance
    - Velocity-based momentum for exploration continuity
    - Adaptive F/CR based on exponential moving average success
    - Bounded archive for diversity maintenance
    - Stagnation-triggered local search on best candidate
    - Adaptive restart mechanism using Thompson Sampling Multi-Armed Bandit
      to automatically select the best restart strategy from 7 candidates
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
        
        # Adaptive restart bandit (Thompson Sampling)
        # 7 restart strategies based on benchmark winners
        self.n_strategies = 7
        self.restart_alpha = np.ones(self.n_strategies, dtype=float)
        self.restart_beta = np.ones(self.n_strategies, dtype=float)
        self.restart_window_size = 50
        self.restart_history = [[] for _ in range(self.n_strategies)]
        self.last_restart_f_best = None
        self.restart_count = 0
        self.current_restart_strategy = None
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
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
            if new_f_best < f_best - 1e-12:
                f_best, x_best = new_f_best, population[new_best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1
            
            velocity = self._update_velocity_batch(population, velocity, x_best)
            
            if stagnation >= self.local_search_interval:
                x_best, f_best = self._apply_adaptive_local_search(
                    func, x_best, f_best, self.local_search_radius
                )
                stagnation = 0
            
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold or stagnation >= self.stagnation_limit:
                old_f_best = f_best
                old_best = x_best.copy()
                
                population = self._adaptive_restart(population, fitness, x_best)
                velocity = self._init_velocity()
                fitness = self._eval_wrapper(func, population)
                
                new_best_idx = self._argmin(fitness)
                new_f_best = float(fitness[new_best_idx])
                if new_f_best < f_best:
                    f_best, x_best = new_f_best, population[new_best_idx].copy()
                
                stagnation = 0
            
            gen += 1
        
        return f_best, x_best
    
    def _adaptive_restart(self, population, fitness, x_best):
        """Select restart strategy using Thompson Sampling and update bandit."""
        f_best = float(np.min(fitness))
        
        # Thompson Sampling: sample from Beta distribution for each strategy
        samples = np.array([
            np.random.beta(max(self.restart_alpha[i], 0.01), max(self.restart_beta[i], 0.01))
            for i in range(self.n_strategies)
        ])
        
        # With small probability, explore randomly (exploration bonus)
        if np.random.random() < 0.1:
            strategy_idx = int(np.random.randint(0, self.n_strategies))
        else:
            strategy_idx = int(np.argmax(samples))
        
        self.current_restart_strategy = strategy_idx
        self.last_restart_f_best = f_best
        
        # Apply selected strategy
        if strategy_idx == 0:
            new_pop = self._restart_strategy_0(population, fitness, x_best)
        elif strategy_idx == 1:
            new_pop = self._restart_strategy_1(population, fitness, x_best)
        elif strategy_idx == 2:
            new_pop = self._restart_strategy_2(population, fitness, x_best)
        elif strategy_idx == 3:
            new_pop = self._restart_strategy_3(population, fitness, x_best)
        elif strategy_idx == 4:
            new_pop = self._restart_strategy_4(population, fitness, x_best)
        elif strategy_idx == 5:
            new_pop = self._restart_strategy_5(population, fitness, x_best)
        else:
            new_pop = self._restart_strategy_6(population, fitness, x_best)
        
        return new_pop
    
    def _update_restart_bandit(self, improvement):
        """Update bandit parameters based on restart outcome."""
        if self.current_restart_strategy is None:
            return
        
        strategy = self.current_restart_strategy
        reward = float(improvement)
        
        # Clamp reward to reasonable range
        reward = float(np.clip(reward, -1000.0, 1000.0))
        
        # Update sliding window
        self.restart_history[strategy].append(reward)
        if len(self.restart_history[strategy]) > self.restart_window_size:
            self.restart_history[strategy] = self.restart_history[strategy][-self.restart_window_size:]
        
        # Compute discounted reward (recent rewards weighted more)
        history = self.restart_history[strategy]
        if len(history) > 0:
            discounted_reward = 0.0
            discount = 0.9
            for i, r in enumerate(reversed(history)):
                discounted_reward += r * (discount ** i)
            discounted_reward /= len(history)
        else:
            discounted_reward = 0.0
        
        # Update Beta distribution parameters
        # Positive reward increases alpha, negative increases beta
        if discounted_reward > 0:
            self.restart_alpha[strategy] += 0.5 * min(discounted_reward / 10.0, 1.0)
        else:
            self.restart_beta[strategy] += 0.5 * min(abs(discounted_reward) / 10.0, 1.0)
        
        # Ensure minimum values to avoid numerical issues
        self.restart_alpha[strategy] = max(self.restart_alpha[strategy], 0.1)
        self.restart_beta[strategy] = max(self.restart_beta[strategy], 0.1)
        
        self.restart_count += 1
        self.current_restart_strategy = None
    
    # === Restart Strategy Implementations ===
    
    def _restart_strategy_0(self, population, fitness, x_best):
        """Variant 01: Replace worst half with random, preserve best."""
        NP = len(population)
        new_pop = population.copy()
        best_idx = self._argmin(fitness)
        
        n_replace = NP // 2
        worst_indices = np.argpartition(fitness, n_replace)[:n_replace]
        
        for idx in worst_indices:
            new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
        
        new_pop[best_idx] = x_best.copy()
        return new_pop
    
    def _restart_strategy_1(self, population, fitness, x_best):
        """Variant 03: Random, opposition-based, and LHS restart."""
        NP = len(population)
        new_pop = population.copy()
        dim = self.dim
        lower, upper = self.lower, self.upper
        
        n_random = NP // 3
        random_indices = np.random.choice(NP, n_random, replace=False)
        for idx in random_indices:
            new_pop[idx] = np.random.uniform(lower, upper, dim)
        
        n_oppose = NP // 3
        remaining_for_oppose = [i for i in range(NP) if i not in random_indices]
        oppose_indices = np.random.choice(remaining_for_oppose, min(n_oppose, len(remaining_for_oppose)), replace=False)
        for idx in oppose_indices:
            boundary_t = np.tanh(np.random.uniform(-3, 3, dim))
            range_val = upper - lower
            new_pop[idx] = (lower + upper) / 2.0 + boundary_t * range_val / 2.0
        
        remaining = [i for i in range(NP) if i not in random_indices and i not in oppose_indices]
        if len(remaining) > 0:
            n_lhs = len(remaining)
            lhs_samples = np.zeros((n_lhs, dim))
            for d in range(dim):
                samples = np.linspace(0, 1, n_lhs + 1)[:-1] + np.random.uniform(0, 1/n_lhs, n_lhs)
                lhs_samples[:, d] = samples
            np.random.shuffle(lhs_samples)
            lhs_scaled = lower + lhs_samples * (upper - lower)
            for i, idx in enumerate(remaining):
                new_pop[idx] = lhs_scaled[i]
        
        return self._clip_to_bounds_batch(new_pop)
    
    def _restart_strategy_2(self, population, fitness, x_best):
        """Variant 04: Half random, half Gaussian perturbation."""
        NP = len(population)
        new_pop = population.copy()
        
        random_count = NP // 2
        for i in range(random_count):
            new_pop[i] = np.random.uniform(self.lower, self.upper, self.dim)
        
        sigma = (self.upper - self.lower) * 0.3
        for i in range(random_count, NP):
            new_pop[i] = x_best + np.random.normal(0, sigma, self.dim)
        
        return self._clip_to_bounds_batch(new_pop)
    
    def _restart_strategy_3(self, population, fitness, x_best):
        """Variant 06: Boundary-focused with stagnation scaling."""
        NP = len(population)
        new_pop = population.copy()
        
        stagnation_gen = getattr(self, 'stagnation', 0)
        distance_scale = 1.0 + min(stagnation_gen / 20.0, 5.0)
        
        for idx in range(NP):
            boundary_t = np.tanh(np.random.uniform(-3, 3, self.dim))
            blend = np.random.random()
            
            if blend < 0.3:
                offset = distance_scale * 50.0 * boundary_t
                new_pop[idx] = x_best + offset
            elif blend < 0.7:
                range_val = self.upper - self.lower
                boundary_pos = (self.lower + self.upper) / 2 + boundary_t * range_val / 2
                new_pop[idx] = boundary_pos
            else:
                new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
        
        return self._clip_to_bounds_batch(new_pop)
    
    def _restart_strategy_4(self, population, fitness, x_best):
        """Variant 07: Opposition, Halton, and Cauchy perturbation."""
        NP = len(population)
        dim = self.dim
        new_pop = population.copy()
        
        centroid = np.mean(population, axis=0)
        pop_spread = np.std(population, axis=0) + 1e-10
        bound_range = self.upper - self.lower
        
        worst_count = NP // 2
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        for i, idx in enumerate(worst_indices):
            strategy = i % 3
            if strategy == 0:
                reflection = 2.0 * centroid - x_best
                new_pop[idx] = reflection + np.random.uniform(-0.2 * bound_range, 0.2 * bound_range, dim)
            elif strategy == 1:
                halton = np.array([(i + 1) / (dim ** k) % 1.0 for k in range(1, dim + 1)])
                new_pop[idx] = self.lower + halton * bound_range
            else:
                cauchy_scale = pop_spread * 2.0
                new_pop[idx] = x_best + np.random.standard_cauchy(dim) * cauchy_scale
        
        return self._clip_to_bounds_batch(new_pop)
    
    def _restart_strategy_5(self, population, fitness, x_best):
        """Variant 08: Error-scaled perturbation with random injection."""
        NP = len(population)
        new_pop = population.copy()
        
        f_best = float(np.min(fitness))
        error_scale = max(1.0, np.log10(max(f_best, 1e-10) + 1e-10))
        perturbation_range = float(np.clip(50.0 * error_scale, 10.0, 500.0))
        
        worst_count = NP // 3
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        for idx in worst_indices:
            scale_factor = perturbation_range * np.random.uniform(0.5, 1.5, self.dim)
            new_pop[idx] = x_best + np.random.uniform(-scale_factor, scale_factor, self.dim)
        
        random_count = max(1, NP // 5)
        random_indices = np.random.choice(NP, random_count, replace=False)
        for idx in random_indices:
            new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
        
        return self._clip_to_bounds_batch(new_pop)
    
    def _restart_strategy_6(self, population, fitness, x_best):
        """Variant 09: Opposition, adaptive perturbation, and Latin hypercube."""
        NP, dim = population.shape
        new_pop = population.copy()
        
        worst_count = max(1, NP // 3)
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        center = (self.lower + self.upper) / 2.0
        range_val = self.upper - self.lower
        current_diversity = self._compute_diversity(population)
        diversity_ratio = float(np.clip(current_diversity / (range_val * 0.1), 0.1, 2.0))
        
        for idx in worst_indices:
            strategy = np.random.random()
            
            if strategy < 0.5:
                opp_point = self.lower + self.upper - population[idx]
                new_pop[idx] = 0.7 * opp_point + 0.3 * np.random.uniform(self.lower, self.upper, dim)
            elif strategy < 0.8:
                scale = max(1.0, min(range_val * 0.5, 50.0 * diversity_ratio))
                new_pop[idx] = x_best + np.random.uniform(-scale, scale, dim)
            else:
                subspace_lower = np.minimum(population[idx], x_best) - range_val * 0.1
                subspace_upper = np.maximum(population[idx], x_best) + range_val * 0.1
                subspace_lower = np.clip(subspace_lower, self.lower, self.upper)
                subspace_upper = np.clip(subspace_upper, self.lower, self.upper)
                u = np.random.random(dim)
                new_pop[idx] = subspace_lower + u * (subspace_upper - subspace_lower)
        
        return self._clip_to_bounds_batch(new_pop)
    
    # === Original methods ===
    
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
            new_fit[idx] = float(trial_fit[idx])
        
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
            F_ema = 0.9 * F_ema + 0.1 * float(np.mean(success_F[-10:]))
            CR_ema = 0.9 * CR_ema + 0.1 * float(np.mean(success_CR[-10:]))
            F = float(np.clip(F_ema + np.random.uniform(-0.1, 0.1), self.F_min, self.F_max))
            CR = float(np.clip(CR_ema + np.random.uniform(-0.1, 0.1), self.CR_min, self.CR_max))
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
```