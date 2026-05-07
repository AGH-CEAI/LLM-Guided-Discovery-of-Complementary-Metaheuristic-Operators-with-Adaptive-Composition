```python
import numpy as np


class AdaptiveSurvivorSelectionOptimizer:
    """
    Adaptive optimizer that automatically selects the best survivor selection
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 4 survivor selection strategies (original, variant_03, variant_04, variant_06)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive operator selection parameters for survivor selection
        self.num_survivor_operators = 4
        self.survivor_operator_names = ['original', 'variant_03', 'variant_04', 'variant_06']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_survivor_operators)
        self.beta = np.ones(self.num_survivor_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_survivor_operators)}
        self.operator_counts = np.zeros(self.num_survivor_operators)
        self.selection_counts = np.zeros(self.num_survivor_operators)
        
        # Runtime state
        self.generation = 0
        self.current_survivor_operator = 0
        self.C = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.mean = None
        self.old_mean = None
        self.population = None
        self.fitness = None
        self.trials = None
        self.trial_fitness = None
        self.f_opt = None
        self.x_opt = None
        self.f_opt_prev = None
        self.stagnation_counter = 0
        self.improvement_ema = 1.0
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
            
            # Select survivor operator using Thompson Sampling
            self._select_operator_thompson()
            
            # Apply selected survivor selection strategy
            self._select_survivors_batch()
            
            self._adapt_step_size()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_survivor_operator = 0
        
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
        if hasattr(self, 'improvement_ema'):
            self.improvement_ema = 1.0
        if hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
        if hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_survivor_operator = int(np.argmax(samples))
        self.operator_counts[self.current_survivor_operator] += 1
        return self.current_survivor_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_survivor_operator
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _select_survivors_batch(self):
        """Dispatch to the selected survivor selection strategy."""
        if self.current_survivor_operator == 0:
            self._select_survivors_original()
        elif self.current_survivor_operator == 1:
            self._select_survivors_variant_03()
        elif self.current_survivor_operator == 2:
            self._select_survivors_variant_04()
        else:
            self._select_survivors_variant_06()
    
    def _select_survivors_original(self):
        """Select survivors via elitist (mu, lambda)-selection (original baseline)."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _select_survivors_variant_03(self):
        """Select survivors via deterministic crowding for diversity maintenance."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        sorted_indices = np.argsort(combined_fit)

        fit_variance = np.var(combined_fit)
        fit_range = np.max(combined_fit) - np.min(combined_fit) + 1e-10
        normalized_variance = fit_variance / fit_range

        base_threshold = (self.ub[0] - self.lb[0]) * 0.005 * np.sqrt(self.dim / 100.0)

        if normalized_variance < 0.01:
            diversity_threshold = base_threshold * 20.0
        elif normalized_variance < 0.05:
            diversity_threshold = base_threshold * 10.0
        elif normalized_variance < 0.1:
            diversity_threshold = base_threshold * 5.0
        else:
            diversity_threshold = base_threshold * 2.0

        survivors_pop = []
        survivors_fit = []

        best_idx = sorted_indices[0]
        survivors_pop.append(combined_pop[best_idx].copy())
        survivors_fit.append(float(combined_fit[best_idx]))

        for idx in sorted_indices[1:]:
            candidate = combined_pop[idx]
            candidate_fit = float(combined_fit[idx])

            min_dist = float('inf')
            most_similar_idx = 0
            for s_idx, surv in enumerate(survivors_pop):
                dist = np.linalg.norm(candidate - surv)
                if dist < min_dist:
                    min_dist = dist
                    most_similar_idx = s_idx

            if min_dist >= diversity_threshold:
                survivors_pop.append(candidate.copy())
                survivors_fit.append(candidate_fit)
            elif candidate_fit < survivors_fit[most_similar_idx]:
                survivors_pop[most_similar_idx] = candidate.copy()
                survivors_fit[most_similar_idx] = candidate_fit

            if len(survivors_pop) >= self.NP:
                break

        if len(survivors_pop) < self.NP:
            for idx in sorted_indices:
                if combined_pop[idx].tolist() not in [p.tolist() for p in survivors_pop]:
                    survivors_pop.append(combined_pop[idx].copy())
                    survivors_fit.append(float(combined_fit[idx]))
                if len(survivors_pop) >= self.NP:
                    break

        self.population = np.array(survivors_pop[:self.NP])
        self.fitness = np.array(survivors_fit[:self.NP])

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _select_survivors_variant_04(self):
        """Select survivors via deterministic crowding: match trials to nearest parents."""
        pop_size = self.NP
        num_trials = min(len(self.trials), pop_size)

        if num_trials == 0 or len(self.population) == 0:
            combined_pop = np.vstack([self.population, self.trials])
            combined_fit = np.concatenate([self.fitness, self.trial_fitness])
            sorted_indices = np.argsort(combined_fit)
            self.population = combined_pop[sorted_indices[:self.NP]]
            self.fitness = combined_fit[sorted_indices[:self.NP]]
        else:
            dists = np.zeros((num_trials, pop_size))
            for i in range(num_trials):
                diffs = self.trials[i] - self.population
                dists[i] = np.sum(diffs * diffs, axis=1)

            nearest_parent = np.argmin(dists, axis=1)

            candidates_pop = []
            candidates_fit = []
            used_parents = set()

            trial_order = np.argsort(self.trial_fitness[:num_trials])

            for ti in trial_order:
                parent_idx = nearest_parent[ti]
                trial_fit = float(self.trial_fitness[ti])
                parent_fit = float(self.fitness[parent_idx])

                if trial_fit <= parent_fit:
                    candidates_pop.append(self.trials[ti])
                    candidates_fit.append(trial_fit)
                    used_parents.add(parent_idx)
                else:
                    if parent_idx not in used_parents:
                        candidates_pop.append(self.population[parent_idx])
                        candidates_fit.append(parent_fit)
                        used_parents.add(parent_idx)

            for i in range(pop_size):
                if i not in used_parents:
                    candidates_pop.append(self.population[i])
                    candidates_fit.append(float(self.fitness[i]))
                    if len(candidates_pop) >= pop_size:
                        break

            candidates_pop = np.array(candidates_pop)
            candidates_fit = np.array(candidates_fit)

            if len(candidates_pop) > pop_size:
                keep_idx = np.argsort(candidates_fit)[:pop_size]
                self.population = candidates_pop[keep_idx]
                self.fitness = candidates_fit[keep_idx]
            elif len(candidates_pop) < pop_size:
                self.population = list(candidates_pop)
                self.fitness = list(candidates_fit)
                remaining = pop_size - len(candidates_pop)
                rand_indices = np.random.choice(pop_size, remaining, replace=True)
                for idx in rand_indices:
                    self.population.append(self.population[idx])
                    self.fitness.append(self.fitness[idx])
                self.population = np.array(self.population)
                self.fitness = np.array(self.fitness)
            else:
                self.population = candidates_pop
                self.fitness = candidates_fit

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _select_survivors_variant_06(self):
        """Select survivors via deterministic crowding with niching to preserve diversity."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        n_trials = len(self.trials)
        n_parents = len(self.population)

        trials_exp = self.trials[:, np.newaxis, :]
        parents_exp = self.population[np.newaxis, :, :]
        dists = np.linalg.norm(trials_exp - parents_exp, axis=2)

        closest_parent = np.argmin(dists, axis=1)

        new_pop = []
        new_fit = []

        for i in range(n_parents):
            new_pop.append(self.population[i])
            new_fit.append(float(self.fitness[i]))

        for i in range(n_trials):
            parent_idx = closest_parent[i]
            trial_fit = float(combined_fit[n_parents + i])
            parent_fit = float(self.fitness[parent_idx])

            if trial_fit < parent_fit:
                new_pop.append(self.trials[i])
                new_fit.append(trial_fit)

        if len(new_pop) > self.NP:
            new_pop = np.array(new_pop)
            new_fit = np.array(new_fit)

            pop_stack = np.vstack(new_pop)
            pdists = np.linalg.norm(pop_stack[:, np.newaxis, :] - pop_stack[np.newaxis, :, :], axis=2)
            avg_dist = np.mean(pdists, axis=1)

            fit_min, fit_max = np.min(new_fit), np.max(new_fit)
            fit_range = max(fit_max - fit_min, 1e-10)
            fit_norm = (new_fit - fit_min) / fit_range

            diversity_weight = 0.05
            selection_score = fit_norm - diversity_weight * (avg_dist / (np.max(avg_dist) + 1e-10))

            best_indices = np.argsort(selection_score)[:self.NP]

            new_pop = pop_stack[best_indices]
            new_fit = new_fit[best_indices]
        else:
            new_pop = np.array(new_pop[:self.NP]) if len(new_pop) >= self.NP else np.array(new_pop)
            new_fit = np.array(new_fit[:self.NP]) if len(new_fit) >= self.NP else np.array(new_fit)

        if len(new_pop) < self.NP:
            deficit = self.NP - len(new_pop)
            combined_sorted = np.argsort(combined_fit)
            padding_indices = combined_sorted[len(new_pop):len(new_pop) + deficit]
            padding_pop = combined_pop[padding_indices]
            padding_fit = combined_fit[padding_indices]
            new_pop = np.vstack([new_pop, padding_pop])
            new_fit = np.concatenate([new_fit, padding_fit])

        self.population = new_pop[:self.NP]
        self.fitness = new_fit[:self.NP]

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = self.fitness[i] - self.trial_fitness[i]
                    total_improvement += max(float(diff), 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_covariance(self):
        """Dispatch to covariance adaptation strategy."""
        self._adapt_covariance_original()
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute population diversity as mean per-dimension std."""
        return np.mean(np.std(self.population, axis=0))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```