```python
import numpy as np

class AdaptiveIslandDifferentialEvolution:
    """
    Hybrid DE with island topology, JADE-style adaptation, and Nelder-Mead local search.
    Strategically different from:
    - AdaptiveMultiStrategyDE: uses island model + explicit local search
    - CMAAdaptiveParticleSwarmOptimizer: DE-based, not PSO
    - AdaptiveTopologyParticleSwarmOptimizer: evolutionary, not swarm intelligence
    
    Key features:
    - Island model topology with periodic migration
    - JADE-style adaptive F/Cr with success history
    - Nelder-Mead local search on top performers
    - Diversity-triggered restarts
    - Stagnation detection with parameter adaptation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np_pop = min(200, max(40, 5 * dim))
        self.n_islands = max(2, min(8, self.np_pop // 20))
        self.island_size = self.np_pop // self.n_islands
        
        self.F = 0.5
        self.Cr = 0.5
        self.mu_F = 0.5
        self.mu_Cr = 0.5
        self.archive = []
        self.archive_size = 1.5 * self.np_pop
        
        self.local_search_interval = 5
        self.local_search_fraction = 0.1
        self.ls_max_evals = 50
        self.ls_reflection = 1.0
        self.ls_contraction = 0.5
        self.ls_expansion = 2.0
        
        self.diversity_threshold = 1e-7
        self.stagnation_threshold = 20
        self.migration_interval = 8
        
        self.F_min, self.F_max = 0.1, 1.0
        self.Cr_min, self.Cr_max = 0.0, 1.0
        self.c = 0.1
        
        self.lb = -100.0
        self.ub = 100.0
        
        self._reset_state()
    
    def _reset_state(self):
        self.generation = 0
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.success_F = []
        self.success_Cr = []
        self.population = None
        self.fitness = None
        self.f_opt = np.inf
        self.x_opt = None
        self.island_best_indices = []
        self.funcall_count = 0
    
    def _initialize_population(self):
        self.population = np.random.uniform(
            self.lb, self.ub, size=(self.np_pop, self.dim)
        )
        self.population = np.clip(self.population, self.lb, self.ub)
    
    def _evaluate_initial_batch(self, func):
        self.fitness = func(self.population)
        self.funcall_count += len(self.fitness)
        
        if len(self.fitness) < len(self.population):
            actual_len = len(self.fitness)
            self.population = self.population[:actual_len]
            self.fitness = self.fitness[:actual_len]
        
        valid_mask = np.isfinite(self.fitness)
        if not np.any(valid_mask):
            raise RuntimeError("All fitness values are NaN/inf on first evaluation")
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        self.best_fitness_history.append(self.f_opt)
    
    def _get_current_bounds(self):
        return self.lb, self.ub
    
    def _mutate_current_to_pbest_batch(self, island_pop, island_indices, p=0.05):
        lb, ub = self._get_current_bounds()
        pop_size = len(island_pop)
        n_select = max(2, int(p * self.np_pop))
        
        sorted_indices = np.argsort(self.fitness)
        pbest_set = sorted_indices[:n_select]
        
        if len(self.archive) > 0:
            combined = np.vstack([self.population, np.array(self.archive)])
        else:
            combined = self.population
        
        candidates = np.random.choice(
            len(combined), size=(pop_size, 3), replace=False
        )
        r1 = combined[candidates[:, 0]]
        r2 = combined[candidates[:, 1]]
        r3 = combined[candidates[:, 2]]
        
        pbest_indices = np.random.choice(pbest_set, size=pop_size)
        x_pbest = self.population[pbest_indices]
        
        F_i = np.random.uniform(self.F_min, self.F_max, size=pop_size)
        mutants = island_pop + F_i.reshape(-1, 1) * (x_pbest - island_pop) + \
                  F_i.reshape(-1, 1) * (r1 - r2)
        
        mutants = np.clip(mutants, lb, ub)
        return mutants, F_i
    
    def _crossover_batch(self, target, mutant, Cr_i=None):
        pop_size = len(target)
        
        if Cr_i is None:
            Cr_i = np.full(pop_size, self.Cr)
        else:
            Cr_i = np.clip(Cr_i, self.Cr_min, self.Cr_max)
        
        Cr_matrix = np.tile(Cr_i.reshape(-1, 1), (1, self.dim))
        rand_mask = np.random.rand(pop_size, self.dim) < Cr_matrix
        
        j_rand = np.random.randint(0, self.dim, size=pop_size)
        j_rand_onehot = np.zeros((pop_size, self.dim))
        j_rand_onehot[np.arange(pop_size), j_rand] = 1
        
        trials = np.where(rand_mask | j_rand_onehot, mutant, target)
        trials = np.clip(trials, self.lb, self.ub)
        
        Cr_values = np.max(Cr_matrix, axis=1)
        return trials, Cr_values
    
    def _select_survivors_batch(self, target, target_fitness, trial, trial_fitness):
        improved_mask = trial_fitness < target_fitness
        
        survivors = np.where(improved_mask[:, np.newaxis], trial, target)
        survivor_fitness = np.minimum(trial_fitness, target_fitness)
        
        new_success_F = []
        new_success_Cr = []
        
        for i in range(len(target)):
            if improved_mask[i]:
                diff = target_fitness[i] - trial_fitness[i]
                if diff > 0:
                    new_success_F.append(self.F)
                    new_success_Cr.append(self.Cr)
        
        return survivors, survivor_fitness, improved_mask, new_success_F, new_success_Cr
    
    def _update_adaptive_parameters(self, success_F, success_Cr):
        if len(success_F) > 0:
            self.success_F.extend(success_F)
            self.success_Cr.extend(success_Cr)
            
            if len(self.success_F) > self.np_pop * 2:
                self.success_F = self.success_F[-self.np_pop * 2:]
                self.success_Cr = self.success_Cr[-self.np_pop * 2:]
            
            if len(self.success_F) > 0:
                self.mu_F = (1 - self.c) * self.mu_F + self.c * np.mean(self.success_F)
                self.mu_Cr = (1 - self.c) * self.mu_Cr + self.c * np.mean(self.success_Cr)
        
        self.F = np.random.normal(self.mu_F, 0.1)
        self.F = np.clip(self.F, self.F_min, self.F_max)
        
        self.Cr = np.random.normal(self.mu_Cr, 0.1)
        self.Cr = np.clip(self.Cr, self.Cr_min, self.Cr_max)
    
    def _update_archive(self, improved_pop, improved_fitness):
        for i, fit in enumerate(improved_fitness):
            if np.isfinite(fit):
                self.archive.append(improved_pop[i].copy())
        
        if len(self.archive) > self.archive_size:
            keep_indices = np.random.choice(
                len(self.archive), size=int(self.archive_size), replace=False
            )
            self.archive = [self.archive[j] for j in keep_indices]
    
    def _split_into_islands(self):
        islands = []
        for i in range(self.n_islands):
            start = i * self.island_size
            end = start + self.island_size if i < self.n_islands - 1 else self.np_pop
            islands.append((start, end))
        return islands
    
    def _migrate_between_islands(self):
        islands = self._split_into_islands()
        migrants = []
        
        for i in range(self.n_islands):
            start, end = islands[i]
            island_pop = self.population[start:end]
            island_fit = self.fitness[start:end]
            
            best_idx = np.argmin(island_fit)
            worst_idx = np.argmax(island_fit)
            
            source_island = (i + 1) % self.n_islands
            src_start, src_end = islands[source_island]
            src_best_idx = src_start + np.argmin(self.fitness[src_start:src_end])
            
            migrants.append((end - 1 if worst_idx == len(island_pop) - 1 else start + worst_idx, 
                           src_best_idx))
        
        for dest_idx, src_idx in migrants:
            self.population[dest_idx] = self.population[src_idx].copy()
    
    def _nelder_mead_local_search(self, x_center):
        n_dims = len(x_center)
        alpha = self.ls_reflection
        gamma = self.ls_expansion
        rho = self.ls_contraction
        sigma = 0.5
        
        simplex = np.zeros((n_dims + 1, n_dims))
        simplex[0] = x_center.copy()
        
        for i in range(1, n_dims + 1):
            simplex[i] = x_center.copy()
            simplex[i, i-1] += np.random.uniform(0.01, 0.1)
        simplex = np.clip(simplex, self.lb, self.ub)
        
        evals = 0
        for _ in range(self.ls_max_evals):
            if evals + n_dims + 1 > self.ls_max_evals:
                break
            
            fitnesses = func(self.population[:1])[:1]
            if len(fitnesses) == 0 or not np.isfinite(fitnesses[0]):
                break
            evals += 1
            
            sorted_idx = np.argsort(fitnesses)
            centroid = np.mean(simplex[:-1], axis=0)
            
            worst = simplex[-1]
            reflected = centroid + alpha * (centroid - worst)
            reflected = np.clip(reflected, self.lb, self.ub)
            
            reflected_fit = func(reflected.reshape(1, -1))
            if len(reflected_fit) == 0:
                break
            evals += 1
            reflected_fit = reflected_fit[0]
            
            if reflected_fit < fitnesses[0]:
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, self.lb, self.ub)
                expanded_fit = func(expanded.reshape(1, -1))
                if len(expanded_fit) == 0:
                    break
                evals += 1
                expanded_fit = expanded_fit[0]
                
                if expanded_fit < reflected_fit:
                    simplex[-1] = expanded
                else:
                    simplex[-1] = reflected
            elif reflected_fit < fitnesses[-2]:
                simplex[-1] = reflected
            else:
                contracted = centroid + rho * (worst - centroid)
                contracted = np.clip(contracted, self.lb, self.ub)
                contracted_fit = func(contracted.reshape(1, -1))
                if len(contracted_fit) == 0:
                    break
                evals += 1
                contracted_fit = contracted_fit[0]
                
                if contracted_fit < fitnesses[-1]:
                    simplex[-1] = contracted
                else:
                    for j in range(1, n_dims + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
        
        best_local_idx = np.argmin(fitnesses)
        return simplex[best_local_idx], fitnesses[best_local_idx]
    
    def _apply_local_search_to_elites(self, func):
        if self.generation % self.local_search_interval != 0:
            return
        
        n_elites = max(1, int(self.local_search_fraction * self.np_pop))
        sorted_indices = np.argsort(self.fitness)[:n_elites]
        
        for idx in sorted_indices:
            if not np.isfinite(self.fitness[idx]):
                continue
            
            if np.random.rand() < 0.3:
                x_center = self.population[idx].copy()
                x_improved, fit_improved = self._nelder_mead_local_search(x_center)
                
                self.funcall_count += 1
                if fit_improved < self.fitness[idx] and np.isfinite(fit_improved):
                    improvement = self.fitness[idx] - fit_improved
                    if improvement > 0:
                        self.population[idx] = x_improved
                        self.fitness[idx] = fit_improved
                        
                        if fit_improved < self.f_opt:
                            self.f_opt = fit_improved
                            self.x_opt = x_improved.copy()
                            self.stagnation_counter = 0
    
    def _compute_diversity(self):
        if len(self.population) < 2:
            return 1.0
        
        mean_pop = np.mean(self.population, axis=0)
        variance = np.mean((self.population - mean_pop) ** 2)
        range_sq = (self.ub - self.lb) ** 2
        
        if range_sq > 0:
            cv = np.sqrt(variance) / (self.ub - self.lb)
        else:
            cv = 0.0
        
        return cv
    
    def _check_stagnation(self):
        if len(self.best_fitness_history) < 2:
            return False
        
        recent_best = self.best_fitness_history[-self.stagnation_threshold:] if \
                      len(self.best_fitness_history) >= self.stagnation_threshold else \
                      self.best_fitness_history
        
        if len(recent_best) >= self.stagnation_threshold:
            improvement = recent_best[0] - recent_best[-1]
            return improvement < 1e-8
        
        return False
    
    def _restart_if_needed(self):
        should_restart = False
        reason = ""
        
        diversity = self._compute_diversity()
        if diversity < self.diversity_threshold:
            should_restart = True
            reason = "low_diversity"
        elif self._check_stagnation():
            should_restart = True
            reason = "stagnation"
        
        if should_restart:
            n_replace = int(0.7 * self.np_pop)
            replace_indices = np.random.choice(
                self.np_pop, size=n_replace, replace=False
            )
            
            new_part = np.random.uniform(
                self.lb, self.ub, size=(n_replace, self.dim)
            )
            self.population[replace_indices] = new_part
            
            self.success_F = []
            self.success_Cr = []
            self.mu_F = 0.5
            self.mu_Cr = 0.5
            self.F = 0.5
            self.Cr = 0.5
            
            self.stagnation_counter = 0
            
            return True, reason
        
        return False, ""
    
    def _update_best_and_stagnation(self):
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        
        if current_best_fitness < self.f_opt:
            improvement = self.f_opt - current_best_fitness
            self.f_opt = current_best_fitness
            self.x_opt = self.population[current_best_idx].copy()
            self.stagnation_counter = 0
            
            if improvement > 1e-10:
                self.best_fitness_history.append(self.f_opt)
        else:
            self.stagnation_counter += 1
    
    def __call__(self, func, stopping_condition):
        self._reset_state()
        self._initialize_population()
        
        self._evaluate_initial_batch(func)
        
        while not stopping_condition():
            islands = self._split_into_islands()
            
            all_trials = []
            all_targets = []
            island_ranges = []
            
            for i in range(self.n_islands):
                start, end = islands[i]
                island_pop = self.population[start:end]
                
                mutants, F_i = self._mutate_current_to_pbest_batch(
                    island_pop, list(range(start, end))
                )
                trials, Cr_i = self._crossover_batch(island_pop, mutants, F_i)
                
                all_trials.append(trials)
                all_targets.append(island_pop)
                island_ranges.append((start, end))
            
            trial_batch = np.vstack(all_trials)
            trial_batch = np.clip(trial_batch, self.lb, self.ub)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trial_batch)
            self.funcall_count += len(trial_fitness)
            
            if len(trial_fitness) < len(trial_batch):
                actual_len = len(trial_fitness)
                trial_batch = trial_batch[:actual_len]
                trial_fitness = trial_fitness[:actual_len]
            
            if len(trial_fitness) == 0:
                break
            
            all_success_F = []
            all_success_Cr = []
            new_population = []
            new_fitness = []
            
            offset = 0
            for i in range(self.n_islands):
                start, end = islands[i]
                island_size = end - start
                
                batch_start = offset
                batch_end = offset + island_size
                
                target = self.population[start:end]
                target_fit = self.fitness[start:end]
                trial = trial_batch[batch_start:batch_end]
                trial_fit = trial_fitness[batch_start:batch_end]
                
                if len(trial_fit) < island_size:
                    trial = trial[:len(trial_fit)]
                    trial_fit = trial_fit[:len(trial_fit)]
                    if len(trial) == 0:
                        new_population.append(target)
                        new_fitness.append(target_fit)
                        offset += island_size
                        continue
                
                survivors, survivor_fit, improved_mask, sF, sCr = \
                    self._select_survivors_batch(target, target_fit, trial, trial_fit)
                
                all_success_F.extend(sF)
                all_success_Cr.extend(sCr)
                
                improved_pop = survivors[improved_mask]
                improved_fit = survivor_fit[improved_mask]
                
                if len(improved_pop) > 0:
                    self._update_archive(improved_pop, improved_fit)
                
                new_population.append(survivors)
                new_fitness.append(survivor_fit)
                offset += island_size
            
            self.population = np.vstack(new_population)
            self.fitness = np.concatenate(new_fitness)
            
            if len(self.population) < self.np_pop:
                deficit = self.np_pop - len(self.population)
               补充 = np.random.uniform(self.lb, self.ub, size=(deficit, self.dim))
                self.population = np.vstack([self.population, 补充])
               补充_fit = func(补充)
                self.funcall_count += len(补充_fit)
                if len(补充_fit) == len(补充):
                   补充_fit = 补充_fit[:deficit]
               补充_fit = np.pad(补充_fit, (0, deficit - len(补充_fit)), 
                               constant_values=np.inf)
                self.fitness = np.concatenate([self.fitness, 补充_fit])
            
            self._update_adaptive_parameters(all_success_F, all_success_Cr)
            
            if self.generation % self.migration_interval == 0:
                self._migrate_between_islands()
            
            self._apply_local_search_to_elites(func)
            
            self._update_best_and_stagnation()
            
            restarted, reason = self._restart_if_needed()
            if restarted:
                reeval = func(self.population)
                self.funcall_count += len(reeval)
                if len(reeval) >= len(self.fitness):
                    self.fitness = reeval[:len(self.fitness)]
                else:
                    self.fitness[:len(reeval)] = reeval
            
            self.generation += 1
            
            if self.generation > 2:
                recent = self.best_fitness_history[-min(5, len(self.best_fitness_history)):]
                if len(recent) >= 3:
                    if recent[-1] >= recent[0] - 1e-10:
                        self.mu_F = min(0.9, self.mu_F + 0.05)
        
        final_fitness = func(self.population)
        self.funcall_count += len(final_fitness)
        
        if len(final_fitness) >= len(self.fitness):
            self.fitness = final_fitness[:len(self.population)]
        
        valid_mask = np.isfinite(self.fitness)
        if np.any(valid_mask):
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.f_opt:
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx].copy()
        
        return self.f_opt, self.x_opt
```