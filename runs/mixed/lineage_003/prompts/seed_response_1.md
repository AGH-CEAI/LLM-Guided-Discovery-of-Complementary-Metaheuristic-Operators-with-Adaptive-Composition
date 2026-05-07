```python
import numpy as np

class AdaptiveCulturalDE:
    """
    Differential Evolution with Cultural Memory and Adaptive Parameters.
    
    Key innovations:
    - Multi-strategy mutation: current-to-pbest/2 with cultural memory
    - Per-individual adaptive F/CR using success-history adaptation
    - Cultural archive storing elite and diverse solutions
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.NP = min(max(5 * dim, 30), 200)
        
        self.F_base = 0.6
        self.CR_base = 0.85
        self.p_best = 0.15
        self.archive = None
        
        self.generation = 0
        self.stagnation_count = 0
        self.last_best_fitness = np.inf
        self.success_history_F = np.ones(6)
        self.success_history_CR = np.ones(6)
        self.history_index = 0
        self.successful_F = []
        self.successful_CR = []
        
    def _initialize_population(self):
        """Initialize population uniformly within bounds."""
        pop = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        return pop
    
    def _clip_to_bounds_batch(self, pop):
        """Clip all individuals to search bounds."""
        return np.clip(pop, self.lower_bound, self.upper_bound)
    
    def _initialize_archive(self):
        """Initialize empty cultural archive."""
        self.archive = []
    
    def _get_pbest_indices(self, fitness, n_best):
        """Get indices of top n_best individuals by fitness."""
        return np.argsort(fitness)[:n_best]
    
    def _sample_random_indices(self, n_indices, n_samples, exclude_indices=None):
        """Sample random indices avoiding excluded ones."""
        if exclude_indices is None:
            exclude_indices = set()
        valid_range = [i for i in range(n_indices) if i not in exclude_indices]
        if len(valid_range) < n_samples:
            valid_range = list(range(n_indices))
        return np.random.choice(valid_range, size=n_samples, replace=False)
    
    def _mutate_current_to_pbest_batch(self, pop, fitness, F_vec):
        """
        Current-to-pbest/2/bin mutation with cultural memory.
        Non-standard: uses 2 difference vectors and pbest guidance.
        """
        n_best = max(1, int(self.p_best * self.NP))
        pbest_indices = self._get_pbest_indices(fitness, n_best)
        pbest = pop[pbest_indices]
        
        NP = self.NP
        dim = self.dim
        all_indices = np.arange(NP)
        
        r1 = np.zeros(NP, dtype=int)
        r2 = np.zeros(NP, dtype=int)
        r3 = np.zeros(NP, dtype=int)
        r4 = np.zeros(NP, dtype=int)
        r5 = np.zeros(NP, dtype=int)
        
        for i in range(NP):
            exclude = {i}
            r1[i] = self._sample_random_indices(NP, 1, exclude)[0]
            exclude.add(r1[i])
            r2[i] = self._sample_random_indices(NP, 1, exclude)[0]
            exclude.add(r2[i])
            r3[i] = self._sample_random_indices(NP, 1, exclude)[0]
            exclude.add(r3[i])
            r4[i] = self._sample_random_indices(NP, 1, exclude)[0]
            exclude.add(r4[i])
            r5[i] = self._sample_random_indices(NP, 1, exclude)[0]
        
        pbest_idx_for_each = np.random.choice(pbest_indices, size=NP)
        pbest_selected = pop[pbest_idx_for_each]
        
        mutant = pop + F_vec.reshape(-1, 1) * (pbest_selected - pop) + \
                 F_vec.reshape(-1, 1) * (pop[r1] - pop[r2]) + \
                 0.5 * F_vec.reshape(-1, 1) * (pop[r3] - pop[r4])
        
        if len(self.archive) > 0:
            archive_array = np.array(self.archive)
            n_archive = len(archive_array)
            r_archive = np.random.randint(0, n_archive, size=NP)
            mutant += 0.3 * F_vec.reshape(-1, 1) * (archive_array[r_archive] - pop[r5])
        
        return mutant
    
    def _crossover_binomial_batch(self, pop, mutant, CR_vec):
        """Binomial crossover with per-individual CR."""
        NP, dim = pop.shape
        trial = np.copy(pop)
        
        j_random = np.random.randint(0, dim, size=NP)
        mask = np.random.random((NP, dim)) < CR_vec.reshape(-1, 1)
        mask[np.arange(NP), j_random] = True
        
        trial = np.where(mask, mutant, pop)
        return trial
    
    def _select_survivors_batch(self, pop, fitness, trial, trial_fitness):
        """Greedy (mu+0) selection: keep better of target and trial."""
        better_mask = trial_fitness < fitness
        new_pop = np.copy(pop)
        new_fitness = np.copy(fitness)
        new_pop[better_mask] = trial[better_mask]
        new_fitness[better_mask] = trial_fitness[better_mask]
        return new_pop, new_fitness
    
    def _update_cultural_archive(self, pop, fitness, trial, trial_fitness, better_mask):
        """Update cultural archive with successful solutions."""
        improved_trials = trial[better_mask]
        
        for sol in improved_trials:
            if len(self.archive) < self.archive_max_size():
                self.archive.append(sol.copy())
            else:
                replace_idx = np.random.randint(0, len(self.archive))
                self.archive[replace_idx] = sol.copy()
        
        if len(self.archive) > 2 * self.NP:
            self.archive = list(np.array(self.archive)[np.random.permutation(len(self.archive))[:2*self.NP]])
    
    def archive_max_size(self):
        """Maximum archive size based on population."""
        return min(3 * self.NP, 500)
    
    def _adapt_parameters_batch(self, fitness, trial_fitness, F_vec, CR_vec, better_mask):
        """
        Success-based parameter adaptation.
        Records successful F/CR values for future adaptation.
        """
        n_success = np.sum(better_mask)
        if n_success > 0:
            successful_F = F_vec[better_mask]
            successful_CR = CR_vec[better_mask]
            
            if len(self.successful_F) > 0:
                self.successful_F = np.concatenate([self.successful_F, successful_F])
                self.successful_CR = np.concatenate([self.successful_CR, successful_CR])
            else:
                self.successful_F = successful_F
                self.successful_CR = successful_CR
            
            max_history = 10 * self.NP
            if len(self.successful_F) > max_history:
                keep_indices = np.random.permutation(len(self.successful_F))[:max_history]
                self.successful_F = self.successful_F[keep_indices]
                self.successful_CR = self.successful_CR[keep_indices]
    
    def _compute_adaptive_F(self):
        """Compute adaptive F values for next generation."""
        if len(self.successful_F) >= 5:
            mean_F = np.mean(self.successful_F)
            std_F = np.std(self.successful_F) + 1e-8
            F_vec = np.random.normal(mean_F, std_F * 0.5, size=self.NP)
            F_vec = np.clip(F_vec, 0.1, 2.0)
        else:
            F_vec = np.random.uniform(0.3, 0.9, size=self.NP)
        return F_vec
    
    def _compute_adaptive_CR(self):
        """Compute adaptive CR values for next generation."""
        if len(self.successful_CR) >= 5:
            mean_CR = np.mean(self.successful_CR)
            std_CR = np.std(self.successful_CR) + 1e-8
            CR_vec = np.random.normal(mean_CR, std_CR * 0.3, size=self.NP)
            CR_vec = np.clip(CR_vec, 0.0, 1.0)
        else:
            CR_vec = np.random.uniform(0.7, 0.95, size=self.NP)
        return CR_vec
    
    def _compute_diversity(self, pop):
        """Compute population diversity via average pairwise distance."""
        if len(pop) < 2:
            return 0.0
        mean_individual = np.mean(pop, axis=0)
        distances = np.sqrt(np.sum((pop - mean_individual) ** 2, axis=1))
        return np.mean(distances)
    
    def _check_stagnation(self, fitness):
        """Check if algorithm is stagnating."""
        current_best = np.min(fitness)
        if current_best < self.last_best_fitness - 1e-10:
            self.stagnation_count = 0
            self.last_best_fitness = current_best
            return False
        else:
            self.stagnation_count += 1
            return self.stagnation_count > 15
    
    def _restart_if_stagnant(self, pop, fitness):
        """Restart population when stagnating, preserving best."""
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        new_pop = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        new_pop[0] = best_individual
        
        for i in range(1, self.NP):
            if len(self.archive) > 0:
                archive_array = np.array(self.archive)
                donor_idx = np.random.randint(0, len(archive_array))
                noise = np.random.normal(0, 0.1, self.dim)
                new_pop[i] = np.clip(archive_array[donor_idx] + noise, 
                                     self.lower_bound, self.upper_bound)
        
        self.stagnation_count = 0
        self.successful_F = np.array([])
        self.successful_CR = np.array([])
        
        return new_pop, best_fitness
    
    def _eval_wrapper(self, pop):
        """Wrapper for evaluation with safety checks."""
        pop = self._clip_to_bounds_batch(pop)
        return pop
    
    def _get_best_solution(self, pop, fitness):
        """Get best solution found."""
        best_idx = np.argmin(fitness)
        return fitness[best_idx], pop[best_idx].copy()
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        pop = self._initialize_population()
        pop = self._eval_wrapper(pop)
        pop = self._clip_to_bounds_batch(pop)
        fitness = func(pop)
        
        if len(fitness) < self.NP:
            fitness = fitness[:len(pop)]
            pop = pop[:len(fitness)]
            self.NP = len(pop)
        
        self._initialize_archive()
        
        while not stopping_condition():
            F_vec = self._compute_adaptive_F()
            CR_vec = self._compute_adaptive_CR()
            
            mutant = self._mutate_current_to_pbest_batch(pop, fitness, F_vec)
            mutant = self._clip_to_bounds_batch(mutant)
            
            trial = self._crossover_binomial_batch(pop, mutant, CR_vec)
            trial = self._clip_to_bounds_batch(trial)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trial)
            
            if len(trial_fitness) < len(trial):
                actual_len = len(trial_fitness)
                trial = trial[:actual_len]
                trial_fitness = trial_fitness[:actual_len]
                pop = pop[:actual_len]
                fitness = fitness[:actual_len]
                self.NP = actual_len
                if self.NP < 5:
                    break
            
            better_mask = trial_fitness < fitness
            
            self._update_cultural_archive(pop, fitness, trial, trial_fitness, better_mask)
            self._adapt_parameters_batch(fitness, trial_fitness, F_vec, CR_vec, better_mask)
            
            pop, fitness = self._select_survivors_batch(pop, fitness, trial, trial_fitness)
            
            self.generation += 1
            
            if stopping_condition():
                break
            
            if self._check_stagnation(fitness):
                diversity = self._compute_diversity(pop)
                if diversity < 0.1 * self.dim:
                    pop, fitness[0] = self._restart_if_stagnant(pop, fitness)
        
        f_opt, x_opt = self._get_best_solution(pop, fitness)
        return f_opt, x_opt
```