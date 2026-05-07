```python
import numpy as np


class AdaptiveStrategyPoolDE:
    """
    Differential Evolution with Adaptive Strategy Pool and Parameter Evolution.
    
    Key features:
    - Pool of 3 mutation strategies with softmax-weighted selection
    - Per-individual adaptive F and CR via exponential moving averages
    - Strategy weight adaptation based on cumulative success tracking
    - Diversity-triggered restart mechanism
    - All operations vectorized over the population
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(200, max(60, 5 * dim))
        self.bounds = np.array([-100.0, 100.0])
        self.p_percentile = 0.2
        
        self.f_best = np.inf
        self.x_best = None
        self.g = 0
        
        self._init_adaptation_state()
    
    def _init_adaptation_state(self):
        self.F_avg = 0.5
        self.CR_avg = 0.5
        self.success_f = []
        self.success_cr = []
        self.improvement_ratios = []
        
        self.strategy_names = ['current_to_best', 'rand', 'current_to_pbest']
        self.strategy_scores = np.ones(len(self.strategy_names))
        self.strategy_probs = np.ones(len(self.strategy_names)) / len(self.strategy_names)
        self.strategy_success = np.zeros(len(self.strategy_names))
        self.strategy_trials = np.zeros(len(self.strategy_names))
        self.current_strategy_idx = 0
        self.last_best_score = np.inf
        self.learning_rate = 0.1
        
        self.diversity_threshold = 0.01
        self.last_restart_gen = 0
    
    def _initialize_population(self):
        pop = np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.np, self.dim)
        )
        return pop
    
    def _mutate_current_to_best_batch(self, pop, best_vec):
        r1, r2 = self._get_random_indices(self.np, exclude_indices=None, count=2)
        diff = pop[r1] - pop[r2]
        return pop + self.F_avg * diff + self.F_avg * (best_vec - pop)
    
    def _mutate_rand_batch(self, pop):
        idx_a, idx_b, idx_c = self._get_random_indices(self.np, count=3)
        return pop[idx_a] + self.F_avg * (pop[idx_b] - pop[idx_c])
    
    def _mutate_current_to_pbest_batch(self, pop, fitness):
        n_pbest = max(1, int(self.p_percentile * self.np))
        pbest_indices = np.argpartition(fitness, n_pbest)[:n_pbest]
        pbest = pop[pbest_indices]
        pbest_selected = pbest[np.random.randint(0, n_pbest, size=self.np)]
        r1, r2 = self._get_random_indices(self.np, count=2)
        diff = pop[r1] - pop[r2]
        return pop + self.F_avg * diff + self.F_avg * (pbest_selected - pop)
    
    def _mutate_batch(self, pop, fitness):
        strategy_idx = self.current_strategy_idx
        self.strategy_trials[strategy_idx] += self.np
        best_vec = pop[np.argmin(fitness)]
        
        if strategy_idx == 0:
            mutants = self._mutate_current_to_best_batch(pop, best_vec)
        elif strategy_idx == 1:
            mutants = self._mutate_rand_batch(pop)
        else:
            mutants = self._mutate_current_to_pbest_batch(pop, fitness)
        
        return np.clip(mutants, self.bounds[0], self.bounds[1])
    
    def _crossover_batch(self, pop, mutants):
        CR = np.clip(self.CR_avg, 0.0, 0.95)
        mask = np.random.random((self.np, self.dim)) < CR
        first_hit = np.argmax(mask, axis=1)
        mask[np.arange(self.np), first_hit] = True
        trials = np.where(mask, mutants, pop)
        return np.clip(trials, self.bounds[0], self.bounds[1])
    
    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        better_mask = trial_fitness < fitness
        new_pop = np.where(better_mask[:, np.newaxis], trials, pop)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        improvements = fitness - trial_fitness
        
        success_mask = better_mask & np.isfinite(trial_fitness)
        if np.any(success_mask):
            self.success_f.extend(self.F_avg for _ in range(np.sum(success_mask)))
            self.success_cr.extend(CR for CR in self.CR_avg for _ in range(np.sum(success_mask)))
            self.improvement_ratios.extend(
                improvements[success_mask] / (np.abs(fitness[success_mask]) + 1e-8)
            )
            self.strategy_success[self.current_strategy_idx] += np.sum(success_mask)
        
        return new_pop, new_fitness, np.mean(improvements)
    
    def _adapt_parameters_batch(self):
        if len(self.success_f) < 5:
            return
        
        avg_improvement = np.mean(self.improvement_ratios[-50:])
        f_weight = np.clip(0.5 + 0.3 * avg_improvement, 0.1, 1.0)
        cr_weight = np.clip(0.5 + 0.2 * avg_improvement, 0.0, 1.0)
        
        if len(self.success_f) >= 10:
            recent_f = self.success_f[-10:]
            recent_cr = self.success_cr[-10:]
            self.F_avg = f_weight * np.mean(recent_f) + (1 - f_weight) * self.F_avg
            self.CR_avg = cr_weight * np.mean(recent_cr) + (1 - cr_weight) * self.CR_avg
        
        self.F_avg = np.clip(self.F_avg, 0.1, 1.0)
        self.CR_avg = np.clip(self.CR_avg, 0.0, 0.95)
    
    def _adapt_strategy_weights(self):
        if self.g - self.last_restart_gen < 10:
            return
        
        total_success = np.sum(self.strategy_success)
        if total_success < 5:
            return
        
        success_rate = self.strategy_success / (self.strategy_trials + 1e-10)
        lr = self.learning_rate
        
        if abs(self.f_best - self.last_best_score) < 1e-6:
            lr *= 0.5
        
        self.strategy_scores = (
            (1 - lr) * self.strategy_scores +
            lr * (success_rate * 100 + 0.1)
        )
        
        exp_scores = np.exp(self.strategy_scores - np.max(self.strategy_scores))
        self.strategy_probs = exp_scores / np.sum(exp_scores)
        
        self.current_strategy_idx = np.random.choice(
            len(self.strategy_names), p=self.strategy_probs
        )
        
        if self.g % 50 == 0:
            self.strategy_success *= 0.9
            self.last_best_score = self.f_best
    
    def _compute_diversity(self, pop):
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        cv = np.std(distances) / (np.mean(distances) + 1e-10)
        return cv
    
    def _restart_if_stagnant(self, pop, fitness):
        if self.g - self.last_restart_gen < 50:
            return pop
        
        diversity = self._compute_diversity(pop)
        if diversity < self.diversity_threshold:
            n_keep = max(1, self.np // 5)
            best_idx = np.argpartition(fitness, n_keep)[:n_keep]
            
            new_pop = self._initialize_population()
            new_pop[:n_keep] = pop[best_idx]
            new_pop[n_keep:] = np.clip(
                pop[n_keep:] + np.random.randn(self.np - n_keep, self.dim) * 20,
                self.bounds[0], self.bounds[1]
            )
            
            self.last_restart_gen = self.g
            self.strategy_scores = np.ones(len(self.strategy_names))
            self.strategy_probs = np.ones(len(self.strategy_names)) / len(self.strategy_names)
            return new_pop
        
        return pop
    
    def _get_random_indices(self, n, exclude_indices=None, count=1):
        if exclude_indices is not None:
            valid = np.setdiff1d(np.arange(n), np.atleast_1d(exclude_indices))
            return tuple(np.random.choice(valid, size=n, replace=True) for _ in range(count))
        return tuple(np.random.randint(0, n, size=n) for _ in range(count))
    
    def _update_best(self, fitness, pop):
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, fitness, np.inf))
        if fitness[best_idx] < self.f_best:
            self.f_best = fitness[best_idx]
            self.x_best = pop[best_idx].copy()
    
    def __call__(self, func, stopping_condition):
        pop = self._initialize_population()
        fitness = func(pop)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[len(fitness):] = np.nan
        
        valid = np.isfinite(fitness)
        if np.any(valid):
            self.f_best = np.min(fitness[valid])
            self.x_best = pop[np.argmin(np.where(valid, fitness, np.inf))].copy()
        
        self.g = 0
        while not stopping_condition():
            self.g += 1
            
            mutants = self._mutate_batch(pop, fitness)
            trials = self._crossover_batch(pop, mutants)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.resize(trial_fitness, len(trials))
                trial_fitness[len(trial_fitness):] = np.nan
            
            pop, fitness, mean_improvement = self._select_survivors_batch(
                pop, fitness, trials, trial_fitness
            )
            self._update_best(fitness, pop)
            
            if self.g % 10 == 0:
                self._adapt_parameters_batch()
            
            self._adapt_strategy_weights()
            
            if self.g % 25 == 0:
                pop = self._restart_if_stagnant(pop, fitness)
                fitness = func(pop)
                if len(fitness) < self.np:
                    fitness = np.resize(fitness, self.np)
                    fitness[len(fitness):] = np.nan
        
        return self.f_best, self.x_best if self.x_best is not None else pop[0]
```