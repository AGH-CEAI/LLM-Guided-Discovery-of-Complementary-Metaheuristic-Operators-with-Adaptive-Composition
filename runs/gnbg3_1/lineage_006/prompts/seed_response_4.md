```python
import numpy as np


class AdaptiveTopologyMemeticOptimizer:
    """
    Hybrid optimizer combining DE-style mutation with adaptive topology,
    local search refinement, and covariance-guided sampling.
    Designed for multi-modal and rugged landscapes.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(8 * dim, 400)
        self.bounds = np.array([-100.0, 100.0])
        self.F_init = 0.5
        self.CR_init = 0.5
        self.local_search_freq = 5
        self.local_search_trials = 3
        self.diversity_threshold = 1e-6
        self.stagnation_limit = 20
        self.F_min, self.F_max = 0.1, 1.0
        self.CR_min, self.CR_max = 0.0, 1.0
        
    def __call__(self, func, stopping_condition):
        pop, fit = self._initialize_population(func)
        x_opt, f_opt = self._update_best(pop, fit, None, None)
        gen, stagnant_gens = 0, 0
        F, CR = self.F_init, self.CR_init
        history = []
        
        while not stopping_condition():
            trials, topo_indices = self._build_trials_batched(pop, fit, F, CR)
            
            if len(trials) < len(pop):
                break
                
            trial_fit = func(trials)
            if len(trial_fit) < len(trials):
                trial_fit = np.append(trial_fit, np.full(len(trials) - len(trial_fit), np.nan))
            
            if stopping_condition():
                break
                
            pop, fit, accepted = self._select_survivors_batch(pop, fit, trials, trial_fit)
            x_opt, f_opt = self._update_best(pop, fit, x_opt, f_opt)
            
            F = self._adapt_mutation_scale(F, accepted)
            CR = self._adapt_crossover_rate(CR, accepted)
            
            if gen % self.local_search_freq == 0:
                pop, fit, x_opt, f_opt = self._apply_local_search_batch(
                    pop, fit, func, x_opt, f_opt
                )
            
            if stopping_condition():
                break
                
            history.append(f_opt)
            stagnant_gens = self._check_stagnation(history, stagnant_gens)
            
            if stagnant_gens >= self.stagnation_limit:
                pop, fit = self._restart_if_stagnant(pop, fit, func, x_opt)
                stagnant_gens = 0
                history = history[-5:]
            
            pop = self._clip_to_bounds(pop)
            gen += 1
            
        return f_opt, x_opt
    
    def _initialize_population(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.np, self.dim))
        pop = self._clip_to_bounds(pop)
        fit = func(pop)
        if len(fit) < self.np:
            fit = np.append(fit, np.full(self.np - len(fit), np.nan))
        return pop, fit
    
    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.bounds[0], self.bounds[1])
    
    def _update_best(self, pop, fit, x_opt, f_opt):
        valid = ~np.isnan(fit)
        if not np.any(valid):
            return x_opt, f_opt
        best_idx = np.argmin(fit[valid])
        best_val = fit[valid][best_idx]
        best_x = pop[valid][best_idx] if x_opt is None else x_opt
        best_f = best_val if f_opt is None else f_opt
        if best_val < f_opt:
            best_x = pop[valid][best_idx]
            best_f = best_val
        return best_x, best_f
    
    def _build_trials_batched(self, pop, fit, F, CR):
        np_pop = len(pop)
        trials = np.empty((np_pop, self.dim))
        topo_indices = np.zeros((np_pop, 4), dtype=int)
        
        for i in range(np_pop):
            indices = self._select_topology_indices(i, fit, np_pop)
            topo_indices[i] = indices
            
            r1, r2, r3, r4 = indices
            p_best = pop[np.argmin(fit[~np.isnan(fit)])]
            p_rand = pop[r1]
            
            strategy = np.random.randint(3)
            if strategy == 0:
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
            elif strategy == 1:
                mutant = p_best + F * (pop[r1] - pop[r2])
            else:
                mutant = pop[r1] + F * (pop[r2] - pop[r3] + pop[r4] - pop[i])
            
            mask = np.random.random(self.dim) < CR
            trials[i] = np.where(mask, mutant, pop[i])
        
        return trials, topo_indices
    
    def _select_topology_indices(self, idx, fit, np_pop):
        valid = ~np.isnan(fit)
        valid_indices = np.where(valid)[0]
        
        if len(valid_indices) < 4:
            indices = np.random.choice(np_pop, 4, replace=False)
            while idx in indices:
                indices = np.random.choice(np_pop, 4, replace=False)
            return indices
        
        fitness_sorted = sorted(valid_indices, key=lambda x: fit[x])
        
        candidates = [i for i in fitness_sorted if i != idx]
        selected = []
        
        selected.append(candidates[np.random.randint(min(3, len(candidates)))])
        
        remaining = [i for i in candidates if i not in selected]
        while len(selected) < 4 and remaining:
            selected.append(remaining[np.random.randint(len(remaining))])
            remaining = [i for i in remaining if i != selected[-1]]
        
        while len(selected) < 4:
            r = np.random.randint(np_pop)
            if r != idx and r not in selected:
                selected.append(r)
        
        return np.array(selected[:4])
    
    def _select_survivors_batch(self, pop, fit, trials, trial_fit):
        accepted = fit > trial_fit
        mask = accepted[:, np.newaxis] if accepted.ndim == 1 else accepted.reshape(-1, 1)
        new_pop = np.where(mask, trials, pop)
        new_fit = np.where(accepted, trial_fit, fit)
        return new_pop, new_fit, accepted
    
    def _adapt_mutation_scale(self, F, accepted):
        success_rate = np.mean(accepted)
        if success_rate > 0.2:
            F = min(F * 1.1, self.F_max)
        elif success_rate < 0.05:
            F = max(F * 0.9, self.F_min)
        return F
    
    def _adapt_crossover_rate(self, CR, accepted):
        success_rate = np.mean(accepted)
        if success_rate > 0.15:
            CR = min(CR + 0.05, self.CR_max)
        elif success_rate < 0.05:
            CR = max(CR - 0.05, self.CR_min)
        return CR
    
    def _apply_local_search_batch(self, pop, fit, func, x_opt, f_opt):
        sorted_indices = np.argsort(fit)
        elite_count = min(self.local_search_trials, self.np // 5)
        
        for i in range(elite_count):
            idx = sorted_indices[i]
            if np.isnan(fit[idx]):
                continue
            
            candidate = pop[idx].copy()
            best_ls = candidate.copy()
            best_f_ls = fit[idx]
            
            step_size = 0.5 * (self.bounds[1] - self.bounds[0]) / (10 + self.dim)
            
            for _ in range(3):
                direction = np.random.randn(self.dim)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                
                new_point = candidate + step_size * direction
                new_point = self._clip_to_bounds(new_point)
                new_f = func(new_point.reshape(1, -1))
                
                if len(new_f) > 0 and new_f[0] < best_f_ls:
                    best_ls = new_point
                    best_f_ls = new_f[0]
                else:
                    new_point = candidate - step_size * direction
                    new_point = self._clip_to_bounds(new_point)
                    new_f = func(new_point.reshape(1, -1))
                    if len(new_f) > 0 and new_f[0] < best_f_ls:
                        best_ls = new_point
                        best_f_ls = new_f[0]
            
            if best_f_ls < fit[idx]:
                pop[idx] = best_ls
                fit[idx] = best_f_ls
                
                if best_f_ls < f_opt:
                    x_opt = best_ls.copy()
                    f_opt = best_f_ls
        
        return pop, fit, x_opt, f_opt
    
    def _check_stagnation(self, history, stagnant_gens):
        if len(history) < 10:
            return 0
        recent_improvement = history[-1] < history[-5] - 1e-12
        return 0 if recent_improvement else stagnant_gens + 1
    
    def _restart_if_stagnant(self, pop, fit, func, x_opt):
        diversity = self._compute_diversity(pop)
        
        if diversity < self.diversity_threshold:
            pop = np.random.uniform(
                self.bounds[0], self.bounds[1], (self.np, self.dim)
            )
        else:
            n_replace = self.np // 3
            replace_idx = np.random.choice(self.np, n_replace, replace=False)
            pop[replace_idx] = np.random.uniform(
                self.bounds[0], self.bounds[1], (n_replace, self.dim)
            )
        
        if x_opt is not None:
            worst_idx = np.argmax(fit)
            pop[worst_idx] = x_opt.copy()
        
        pop = self._clip_to_bounds(pop)
        fit = func(pop)
        if len(fit) < self.np:
            fit = np.append(fit, np.full(self.np - len(fit), np.nan))
        
        return pop, fit
    
    def _compute_diversity(self, pop):
        if len(pop) < 2:
            return 1.0
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances) + 1e-10
    
    def _sample_trials_batch(self, pop, mean, cov_factor, count):
        base = np.random.randn(count, self.dim)
        scaled = base * cov_factor
        trials = pop[np.random.randint(len(pop), size=count))] + scaled
        return self._clip_to_bounds(trials)
```