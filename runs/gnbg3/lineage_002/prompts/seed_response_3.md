```python
import numpy as np


class AdaptiveMultiStrategyPSO:
    """
    Hybrid optimizer combining PSO velocity dynamics with DE-style mutations
    and adaptive strategy selection based on historical performance.
    
    Key design decisions (different from DERing/CMA-ES):
    - PSO inertia + cognitive/social terms as base dynamics
    - DE mutations overlaid on PSO velocity
    - Strategy selection via softmax-weighted probabilities
    - Covariance-like step size adaptation
    - Diversity-based restart mechanism
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = kwargs.get('np', min(120, max(60, 4 * dim)))
        self.x_min = -100.0
        self.x_max = 100.0
        
        # PSO parameters (adaptive)
        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494
        
        # DE parameters (adaptive)
        self.f_scale = 0.5
        self.cr_base = 0.9
        
        # Strategy system
        self.n_strategies = 6
        self.strategy_scores = np.ones(self.n_strategies)
        self.strategy_counts = np.zeros(self.n_strategies, dtype=int)
        self.history_len = 5
        self.improvement_history = []
        
        # Restart system
        self.min_diversity = 1e-6
        self.stagnation_limit = 15
        self.stagnation_count = 0
        
        # State
        self.pop = None
        self.fit = None
        self.vel = None
        self.pbest = None
        self.pbest_fit = None
        self.gbest = None
        self.gbest_fit = np.inf
        self.gen = 0
        self.active_count = 0

    def _initialize_population(self, func):
        self.pop = np.random.uniform(self.x_min, self.x_max, (self.np, self.dim))
        self.pop = np.clip(self.pop, self.x_min, self.x_max)
        self.fit = func(self.pop)
        if len(self.fit) < self.np:
            self.fit = np.pad(self.fit, (0, self.np - len(self.fit)), constant_values=np.inf)
        self.active_count = min(len(self.fit), self.np)
        
        self.vel = np.zeros((self.np, self.dim))
        self.pbest = self.pop.copy()
        self.pbest_fit = self.fit.copy()
        
        best_idx = np.argmin(self.fit[:self.active_count])
        self.gbest = self.pop[best_idx].copy()
        self.gbest_fit = self.fit[best_idx]

    def _compute_diversity(self):
        centroid = np.mean(self.pop[:self.active_count], axis=0)
        distances = np.linalg.norm(
            self.pop[:self.active_count] - centroid, axis=1
        )
        return np.mean(distances)

    def _select_strategy_batch(self):
        probs = self.strategy_scores / np.sum(self.strategy_scores)
        return np.random.choice(self.n_strategies, size=self.np, p=probs)

    def _mutate_velocity_psu_batch(self):
        r1 = np.random.rand(self.np, self.dim)
        r2 = np.random.rand(self.np, self.dim)
        
        cognitive = self.c1 * r1 * (self.pbest - self.pop)
        
        lbest_idx = (np.arange(self.np) + 1) % self.np
        social = self.c2 * r2 * (self.pbest[lbest_idx] - self.pop)
        
        self.vel = self.w * self.vel + cognitive + social

    def _mutate_de_rand1_batch(self, indices, strategies):
        mask = strategies == 0
        idx_mask = indices[mask]
        if len(idx_mask) == 0:
            return
        
        n = len(idx_mask)
        perm = np.random.permutation(self.np)
        r1, r2, r3 = perm[:3*n].reshape(3, n, 1)
        r1, r2, r3 = r1.squeeze(), r2.squeeze(), r3.squeeze()
        
        f = self.f_scale * (0.9 + 0.2 * np.random.rand(n, 1))
        mutation = self.pop[r1] + f * (self.pop[r2] - self.pop[r3])
        mutation = np.clip(mutation, self.x_min, self.x_max)
        self.vel[idx_mask] = mutation - self.pop[idx_mask]

    def _mutate_de_target_best_batch(self, indices, strategies):
        mask = strategies == 1
        idx_mask = indices[mask]
        if len(idx_mask) == 0:
            return
        
        n = len(idx_mask)
        perm = np.random.permutation(self.np)
        r1, r2 = perm[:2*n].reshape(2, n, 1)
        r1, r2 = r1.squeeze(), r2.squeeze()
        
        f = self.f_scale * (0.8 + 0.4 * np.random.rand(n, 1))
        mutation = self.pop[r1] + f * (self.gbest - self.pop[r1]) + f * (self.pop[r2] - self.pop[np.random.randint(0, self.np, n)])
        mutation = np.clip(mutation, self.x_min, self.x_max)
        self.vel[idx_mask] = mutation - self.pop[idx_mask]

    def _mutate_de_current_pbest_batch(self, indices, strategies):
        mask = strategies == 2
        idx_mask = indices[mask]
        if len(idx_mask) == 0:
            return
        
        n = len(idx_mask)
        n_elite = max(3, self.np // 10)
        sorted_idx = np.argsort(self.fit[:self.active_count])
        elite_pool = sorted_idx[:n_elite]
        
        pbest_idx = elite_pool[np.random.randint(0, n_elite, n)]
        r1, r2 = np.random.randint(0, self.np, (2, n))
        
        f = self.f_scale * (0.6 + 0.6 * np.random.rand(n, 1))
        mutation = self.pop[idx_mask] + f * (self.pbest[pbest_idx] - self.pop[idx_mask]) + f * (self.pop[r1] - self.pop[r2])
        mutation = np.clip(mutation, self.x_min, self.x_max)
        self.vel[idx_mask] = mutation - self.pop[idx_mask]

    def _mutate_de_rings_batch(self, indices, strategies):
        mask = strategies == 3
        idx_mask = indices[mask]
        if len(idx_mask) == 0:
            return
        
        n = len(idx_mask)
        r1 = (indices - 2) % self.np
        r2 = (indices - 1) % self.np
        r3 = (indices + 1) % self.np
        r4 = (indices + 2) % self.np
        
        f = self.f_scale * (0.7 + 0.3 * np.random.rand(n, 1))
        mutation = self.pop[r1] + f * (self.pop[r2] - self.pop[r3]) + f * (self.pop[r4] - self.pop[np.random.randint(0, self.np, n)])
        mutation = np.clip(mutation, self.x_min, self.x_max)
        self.vel[idx_mask] = mutation - self.pop[idx_mask]

    def _mutate_de_grandchild_batch(self, indices, strategies):
        mask = strategies == 4
        idx_mask = indices[mask]
        if len(idx_mask) == 0:
            return
        
        n = len(idx_mask)
        perm = np.random.permutation(self.np)
        r1, r2, r3, r4, r5 = perm[:5*n].reshape(5, n, 1)
        r1, r2, r3, r4, r5 = r1.squeeze(), r2.squeeze(), r3.squeeze(), r4.squeeze(), r5.squeeze()
        
        f1 = self.f_scale * (0.6 + 0.4 * np.random.rand(n, 1))
        f2 = self.f_scale * (0.5 + 0.5 * np.random.rand(n, 1))
        mutation = self.pop[r1] + f1 * (self.pop[r2] - self.pop[r3]) + f2 * (self.pop[r4] - self.pop[r5])
        mutation = np.clip(mutation, self.x_min, self.x_max)
        self.vel[idx_mask] = mutation - self.pop[idx_mask]

    def _mutate_chaotic_batch(self, indices, strategies):
        mask = strategies == 5
        idx_mask = indices[mask]
        if len(idx_mask) == 0:
            return
        
        n = len(idx_mask)
        r = np.random.rand(n, 1)
        chaos = 4 * r * (1 - r)
        
        centroid = np.mean(self.pop[:self.active_count], axis=0)
        mutation = centroid + self.f_scale * chaos * (self.pop[np.random.randint(0, self.np, (n, 1))] - centroid)
        mutation = np.clip(mutation, self.x_min, self.x_max)
        self.vel[idx_mask] = mutation - self.pop[idx_mask]

    def _apply_velocity_batch(self):
        max_vel = 0.2 * (self.x_max - self.x_min)
        self.vel = np.clip(self.vel, -max_vel, max_vel)
        self.pop = self.pop + self.vel
        self.pop = np.clip(self.pop, self.x_min, self.x_max)

    def _crossover_batch(self, trials):
        cr = self.cr_base * (0.8 + 0.4 * np.random.rand(self.np, 1))
        cr = np.clip(cr, 0.3, 1.0)
        
        mask = np.random.rand(self.np, self.dim) < cr
        first_idx = np.random.randint(0, self.dim, size=self.np)
        first_mask = np.zeros((self.np, self.dim), dtype=bool)
        first_mask[np.arange(self.np), first_idx] = True
        
        final_mask = mask | first_mask
        offspring = np.where(final_mask, trials, self.pop)
        return np.clip(offspring, self.x_min, self.x_max)

    def _select_survivors_batch(self, trials, trial_fit):
        better = trial_fit < self.fit
        self.pop = np.where(better[:, np.newaxis], trials, self.pop)
        self.fit = np.where(better, trial_fit, self.fit)
        return np.sum(better)

    def _update_personal_best_batch(self):
        improved = self.fit < self.pbest_fit
        self.pbest = np.where(improved[:, np.newaxis], self.pop, self.pbest)
        self.pbest_fit = np.where(improved, self.fit, self.pbest_fit)

    def _update_global_best_batch(self):
        best_local_idx = np.argmin(self.fit[:self.active_count])
        if self.fit[best_local_idx] < self.gbest_fit:
            self.gbest = self.pop[best_local_idx].copy()
            self.gbest_fit = self.fit[best_local_idx]

    def _update_strategy_scores(self, n_improved):
        improvement_rate = n_improved / max(1, self.np)
        decay = 0.95
        boost = 1.05
        
        if improvement_rate > 0.2:
            self.strategy_scores *= decay
            self.strategy_scores += 0.1
        else:
            self.strategy_scores *= (2.0 - decay)
        
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
        self.improvement_history.append(improvement_rate)
        if len(self.improvement_history) > self.history_len:
            self.improvement_history.pop(0)

    def _adapt_parameters(self):
        if len(self.improvement_history) < 2:
            return
        
        recent_trend = np.mean(self.improvement_history[-2:])
        if recent_trend > 0.15:
            self.w *= 0.98
            self.f_scale *= 1.02
        elif recent_trend < 0.05:
            self.w *= 1.02
            self.f_scale *= 0.98
            self.c1 *= 1.01
            self.c2 *= 1.01
        
        self.w = np.clip(self.w, 0.3, 0.9)
        self.f_scale = np.clip(self.f_scale, 0.3, 1.5)
        self.c1 = np.clip(self.c1, 1.0, 2.5)
        self.c2 = np.clip(self.c2, 1.0, 2.5)

    def _restart_if_needed(self):
        diversity = self._compute_diversity()
        
        if len(self.improvement_history) >= 3:
            avg_improvement = np.mean(self.improvement_history[-3:])
            if avg_improvement < 0.02:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        else:
            self.stagnation_count = 0
        
        should_restart = (diversity < self.min_diversity or 
                         self.stagnation_count >= self.stagnation_limit)
        
        if should_restart:
            n_replace = self.np // 2
            replace_idx = np.random.permutation(self.np)[:n_replace]
            self.pop[replace_idx] = np.random.uniform(
                self.x_min, self.x_max, (n_replace, self.dim)
            )
            self.vel[replace_idx] = np.zeros((n_replace, self.dim))
            self.stagnation_count = 0
            self.improvement_history.clear()
            self.strategy_scores = np.ones(self.n_strategies)

    def _build_trial_batch(self):
        indices = np.arange(self.np)
        strategies = self._select_strategy_batch()
        
        self._mutate_velocity_psu_batch()
        self._mutate_de_rand1_batch(indices, strategies)
        self._mutate_de_target_best_batch(indices, strategies)
        self._mutate_de_current_pbest_batch(indices, strategies)
        self._mutate_de_rings_batch(indices, strategies)
        self._mutate_de_grandchild_batch(indices, strategies)
        self._mutate_chaotic_batch(indices, strategies)
        
        self._apply_velocity_batch()
        
        perm = np.random.permutation(self.np)
        r1, r2 = perm[:2*self.np].reshape(2, self.np, 1)
        r1, r2 = r1.squeeze(), r2.squeeze()
        
        f = self.f_scale * (0.8 + 0.4 * np.random.rand(self.np, 1))
        base_mutants = self.pop[r1] + f * (self.pop[r2] - self.pop[np.random.randint(0, self.np, self.np)])
        base_mutants = np.clip(base_mutants, self.x_min, self.x_max)
        
        return self._crossover_batch(base_mutants)

    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        
        while not stopping_condition():
            trials = self._build_trial_batch()
            trial_fit = func(trials)
            
            if len(trial_fit) < self.np:
                actual_len = len(trial_fit)
                trial_fit = np.pad(trial_fit, (0, self.np - actual_len), constant_values=np.inf)
                trials = trials[:actual_len] if actual_len > 0 else trials
                if actual_len == 0:
                    break
            
            if stopping_condition():
                break
            
            n_improved = self._select_survivors_batch(trials, trial_fit)
            self._update_personal_best_batch()
            self._update_global_best_batch()
            self._update_strategy_scores(n_improved)
            self._adapt_parameters()
            self._restart_if_needed()
            self.gen += 1
        
        return self.gbest_fit, self.gbest
```