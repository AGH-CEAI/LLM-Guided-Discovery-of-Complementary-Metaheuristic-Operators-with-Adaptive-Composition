```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Particle Swarm Optimizer with adaptive ring topology, covariance-guided exploration,
    and quantum-inspired velocity perturbation. Uses batched operations throughout.
    """
    
    def __init__(self, dim, np_ratio=6, seed=None):
        self.dim = dim
        self.np = max(20, min(100, np_ratio * dim))
        self.rng = np.random.default_rng(seed)
        self.bounds = np.array([-100.0, 100.0])
        self._reset_state()
    
    def _reset_state(self):
        self.population = None
        self.velocity = None
        self.fitness = None
        self.pbest = None
        self.pbest_fit = None
        self.gbest = None
        self.gbest_fit = np.inf
        self.topo_k = max(3, self.np // 10)
        self.inertia = 0.729
        self.c1 = 1.494
        self.c2 = 1.494
        self.quantum_strength = 0.5
        self.cov_trace = np.eye(self.dim)
        self.improvement_history = []
        self.stagnation_count = 0
        self.gen = 0
    
    def _initialize_population(self):
        width = self.bounds[1] - self.bounds[0]
        self.population = self.rng.uniform(
            self.bounds[0], self.bounds[1], size=(self.np, self.dim)
        )
        self.velocity = self.rng.uniform(
            -width * 0.1, width * 0.1, size=(self.np, self.dim)
        )
        self.pbest = self.population.copy()
        self.pbest_fit = np.full(self.np, np.inf)
    
    def _evaluate_batch(self, pop, force=False):
        if not force and self._check_budget_exhausted():
            return np.array([])
        fit = self.func(pop)
        if len(fit) < len(pop):
            return fit
        return fit
    
    def _check_budget_exhausted(self):
        return self.stopping_condition.budget_exhausted if hasattr(self.stopping_condition, 'budget_exhausted') else False
    
    def _build_ring_topology_batch(self):
        n = self.np
        k = self.topo_k
        lbest_indices = np.zeros((n, k + 1), dtype=int)
        for i in range(n):
            neighbors = [(i - j) % n for j in range(1, k + 1)]
            lbest_indices[i] = np.array([i] + neighbors)
        return lbest_indices
    
    def _compute_lbest_batch(self, lbest_indices, pbest_fit):
        nbests = np.zeros(len(lbest_indices), dtype=int)
        for i, indices in enumerate(lbest_indices):
            nbests[i] = indices[np.argmin(pbest_fit[indices])]
        return nbests
    
    def _update_velocity_batch(self, lbest_indices, pbest_fit):
        nbests = self._compute_lbest_batch(lbest_indices, pbest_fit)
        cognitive = self.c1 * self.rng.uniform(0, 1, (self.np, self.dim)) * (self.pbest - self.population)
        social = self.c2 * self.rng.uniform(0, 1, (self.np, self.dim)) * (self.pbest[nbests] - self.population)
        self.velocity = self.inertia * self.velocity + cognitive + social
        self._apply_quantum_perturbation_batch()
        max_vel = (self.bounds[1] - self.bounds[0]) * 0.2
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
    
    def _apply_quantum_perturbation_batch(self):
        if self.gen < 5:
            return
        div = self._compute_diversity_batch()
        if div < 0.3:
            q_strength = self.quantum_strength * (1 - div / 0.3)
            quantum_noise = self.rng.normal(0, q_strength, size=self.velocity.shape)
            self.velocity += quantum_noise * self.velocity.std(axis=0)
    
    def _update_position_batch(self):
        self.population = self.population + self.velocity
        self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
    
    def _select_pbest_batch(self, fit):
        improved = fit < self.pbest_fit
        self.pbest_fit = np.where(improved, fit, self.pbest_fit)
        self.pbest = np.where(improved[:, np.newaxis], self.population, self.pbest)
        g_idx = np.argmin(self.pbest_fit)
        if self.pbest_fit[g_idx] < self.gbest_fit:
            old_best = self.gbest_fit
            self.gbest = self.pbest[g_idx].copy()
            self.gbest_fit = self.pbest_fit[g_idx]
            self.stagnation_count = 0
            return self.gbest_fit - old_best
        return 0.0
    
    def _adapt_step_size_batch(self, improvement):
        self.improvement_history.append(improvement)
        if len(self.improvement_history) > 5:
            self.improvement_history.pop(0)
        recent = self.improvement_history[-5:] if len(self.improvement_history) >= 5 else self.improvement_history
        avg_imp = np.mean(recent) if recent else 0.0
        if avg_imp < -1e-6:
            self.inertia = max(0.3, self.inertia * 0.9)
            self.c1 = min(2.5, self.c1 * 1.05)
            self.c2 = min(2.5, self.c2 * 1.05)
        elif avg_imp > 1e-4:
            self.inertia = min(0.9, self.inertia * 1.02)
        self.stagnation_count += 1
    
    def _adapt_topology_batch(self):
        div = self._compute_diversity_batch()
        if div < 0.2:
            self.topo_k = min(self.np - 1, self.topo_k + 1)
        elif div > 0.6 and self.topo_k > 2:
            self.topo_k = max(2, self.topo_k - 1)
    
    def _adapt_covariance_batch(self):
        if self.gen % 10 != 0 or self.gen == 0:
            return
        if self.population is None or len(self.population) < 3:
            return
        try:
            centered = self.population - self.population.mean(axis=0)
            cov = np.cov(centered.T)
            if cov.shape == (self.dim, self.dim) and np.isfinite(cov).all():
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.clip(eigvals, 1e-10, None)
                self.cov_trace = eigvecs @ np.diag(eigvals + 1e-8) @ eigvecs.T
                self.cov_trace = (self.cov_trace + self.cov_trace.T) / 2
        except np.linalg.LinAlgError:
            pass
    
    def _sample_trials_batch(self):
        div = self._compute_diversity_batch()
        if div < 0.25 and self.gen > 10:
            try:
                L = np.linalg.cholesky(self.cov_trace * 0.1 + np.eye(self.dim) * 1e-6)
                noise = self.rng.standard_normal((self.np, self.dim)) @ L.T
                cov_trial = self.population + noise * (self.bounds[1] - self.bounds[0]) * 0.05
                cov_trial = np.clip(cov_trial, self.bounds[0], self.bounds[1])
                return cov_trial
            except np.linalg.LinAlgError:
                pass
        return None
    
    def _compute_diversity_batch(self):
        if self.population is None or len(self.population) < 2:
            return 1.0
        fit_std = np.std(self.fitness)
        fit_mean = np.abs(np.mean(self.fitness))
        if fit_mean > 1e-10:
            cv = fit_std / fit_mean
            return np.clip(cv, 0.0, 2.0) / 2.0
        return 0.5
    
    def _restart_if_stagnant_batch(self):
        if self.stagnation_count < 30:
            return False
        if self.gbest is not None:
            n_replace = self.np // 2
            replace_idx = self.rng.choice(self.np, n_replace, replace=False)
            width = self.bounds[1] - self.bounds[0]
            self.population[replace_idx] = self.rng.uniform(
                self.bounds[0], self.bounds[1], size=(n_replace, self.dim)
            )
            self.velocity[replace_idx] = self.rng.uniform(
                -width * 0.1, width * 0.1, size=(n_replace, self.dim)
            )
        self.stagnation_count = 0
        self.inertia = 0.729
        self.c1 = 1.494
        self.c2 = 1.494
        return True
    
    def _select_survivors_batch(self, trial_pop, trial_fit):
        if len(trial_fit) < len(trial_pop):
            keep_n = len(trial_fit)
            trial_pop = trial_pop[:keep_n]
            trial_fit = trial_fit[:keep_n]
        if len(trial_pop) < len(self.population):
            self.population = self.population[:len(trial_pop)]
            self.velocity = self.velocity[:len(trial_pop)]
            self.fitness = self.fitness[:len(trial_pop)]
            self.pbest = self.pbest[:len(trial_pop)]
            self.pbest_fit = self.pbest_fit[:len(trial_pop)]
        better = trial_fit < self.fitness
        self.population = np.where(better[:, np.newaxis], trial_pop, self.population)
        self.fitness = np.where(better, trial_fit, self.fitness)
        self.velocity = np.where(better[:, np.newaxis], trial_pop - self.population + self.velocity * 0.5, self.velocity)
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self.stopping_condition = stopping_condition
        self._reset_state()
        self._initialize_population()
        self.fitness = self._evaluate_batch(self.population, force=True)
        if len(self.fitness) < self.np:
            self.np = len(self.fitness)
            self.population = self.population[:self.np]
            self.velocity = self.velocity[:self.np]
            self.pbest = self.pbest[:self.np]
            self.pbest_fit = self.pbest_fit[:self.np]
        self._select_pbest_batch(self.fitness)
        while not stopping_condition():
            if self._check_budget_exhausted():
                break
            self.gen += 1
            lbest_indices = self._build_ring_topology_batch()
            self._update_velocity_batch(lbest_indices, self.pbest_fit)
            self._update_position_batch()
            trials = self.population.copy()
            cov_trials = self._sample_trials_batch()
            if cov_trials is not None:
                trials = np.vstack([trials, cov_trials])
            trial_fit = self._evaluate_batch(trials)
            if len(trial_fit) < len(trials):
                trial_fit = trial_fit[:len(trial_fit)]
                trials = trials[:len(trial_fit)]
            if len(trial_fit) == 0:
                break
            if len(trial_fit) < len(self.population):
                keep_n = len(trial_fit)
                self.population = self.population[:keep_n]
                self.velocity = self.velocity[:keep_n]
                self.fitness = self.fitness[:keep_n]
                self.pbest = self.pbest[:keep_n]
                self.pbest_fit = self.pbest_fit[:keep_n]
                self.np = keep_n
            self.fitness = trial_fit[:self.np]
            improvement = self._select_pbest_batch(self.fitness)
            self._select_survivors_batch(trials, trial_fit)
            self._adapt_step_size_batch(improvement)
            self._adapt_topology_batch()
            self._adapt_covariance_batch()
            self._restart_if_stagnant_batch()
        return self.gbest_fit, self.gbest
```