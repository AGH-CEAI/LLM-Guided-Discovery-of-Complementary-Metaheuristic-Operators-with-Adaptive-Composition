```python
import numpy as np


class AdaptiveCovarianceSwarmOptimizer:
    """
    Adaptive Covariance Swarm Optimizer (ACSO)
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with multiple trial generation strategies and adaptive operator selection.
    Different from CompositeDEOptimizer by using covariance-matrix-guided
    sampling, niching-based diversity maintenance, and adaptive operator
    weights based on multi-armed bandit selection.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(kwargs.get('np', 8 * dim), 300)
        self.bounds = kwargs.get('bounds', (-100.0, 100.0))
        self.pops_frac = kwargs.get('pops_frac', 0.5)
        
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize all mutable state variables."""
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.best_fitness = np.inf
        self.best_position = None
        
        # CMA-ES style covariance and step size
        self.mean = None
        self.cov = None
        self.step_size = 1.0
        self.pc = None
        self.ps = None
        self.cov_decay = 2e-3
        self.cs = None
        self.cc = None
        self.mu_eff = None
        
        # Adaptive operator system (multi-armed bandit)
        self.num_operators = 6
        self.operator_weights = np.ones(self.num_operators)
        self.operator_rewards = np.zeros(self.num_operators)
        self.operator_counts = np.zeros(self.num_operators, dtype=int)
        self.epsilon = 0.1
        
        # Diversity and stagnation tracking
        self.diversity_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 60
        self.generation = 0
        self.improvement_history = []
        
        # Success history for adaptation
        self.success_buffer = []
        self.trial_success = np.zeros(self.num_operators)
        self.recent_success_rate = 0.25
    
    def _initialize_population(self):
        """Initialize population within bounds using stratified sampling."""
        lb, ub = self.bounds
        samples = self._stratified_sampling(self.np, self.dim)
        self.population = lb + (ub - lb) * samples
        self._initialize_cma_state()
        self.fitness = np.full(self.np, np.inf)
    
    def _stratified_sampling(self, n, d):
        """Generate stratified samples for better space coverage."""
        samples = np.zeros((n, d))
        for dim_idx in range(d):
            perm = np.random.permutation(n)
            offsets = (perm + np.random.uniform(0, 1, n)) / n
            samples[:, dim_idx] = offsets
        rng = np.random.default_rng(42)
        samples = rng.permuted(samples, axis=1)
        return samples
    
    def _initialize_cma_state(self):
        """Initialize CMA-ES state variables."""
        self.mean = np.mean(self.population, axis=0)
        self.cov = np.eye(self.dim) * (self.bounds[1] - self.bounds[0]) ** 2 * 0.1
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.mu_eff = self.np / 10
        self.cs = (self.mu_eff + 2) / (d := self.dim + mu_eff := self.mu_eff + 4)
        self.cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
    
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return best solution found."""
        self._initialize_population()
        self._evaluate_population(func, stopping_condition)
        
        while not stopping_condition():
            self._generation_step(func, stopping_condition)
            if self._check_stagnation():
                self._restart_adaptive(func)
        
        return self.best_fitness, self.best_position
    
    def _generation_step(self, func, stopping_condition):
        """Execute one generation: update CMA state, generate trials, select."""
        self._update_cma_es()
        self._generate_and_select_trials(func, stopping_condition)
        
        if stopping_condition():
            return
        
        self._update_adaptations()
        self.generation += 1
    
    def _evaluate_population(self, func, stopping_condition=None):
        """Evaluate fitness for entire population in one batched call."""
        self.fitness = func(self.population)
        
        if len(self.fitness) < self.np:
            self.fitness = np.pad(self.fitness, (0, self.np - len(self.fitness)), 
                                  constant_values=np.inf)
            if stopping_condition and stopping_condition():
                return
        
        self._update_best_tracking()
    
    def _update_best_tracking(self):
        """Track the best solution found so far."""
        valid = np.isfinite(self.fitness)
        if not np.any(valid):
            return
        
        current_best_idx = np.argmin(self.fitness * valid + np.where(valid, 0, np.inf))
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[current_best_idx]
            self.best_position = self.population[current_best_idx].copy()
            self.best_idx = current_best_idx
            self.stagnation_counter = 0
            return
        
        self.stagnation_counter += 1
    
    def _update_cma_es(self):
        """Update CMA-ES state: mean, covariance matrix, step size."""
        weights = self._compute_selection_weights()
        
        old_mean = self.mean.copy()
        self.mean = self._weighted_recombination(weights)
        
        self._update_evolution_paths(old_mean)
        self._update_covariance_matrix(weights)
        self._adapt_step_size()
    
    def _compute_selection_weights(self):
        """Compute weights for weighted recombination based on fitness ranking."""
        ranks = np.argsort(np.argsort(self.fitness))
        weights = np.maximum(0, self.np / 2 - ranks)
        weights = weights / np.maximum(np.sum(weights), 1e-10)
        weights = np.clip(weights, 0, 2.0 / self.np)
        return weights
    
    def _weighted_recombination(self, weights):
        """Compute weighted mean of population."""
        return np.sum(self.population * weights[:, np.newaxis], axis=0)
    
    def _update_evolution_paths(self, old_mean):
        """Update evolution paths pc and ps."""
        y_diff = (self.mean - old_mean) / max(self.step_size, 1e-10)
        
        c_sig = self.cs
        self.ps = (1 - c_sig) * self.ps + np.sqrt(c_sig * (2 - c_sig)) * y_diff
        
        c_c = self.cc
        self.pc = (1 - c_c) * self.pc + np.sqrt(c_c * (2 - c_c)) * y_diff
    
    def _update_covariance_matrix(self, weights):
        """Update covariance matrix using rank-1 and rank-mu updates."""
        c_1 = 1 / (6 * self.dim ** 2)
        c_mu = max(c_1, (self.mu_eff - 2 + 1 / self.mu_eff) / 
                   (self.dim ** 2 + self.mu_eff + 2))
        
        rank_one = np.outer(self.pc, self.pc)
        
        diff = self.population - old_mean := self.mean
        rank_mu = np.sum(
            weights[:, np.newaxis, np.newaxis] * 
            np.einsum('ij,ik->ijk', diff, diff), axis=0
        )
        
        self.cov = (1 - c_1 - c_mu) * self.cov + c_1 * rank_one + c_mu * rank_mu
        
        self.cov = self._ensure_covariance_positive_definite()
    
    def _ensure_covariance_positive_definite(self):
        """Ensure covariance matrix remains positive definite."""
        cov = (self.cov + self.cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = np.min(eigvals)
        
        if min_eig < 1e-10:
            cov += (1e-9 - min_eig) * np.eye(self.dim)
        
        return cov
    
    def _adapt_step_size(self):
        """Adapt step size based on evolution path length."""
        target = (1.4 - 1 / (self.dim + 1)) * np.sqrt(self.dim)
        actual = np.linalg.norm(self.ps)
        
        ratio = actual / max(target, 1e-10)
        decay = 1.0 + self.cs * (ratio - 1.0)
        
        self.step_size *= np.clip(decay, 0.6, 1.6)
        self.step_size = np.clip(self.step_size, 1e-10, 
                                  self.bounds[1] - self.bounds[0])
    
    def _generate_and_select_trials(self, func, stopping_condition):
        """Generate trials using multiple operators and select survivors."""
        trials, op_indices = self._sample_trials_batch()
        
        if len(trials) == 0:
            return
        
        trials = self._clip_to_bounds(trials)
        trial_fitness = func(trials)
        
        if len(trial_fitness) < len(trials):
            return
        
        if stopping_condition():
            return
        
        self._update_operator_rewards(op_indices, trial_fitness)
        self._select_survivors_batch(trials, trial_fitness, op_indices)
    
    def _sample_trials_batch(self):
        """Sample trial solutions using adaptive operator selection."""
        trials_list = []
        op_indices_list = []
        
        ops_per_type = max(1, self.np // self.num_operators)
        
        for op_id in range(self.num_operators):
            n_trials = ops_per_type if op_id < self.num_operators - 1 else \
                       self.np - len(trials_list)
            
            if n_trials <= 0:
                continue
            
            trials_op = self._generate_operator_trials(op_id, n_trials)
            trials_list.append(trials_op)
            op_indices_list.extend([op_id] * n_trials)
        
        if not trials_list:
            return np.empty((0, self.dim)), np.array([], dtype=int)
        
        trials = np.vstack(trials_list)
        op_indices = np.array(op_indices_list, dtype=int)
        
        return trials, op_indices
    
    def _generate_operator_trials(self, op_id, n_trials):
        """Generate trials using specific operator."""
        generators = [
            self._operator_covariance_sampling,
            self._operator_best_difference,
            self._operator_current_to_pbest,
            self._operator_ring_topology,
            self._operator_success_history,
            self._operator_diversity_guided,
        ]
        
        generator = generators[op_id % len(generators)]
        return generator(n_trials)
    
    def _operator_covariance_sampling(self, n):
        """Sample from CMA-ES distribution."""
        try:
            L = np.linalg.cholesky(self.cov)
            z = np.random.randn(n, self.dim)
            samples = self.mean + self.step_size * (z @ L.T)
            return samples
        except np.linalg.LinAlgError:
            return self._fallback_sampling(n)
    
    def _operator_best_difference(self, n):
        """DE/rand/1 style with best individual influence."""
        indices = np.random.choice(self.np, size=(n, 4), replace=True)
        r0, r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
        
        f = np.random.uniform(0.5, 1.0, n)
        cr = np.random.uniform(0.7, 0.95, n)
        
        trials = self.population[r0].copy()
        j_rand = np.random.randint(0, self.dim, n)
        
        mask = np.random.rand(n, self.dim) < cr[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        diff = self.population[r1] - self.population[r2]
        best_contrib = 0.3 * (self.best_position - self.population[r0])
        
        trials = np.where(mask, self.population[r0] + f[:, np.newaxis] * diff + best_contrib, trials)
        return trials
    
    def _operator_current_to_pbest(self, n):
        """DE/current-to-pbest (jDE variant)."""
        p = 0.1
        f = np.random.uniform(0.3, 0.9, n)
        cr = np.random.uniform(0.5, 0.9, n)
        
        pbest_indices = np.random.choice(self.np, size=n, replace=False,
                                         p=self._softmax_weights())
        
        indices = np.random.choice(self.np, size=(n, 2), replace=True)
        r1, r2 = indices[:, 0], indices[:, 1]
        
        trials = self.population.copy()
        j_rand = np.random.randint(0, self.dim, n)
        
        mask = np.random.rand(n, self.dim) < cr[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        base = self.population
        diff = f[:, np.newaxis] * (self.population[pbest_indices] - base + 
                                    self.population[r1] - self.population[r2])
        
        trials = np.where(mask, base + diff, trials)
        return trials[:n]
    
    def _operator_ring_topology(self, n):
        """Ring topology mutation (like SHADE)."""
        f = np.random.uniform(0.5, 1.0, n)
        cr = np.random.uniform(0.6, 0.95, n)
        
        sorted_idx = np.argsort(self.fitness)
        ring_neighbor = np.roll(sorted_idx, 1)
        
        r1 = np.random.choice(ring_neighbor, size=n, replace=True)
        r2 = np.random.choice(ring_neighbor, size=n, replace=True)
        
        trials = self.population.copy()
        j_rand = np.random.randint(0, self.dim, n)
        
        mask = np.random.rand(n, self.dim) < cr[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        diff = f[:, np.newaxis] * (self.population[r1] - self.population[r2])
        trials = np.where(mask, self.population + diff, trials)
        return trials[:n]
    
    def _operator_success_history(self, n):
        """Use success history to guide mutation (inspired by LSHADE)."""
        f = np.clip(np.random.normal(0.5, 0.3, n), 0.1, 1.0)
        cr = np.clip(np.random.normal(0.5, 0.3, n), 0.0, 1.0)
        
        sorted_fitness = np.argsort(self.fitness)
        top_half = sorted_fitness[:max(1, self.np // 2)]
        
        if len(self.success_buffer) > 10:
            recent_f = np.median([x[0] for x in self.success_buffer[-20:]])
            recent_cr = np.median([x[1] for x in self.success_buffer[-20:]])
            f = np.clip(f * 0.5 + recent_f * 0.5, 0.1, 1.0)
            cr = np.clip(cr * 0.5 + recent_cr * 0.5, 0.0, 1.0)
        
        indices = np.random.choice(self.np, size=(n, 3), replace=True)
        r0, r1, r2 = indices[:, 0], indices[:, 1], indices[:, 2]
        
        trials = self.population[r0].copy()
        j_rand = np.random.randint(0, self.dim, n)
        
        mask = np.random.rand(n, self.dim) < cr[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        diff = f[:, np.newaxis] * (self.population[r1] - self.population[r2])
        trials = np.where(mask, self.population[r0] + diff, trials)
        return trials
    
    def _operator_diversity_guided(self, n):
        """Mutate towards diverse regions of search space."""
        diversity_scores = self._compute_per_individual_diversity()
        probs = diversity_scores / (np.sum(diversity_scores) + 1e-10)
        
        f = np.random.uniform(0.4, 1.0, n)
        cr = np.random.uniform(0.5, 0.9, n)
        
        indices = np.random.choice(self.np, size=(n, 3), replace=True, p=probs)
        r0, r1, r2 = indices[:, 0], indices[:, 1], indices[:, 2]
        
        trials = self.population[r0].copy()
        j_rand = np.random.randint(0, self.dim, n)
        
        mask = np.random.rand(n, self.dim) < cr[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        diff = f[:, np.newaxis] * (self.population[r1] - self.population[r2])
        trials = np.where(mask, self.population[r0] + diff, trials)
        return trials
    
    def _compute_per_individual_diversity(self):
        """Compute diversity score for each individual."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return distances + 1e-10
    
    def _softmax_weights(self):
        """Compute softmax probability distribution from fitness."""
        neg_fitness = -self.fitness + np.max(-self.fitness) + 1
        exp_weights = np.exp(neg_fitness * 2)
        return exp_weights / np.sum(exp_weights)
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.bounds[0], self.bounds[1])
    
    def _select_survivors_batch(self, trials, trial_fitness, op_indices):
        """Select survivors using greedy selection with diversity bonus."""
        improved = trial_fitness < self.fitness
        
        self.population = np.where(improved[:, np.newaxis], trials, self.population)
        self.fitness = np.where(improved, trial_fitness, self.fitness)
        
        for op_id in range(self.num_operators):
            mask = op_indices == op_id
            if np.any(mask):
                self.trial_success[op_id] = np.sum(improved[mask]) / np.sum(mask)
        
        self._update_best_tracking()
    
    def _update_operator_rewards(self, op_indices, trial_fitness):
        """Update operator rewards based on trial success."""
        for op_id in range(self.num_operators):
            mask = op_indices == op_id
            if not np.any(mask):
                continue
            
            op_fitness = trial_fitness[mask]
            improved = op_fitness < self.fitness[mask]
            success_rate = np.mean(improved)
            
            improvement = np.mean(np.maximum(0, self.fitness[mask] - op_fitness))
            
            alpha = 0.1
            self.operator_rewards[op_id] = (1 - alpha) * self.operator_rewards[op_id] + \
                                            alpha * (success_rate * 10 + improvement * 0.1)
    
    def _update_adaptations(self):
        """Update adaptive mechanisms after each generation."""
        self._update_operator_weights()
        self._record_diversity()
        self._adapt_covariance()
        self._update_success_history_buffer()
    
    def _update_operator_weights(self):
        """Update operator selection weights using UCB1-inspired formula."""
        total_counts = np.sum(self.operator_counts) + 1
        
        ucb_scores = self.operator_rewards + np.sqrt(
            2 * np.log(total_counts) / (self.operator_counts + 1)
        )
        
        self.operator_weights = 0.9 * self.operator_weights + 0.1 * ucb_scores
        self.operator_weights = np.clip(self.operator_weights, 0.01, 10)
    
    def _record_diversity(self):
        """Record population diversity for monitoring."""
        diversity = self._compute_diversity()
        self.diversity_history.append(diversity)
        
        if len(self.diversity_history) > 100:
            self.diversity_history.pop(0)
    
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_covariance(self):
        """Adapt covariance matrix decay rate based on success."""
        if len(self.diversity_history) < 10:
            return
        
        recent_div = np.mean(self.diversity_history[-10:])
        older_div = np.mean(self.diversity_history[-50:-10]) if len(self.diversity_history) > 50 else recent_div
        
        if recent_div < older_div * 0.9:
            self.cov_decay = min(0.05, self.cov_decay * 1.5)
        else:
            self.cov_decay = max(1e-4, self.cov_decay * 0.95)
    
    def _update_success_history_buffer(self):
        """Update success history buffer for LSHADE-style adaptation."""
        if len(self.improvement_history) > 0:
            recent_f = np.mean([x[0] for x in self.improvement_history[-20:]])
            recent_cr = np.mean([x[1] for x in self.improvement_history[-20:]])
            
            if len(self.success_buffer) > 50:
                self.success_buffer.pop(0)
            self.success_buffer.append((recent_f, recent_cr))
    
    def _check_stagnation(self):
        """Check if the optimization has stagnated."""
        if self.stagnation_counter > self.stagnation_threshold:
            return True
        
        if len(self.diversity_history) > 30:
            recent = np.mean(self.diversity_history[-10:])
            older = np.mean(self.diversity_history[-30:-10])
            
            if recent < older * 0.3 and self.stagnation_counter > 20:
                return True
        
        return False
    
    def _restart_adaptive(self, func):
        """Perform adaptive restart to escape local optima."""
        if len(self.diversity_history) > 0:
            best_div = max(self.diversity_history)
            current_div = self.diversity_history[-1]
            
            if current_div < best_div * 0.5:
                self._reinitialize_population_smart()
            else:
                self._reinitialize_population_diverse()
        else:
            self._reinitialize_population_smart()
        
        self._reset_cma_state()
        self._reset_adaptation_state()
        self._evaluate_population(func)
    
    def _reinitialize_population_smart(self):
        """Reinitialize around current best solution."""
        lb, ub = self.bounds
        radius = min(10, (ub - lb) * 0.1)
        
        self.population = np.random.uniform(
            self.best_position - radius, self.best_position + radius,
            size=(self.np, self.dim)
        )
        self.population = self._clip_to_bounds(self.population)
        
        self.population[0] = self.best_position.copy()
    
    def _reinitialize_population_diverse(self):
        """Reinitialize population with focus on diversity."""
        lb, ub = self.bounds
        
        samples = self._stratified_sampling(self.np, self.dim)
        self.population = lb + (ub - lb) * samples
        
        self.population[0] = self.best_position.copy()
    
    def _reset_cma_state(self):
        """Reset CMA-ES state variables."""
        self.mean = np.mean(self.population, axis=0)
        self.cov = np.eye(self.dim) * ((self.bounds[1] - self.bounds[0]) ** 2 * 0.1)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.step_size = 1.0
        self.cov_decay = 2e-3
    
    def _reset_adaptation_state(self):
        """Reset adaptation-related state."""
        self.stagnation_counter = 0
        self.diversity_history = []
        self.improvement_history = []
        self.success_buffer = []
        self.operator_rewards = np.zeros(self.num_operators)
        self.operator_counts = np.zeros(self.num_operators, dtype=int)
    
    def _fallback_sampling(self, n):
        """Fallback sampling when covariance fails."""
        lb, ub = self.bounds
        center = (lb + ub) / 2
        scale = (ub - lb) / 4
        
        return np.clip(center + scale * np.random.randn(n, self.dim), lb, ub)
```