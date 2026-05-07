import numpy as np


class AdaptiveHybridDEWithVelocity:
    """
    Adaptive Hybrid Differential Evolution with Velocity and Multi-Strategy Mutation.
    Combines DE backbone with PSO-like velocity term and adaptive parameter control.
    """

    def __init__(self, dim, NP=None, F=0.5, Cr=0.5, max_velocity=0.1):
        self.dim = dim
        self.NP = NP if NP is not None else max(4 * dim, 40)
        self.NP = min(self.NP, 400)
        self.F_init = F
        self.Cr_init = Cr
        self.max_velocity = max_velocity
        self.bounds = np.array([-100.0, 100.0])
        self.p_best_weight = 0.1
        self.success_history_F = []
        self.success_history_Cr = []

    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)

    def _initialize_population(self):
        rng = np.random.default_rng()
        lo, hi = self.bounds[0], self.bounds[1]
        pop = rng.uniform(lo, hi, size=(self.NP, self.dim))
        return pop

    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.bounds[0], self.bounds[1])

    def _mutate_current_to_pbest_batch(self, pop, p_best, indices, F):
        n_trials = len(indices)
        rng = np.random.default_rng()
        p_idx = rng.integers(0, self.NP, size=n_trials)
        r1 = rng.integers(0, self.NP, size=n_trials)
        r2 = rng.integers(0, self.NP, size=n_trials)
        while np.any(r1 == p_idx):
            r1 = rng.integers(0, self.NP, size=n_trials)
        while np.any((r2 == p_idx) | (r2 == r1)):
            r2 = rng.integers(0, self.NP, size=n_trials)
        base = pop[p_idx]
        pbest_vec = p_best[p_idx]
        diff1 = pop[r1] - pop[r1]
        diff2 = pop[r2] - pop[r2]
        mutant = base + F[:, np.newaxis] * (pbest_vec - base) + F[:, np.newaxis] * (pop[r1] - pop[r2])
        return mutant

    def _mutate_rand_batch(self, pop, indices, F):
        n_trials = len(indices)
        rng = np.random.default_rng()
        r1 = rng.integers(0, self.NP, size=n_trials)
        r2 = rng.integers(0, self.NP, size=n_trials)
        r3 = rng.integers(0, self.NP, size=n_trials)
        while np.any((r2 == r1) | (r3 == r1) | (r3 == r2)):
            r2 = rng.integers(0, self.NP, size=n_trials)
            r3 = rng.integers(0, self.NP, size=n_trials)
        mutant = pop[r1] + F[:, np.newaxis] * (pop[r2] - pop[r3])
        return mutant

    def _mutate_de_current_to_best_batch(self, pop, best_idx, indices, F):
        n_trials = len(indices)
        rng = np.random.default_rng()
        r1 = rng.integers(0, self.NP, size=n_trials)
        r2 = rng.integers(0, self.NP, size=n_trials)
        while np.any((r1 == best_idx) | (r2 == best_idx)):
            r1 = rng.integers(0, self.NP, size=n_trials)
            r2 = rng.integers(0, self.NP, size=n_trials)
        while np.any(r2 == r1):
            r2 = rng.integers(0, self.NP, size=n_trials)
        base = pop[best_idx]
        mutant = base + F[:, np.newaxis] * (pop[r1] - pop[r2])
        return mutant

    def _compute_velocity_batch(self, pop, p_best, inertia, rng):
        global_best = p_best[np.argmin(self._current_fitness)]
        best_pos = global_best
        r1_vel = rng.random((self.NP, self.dim))
        r2_vel = rng.random((self.NP, self.dim))
        cognitive = self.p_best_weight * r1_vel * (p_best - pop)
        r3_vel = rng.random((self.NP, self.dim))
        social = self.p_best_weight * r3_vel * (best_pos - pop)
        velocity = inertia * velocity + cognitive + social
        return velocity

    def _position_update_temporal_drift(self, pop, fitness, rng):
        sorted_idx = np.argsort(fitness)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(self.NP)
        drift_strength = 0.05 * (1 - ranks / self.NP)[:, np.newaxis]
        drift = rng.normal(0, drift_strength, size=pop.shape)
        updated = pop + drift
        return updated

    def _position_update_fitness_rank(self, pop, fitness, rng):
        normalized_rank = np.argsort(np.argsort(fitness)) / self.NP
        step_scale = 10.0 * (1 - normalized_rank)[:, np.newaxis]
        perturbation = rng.uniform(-1, 1, size=pop.shape) * step_scale
        updated = pop + perturbation
        return updated

    def _crossover_batch(self, target, mutant, Cr):
        rng = np.random.default_rng()
        n_trials = target.shape[0]
        mask = rng.random(target.shape) < Cr[:, np.newaxis]
        j_rand = rng.integers(0, self.dim, size=n_trials)
        for i in range(n_trials):
            mask[i, j_rand[i]] = True
        offspring = np.where(mask, mutant, target)
        return offspring

    def _adapt_inertia_weight(self, gen, max_gen):
        progress = gen / max_gen
        return 0.9 - 0.5 * progress

    def _adapt_F_Cr(self, fitness, trial_fitness):
        improved = trial_fitness < fitness
        if np.any(improved):
            self.success_history_F.append(self._current_F[improved])
            self.success_history_Cr.append(self._current_Cr[improved])
        if len(self.success_history_F) > 5:
            self.success_history_F.pop(0)
            self.success_history_Cr.pop(0)
        mean_sf = np.mean(self.success_history_F)
        mean_scr = np.mean(self.success_history_Cr)
        self._current_F = np.clip(mean_sf + 0.1 * np.random.randn(self.NP), 0.1, 1.0)
        self._current_Cr = np.clip(mean_scr + 0.1 * np.random.randn(self.NP), 0.0, 1.0)

    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        improved = trial_fitness < fitness
        new_pop = pop.copy()
        new_pop[improved] = trials[improved]
        new_fitness = fitness.copy()
        new_fitness[improved] = trial_fitness[improved]
        return new_pop, new_fitness

    def _compute_diversity(self, pop):
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances)

    def _restart_if_stagnant(self, pop, fitness, stagnation_gen):
        if stagnation_gen > 50 and self._compute_diversity(pop) < 5.0:
            rng = np.random.default_rng()
            n_replace = self.NP // 4
            replace_idx = rng.integers(0, self.NP, size=n_replace)
            lo, hi = self.bounds
            pop[replace_idx] = rng.uniform(lo, hi, size=(n_replace, self.dim))
            best_idx = np.argmin(fitness)
            pop[best_idx] = self._best_solution
            fitness[replace_idx] = np.inf
            return 0
        return stagnation_gen + 1

    def _optimize(self, func, stopping_condition):
        rng = np.random.default_rng()
        pop = self._initialize_population()
        pop = self._clip_to_bounds(pop)
        fitness = func(pop)
        self._current_fitness = fitness.copy()
        self._current_F = np.full(self.NP, self.F_init)
        self._current_Cr = np.full(self.NP, self.Cr_init)
        self._best_solution = pop[np.argmin(fitness)].copy()
        best_fitness = np.min(fitness)
        best_idx = np.argmin(fitness)
        gen = 0
        stagnation_gen = 0
        max_velocity = self.max_velocity * 200
        velocity = np.zeros((self.NP, self.dim))
        p_best = pop.copy()
        p_best_fitness = fitness.copy()
        while not stopping_condition():
            gen += 1
            inertia = self._adapt_inertia_weight(gen, 1000)
            n_trials = self.NP
            trial_indices = np.arange(n_trials)
            F = self._current_F
            Cr = self._current_Cr
            if gen % 3 == 0:
                mutant = self._mutate_rand_batch(pop, trial_indices, F)
            elif gen % 3 == 1:
                mutant = self._mutate_current_to_pbest_batch(pop, p_best, trial_indices, F)
            else:
                mutant = self._mutate_de_current_to_best_batch(pop, best_idx, trial_indices, F)
            mutant = self._clip_to_bounds(mutant)
            r_pbest = rng.random((n_trials, self.dim))
            cognitive = self.p_best_weight * r_pbest * (p_best[trial_indices] - mutant)
            r_global = rng.random((n_trials, self.dim))
            global_best = self._best_solution
            social = self.p_best_weight * r_global * (global_best - mutant)
            step = inertia * (cognitive + social)
            velocity_magnitude = np.linalg.norm(step, axis=1, keepdims=True)
            scale = np.clip(velocity_magnitude, 0, max_velocity)
            step = step * (scale / (velocity_magnitude + 1e-10))
            mutant = mutant + step
            mutant = self._clip_to_bounds(mutant)
            trials = self._crossover_batch(pop, mutant, Cr)
            trials = self._clip_to_bounds(trials)
            if len(trials) == 0:
                break
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                actual_len = len(trial_fitness)
                trials = trials[:actual_len]
                trial_fitness = trial_fitness[:actual_len]
                if stopping_condition():
                    break
            self._adapt_F_Cr(fitness[:actual_len], trial_fitness)
            pop, fitness = self._select_survivors_batch(pop, fitness, trials, trial_fitness)
            self._current_fitness = fitness
            improved = trial_fitness < fitness[:actual_len]
            p_best_update = pop[:actual_len][improved]
            p_best_fitness_update = fitness[:actual_len][improved]
            if len(p_best_update) > 0:
                p_best_idx = np.where(improved)[0]
                p_best[p_best_idx] = p_best_update
                p_best_fitness[p_best_idx] = p_best_fitness_update
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                self._best_solution = pop[current_best_idx].copy()
                best_idx = current_best_idx
                stagnation_gen = 0
            else:
                stagnation_gen += 1
            stagnation_gen = self._restart_if_stagnant(pop, fitness, stagnation_gen)
            if stopping_condition():
                break
        return best_fitness, self._best_solution
