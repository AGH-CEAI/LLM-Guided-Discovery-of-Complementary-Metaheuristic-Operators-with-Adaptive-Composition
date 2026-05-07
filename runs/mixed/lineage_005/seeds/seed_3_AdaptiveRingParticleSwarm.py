import numpy as np


class AdaptiveRingParticleSwarm:
    """Particle Swarm Optimizer with ring topology, adaptive parameters, and archive."""

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        NP = kwargs.get('pop_size', 6 * dim)
        self.NP = min(max(NP, 20), 300)

        self.lower = -100.0
        self.upper = 100.0
        self.v_max = kwargs.get('v_max', 20.0)

        self._init_adaptive_params()
        self._init_stagnation_tracking()
        self._init_archive()

    def _init_adaptive_params(self):
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.c3 = 1.5
        self.alpha = 0.5
        self.chi = 0.1

        self.ema_c1 = 0.0
        self.ema_c2 = 0.0
        self.ema_c3 = 0.0
        self.ema_alpha = 0.0
        self.ema_chi = 0.0

    def _init_stagnation_tracking(self):
        self.stagnation_counter = 0
        self.stagnation_limit = 30
        self.diversity_floor = 1e-6
        self.prev_global_best = np.inf

    def _init_archive(self):
        self.archive = None
        self.archive_fit = None
        self.archive_capacity = max(10, self.NP // 4)

    def _initialize_population(self):
        uniform = np.random.default_rng().uniform
        pop = uniform(self.lower, self.upper, (self.NP, self.dim))
        return pop

    def _initialize_velocities(self, pop):
        range_scale = (self.upper - self.lower) * 0.05
        vel = np.random.default_rng().uniform(-range_scale, range_scale, (self.NP, self.dim))
        return vel

    def _build_ring_topology(self):
        indices = np.arange(self.NP)
        left = np.roll(indices, 1)
        right = np.roll(indices, -1)
        return left, right

    def _eval_wrapper(self, pop):
        clipped = np.clip(pop, self.lower, self.upper)
        fitness = self._objective(clipped)
        return clipped, fitness

    def _objective(self, pop):
        return np.sum(pop ** 2, axis=1)

    def _initialize_personal_best(self, pop, fitness):
        self.personal_best_pos = pop.copy()
        self.personal_best_fit = fitness.copy()
        gbest_idx = np.argmin(fitness)
        self.global_best_pos = pop[gbest_idx].copy()
        self.global_best_fit = fitness[gbest_idx]

    def _compute_neighborhood_best_batch(self, pbest_pos, left, right):
        left_fit = self.personal_best_fit[left]
        right_fit = self.personal_best_fit[right]
        pbest_fit = self.personal_best_fit

        combined_fit = np.stack([left_fit, pbest_fit, right_fit], axis=1)
        combined_pos = np.stack([pbest_pos[left], pbest_pos, pbest_pos[right]], axis=1)

        nb_idx = np.argmin(combined_fit, axis=1)
        row_indices = np.arange(self.NP)
        nbest_pos = combined_pos[row_indices, nb_idx]
        return nbest_pos

    def _mutate_current_to_pbest_with_culture(self, pop, vel, left, right):
        pbest_pos = self.personal_best_pos
        nbest_pos = self._compute_neighborhood_best_batch(pbest_pos, left, right)
        gbest_pos = self.global_best_pos

        rand = np.random.default_rng().random

        r1, r2, r3, r4 = rand((4, self.NP, self.dim))
        r5 = rand((self.NP, self.dim))

        pbest_factor = r1 * (pbest_pos - pop)
        nbest_factor = r2 * (nbest_pos - pop)
        gbest_factor = r3 * (gbest_pos - pop)
        velocity_factor = self.chi * vel
        random_factor = self.alpha * (self.upper - self.lower) * (r4 - 0.5)

        trials = pop + velocity_factor + pbest_factor + nbest_factor + gbest_factor + random_factor
        trials = np.clip(trials, self.lower, self.upper)
        return trials

    def _update_velocity_batch(self, pop, vel, left, right):
        pbest_pos = self.personal_best_pos
        nbest_pos = self._compute_neighborhood_best_batch(pbest_pos, left, right)
        gbest_pos = self.global_best_pos

        rand = np.random.default_rng().random

        v_cognitive = self.c1 * rand((self.NP, self.dim)) * (pbest_pos - pop)
        v_social = self.c2 * rand((self.NP, self.dim)) * (nbest_pos - pop)
        v_global = self.c3 * rand((self.NP, self.dim)) * (gbest_pos - pop)

        new_vel = self.w * vel + v_cognitive + v_social + v_global
        new_vel = np.clip(new_vel, -self.v_max, self.v_max)
        return new_vel

    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        better_mask = trial_fitness < fitness
        new_pop = np.where(better_mask[:, np.newaxis], trials, pop)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        return new_pop, new_fitness

    def _compute_reward(self):
        improvement = self.prev_global_best - self.global_best_fit
        if improvement > 1e-10:
            reward = 1.0
        elif improvement < -1e-10:
            reward = -1.0
        else:
            reward = 0.0
        self.prev_global_best = self.global_best_fit
        return reward

    def _adapt_parameters_batch(self, reward):
        beta1, beta2, beta3 = 0.1, 0.1, 0.05

        self.ema_c1 = 0.9 * self.ema_c1 + 0.1 * (self.c1 + beta1 * reward)
        self.ema_c2 = 0.9 * self.ema_c2 + 0.1 * (self.c2 + beta2 * reward)
        self.ema_c3 = 0.9 * self.ema_c3 + 0.1 * (self.c3 + beta3 * reward)

        self.c1 = np.clip(self.ema_c1, 0.5, 3.0)
        self.c2 = np.clip(self.ema_c2, 0.5, 3.0)
        self.c3 = np.clip(self.ema_c3, 0.0, 2.0)

        self.w = np.clip(self.w * (1.0 + 0.05 * reward), 0.1, 0.9)

    def _update_archive_batch(self, pop, fitness):
        if self.archive is None:
            top_k = min(self.archive_capacity, self.NP)
            order = np.argsort(fitness)[:top_k]
            self.archive = pop[order].copy()
            self.archive_fit = fitness[order].copy()
        else:
            combined_pop = np.vstack([self.archive, pop])
            combined_fit = np.concatenate([self.archive_fit, fitness])
            order = np.argsort(combined_fit)[:self.archive_capacity]
            self.archive = combined_pop[order].copy()
            self.archive_fit = combined_fit[order].copy()

    def _apply_adaptive_local_search(self, pop, fitness, iteration):
        if iteration % 20 != 0:
            return pop, fitness

        top_count = max(1, self.NP // 10)
        top_indices = np.argsort(fitness)[:top_count]

        radius = (self.upper - self.lower) * 0.05 * np.exp(-iteration / 500)

        for idx in top_indices:
            center = pop[idx]
            candidates = center + np.random.default_rng().uniform(
                -radius, radius, (5, self.dim)
            )
            candidates = np.clip(candidates, self.lower, self.upper)

            if len(candidates) < 5:
                continue

            cand_fitness = self._objective(candidates)
            best_local_idx = np.argmin(cand_fitness)

            if cand_fitness[best_local_idx] < fitness[idx]:
                pop[idx] = candidates[best_local_idx]
                fitness[idx] = cand_fitness[best_local_idx]

        return pop, fitness

    def _compute_diversity(self, pop):
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances)

    def _restart_if_stagnant(self, pop, fitness, left, right):
        self.stagnation_counter += 1

        should_restart = (
            self.stagnation_counter >= self.stagnation_limit or
            self._compute_diversity(pop) < self.diversity_floor
        )

        if not should_restart:
            return pop, fitness

        num_replace = self.NP // 2
        keep_indices = np.argsort(fitness)[:self.NP - num_replace]

        new_portion = self._initialize_population()[:num_replace]
        new_portion, new_fitness = self._eval_wrapper(new_portion)

        pop = np.vstack([pop[keep_indices], new_portion])
        fitness = np.concatenate([fitness[keep_indices], new_fitness])

        self._initialize_personal_best(pop, fitness)
        self.stagnation_counter = 0

        return pop, fitness

    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.lower, self.upper)

    def __call__(self, func, stopping_condition):
        self._objective = func

        pop = self._initialize_population()
        pop, fitness = self._eval_wrapper(pop)
        vel = self._initialize_velocities(pop)

        self._initialize_personal_best(pop, fitness)
        left, right = self._build_ring_topology()

        iteration = 0
        while not stopping_condition():
            iteration += 1

            trials = self._mutate_current_to_pbest_with_culture(pop, vel, left, right)
            trials = self._clip_to_bounds_batch(trials)
            trial_fitness = func(trials)

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]

            if stopping_condition():
                break

            pop, fitness = self._select_survivors_batch(pop, fitness, trials, trial_fitness)

            improved = fitness < self.personal_best_fit
            self.personal_best_fit[improved] = fitness[improved]
            self.personal_best_pos[improved] = pop[improved]

            gbest_idx = np.argmin(fitness)
            if fitness[gbest_idx] < self.global_best_fit:
                self.global_best_fit = fitness[gbest_idx]
                self.global_best_pos = pop[gbest_idx].copy()
                self.stagnation_counter = 0

            pop, fitness = self._apply_adaptive_local_search(pop, fitness, iteration)

            self._update_archive_batch(pop, fitness)

            reward = self._compute_reward()
            self._adapt_parameters_batch(reward)

            vel = self._update_velocity_batch(pop, vel, left, right)

            pop, fitness = self._restart_if_stagnant(pop, fitness, left, right)

        return self.global_best_fit, self.global_best_pos.copy()
