Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_survivors_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.778120e-08            1.000000e-08            2.153045e+01            1.000000e-08            
1      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.298154e+00            1.000000e-08            5.240284e-05            1.000000e-08            5.010743e+01            1.160820e-07            
2      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            4.332576e-06            1.000000e-08            2.336220e+02            5.075411e-06            
3      1.386328e-04            1.381477e-04            1.339541e-04            1.358686e-04            1.360789e-04            2.706708e-01            1.388810e-04            1.869807e-04            1.341045e-04            1.243746e+02            1.103108e+00            
4      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.480596e+00            1.000000e-08            1.409805e+00            1.000000e-08            4.199602e+00            1.000000e-08            
5      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            5.106541e-04            1.000000e-08            9.844254e-04            1.000000e-08            2.677000e+05            5.036437e-02            
6      5.699103e-08            5.883916e-08            5.772301e-08            5.513192e-08            5.763098e-08            1.555625e-02            5.665906e-08            1.382607e-05            5.532691e-08            1.285964e-03            5.861464e-08            
7      4.073435e-06            3.919647e-06            4.302496e-06            3.954805e-06            4.316768e-06            4.404631e-01            3.791848e-06            1.409847e-03            3.797614e-06            1.412021e+02            3.050815e+00            
8      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            5.327560e-04            9.382816e-01            1.000000e-08            9.952202e-04            1.000000e-08            3.039968e+01            1.277965e+00            
9      2.081100e-04            2.061984e-04            7.851408e-04            2.026437e-04            5.741322e-04            5.135196e-01            2.871744e-04            7.112140e-02            2.092999e-04            1.781504e-01            1.885145e-04            
10     1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.004487e-05            1.000000e-08            6.766901e-07            1.000000e-08            8.191659e+00            2.819053e+00            
11     1.751516e+00            3.270351e+00            2.871043e+00            3.647202e+00            2.440656e+00            2.632883e+00            2.389002e+00            1.472470e+00            3.304094e+00            3.322307e+01            3.214915e+00            
12     4.567611e-06            4.589243e-06            4.805242e-06            4.242015e-06            5.145553e-06            8.442432e-03            4.384884e-06            8.500298e-05            4.399438e-06            4.022792e+01            8.836398e-01            
13     1.335045e+00            1.444190e+00            1.434350e+00            1.537284e+00            1.377922e+00            2.851552e+00            1.487792e+00            1.454117e+00            1.530147e+00            3.074797e+00            1.428465e+00            
14     2.448680e+00            2.457374e+00            2.521934e+00            2.471661e+00            2.480152e+00            2.732941e+00            2.399647e+00            2.510586e+00            2.419243e+00            2.661256e+00            2.829849e+00            
15     1.230940e+00            1.119828e+00            1.108635e+00            1.049433e+00            9.825813e-01            3.158959e+00            1.169142e+00            1.249967e+00            1.150659e+00            2.684132e+00            1.180047e+00            
16     1.418927e+02            1.116318e+02            1.319800e+02            1.308728e+02            4.622237e+01            5.547599e+01            2.050746e+02            9.775873e+01            1.294471e+02            4.914886e+01            6.916177e+01            
17     3.120103e+00            3.120103e+00            3.120103e+00            3.120103e+00            3.120103e+00            3.120151e+00            3.120103e+00            3.120125e+00            3.120103e+00            5.237442e+03            3.120103e+00            
18     2.671167e+00            2.671167e+00            2.671167e+00            2.671167e+00            2.671167e+00            5.674782e+00            2.671167e+00            2.671167e+00            2.671167e+00            2.719459e+01            2.673303e+00            
19     8.810302e+00            6.753502e+00            1.246118e+01            6.418511e+00            6.045520e+00            6.683722e+00            6.399006e+00            6.158164e+00            6.398505e+00            7.026916e+00            6.342837e+00            
20     1.395828e+01            1.309053e+01            8.426035e+00            1.408862e+01            5.785887e+00            3.968466e+00            8.302521e+00            6.740209e+00            8.940169e+00            4.613256e+00            7.969387e+00            
21     4.203408e+00            4.179989e+00            4.193486e+00            4.292849e+00            4.167994e+00            5.802390e+00            4.235126e+00            4.155325e+00            4.197457e+00            4.212163e+00            4.423118e+00            
22     2.296287e+00            2.224552e+00            2.161306e+00            2.297168e+00            2.198612e+00            2.675686e+00            2.259879e+00            2.254423e+00            2.299481e+00            3.309642e+00            2.335738e+00            
23     1.626722e+01            1.461811e+01            7.551014e+00            1.428481e+01            2.226440e+01            2.991997e+01            2.739486e+01            1.931987e+01            1.225658e+01            5.941606e+00            1.805033e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: SKIPPED (trivial)
Task  2: SKIPPED (trivial)
Task  3: variant_02_idea_0.py  (error=1.339541e-04)
Task  4: SKIPPED (trivial)
Task  5: SKIPPED (trivial)
Task  6: SKIPPED (trivial)
Task  7: variant_06_idea_0.py  (error=3.791848e-06)
Task  8: SKIPPED (trivial)
Task  9: variant_10_idea_0.py  (error=1.885145e-04)
Task 10: SKIPPED (trivial)
Task 11: variant_07_idea_0.py  (error=1.472470e+00)
Task 12: variant_03_idea_0.py  (error=4.242015e-06)
Task 13: original.py  (error=1.335045e+00)
Task 14: variant_06_idea_0.py  (error=2.399647e+00)
Task 15: variant_04_idea_0.py  (error=9.825813e-01)
Task 16: variant_04_idea_0.py  (error=4.622237e+01)
Task 17: original.py  (error=3.120103e+00)
Task 18: original.py  (error=2.671167e+00)
Task 19: variant_04_idea_0.py  (error=6.045520e+00)
Task 20: variant_05_idea_0.py  (error=3.968466e+00)
Task 21: variant_07_idea_0.py  (error=4.155325e+00)
Task 22: variant_02_idea_0.py  (error=2.161306e+00)
Task 23: variant_09_idea_0.py  (error=5.941606e+00)

WIN COUNTS:
  original.py: 3 wins
  variant_04_idea_0.py: 3 wins
  variant_02_idea_0.py: 2 wins
  variant_06_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_10_idea_0.py: 1 wins
  variant_03_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (2 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                     archive, f_values, cr_values, valid_mask):
        """Selection with probabilistic acceptance of worse solutions and crowding-based diversity."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        # Adaptive temperature for accepting worse solutions
        generation = getattr(self, 'generation', 0)
        # Start with moderate acceptance, decay over time but never fully zero
        T = max(1e-10, 1.0 * np.exp(-generation / 100.0))

        # Compute pairwise distances for crowding (to nearest neighbor)
        from scipy.spatial.distance import cdist
        try:
            dists = cdist(population[:n], population[:n])
            np.fill_diagonal(dists, np.inf)
            crowd_dist = np.min(dists, axis=1)
            # Normalize crowding distances
            cd_max = np.max(crowd_dist) + 1e-30
            crowd_scores = crowd_dist / cd_max  # 0=crowded, 1=isolated
        except Exception:
            crowd_scores = np.ones(n) * 0.5

        for i in range(n):
            if not valid_mask[i]:
                continue

            delta = trial_fitness[i] - fitness[i]

            if delta <= 0:
                # Trial is better or equal — always accept
                if delta < 0:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(delta))
                # Archive old individual
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i].copy()
                fitness[i] = trial_fitness[i]
            else:
                # Trial is worse — accept probabilistically
                # Higher acceptance for crowded individuals (low crowd_score means crowded)
                # Scale delta by fitness range for normalization
                fit_range = np.max(fitness[:n]) - np.min(fitness[:n]) + 1e-30
                normalized_delta = delta / fit_range

                # Crowded individuals more likely to accept worse (escape local basin)
                crowding_boost = 1.0 + 2.0 * (1.0 - crowd_scores[i])  # boost for crowded

                accept_prob = np.exp(-normalized_delta / (T * crowding_boost + 1e-30))
                accept_prob = min(accept_prob, 0.3)  # cap at 30% to avoid too much randomness

                if np.random.random() < accept_prob:
                    # Archive old individual before replacement
                    if len(archive) < self.archive_max:
                        archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                    else:
                        replace_idx = np.random.randint(0, self.archive_max)
                        archive[replace_idx] = population[i]
                    population[i] = trials[i].copy()
                    fitness[i] = trial_fitness[i]
                    # Don't record as success for parameter adaptation

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                     archive, f_values, cr_values, valid_mask):
        """Crowding selection with SA-style acceptance of worse solutions for diversity."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)
        # Temperature schedule: starts high, decays slowly
        T_initial = 1.0
        T_min = 1e-10
        temperature = max(T_min, T_initial * (0.995 ** generation))

        # Fitness range for normalization
        fit_range = np.ptp(fitness[:n])
        if fit_range < 1e-30:
            fit_range = 1.0

        for i in range(n):
            if not valid_mask[i]:
                continue

            if trial_fitness[i] <= fitness[i]:
                # Standard greedy acceptance
                if trial_fitness[i] < fitness[i]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))
                # Archive old individual
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                # Probabilistic acceptance of worse solutions (SA-style)
                # with crowding: replace the nearest neighbor if accepted
                delta = (trial_fitness[i] - fitness[i]) / (fit_range + 1e-30)
                accept_prob = np.exp(-delta / (temperature + 1e-30))

                if np.random.random() < accept_prob * 0.15:  # scale down acceptance rate
                    # Find the most similar individual in population (crowding)
                    dists = np.sum((population[:n] - trials[i]) ** 2, axis=1)
                    dists[i] = -1  # exclude self initially
                    # Among the closest 10% of population, pick the worst fitness
                    k = max(2, n // 10)
                    nearest_indices = np.argpartition(dists, -k)[-k:]  # largest because i=-1
                    # Actually get nearest (smallest positive distances)
                    dists[i] = np.inf
                    nearest_indices = np.argpartition(dists, k)[:k]
                    # Replace the worst-fitness among nearest neighbors
                    worst_neighbor = nearest_indices[np.argmax(fitness[nearest_indices])]

                    if trial_fitness[i] < fitness[worst_neighbor]:
                        # Archive replaced individual
                        if len(archive) < self.archive_max:
                            archive = np.vstack([archive, population[worst_neighbor:worst_neighbor+1]]) if len(archive) > 0 else population[worst_neighbor:worst_neighbor+1].copy()
                        else:
                            replace_idx = np.random.randint(0, self.archive_max)
                            archive[replace_idx] = population[worst_neighbor]
                        population[worst_neighbor] = trials[i]
                        fitness[worst_neighbor] = trial_fitness[i]
                        success_f.append(f_values[i])
                        success_cr.append(cr_values[i])
                        success_delta.append(abs(fitness[worst_neighbor] - trial_fitness[i]))

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_04_idea_0.py (3 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                     archive, f_values, cr_values, valid_mask):
        """Crowding-based selection with probabilistic worse-acceptance to escape local optima."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)
        # Decaying acceptance probability for worse solutions
        accept_temp = max(1e-15, 1.0 * np.exp(-generation / 30.0))

        for i in range(n):
            if not valid_mask[i]:
                continue

            # Find nearest neighbor in population (crowding)
            dists = np.sum((population - trials[i]) ** 2, axis=1)
            dists[i] = -1.0  # mark self
            nearest_idx = np.argmax(dists >= 0)  # init
            min_dist = np.inf
            for j in range(n):
                if j != i and dists[j] >= 0 and dists[j] < min_dist:
                    min_dist = dists[j]
                    nearest_idx = j

            # Use crowding: compare trial with nearest individual
            target_idx = nearest_idx if np.random.random() < 0.5 else i

            if trial_fitness[i] <= fitness[target_idx]:
                # Standard improvement
                if trial_fitness[i] < fitness[target_idx]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[target_idx] - trial_fitness[i]))
                # Archive old individual
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[target_idx:target_idx+1]]) if len(archive) > 0 else population[target_idx:target_idx+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[target_idx]
                population[target_idx] = trials[i]
                fitness[target_idx] = trial_fitness[i]
            elif target_idx == i:
                # Probabilistic acceptance of worse solutions (SA-like)
                delta = trial_fitness[i] - fitness[i]
                # Normalize by fitness scale
                scale = max(abs(fitness[i]), 1e-30)
                rel_delta = delta / scale
                accept_prob = np.exp(-rel_delta / (accept_temp + 1e-30))
                accept_prob = min(accept_prob, 0.05)  # cap at 5%

                if np.random.random() < accept_prob:
                    if len(archive) < self.archive_max:
                        archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                    else:
                        replace_idx = np.random.randint(0, self.archive_max)
                        archive[replace_idx] = population[i]
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                     archive, f_values, cr_values, valid_mask):
        """Crowding selection: trial replaces nearest neighbor if better, with diversity bonus."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        # Compute pairwise distances between each trial and all population members
        # Use crowding: each trial competes with its nearest population member
        for i in range(n):
            if not valid_mask[i]:
                continue

            # Find nearest neighbor in population to this trial
            diffs = population - trials[i]
            distances = np.sum(diffs ** 2, axis=1)
            # Exclude self from consideration with small probability to allow self-replacement too
            nearest_idx = np.argmin(distances)

            # Decide target: use nearest neighbor for crowding (70% of time) or parent (30%)
            if np.random.random() < 0.7:
                target_idx = nearest_idx
            else:
                target_idx = i

            target_fit = fitness[target_idx]

            # Accept if trial is better than target
            # Also accept slightly worse with small probability for diversity (simulated annealing style)
            accept = False
            if trial_fitness[i] <= target_fit:
                accept = True
            else:
                # Probabilistic acceptance for diversity - decreases over generations
                delta = trial_fitness[i] - target_fit
                temperature = max(1e-15, 1.0 / (1.0 + generation * 0.1))
                # Scale delta by fitness magnitude for robustness
                scale = max(abs(target_fit), 1e-30)
                prob = np.exp(-delta / (scale * temperature + 1e-30))
                prob = min(prob, 0.05)  # Cap acceptance probability
                if np.random.random() < prob:
                    accept = True

            if accept:
                if trial_fitness[i] < fitness[target_idx]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[target_idx] - trial_fitness[i]))

                # Archive the replaced individual
                old_individual = population[target_idx:target_idx+1].copy()
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, old_individual]) if len(archive) > 0 else old_individual.copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[target_idx]

                population[target_idx] = trials[i].copy()
                fitness[target_idx] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                     archive, f_values, cr_values, valid_mask):
        """SA-like acceptance with crowding distance to escape local optima on multimodal functions."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        # Compute fitness range for temperature scaling
        fit_valid = fitness[~np.isnan(fitness)]
        if len(fit_valid) > 1:
            fit_range = np.max(fit_valid) - np.min(fit_valid)
        else:
            fit_range = 1.0
        fit_range = max(fit_range, 1e-30)

        # Temperature: starts high, decays slowly — allows worse solutions early on
        # Higher base temperature than typical SA to really enable exploration
        temperature = fit_range * 0.3 * np.exp(-generation / 150.0)
        temperature = max(temperature, 1e-30)

        for i in range(n):
            if not valid_mask[i]:
                continue

            delta = trial_fitness[i] - fitness[i]

            if delta <= 0:
                # Trial is better or equal — always accept
                if delta < 0:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(delta))

                # Archive old individual
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]

                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                # Trial is worse — accept with Boltzmann probability
                # Scale delta by fitness range for normalization
                normalized_delta = delta / fit_range
                accept_prob = np.exp(-normalized_delta / (temperature / fit_range + 1e-30))
                accept_prob = min(accept_prob, 0.15)  # Cap acceptance of worse solutions

                if np.random.random() < accept_prob:
                    # Accept the worse solution to escape local optima
                    # Don't record as "success" for parameter adaptation
                    # but do archive the old solution
                    if len(archive) < self.archive_max:
                        archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                    else:
                        replace_idx = np.random.randint(0, self.archive_max)
                        archive[replace_idx] = population[i]

                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """SA-inspired acceptance with crowding replacement for diversity."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        # Temperature schedule: starts high, decays slowly
        # High initial temp allows accepting worse solutions early on
        T_init = max(1.0, np.max(np.abs(fitness[~np.isnan(fitness)])) * 0.1) if np.any(~np.isnan(fitness)) else 10.0
        T = T_init * np.exp(-generation / 150.0)
        T = max(T, 1e-15)

        # Fraction of population using crowding replacement vs direct replacement
        crowding_prob = 0.3

        for i in range(n):
            if not valid_mask[i]:
                continue

            # Standard greedy acceptance
            if trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))

                # Archive old individual
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]

                # Crowding: replace nearest neighbor instead of parent sometimes
                if np.random.random() < crowding_prob and n > 4:
                    dists = np.sum((population - trials[i]) ** 2, axis=1)
                    dists[i] = -1  # ensure parent is eligible
                    nearest = np.argmin(np.where(dists >= 0, dists, np.inf))
                    if trial_fitness[i] <= fitness[nearest]:
                        population[nearest] = trials[i]
                        fitness[nearest] = trial_fitness[i]
                    else:
                        population[i] = trials[i]
                        fitness[i] = trial_fitness[i]
                else:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]
            else:
                # Probabilistic acceptance of worse solutions (SA-like)
                delta = trial_fitness[i] - fitness[i]
                # Normalize delta by fitness scale
                scale = max(abs(fitness[i]), 1e-30)
                normalized_delta = delta / scale

                accept_prob = np.exp(-normalized_delta / (T / scale + 1e-30))
                accept_prob = min(accept_prob, 0.15)  # cap acceptance probability

                if np.random.random() < accept_prob:
                    # Archive old individual
                    if len(archive) < self.archive_max:
                        archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                    else:
                        replace_idx = np.random.randint(0, self.archive_max)
                        archive[replace_idx] = population[i]
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)
        # Temperature decays but stays warm enough for exploration
        T = max(0.01, 10.0 * np.exp(-generation / 100.0))

        # Compute fitness range for normalization
        fit_range = max(np.ptp(fitness[np.isfinite(fitness)]), 1e-30) if np.any(np.isfinite(fitness)) else 1.0

        replaced = np.zeros(n, dtype=bool)

        for i in range(n):
            if not valid_mask[i]:
                continue

            # Find nearest neighbor in population (crowding)
            dists = np.sum((population - trials[i]) ** 2, axis=1)
            dists[replaced] = np.inf  # Don't replace already-replaced individuals
            nearest = np.argmin(dists)

            delta = trial_fitness[i] - fitness[nearest]

            if delta <= 0:
                # Trial is better than nearest neighbor - always accept
                if fitness[nearest] - trial_fitness[i] > 1e-30:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))

                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[nearest:nearest+1]]) if len(archive) > 0 else population[nearest:nearest+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[nearest]

                population[nearest] = trials[i]
                fitness[nearest] = trial_fitness[i]
                replaced[nearest] = True
            else:
                # Boltzmann acceptance for worse solutions (normalized)
                normalized_delta = delta / fit_range
                accept_prob = np.exp(-normalized_delta / max(T, 1e-30))
                accept_prob = min(accept_prob, 0.15)  # Cap acceptance of worse solutions

                if np.random.random() < accept_prob:
                    if len(archive) < self.archive_max:
                        archive = np.vstack([archive, population[nearest:nearest+1]]) if len(archive) > 0 else population[nearest:nearest+1].copy()
                    else:
                        replace_idx = np.random.randint(0, self.archive_max)
                        archive[replace_idx] = population[nearest]

                    population[nearest] = trials[i]
                    fitness[nearest] = trial_fitness[i]
                    replaced[nearest] = True

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """SA-style acceptance with fitness sharing for escaping local optima on multimodal landscapes."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        # Simulated annealing temperature: starts high, decays slowly
        # High initial temperature allows accepting much worse solutions
        T_initial = 100.0
        T_min = 1e-10
        decay_rate = 0.02
        temperature = max(T_min, T_initial * np.exp(-decay_rate * generation))

        # Compute fitness sharing radius (niche radius) based on population spread
        pop_center = np.mean(population[:n], axis=0)
        pop_dists = np.sqrt(np.sum((population[:n] - pop_center)**2, axis=1))
        sigma_share = max(1e-10, np.median(pop_dists) * 0.5)

        # Compute shared fitness: penalize crowded regions to maintain diversity
        shared_fitness = fitness[:n].copy()
        for i in range(n):
            dists = np.sqrt(np.sum((population[:n] - population[i])**2, axis=1))
            sharing = np.sum(np.maximum(0, 1.0 - (dists / sigma_share)**2))
            sharing = max(1.0, sharing)
            shared_fitness[i] = fitness[i] * sharing

        for i in range(n):
            if not valid_mask[i]:
                continue

            # Compute shared trial fitness
            trial_dists = np.sqrt(np.sum((population[:n] - trials[i])**2, axis=1))
            trial_sharing = np.sum(np.maximum(0, 1.0 - (trial_dists / sigma_share)**2))
            trial_sharing = max(1.0, trial_sharing)
            trial_shared = trial_fitness[i] * trial_sharing

            accept = False
            if trial_fitness[i] <= fitness[i]:
                accept = True
            else:
                # SA acceptance based on shared fitness difference
                delta = trial_shared - shared_fitness[i]
                if temperature > T_min:
                    prob = np.exp(-delta / (temperature + 1e-30))
                    prob = min(prob, 0.3)  # Cap acceptance probability
                    if np.random.random() < prob:
                        accept = True

            if accept:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))

                # Archive old individual
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]

                population[i] = trials[i].copy()
                fitness[i] = trial_fitness[i]
                # Update shared fitness for subsequent comparisons
                shared_fitness[i] = trial_shared

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, and intelligent restart mechanism.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(8 * dim, 300)
        self.archive_max = self.np_size
        self.memory_size = 5
        self.p_best_rate = 0.15
        self.restart_threshold = 50
        self.min_pop_size = max(4, dim // 2)

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        population, fitness = self._initialize_population(func, stopping_condition)
        if stopping_condition():
            return self.f_opt, self.x_opt

        self._update_best(population, fitness)
        archive = np.empty((0, self.dim))
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        prev_best_fitness = self.f_opt

        while not stopping_condition():
            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch()

            # Generate mutant vectors
            mutants = self._mutate_batch(population, fitness, archive, f_values, strategy_mask)

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values)

            # Clip to bounds
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                self._safe_update_best(trials, trial_fitness)
                break

            # Handle truncated results
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                f_values = f_values[:len(trial_fitness)]
                cr_values = cr_values[:len(trial_fitness)]

            valid = ~np.isnan(trial_fitness)
            
            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Update parameter memory
            memory_f, memory_cr, memory_idx = self._update_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta
            )

            self._update_best(population, fitness)

            # Detect stagnation
            stagnant = self._detect_stagnation(prev_best_fitness)
            if stagnant:
                population, fitness, archive, memory_f, memory_cr, memory_idx = (
                    self._restart(func, stopping_condition, population, fitness)
                )
                if stopping_condition():
                    break

            prev_best_fitness = self.f_opt
            self.generation += 1

            # Periodically try opposition-based learning
            if self.generation % 25 == 0 and not stopping_condition():
                population, fitness = self._opposition_based_jump(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
                self._update_best(population, fitness)

        return self.f_opt, self.x_opt

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values with generation-adaptive scaling for better convergence."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Adaptive scale factor: starts at 0.1, decreases over generations for finer tuning
        # but never goes below 0.02 to maintain some exploration
        base_scale = 0.1
        generation = getattr(self, 'generation', 0)
        scale_f = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))
        scale_cr = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))

        # Vectorized Cauchy distribution for F with rejection sampling
        mu_f = memory_f[indices]
        f_values = np.zeros(n)
        remaining = np.ones(n, dtype=bool)
        max_attempts = 100
        attempt = 0

        while np.any(remaining) and attempt < max_attempts:
            count = np.sum(remaining)
            # Cauchy samples: tan(pi * (U - 0.5))
            u = np.random.uniform(0, 1, size=count)
            cauchy_samples = np.tan(np.pi * (u - 0.5))
            candidates = mu_f[remaining] + scale_f * cauchy_samples

            # Accept those > 0
            valid = candidates > 0
            idx_remaining = np.where(remaining)[0]
            accepted = idx_remaining[valid]
            f_values[accepted] = np.minimum(candidates[valid], 1.0)
            remaining[accepted] = False
            attempt += 1

        # Fallback for any remaining (shouldn't happen but be safe)
        if np.any(remaining):
            f_values[remaining] = np.clip(mu_f[remaining], 0.01, 1.0)

        # Normal distribution for CR with adaptive scale
        cr_values = np.random.normal(memory_cr[indices], scale_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        # For some fraction of population, use more explorative parameters
        # This helps escape local optima on multimodal functions
        explore_mask = np.random.random(n) < 0.1
        if np.any(explore_mask):
            n_explore = np.sum(explore_mask)
            # Use larger F for exploration
            f_values[explore_mask] = np.clip(
                np.random.uniform(0.5, 1.0, size=n_explore), 0.1, 1.0
            )
            # Use higher CR for exploration (more dimensions from donor)
            cr_values[explore_mask] = np.clip(
                np.random.uniform(0.8, 1.0, size=n_explore), 0.0, 1.0
            )

        return f_values, cr_values

    def _select_strategies_batch(self):
        """Select mutation strategy for each individual. Returns boolean mask."""
        # True = current-to-pbest/1, False = rand-to-pbest/1
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using mixed strategies for the whole population."""
        n = len(population)
        dim = self.dim

        # Sort by fitness for p-best selection
        sorted_idx = np.argsort(fitness)
        p = max(2, int(np.ceil(self.p_best_rate * n)))

        # Select p-best indices for each individual
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Select r1 != i
        r1 = self._random_indices_not_equal(n, np.arange(n))

        # Union of population and archive for r2
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Select r2 from union, r2 != i and r2 != r1
        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        # Strategy 1: current-to-pbest/1 (for strategy_mask == True)
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            mutants[idx1] = (
                population[idx1]
                + f_col[idx1] * (population[pbest_idx[idx1]] - population[idx1])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            )

        # Strategy 2: rand-to-pbest/1 (for strategy_mask == False)
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            mutants[idx2] = (
                population[r_base]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            )

        return mutants

    def _random_indices_not_equal(self, n, exclude_indices):
        """Generate random indices in [0, n) that differ from exclude_indices."""
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to whole population at once."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        # Ensure at least one dimension is from mutant
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds with midpoint reflection."""
        # For out-of-bounds, reflect towards the center
        too_low = trials < self.lb
        too_high = trials > self.ub
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """Greedy selection: replace if trial is better. Track successful params."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        improved = np.zeros(n, dtype=bool)
        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    improved[i] = True
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))
                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    def _update_memory(self, memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta):
        """Update parameter memory using weighted Lehmer mean."""
        if len(success_f) == 0:
            return memory_f, memory_cr, memory_idx

        weights = success_delta / (np.sum(success_delta) + 1e-30)

        # Weighted Lehmer mean for F
        lehmer_f = np.sum(weights * success_f ** 2) / (np.sum(weights * success_f) + 1e-30)
        memory_f[memory_idx] = lehmer_f

        # Weighted arithmetic mean for CR
        mean_cr = np.sum(weights * success_cr)
        memory_cr[memory_idx] = mean_cr

        memory_idx = (memory_idx + 1) % self.memory_size
        return memory_f, memory_cr, memory_idx

    def _detect_stagnation(self, prev_best_fitness):
        """Detect if the algorithm is stagnating."""
        improvement = prev_best_fitness - self.f_opt
        if improvement < 1e-12 * (abs(prev_best_fitness) + 1e-30):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        return self.stagnation_counter >= self.restart_threshold

    def _restart(self, func, stopping_condition, population, fitness):
        """Partial restart: keep best individuals, reinitialize rest with opposition."""
        self.stagnation_counter = 0
        n = len(population)
        keep_count = max(2, n // 5)

        sorted_idx = np.argsort(fitness)
        elite = population[sorted_idx[:keep_count]].copy()
        elite_fit = fitness[sorted_idx[:keep_count]].copy()

        reinit_count = n - keep_count
        # Generate new individuals using quasi-opposition
        new_pop = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

        # Quasi-opposition: reflect around population center
        center = np.mean(elite, axis=0)
        opp_pop = 2.0 * center - new_pop
        opp_pop = np.clip(opp_pop, self.lb, self.ub)

        # Choose random mix of new and opposition
        mix_mask = np.random.random(reinit_count) < 0.5
        combined_new = np.where(mix_mask[:, np.newaxis], opp_pop, new_pop)

        if stopping_condition():
            population = np.vstack([elite, combined_new])
            fitness_new = np.full(reinit_count, np.inf)
            return population, np.concatenate([elite_fit, fitness_new]), np.empty((0, self.dim)), np.full(self.memory_size, 0.5), np.full(self.memory_size, 0.5), 0

        combined_new = self._clip_to_bounds(combined_new)
        new_fitness = func(combined_new)
        if len(new_fitness) < len(combined_new):
            combined_new = combined_new[:len(new_fitness)]

        self._safe_update_best(combined_new, new_fitness)

        population = np.vstack([elite, combined_new[:len(new_fitness)]])
        fitness_all = np.concatenate([elite_fit, new_fitness])

        # Reset memory
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        archive = np.empty((0, self.dim))

        self.np_size = len(population)

        return population, fitness_all, archive, memory_f, memory_cr, memory_idx

    def _opposition_based_jump(self, population, fitness, func, stopping_condition):
        """Apply opposition-based learning to diversify population."""
        center = (self.lb + self.ub) / 2.0
        opp_population = 2.0 * center - population
        opp_population = self._clip_to_bounds(opp_population)

        # Add random perturbation
        noise = np.random.normal(0, 1.0, opp_population.shape)
        opp_population = self._clip_to_bounds(opp_population + noise)

        if stopping_condition():
            return population, fitness

        opp_fitness = func(opp_population)
        if len(opp_fitness) < len(opp_population):
            opp_population = opp_population[:len(opp_fitness)]

        self._safe_update_best(opp_population, opp_fitness)

        # Combine and select best
        n = min(len(population), len(opp_population))
        combined_pop = np.vstack([population[:n], opp_population[:n]])
        combined_fit = np.concatenate([fitness[:n], opp_fitness[:n]])

        sorted_idx = np.argsort(combined_fit)
        best_idx = sorted_idx[:len(population)]
        return combined_pop[best_idx], combined_fit[best_idx]

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance."""
        center = np.mean(population, axis=0)
        diffs = population - center
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _update_best(self, population, fitness):
        """Update global best from population."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fitness = fitness[valid]
        valid_pop = population[valid]
        best_idx = np.argmin(valid_fitness)
        if valid_fitness[best_idx] < self.f_opt:
            self.f_opt = valid_fitness[best_idx]
            self.x_opt = valid_pop[best_idx].copy()

    def _safe_update_best(self, candidates, fitnesses):
        """Safely update best considering possible NaN values."""
        if len(fitnesses) == 0:
            return
        valid = ~np.isnan(fitnesses)
        if not np.any(valid):
            return
        best_idx = np.nanargmin(fitnesses)
        if fitnesses[best_idx] < self.f_opt:
            self.f_opt = fitnesses[best_idx]
            self.x_opt = candidates[best_idx].copy()

```