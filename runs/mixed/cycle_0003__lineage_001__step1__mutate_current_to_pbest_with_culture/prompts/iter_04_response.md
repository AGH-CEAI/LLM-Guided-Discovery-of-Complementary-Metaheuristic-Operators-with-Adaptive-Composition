**Idea: Multi-Candidate Tournament Mutation with Archive Diversity**

Instead of a single mutation direction, generate 3 diverse mutation candidates (p_best-based, random-based, and archive-based) and select the best candidate per individual using fitness evaluation. This breaks the convergence trap on the worst tasks (12, 8, 21) which are likely stuck in local optima due to over-reliance on current population structure.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None], 
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1], 
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        # Candidate 1: Standard current-to-pbest
        cand1 = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])
        
        # Candidate 2: Random-base (exploration-focused, escapes local optima)
        idx_rand = np.random.randint(0, self.NP, (self.NP, 2))
        idx_rand[:, 1:] = np.where(idx_rand[:, 1:] == np.arange(self.NP)[:, None], 
                                   (idx_rand[:, 1:] + 1) % self.NP, idx_rand[:, 1:])
        r3, r4 = idx_rand[:, 0], idx_rand[:, 1]
        cand2 = population[r3] + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Candidate 3: Archive-influenced (diversity-driven)
        if hasattr(self, 'archive') and len(self.archive) > 0:
            archive_arr = np.array(self.archive)
            n_archive = min(len(archive_arr), self.NP)
            archive_sample_idx = np.random.randint(0, n_archive, size=self.NP)
            archive_sample = archive_arr[archive_sample_idx]
            F_archive = np.clip(self.F * 1.5, 0.3, 2.0)
            cand3 = population + F_archive * (archive_sample - population) + self.F * (population[r1] - population[r2])
        else:
            cand3 = cand1.copy()

        cand1 = np.clip(cand1, self.lower, self.upper)
        cand2 = np.clip(cand2, self.lower, self.upper)
        cand3 = np.clip(cand3, self.lower, self.upper)

        # Estimate candidate fitness using distance-based proxy to current best
        current_best_vec = population[np.argmin(fitness)]
        dist1 = np.linalg.norm(cand1 - current_best_vec, axis=1)
        dist2 = np.linalg.norm(cand2 - current_best_vec, axis=1)
        dist3 = np.linalg.norm(cand3 - current_best_vec, axis=1)

        # Also use directional component toward p_best
        dir1 = np.sum((p_best - population) * (cand1 - population), axis=1)
        dir2 = np.sum((p_best - population) * (cand2 - population), axis=1)
        dir3 = np.sum((p_best - population) * (cand3 - population), axis=1)

        # Composite score: prefer closer to p_best but with exploration bonus
        score1 = dist1 + 0.5 * dir1
        score2 = dist2 + 0.5 * dir2 - 0.3 * dist2  # exploration bonus
        score3 = dist3 + 0.5 * dir3

        # Per-individual tournament selection of best candidate
        donors = np.empty_like(population)
        for i in range(self.NP):
            scores = [score1[i], score2[i], score3[i]]
            best_cand_idx = np.argmin(scores)
            if best_cand_idx == 0:
                donors[i] = cand1[i]
            elif best_cand_idx == 1:
                donors[i] = cand2[i]
            else:
                donors[i] = cand3[i]

        # Light cultural memory influence (only if it shows promise)
        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            # Only apply if cultural direction is different from current
            cultural_diff = np.linalg.norm(cultural_contribution)
            if cultural_diff > 1e-6:
                blend_weight = 0.05
                donors = (1 - blend_weight) * donors + blend_weight * cultural_contribution

        return np.clip(donors, self.lower, self.upper)
```