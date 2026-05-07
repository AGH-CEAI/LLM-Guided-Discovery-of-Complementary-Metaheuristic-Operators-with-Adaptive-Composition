**Idea: Opposition-Based Niching Restart**
Replace the worst third with opposition-based and archive-diverse solutions far from the current population centroid, escaping local optima traps.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        worst_count = NP // 3
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        # Compute population statistics for opposition-based restart
        centroid = np.mean(population, axis=0)
        pop_range = np.array([self.lower, self.upper])
        
        # Generate diverse restart candidates
        candidates = []
        
        # 1. Opposition-based: mirror worst individuals through centroid
        for idx in worst_indices:
            worst = new_pop[idx]
            opp = centroid + (centroid - worst)
            opp = np.clip(opp, self.lower, self.upper)
            candidates.append(opp)
        
        # 2. Boundary-far: points near opposite bounds
        for _ in range(len(worst_indices) // 2):
            # Random point near opposite boundary from centroid
            boundary = np.where(centroid > 0, self.lower, self.upper)
            far_point = boundary + np.random.uniform(0, 0.3) * (pop_range[1] - pop_range[0]) * np.sign(boundary - centroid)
            far_point = np.clip(far_point, self.lower, self.upper)
            candidates.append(far_point)
        
        # 3. Archive-based diversity: sample from archive extremes
        if len(self.archive) > 0:
            arch_centroid = np.mean(self.archive, axis=0)
            for _ in range(len(worst_indices) // 2):
                # Point opposite to archive centroid
                opp_arch = 2 * centroid - arch_centroid + np.random.uniform(-20, 20, self.dim)
                opp_arch = np.clip(opp_arch, self.lower, self.upper)
                candidates.append(opp_arch)
        
        # 4. Fill remaining with fully random (escape any basin)
        while len(candidates) < len(worst_indices):
            candidates.append(np.random.uniform(self.lower, self.upper, self.dim))
        
        # Assign candidates to worst positions
        for i, idx in enumerate(worst_indices):
            if i < len(candidates):
                new_pop[idx] = candidates[i]
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```