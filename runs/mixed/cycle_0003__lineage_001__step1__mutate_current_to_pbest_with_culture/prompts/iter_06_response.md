**Idea: Stagnation-Triggered Adaptive Exploration with Elite Perturbation**

The worst unsolved tasks (12, 8, 21, 13) show errors stuck at 1e+01 to 1e+03 across ALL variants, meaning the population is trapped in local optima. The key insight is that the current operator uses fixed `self.F` and a fixed 15% cultural blend weight — this provides insufficient exploration push when the population converges. This variant detects stagnation and diversity collapse, then applies a fundamentally different mutation mode: aggressive elite-guided perturbation with dynamic scaling, breaking the population out of local basins.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
        # Detect stagnation and low diversity to trigger exploration mode
        diversity = np.std(population, axis=0).mean()
        is_stagnant = self.stagnation_count > 10
        is_low_diversity = diversity < 5.0
        
        # Adaptive F: increases with generation for more exploration, boosted further when stuck
        base_F = self.F
        gen_factor = min(self.generation / 200.0, 0.5)
        if is_stagnant or is_low_diversity:
            exploration_F = min(base_F * (2.0 + gen_factor), 2.0)
        else:
            exploration_F = base_F * (1.0 + gen_factor)
        
        # Select top performers for elite-guided perturbation
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        elite_indices = sorted_idx[:n_best]
        
        # Weighted elite selection (fitter = higher selection probability)
        elite_weights = 1.0 / (fitness[elite_indices] - fitness[elite_indices].min() + 1e-10)
        elite_weights = elite_weights / elite_weights.sum()
        
        # Choose p_best from weighted elite (not uniform random)
        p_best_idx = elite_indices[np.random.choice(n_best, size=self.NP, p=elite_weights)]
        p_best = population[p_best_idx]
        
        # Generate diverse random indices (avoid self)
        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]
        
        # Core DE mutation with adaptive F
        base = population + exploration_F * (p_best - population) + exploration_F * (population[r1] - population[r2])
        
        # Elite perturbation: add directed perturbation toward best individuals
        if is_stagnant or is_low_diversity:
            # Find global best for directed escape
            global_best_idx = sorted_idx[0]
            global_best = population[global_best_idx]
            
            # Perturbation toward global best with random component
            perturb_strength = min(0.3 * exploration_F, 0.5)
            direction = global_best - population
            direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
            direction_norm = np.where(direction_norm < 1e-10, 1.0, direction_norm)
            direction_normalized = direction / direction_norm
            
            # Random jump vectors for escape
            random_vectors = np.random.randn(self.NP, self.dim)
            random_norm = np.linalg.norm(random_vectors, axis=1, keepdims=True)
            random_norm = np.where(random_norm < 1e-10, 1.0, random_norm)
            random_normalized = random_vectors / random_norm
            
            # Blend: 60% directed toward best, 40% random escape
            escape_vectors = 0.6 * direction_normalized + 0.4 * random_normalized
            escape_vectors = escape_vectors / (np.linalg.norm(escape_vectors, axis=1, keepdims=True) + 1e-10)
            
            # Apply perturbation scaled by distance to global best
            distances_to_best = np.linalg.norm(population - global_best, axis=1, keepdims=True)
            jump_magnitude = perturb_strength * (1.0 + distances_to_best / (np.linalg.norm(population, axis=1, keepdims=True) + 1e-10))
            
            elite_perturbation = population + jump_magnitude * escape_vectors
            elite_perturbation = np.clip(elite_perturbation, self.lower, self.upper)
            
            # Blend base mutation with elite perturbation
            blend_weight = min(0.4 + 0.3 * gen_factor, 0.7)
            donors = (1 - blend_weight) * base + blend_weight * elite_perturbation
        else:
            donors = base
        
        return donors
```