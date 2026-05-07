**Idea: Worst-Focused Jump with Age-Scaled Explosion**
Uses the worst individual as mutation base (opposite of p-best), applies worst-to-top difference vectors, age-exponential F decay, and probabilistic full random restart to escape massive local optima traps blocking tasks 12, 8, 16, 21.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
    age = max(1, self.generation)
    F_age = min(2.5, 0.5 * np.exp(age / 50.0))
    
    sorted_idx = np.argsort(fitness)
    best_idx = sorted_idx[0]
    worst_idx = sorted_idx[-1]
    
    best_vec = population[best_idx]
    worst_vec = population[worst_idx]
    
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None], 
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1], 
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]
    
    jump_mask = np.random.rand(self.NP) < 0.25
    
    top_n = max(1, self.NP // 4)
    top_indices = sorted_idx[:top_n]
    
    if len(top_indices) > 1:
        top_a = top_indices[np.random.randint(0, len(top_indices), size=self.NP)]
        top_b = top_indices[np.random.randint(0, len(top_indices), size=self.NP)]
    else:
        top_a = np.full(self.NP, best_idx)
        top_b = np.full(self.NP, best_idx)
    
    top_a_vec = population[top_a]
    top_b_vec = population[top_b]
    
    top_diff = top_a_vec - top_b_vec
    
    worst_to_top = best_vec - worst_vec
    
    base = np.where(jump_mask[:, None], 
                    worst_vec + F_age * worst_to_top + F_age * (population[r1] - population[r2]),
                    worst_vec + F_age * top_diff)
    
    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        cultural_perturbation = cultural_contribution - np.mean(population, axis=0)
        base = base + 0.2 * cultural_perturbation
    
    if np.random.rand() < 0.15:
        base = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
    
    return base
```