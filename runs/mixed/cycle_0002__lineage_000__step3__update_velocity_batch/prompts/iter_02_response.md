**Idea: Comprehensive Learning PSO Velocity Update**

Uses CLPSO's dimension-wise best selection to prevent premature convergence on multimodal tasks. Each dimension learns from a *different* particle's personal best, chosen by fitness ranking with probability proportional to improvement potential. This fundamentally differs from the original's single-attractor-per-particle approach, enabling escape from deceptive local optima that trap the original (especially visible on Tasks 12, 16, 8 with errors >100).

```python
def _update_velocity_batch(self, inertia):
    """CLPSO-style: dimension-wise best selection for enhanced exploration."""
    pop_size, dim = self.pop_size, self.dim
    
    # Compute ranks: lower rank = better fitness (0 = best)
    ranks = np.argsort(np.argsort(self.personal_best_fit))
    
    # Learning probability inversely correlated with rank (better particles more likely to teach)
    # Add small epsilon to avoid division by zero for worst particles
    top_k = max(2, pop_size // 4)
    probs = np.clip((top_k - ranks) / (top_k + 1), 0.1, 0.9)
    
    # Determine which dimensions learn from which particle
    learn_mask = np.random.uniform(0, 1, size=(pop_size, dim)) < probs[:, np.newaxis]
    
    # For each particle, select a random "teacher" from better-ranked particles for each dimension
    # Teachers are drawn from top 40% of population to ensure quality
    n_teachers = max(2, pop_size // 3)
    teacher_pool = np.argsort(self.personal_best_fit)[:n_teachers]
    
    # Select teacher for each (particle, dimension) where learn_mask is True
    teachers = np.random.choice(teacher_pool, size=(pop_size, dim))
    
    # Build dimension-wise best: use personal_best for non-learning dims, chosen teacher for learning dims
    dim_best = np.where(learn_mask, 
                        self.personal_best[teachers, np.arange(dim)],
                        self.personal_best)
    
    # CLPSO velocity update: V = w*V + rand * (dim_best - X)
    r = np.random.uniform(0, 1, size=(pop_size, dim))
    cognitive = self.c1 * r * (dim_best - self.population)
    
    # Add small Gaussian perturbation for diversity (especially helpful for fine convergence)
    mutation_scale = 0.01 * (self.upper - self.lower) * self.step_size
    perturbation = np.random.normal(0, mutation_scale, size=(pop_size, dim))
    
    self.velocity = inertia * self.velocity + cognitive + perturbation
    
    # Velocity clamping with adaptive bounds
    max_vel = (self.upper - self.lower) * 0.15
    self.velocity = np.clip(self.velocity, -max_vel, max_vel)
```