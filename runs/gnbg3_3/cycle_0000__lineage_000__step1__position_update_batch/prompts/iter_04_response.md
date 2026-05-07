**Idea: Rank-weighted position update with fitness-guided momentum**

This approach modifies position updates using fitness percentile ranks. Top-ranked particles get conservative updates (exploitation), while lower-ranked particles get amplified updates (exploration). This is fundamentally different from geometry-based approaches because it uses *fitness rankings* to modulate movement, not distances or spatial relationships.

```python
def _position_update_batch(self):
    """Update positions using rank-based fitness weighting."""
    # Compute fitness percentile rank (0 = worst, 1 = best)
    fitness_ranks = 1.0 - (np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1))
    
    # Compute relative fitness improvement vs personal best
    improvement = (self.personal_best_fitness - self.current_fitness) / (
        np.abs(self.personal_best_fitness) + 1e-10
    )
    improvement = np.clip(improvement, -1.0, 1.0)
    
    # Rank-weighted momentum: successful particles reduce velocity, struggling ones amplify
    momentum_scale = 0.5 + 0.5 * fitness_ranks  # Range [0.5, 1.0]
    
    # Apply improvement signal as directional adjustment
    momentum_sign = np.where(improvement > 0, 1.0, 0.7)  # Keep direction if improving, dampen if worsening
    
    # Scaled position delta
    scaled_velocity = self.velocity * momentum_scale * momentum_sign[:, np.newaxis]
    
    # Fitness-guided perturbation for bottom 30% particles (rank < 0.3)
    exploration_mask = fitness_ranks < 0.3
    n_explore = np.sum(exploration_mask)
    
    if n_explore > 0:
        # Generate rank-based perturbations (no distance computation)
        explore_strength = (0.3 - fitness_ranks[exploration_mask]) / 0.3  # Stronger for worse ranks
        random_perturbation = np.random.uniform(-1, 1, (n_explore, self.dim)) * explore_strength[:, np.newaxis] * self.v_max * 0.5
        scaled_velocity[exploration_mask] += random_perturbation
    
    # Update positions
    new_population = self.population + scaled_velocity
    self.population = self._clip_to_bounds(new_population)
```