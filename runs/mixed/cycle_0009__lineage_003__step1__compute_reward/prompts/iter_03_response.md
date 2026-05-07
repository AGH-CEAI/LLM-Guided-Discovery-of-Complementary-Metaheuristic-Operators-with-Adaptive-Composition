**Idea: Top-Performer Weighted Reward with Diversity Bonus**

A fundamentally different reward strategy that prioritizes the best individual improvements (superlinear weighting) and adds a diversity bonus when trial solutions explore new regions. This targets the worst unsolved tasks (12, 16, 8, 21, 23, 13, 14, 20) which appear stuck in local optima—rewarding only top improvements (not averages) signals the operator to preserve and amplify breakthrough moves while diversity bonuses encourage escaping deceptive basins.

```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on top individual improvements with diversity bonus"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        
        # Identify successful improvements
        improved_mask = improvements > 0
        n_improved = int(np.sum(improved_mask))
        
        if n_improved == 0:
            reward = -0.01
        else:
            # Get top individual improvements only (not average)
            top_k = max(1, min(n_improved, max(1, self.NP // 4)))
            top_improvements = np.sort(improvements[improved_mask])[-top_k:]
            
            # Superlinear weighting: reward the best improvements disproportionately
            # This signals "breakthrough" moves vs marginal gains
            weights = np.linspace(1.0, 3.0, len(top_improvements))
            weighted_sum = float(np.sum(top_improvements * weights))
            
            # Normalize by a reference scale (median absolute fitness)
            ref_scale = float(np.median(np.abs(fitness[np.isfinite(fitness)])) + 1.0)
            reward = weighted_sum / (ref_scale * float(n_improved) + 1e-10)
            
            # Add diversity bonus: reward when trial solutions explore new regions
            # This helps escape local optima on multi-modal tasks
            if hasattr(self, 'cultural_memory') and self.generation > 0:
                diversity_scores = []
                for i in range(len(trial_fitness)):
                    if trial_fitness[i] < fitness[i]:  # Only for improving solutions
                        dist_to_pop = np.min(np.linalg.norm(
                            trial_fitness[i] - fitness, axis=1
                        ) + 1e-10) if len(fitness) > 1 else 1.0
                        diversity_scores.append(dist_to_pop)
                
                if len(diversity_scores) > 0:
                    max_dist = max(np.max(diversity_scores), 1e-10)
                    diversity_bonus = float(np.mean(diversity_scores) / max_dist) * 0.2
                    reward += diversity_bonus

        return float(np.clip(reward, -1.0, 1.0))
```