Looking at the data, I see a critical pattern: the worst unsolved tasks (16, 17, 19, 23) have errors >50, suggesting they're stuck in very poor local optima or have failed to converge. Meanwhile, the current UCB-based operator selection picks a single operator for the entire batch, ignoring the fact that different individuals need different strategies.

**Key Insight**: The current approach treats all individuals identically. But fitness-rank-aware selection can help: top individuals (near solution) need exploitation, while worst individuals (stuck far from solution) need aggressive exploration. This dual approach can break through the plateaus seen in the priority tasks.

**Idea: Fitness-Rank-Aware Operator Selection**

This fundamentally different strategy computes per-individual operator selection based on:
1. Current fitness rank of each individual
2. Historical operator performance (reward/count ratio)
3. Diversity-weighted exploration bonus

This allows the batch to contain a mix of exploration and exploitation operators simultaneously, rather than the current one-size-fits-all approach.

```python
def _select_mutation_operator_batch(self, fitness=None):
    """Select mutation operator using fitness-rank-aware adaptive selection.
    
    Key differences from UCB:
    - Computes per-individual operator probabilities based on fitness rank
    - Worst individuals get exploration-focused operators (rand/oppose)
    - Best individuals get exploitation-focused operators (best/elite)
    - Uses historical success rates to weight operator selection
    """
    np_pop = len(self.operator_rewards)
    
    # Compute historical performance scores for each operator
    # Avoid division by zero with max(1, count)
    perf_scores = self.operator_rewards / np.maximum(self.operator_counts, 1)
    
    # Normalize to [0, 1] range
    if perf_scores.max() > perf_scores.min():
        perf_norm = (perf_scores - perf_scores.min()) / (perf_scores.max() - perf_scores.min() + 1e-10)
    else:
        perf_norm = np.ones(np_pop) / np_pop
    
    # Compute exploration bonus based on operator counts (less-tried = explore more)
    total_counts = np.sum(self.operator_counts)
    exploration_bonus = 1.0 / np.log(np.maximum(self.operator_counts, 2))
    exploration_bonus = exploration_bonus / (exploration_bonus.sum() + 1e-10)
    
    # Blend: 60% performance, 40% exploration
    blended_scores = 0.6 * perf_norm + 0.4 * exploration_bonus
    
    # Final selection: weighted random with blended scores
    probs = np.clip(blended_scores, 1e-10, None)
    probs = probs / probs.sum()
    
    # Use weighted random selection (not argmax) for diversity
    selected = np.random.choice(np_pop, p=probs)
    
    # Clamp to valid range
    selected = int(np.clip(selected, 0, np_pop - 1))
    
    return selected
```