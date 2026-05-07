Looking at the data, I see a critical pattern: the WORST unsolved tasks (16, 17, 20, 23, 19) all have errors > 10, suggesting they are highly multi-modal, deceptive, or ill-conditioned problems. The current Thompson Sampling with Beta distributions is exploiting operators that perform well on easier tasks but failing to explore solutions for the hardest ones.

The key insight is that the current selection mechanism is **greedy** — it samples from Beta distributions that favor previously-rewarded operators. For tasks stuck at error ~1e+1, we need **thorough exploration** across ALL operators, not exploitation of partial solutions.

**Idea: Deterministic Round-Robin with Diversity Boost**
Instead of Thompson Sampling (which is exploitative), use a deterministic scheduler that cycles through operators, ensuring every operator gets equal trials on each task. This guarantees the adaptive meta-algorithm downstream can evaluate each operator's true potential on each task, rather than being misled by premature convergence.

```python
def _select_operator_thompson(self):
    """Select operator using deterministic round-robin with periodic random perturbation."""
    # Deterministic cycle: ensure each operator gets equal opportunity per task
    cycle_position = self.generation % self.num_operators
    
    # Small probability of random perturbation to maintain some diversity
    if hasattr(self, 'total_selections'):
        self.total_selections += 1
    else:
        self.total_selections = 0
    
    # 10% chance of random selection to prevent complete determinism
    if np.random.random() < 0.1:
        self.current_operator = int(np.random.randint(0, self.num_operators))
    else:
        self.current_operator = int(cycle_position)
    
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```